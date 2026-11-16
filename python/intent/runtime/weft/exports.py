"""Whole-invocation entry and autotune measurement over the declared native ABI."""


def native_exports(contract):
    signature, arguments = ["void *context"], ["context"]
    carriers = {"bool": "_Bool", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t",
                "i64": "int64_t", "f32": "float", "f64": "double"}
    for slot in contract.abi.slots:
        signature.append(f"{'void *' if slot.role == 'pointer' else carriers[slot.carrier] + ' '}{slot.name}")
        arguments.append(slot.name)
    declaration, invocation = ", ".join(signature), ", ".join(arguments)
    mutable = contract.abi.trial_regions()
    sections = ["#include <stdint.h>\n#include <time.h>\n#include <fenv.h>\n"
                "#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n"]
    for candidate in contract.facts.candidates:
        entry = candidate.entry
        sections.append(f"extern void {entry}({declaration});\n"
                        f"void {entry}_invoke({declaration}) {{\n"
                        "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
                        f"  {entry}({invocation});\n  fesetenv(&saved);\n}}\n"
                        f"double {entry}_benchmark({declaration}, int64_t repetitions) {{\n"
                        "  if (repetitions <= 0) return NAN;\n")
        if not mutable:
            sections.append("  struct timespec begin, end;\n"
                "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
                "  clock_gettime(CLOCK_MONOTONIC, &begin);\n"
                "  for (int64_t i = 0; i < repetitions; ++i)\n"
                f"    {entry}({invocation});\n"
                "  clock_gettime(CLOCK_MONOTONIC, &end); fesetenv(&saved);\n"
                "  return ((end.tv_sec-begin.tv_sec)*1.e3+(end.tv_nsec-begin.tv_nsec)*1.e-6)/repetitions;\n}\n")
            continue
        for region in mutable:
            name = region.pointer.name
            sections.append(f"  size_t bytes_{name} = {region.element_bytes};\n")
            for extent in region.extents:
                sections.append(f"  if ({extent.name}<0 || (bytes_{name} && (uint64_t){extent.name}>SIZE_MAX/bytes_{name})) return NAN;\n"
                                f"  bytes_{name} *= (size_t){extent.name};\n")
        for region in mutable:
            sections.append(f"  void *saved_{region.pointer.name} = NULL;\n")
        for region in mutable:
            name = region.pointer.name
            sections.append(f"  if (bytes_{name}) {{ saved_{name}=malloc(bytes_{name});\n"
                            f"    if (!saved_{name}) goto allocation_failed;\n"
                            f"    memcpy(saved_{name}, {name}, bytes_{name}); }}\n")
        sections.append("  double elapsed=0.0; struct timespec begin,end;\n"
                        "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
                        "  for (int64_t i=0; i<repetitions; ++i) {\n")
        for region in mutable:
            name = region.pointer.name
            sections.append(f"    if (bytes_{name}) memcpy({name},saved_{name},bytes_{name});\n")
        sections.append(f"    clock_gettime(CLOCK_MONOTONIC,&begin); {entry}({invocation});\n"
                        "    clock_gettime(CLOCK_MONOTONIC,&end);\n"
                        "    elapsed+=(end.tv_sec-begin.tv_sec)*1.e3+(end.tv_nsec-begin.tv_nsec)*1.e-6;\n  }\n")
        for region in mutable:
            name = region.pointer.name
            sections.append(f"  if (bytes_{name}) memcpy({name},saved_{name},bytes_{name}); free(saved_{name});\n")
        sections.append("  fesetenv(&saved); return elapsed/repetitions;\nallocation_failed:\n")
        for region in mutable:
            sections.append(f"  free(saved_{region.pointer.name});\n")
        sections.append("  return NAN;\n}\n")
    return "".join(sections)
