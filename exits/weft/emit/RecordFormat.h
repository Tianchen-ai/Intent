#pragma once
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace CI_UNIT_NAMESPACE {
// Weft consumes the physical fields and the logical values of a K record
// independently. The two six-bit fields share the twelve scale-header bytes.
inline void declareRecord(llvm::raw_ostream &s, bool weight) {
  struct Field {
    std::string name, type, layout;
    int extent, offset, bits;
  };
  const std::string natural = "[{kind=\"natural\"}]";
  auto joined = [](int role) {
    return "[{kind=\"joined\",fields=2:i64,low_bits=4:i64,order=\"lo_first\","
           "size=4:i64,role=" +
           std::to_string(role) + ":i64}]";
  };
  const std::string nibble = "[{kind=\"grouped\",size=64:i64},{kind="
                             "\"layered\",size=32:i64,order=\"lo_first\"}]";
  std::vector<Field> fields =
      weight ? std::vector<Field>{{"d", "f16", natural, 1, 0, 16},
                                  {"dmin", "f16", natural, 1, 16, 16},
                                  {"sc", "ui6", joined(0), 8, 32, 96},
                                  {"m", "ui6", joined(1), 8, 32, 96},
                                  {"q", "ui4", nibble, 256, 128, 1024}}
             : std::vector<Field>{{"ds", "f32", natural, 1, 0, 32},
                                  {"q", "si8", natural, 256, 32, 2048},
                                  {"bsum", "si16", natural, 16, 2080, 256}};
  std::vector<std::string> names, types, layouts, shapes, offsets, bits;
  for (auto &f : fields) {
    names.push_back("\"" + f.name + "\"");
    types.push_back(f.type);
    layouts.push_back(f.layout);
    shapes.push_back(f.extent == 1
                         ? "array<i64>"
                         : "array<i64:" + std::to_string(f.extent) + ">");
    offsets.push_back(std::to_string(f.offset));
    bits.push_back(std::to_string(f.bits));
  }
  auto name = weight ? "Q4_K" : "Q8_K";
  s << "\"weft_kernel.encoding_decl\"() <{sym_name=\"" << name
    << "\",kind=\"base\",layout_identity=\"" << name
    << "\",bit_order=\"lsb_first\",byte_order=\"little\",alignment="
    << (weight ? 2 : 4)
    << ":i64,elements=256:i64,storage_bits=" << (weight ? 144 : 292) * 8
    << ":i64,field_names=[" << llvm::join(names, ",") << "],field_types=["
    << llvm::join(types, ",") << "],field_shapes=[" << llvm::join(shapes, ",")
    << "],field_layouts=[" << llvm::join(layouts, ",")
    << "],field_bit_offsets=array<i64:" << llvm::join(offsets, ",")
    << ">,field_storage_bits=array<i64:" << llvm::join(bits, ",")
    << ">,padding=array<i64>}>:()->()\n";
}
} // namespace CI_UNIT_NAMESPACE
