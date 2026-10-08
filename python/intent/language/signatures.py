from __future__ import annotations

from inspect import Parameter, Signature


def _signature(
    positional: tuple[str, ...],
    *,
    optional_positional: tuple[tuple[str, object], ...] = (),
    required: tuple[str, ...] = (),
    defaults: tuple[tuple[str, object], ...] = (),
    return_annotation: object = Signature.empty,
) -> Signature:
    return Signature(
        [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in positional]
        + [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, default=value)
           for name, value in optional_positional]
        + [Parameter(name, Parameter.KEYWORD_ONLY) for name in required]
        + [
            Parameter(name, Parameter.KEYWORD_ONLY, default=value)
            for name, value in defaults
        ],
        return_annotation=return_annotation,
    )


INTRINSIC_SIGNATURES = {
    "domain": _signature(("start", "stop"), optional_positional=(("step", 1),),
                         return_annotation="logical domain descriptor, not a tensor"),
    "parallel": _signature(("source",), return_annotation="parallel iteration"),
    "indices": _signature(("region",), return_annotation="index tensor or tuple of index tensors"),
    "end": _signature(("region",)),
    "assume_in_bounds": _signature(("index", "view", "axis")),
    "reshape": _signature(("value", "shape"),
                          return_annotation="tensor of requested shape; tensor input required; row-major element order preserved"),
    "join": _signature(("lhs", "rhs")),
    "transpose": _signature(("value",), optional_positional=(("permutation", None),),
                            return_annotation="tensor with permuted axes; omitted permutation reverses all axes"),
    "full": _signature(("shape", "fill", "dtype"), return_annotation="immutable tensor"),
    "zeros": _signature(("shape", "dtype"), return_annotation="immutable tensor"),
    "buffer": _signature(("shape", "dtype"), optional_positional=(("init", None),),
                         return_annotation="mutable buffer"),
    "record": Signature([Parameter("fields", Parameter.VAR_KEYWORD)]),
    "cast": _signature(("value", "dtype")),
    "bitcast": _signature(("value", "dtype")),
    "mask": _signature(("value", "valid", "fill")),
    "select": _signature(("condition", "true_value", "false_value")),
    **{name: _signature(("value",)) for name in
       ("exp", "log", "log1p", "lgamma", "sin", "asin", "cos", "floor", "erf", "erfc", "i0", "rsqrt", "sqrt", "sigmoid", "abs")},
    **{name: _signature(("lhs", "rhs")) for name in
       ("add", "maximum", "minimum", "maximum_num", "minimum_num")},
    "reduce": _signature(("value", "axis", "identity", "combine"),
                         optional_positional=(("combine_operands", ()), ("acc_dtype", None))),
    "arg_reduce.max": _signature(
        ("value", "axis"), optional_positional=(("acc_dtype", None),),
        return_annotation="(value, I.index indices); reduced axes removed; full reduction gives scalars",
    ),
    "contract": _signature(("lhs", "rhs", "reduce", "acc_dtype"), optional_positional=(("batch", ()),)),
    "region_fold": _signature(("source", "axis", "summarize", "combine", "identity"),
                              optional_positional=(("operands", ()),)),
    "region_scan": _signature(("source", "axis", "summarize", "combine", "identity", "initial_state", "apply", "emit"),
                              optional_positional=(("operands", ()),)),
    "histogram": _signature(("values", "bins", "valid", "count_dtype")),
    "gather": _signature(("source", "index"), optional_positional=(("valid", True), ("fill", 0))),
    "store": _signature(("target", "index", "value")),
    "mutable_load": _signature(("target", "index")),
    "random.bits": _signature(("seed", "logical_counter")),
    "random.uniform": _signature(("seed", "logical_counter", "dtype")),
    "quantize": _signature(("values",), required=("format",)),
    "quantized_dot": _signature(
        ("lhs", "rhs"), required=("lhs_format", "rhs_format", "acc_dtype")
    ),
    "fdiv": _signature(
        ("lhs", "rhs"), defaults=(("approximate", False), ("flush_to_zero", False))
    ),
    "exp2": _signature(
        ("value",), defaults=(("approximate", False), ("flush_to_zero", False))
    ),
    "tanh": _signature(("value",), defaults=(("approximate", False),)),
    "dot": _signature(("lhs", "rhs"), required=("acc_dtype",)),
    "matvec": _signature(
        ("matrix", "vector"), required=("acc_dtype",), defaults=(("transpose", False),)
    ),
    "vecmat": _signature(
        ("vector", "matrix"), required=("acc_dtype",), defaults=(("transpose", False),)
    ),
    "matmul": _signature(
        ("lhs", "rhs"),
        required=("acc_dtype",),
        defaults=(("transpose_lhs", False), ("transpose_rhs", False)),
    ),
    "outer": _signature(("lhs", "rhs")),
    "reduce.sum": _signature(
        ("value",), required=("axis",), defaults=(("acc_dtype", None),),
        return_annotation="scalar when all axes are reduced; otherwise tensor with reduced axes removed",
    ),
    "reduce.max": _signature(
        ("value",), required=("axis",), defaults=(("acc_dtype", None),),
        return_annotation="scalar when all axes are reduced; otherwise tensor with reduced axes removed",
    ),
    "reduce.any": _signature(("value",), required=("axis",)),
    "reduce.all": _signature(("value",), required=("axis",)),
    "cumsum": _signature(
        ("value",),
        required=("axis",),
        defaults=(("inclusive", True), ("reverse", False), ("acc_dtype", None)),
    ),
    "cummax": _signature(
        ("value",),
        required=("axis",),
        defaults=(("inclusive", True), ("reverse", False), ("acc_dtype", None)),
    ),
    "scaled_matmul": _signature(
        ("lhs", "lhs_scale", "rhs", "rhs_scale"),
        required=("lhs_format", "rhs_format", "group_size", "acc_dtype"),
    ),
    "sparse_matmul": _signature(
        ("compressed", "metadata", "rhs"), required=("format", "acc_dtype")
    ),
}
