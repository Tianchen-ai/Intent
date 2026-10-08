from __future__ import annotations

import math
from dataclasses import dataclass

from intent.diagnostics import LanguageUseError

from .signatures import INTRINSIC_SIGNATURES


@dataclass(frozen=True, slots=True)
class Intrinsic:
    name: str

    def __getattr__(self, member: str):
        if member == "__signature__":
            return INTRINSIC_SIGNATURES.get(self.name)
        raise AttributeError(member)

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"I.{self.name} is only valid inside an @intent.kernel or @intent.fn body"
        )

    def __repr__(self) -> str:
        return f"I.{self.name}"


@dataclass(frozen=True, slots=True)
class IntrinsicNamespace:
    name: str
    members: tuple[str, ...]

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"I.{self.name} is only valid inside an @intent.kernel or @intent.fn body"
        )

    def __getattr__(self, member: str) -> Intrinsic:
        if member not in self.members:
            raise AttributeError(member)
        return Intrinsic(f"{self.name}.{member}")

    def __repr__(self) -> str:
        return f"I.{self.name}"


@dataclass(frozen=True, slots=True)
class ScaledFormat:
    name: str

    def __repr__(self) -> str:
        return f"I.{self.name}"


@dataclass(frozen=True, slots=True)
class QuantFormat:
    name: str


@dataclass(frozen=True, slots=True)
class QuantFormats:
    q4_k: QuantFormat = QuantFormat("q4_k")
    q8_k: QuantFormat = QuantFormat("q8_k")


quant = QuantFormats()
quantize = Intrinsic("quantize")
quantized_dot = Intrinsic("quantized_dot")


domain = Intrinsic("domain")
parallel = Intrinsic("parallel")
indices = Intrinsic("indices")
end = Intrinsic("end")
assume_in_bounds = Intrinsic("assume_in_bounds")

reshape = Intrinsic("reshape")
join = Intrinsic("join")
transpose = Intrinsic("transpose")
full = Intrinsic("full")
zeros = Intrinsic("zeros")
record = Intrinsic("record")
cast = Intrinsic("cast")
bitcast = Intrinsic("bitcast")
mask = Intrinsic("mask")
select = Intrinsic("select")

exp = Intrinsic("exp")
exp2 = Intrinsic("exp2")
log = Intrinsic("log")
log1p = Intrinsic("log1p")
lgamma = Intrinsic("lgamma")
sin = Intrinsic("sin")
asin = Intrinsic("asin")
cos = Intrinsic("cos")
floor = Intrinsic("floor")
erf = Intrinsic("erf")
erfc = Intrinsic("erfc")
i0 = Intrinsic("i0")
rsqrt = Intrinsic("rsqrt")
sqrt = Intrinsic("sqrt")
sigmoid = Intrinsic("sigmoid")
tanh = Intrinsic("tanh")
abs = Intrinsic("abs")
maximum = Intrinsic("maximum")
minimum = Intrinsic("minimum")
maximum_num = Intrinsic("maximum_num")
minimum_num = Intrinsic("minimum_num")
add = Intrinsic("add")
fdiv = Intrinsic("fdiv")

reduce = IntrinsicNamespace("reduce", ("max", "sum", "any", "all"))
arg_reduce = IntrinsicNamespace("arg_reduce", ("max",))
scan = Intrinsic("scan")
region_fold = Intrinsic("region_fold")
region_scan = Intrinsic("region_scan")
contract = Intrinsic("contract")
dot = Intrinsic("dot")
matvec = Intrinsic("matvec")
vecmat = Intrinsic("vecmat")
matmul = Intrinsic("matmul")
outer = Intrinsic("outer")
cumsum = Intrinsic("cumsum")
cummax = Intrinsic("cummax")
scaled_contract = Intrinsic("scaled_contract")
scaled_matmul = Intrinsic("scaled_matmul")
sparse_contract = Intrinsic("sparse_contract")
sparse_matmul = Intrinsic("sparse_matmul")
sparse_contract_2to4 = Intrinsic("sparse_contract_2to4")
histogram = Intrinsic("histogram")

e2m1 = ScaledFormat("e2m1")
e4m3 = ScaledFormat("e4m3")
e8m0 = ScaledFormat("e8m0")
sparse = IntrinsicNamespace("sparse", ("one_of_two", "two_of_four"))

gather = Intrinsic("gather")
scatter_unique = Intrinsic("scatter_unique")
scatter_reduce = Intrinsic("scatter_reduce")

buffer = Intrinsic("buffer")
store = Intrinsic("store")
mutable_load = Intrinsic("mutable_load")
atomic = IntrinsicNamespace(
    "atomic",
    (
        "load",
        "store",
        "exchange",
        "add",
        "max",
        "min",
        "and_",
        "or_",
        "xor",
        "compare_exchange",
    ),
)

ragged = Intrinsic("ragged")
members = Intrinsic("members")
random = IntrinsicNamespace("random", ("bits", "uniform"))

inf = math.inf
LOG2E = math.log2(math.e)


INTRINSICS = {
    intrinsic.name: intrinsic
    for intrinsic in (
        domain,
        parallel,
        indices,
        end,
        assume_in_bounds,
        reshape,
        join,
        transpose,
        full,
        zeros,
        record,
        cast,
        bitcast,
        mask,
        select,
        exp,
        exp2,
        log,
        log1p,
        lgamma,
        sin,
        asin,
        cos,
        floor,
        erf,
        erfc,
        i0,
        rsqrt,
        sqrt,
        sigmoid,
        tanh,
        abs,
        maximum,
        minimum,
        maximum_num,
        minimum_num,
        add,
        fdiv,
        reduce,
        reduce.max,
        reduce.sum,
        reduce.any,
        reduce.all,
        arg_reduce,
        arg_reduce.max,
        scan,
        region_fold,
        region_scan,
        contract,
        quantize,
        quantized_dot,
        dot,
        matvec,
        vecmat,
        matmul,
        outer,
        cumsum,
        cummax,
        scaled_contract,
        scaled_matmul,
        sparse_contract,
        sparse_matmul,
        sparse_contract_2to4,
        histogram,
        sparse,
        sparse.one_of_two,
        sparse.two_of_four,
        gather,
        scatter_unique,
        scatter_reduce,
        buffer,
        store,
        mutable_load,
        atomic,
        atomic.load,
        atomic.store,
        atomic.exchange,
        atomic.add,
        atomic.max,
        atomic.min,
        atomic.and_,
        atomic.or_,
        atomic.xor,
        atomic.compare_exchange,
        ragged,
        members,
        random,
        random.bits,
        random.uniform,
    )
}
