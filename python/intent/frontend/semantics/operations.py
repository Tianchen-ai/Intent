from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from enum import IntEnum


class OperationKind(Enum):
    CONSTANT = "constant"
    DIM = "dim"

    DOMAIN = "domain"
    DOMAIN_PRODUCT = "domain_product"
    SUBREGION = "subregion"
    INDICES = "indices"
    REGION_END = "region_end"
    ASSUME_IN_BOUNDS = "assume_in_bounds"

    PARALLEL = "parallel"
    IF = "if"
    FOR = "for"
    WHILE = "while"
    YIELD = "yield"
    CONDITION = "condition"

    VIEW_LOAD = "view_load"
    VIEW_STORE = "view_store"
    RESHAPE = "reshape"
    JOIN = "join"
    TRANSPOSE = "transpose"
    BROADCAST = "broadcast"
    FULL = "full"
    MAKE_TUPLE = "make_tuple"
    MAKE_RECORD = "make_record"
    EXTRACT = "extract"

    UNARY = "unary"
    BINARY = "binary"
    COMPARE = "compare"
    SELECT = "select"
    CAST = "cast"
    BITCAST = "bitcast"
    MASK = "mask"

    REDUCE = "reduce"
    SCAN = "scan"
    REGION_FOLD = "region_fold"
    REGION_SCAN = "region_scan"
    CONTRACT = "contract"
    QUANTIZE = "quantize"
    QUANTIZED_DOT = "quantized_dot"
    SCALED_CONTRACT = "scaled_contract"
    SPARSE_CONTRACT = "sparse_contract"
    HISTOGRAM = "histogram"

    GATHER = "gather"
    SCATTER_UNIQUE = "scatter_unique"
    SCATTER_REDUCE = "scatter_reduce"

    BUFFER = "buffer"
    BUFFER_LOAD = "buffer_load"
    BUFFER_STORE = "buffer_store"
    ATOMIC_LOAD = "atomic_load"
    ATOMIC_STORE = "atomic_store"
    ATOMIC_RMW = "atomic_rmw"
    ATOMIC_COMPARE_EXCHANGE = "atomic_compare_exchange"
    RANDOM_BITS = "random_bits"

    RETURN = "return"


TERMINATORS = {
    OperationKind.YIELD,
    OperationKind.CONDITION,
    OperationKind.RETURN,
}

REGION_OPS = {
    OperationKind.PARALLEL,
    OperationKind.IF,
    OperationKind.FOR,
    OperationKind.WHILE,
    OperationKind.REDUCE,
    OperationKind.SCAN,
    OperationKind.REGION_FOLD,
    OperationKind.REGION_SCAN,
    OperationKind.SCATTER_REDUCE,
}

STRUCTURED_OPS = {
    OperationKind.REDUCE,
    OperationKind.SCAN,
    OperationKind.REGION_FOLD,
    OperationKind.REGION_SCAN,
    OperationKind.CONTRACT,
    OperationKind.QUANTIZE,
    OperationKind.QUANTIZED_DOT,
    OperationKind.SCALED_CONTRACT,
    OperationKind.SPARSE_CONTRACT,
    OperationKind.PARALLEL,
    OperationKind.HISTOGRAM,
}

EFFECTFUL_OPS = {
    OperationKind.VIEW_LOAD,
    OperationKind.VIEW_STORE,
    OperationKind.SCATTER_UNIQUE,
    OperationKind.SCATTER_REDUCE,
    OperationKind.BUFFER_LOAD,
    OperationKind.BUFFER_STORE,
    OperationKind.ATOMIC_LOAD,
    OperationKind.ATOMIC_STORE,
    OperationKind.ATOMIC_RMW,
    OperationKind.ATOMIC_COMPARE_EXCHANGE,
}


class UnaryOperator(IntEnum):
    NEGATE = 0
    NOT = 1
    EXP = 2
    EXP2 = 3
    LOG = 4
    SIN = 5
    COS = 6
    FLOOR = 7
    ERF = 8
    RSQRT = 9
    SIGMOID = 10
    TANH = 11
    ABS = 12
    SQRT = 13
    LGAMMA = 14
    LOG1P = 15
    ERFC = 16
    I0 = 17
    ASIN = 18


class BinaryOperator(IntEnum):
    ADD = 0
    SUBTRACT = 1
    MULTIPLY = 2
    TRUE_DIVIDE = 3
    FLOOR_DIVIDE = 4
    REMAINDER = 5
    POWER = 6
    MAXIMUM = 7
    MINIMUM = 8
    MAXIMUM_NUM = 9
    MINIMUM_NUM = 10
    LOGICAL_AND = 11
    LOGICAL_OR = 12
    BITWISE_AND = 13
    BITWISE_OR = 14
    BITWISE_XOR = 15
    LEFT_SHIFT = 16
    RIGHT_SHIFT = 17


class ComparePredicate(IntEnum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5


class AtomicOrdering(IntEnum):
    RELAXED = 0
    ACQUIRE = 1
    RELEASE = 2
    ACQ_REL = 3


class AtomicRMWKind(IntEnum):
    EXCHANGE = 0
    ADD = 1
    MAX = 2
    MIN = 3
    AND = 4
    OR = 5
    XOR = 6


class ScaledFormatKind(IntEnum):
    E2M1 = 0
    E4M3 = 1
    E8M0 = 2


class QuantFormatKind(IntEnum):
    Q4_K = 0
    Q8_K = 1


class IndexTermKind(IntEnum):
    FULL_SLICE = 0
    NEW_AXIS = 1
    STATIC_INDEX = 2
    VALUE_INDEX = 3
    REGION_INDEX = 4
    SLICE = 5


class ShapeExprKind(IntEnum):
    STATIC = 0
    SSA_EXTENT = 1
    INFERRED = 2


@dataclass(frozen=True, slots=True)
class ShapeExpr:
    kind: ShapeExprKind
    dimension: int
    payload: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ShapeExprKind):
            raise TypeError("shape expression kind must be a ShapeExprKind")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.dimension, self.payload)
        ):
            raise TypeError("shape expression fields must be integers")
        if self.kind is ShapeExprKind.STATIC:
            if self.dimension <= 0 or self.payload < 0:
                raise ValueError(
                    "static shape expression requires a positive identity and non-negative extent"
                )
        elif self.dimension <= 0:
            raise ValueError("dynamic shape expression requires a positive identity")
        elif self.kind is ShapeExprKind.SSA_EXTENT and self.payload < 0:
            raise ValueError(
                "SSA shape expression requires a non-negative operand position"
            )
        elif self.kind is ShapeExprKind.INFERRED and self.payload != -1:
            raise ValueError("inferred shape expression uses payload -1")


@dataclass(frozen=True, slots=True)
class ShapeRelation:
    axes: tuple[ShapeExpr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        if any(not isinstance(axis, ShapeExpr) for axis in self.axes):
            raise TypeError("shape relation axes must be ShapeExpr values")


@dataclass(frozen=True, slots=True)
class IndexTerm:
    kind: IndexTermKind
    operand_positions: tuple[int | None, ...] = ()
    static_values: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operand_positions", tuple(self.operand_positions))
        object.__setattr__(self, "static_values", tuple(self.static_values))
        if any(
            position is not None
            and (isinstance(position, bool) or not isinstance(position, int) or position < 0)
            for position in self.operand_positions
        ):
            raise ValueError("index operand positions must be non-negative")
        if self.kind in (IndexTermKind.FULL_SLICE, IndexTermKind.NEW_AXIS):
            if self.operand_positions or self.static_values:
                raise ValueError(f"{self.kind.value} index term carries no payload")
        elif self.kind is IndexTermKind.STATIC_INDEX:
            if self.operand_positions or len(self.static_values) != 1:
                raise ValueError("static index requires exactly one static value")
            value = self.static_values[0]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("static index must be an integer")
        elif self.kind in (IndexTermKind.VALUE_INDEX, IndexTermKind.REGION_INDEX):
            if (
                len(self.operand_positions) != 1
                or self.operand_positions[0] is None
                or self.static_values
            ):
                raise ValueError(f"{self.kind.value} requires exactly one operand")
        elif self.kind is IndexTermKind.SLICE:
            if len(self.operand_positions) != 3 or len(self.static_values) != 3:
                raise ValueError("slice index must encode start/stop/step slots")
            for operand_position, static_value in zip(
                self.operand_positions, self.static_values
            ):
                if operand_position is not None and static_value is not None:
                    raise ValueError("slice slot cannot be both dynamic and static")
                if static_value is not None and (
                    isinstance(static_value, bool)
                    or not isinstance(static_value, int)
                ):
                    raise TypeError("static slice values must be integers or None")
            if self.static_values[2] == 0:
                raise ValueError("slice step cannot be zero")


@dataclass(frozen=True, slots=True)
class IndexRelation:
    source_rank: int
    result_rank: int
    result_dimensions: tuple[int, ...]
    terms: tuple[IndexTerm, ...]

    def __post_init__(self) -> None:
        for name, rank in (("source", self.source_rank), ("result", self.result_rank)):
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError(f"index relation {name} rank must be an integer")
            if rank < 0:
                raise ValueError(f"index relation {name} rank must be non-negative")
        object.__setattr__(self, "result_dimensions", tuple(self.result_dimensions))
        if len(self.result_dimensions) != self.result_rank or any(
            isinstance(identity, bool)
            or not isinstance(identity, int)
            or identity <= 0
            for identity in self.result_dimensions
        ):
            raise ValueError(
                "index relation requires one positive dimension identity per result axis"
            )
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms and (self.source_rank or self.result_rank):
            raise ValueError("non-scalar index relation requires at least one term")
        if any(not isinstance(term, IndexTerm) for term in self.terms):
            raise TypeError("index relation terms must be IndexTerm values")
        consuming = sum(term.kind is not IndexTermKind.NEW_AXIS for term in self.terms)
        if consuming != self.source_rank:
            raise ValueError("index relation must consume every source axis exactly once")
