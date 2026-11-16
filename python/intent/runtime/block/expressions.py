from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "floor_div": "//"}


def _integer(value):
    if type(value) is not int:
        raise TypeError("physical quantity must be an integer")
    if not -(1 << 63) <= value < 1 << 63:
        raise OverflowError("physical quantity exceeds signed 64-bit range")
    return value


def _select(condition, yes, no):
    # Function arguments are evaluated eagerly, as in the native physical
    # expression evaluator. Only requirement activation skips an expression.
    return yes if condition else no


def _source(expression: dict, references: set[int | str], *, checked: bool = False) -> str:
    def result(source):
        return f"_integer({source})" if checked else source

    kind = expression["kind"]
    if kind == "constant":
        value = expression["value"]
        if type(value) is not int:
            raise TypeError("physical constant must be an integer")
        return result(repr(value))
    if kind == "argument":
        identity = expression["id"]
        if type(identity) is not int or not 0 < identity < 1 << 64:
            raise TypeError("physical argument reference must be a positive unsigned 64-bit integer")
        references.add(identity)
        return result(f"values[{identity}]")
    if kind == "parameter":
        symbol = expression["symbol"]
        if not isinstance(symbol, str) or not symbol:
            raise TypeError("physical parameter reference must name a declared parameter")
        references.add(symbol)
        return result(f"values[{symbol!r}]")
    operands = tuple(_source(operand, references, checked=checked) for operand in expression["operands"])
    if kind in _BINARY:
        left, right = operands
        return result(f"({left} {_BINARY[kind]} {right})")
    if kind == "ceil_div":
        left, right = operands
        return result(f"(-(-{left} // {right}))")
    if kind in {"min", "max"}:
        left, right = operands
        return result(f"{kind}({left}, {right})")
    if kind == "select":
        condition, yes, no = operands
        return result(f"_select({condition}, {yes}, {no})") if checked else f"({yes} if {condition} else {no})"
    if kind == "next_power_of_two":
        (value,) = operands
        return result(f"(1 << (max({value}, 1) - 1).bit_length())")
    raise NotImplementedError(f"unknown physical expression kind: {kind}")


def _compile(source):
    return eval(compile(f"lambda values: {source}", "<intent physical expression>", "eval"),
                {"__builtins__": {}, "min": min, "max": max,
                 "_integer": _integer, "_select": _select})


def _nonnegative_bound(expression, exact):
    kind = expression["kind"]
    children = ()
    if kind in {"add", "multiply"}:
        children = tuple(_nonnegative_bound(child, _compile(_source(child, set(), checked=True)))
                         for child in expression["operands"])

    def evaluate(values, limit):
        try:
            quantity = exact(values)
        except (ArithmeticError, KeyError):
            if not children:
                return None
            left, right = (child(values, limit) for child in children)
            if left is None or right is None:
                return None
            quantity = left + right if kind == "add" else left * right
        return None if quantity < 0 else min(limit + 1, quantity)

    return evaluate


@dataclass(frozen=True, slots=True)
class Expression:
    """A compiler expression bound once, with live invocation values on use."""

    evaluate: Callable[[Mapping[int | str, object]], object]
    references: frozenset[int | str]
    nonnegative_bound: Callable[[Mapping[int | str, object], int], int | None] | None = None

    @classmethod
    def read(cls, expression: dict, *, checked: bool = False) -> Expression:
        references: set[int | str] = set()
        function = _compile(_source(expression, references, checked=checked))
        return cls(function, frozenset(references),
                   _nonnegative_bound(expression, function) if checked else None)

    def __call__(self, values: Mapping[int | str, object]):
        return self.evaluate(values)


def read_expressions(expressions: list[dict]) -> tuple[Expression, ...]:
    return tuple(Expression.read(expression) for expression in expressions)


def evaluate_shape(expressions: tuple[Expression, ...], values: Mapping[int | str, object]) -> tuple:
    return tuple(expression(values) for expression in expressions)
