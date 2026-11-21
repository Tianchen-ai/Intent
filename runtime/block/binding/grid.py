"""Evaluate the block instance mapping published in a compiled call ABI."""
def accepts(guard, shape):
    width = shape[1]
    return guard["min_width"] <= width and (guard["max_width"] is None or width <= guard["max_width"])


def grid(mapping, shape, row):
    values = {"M": shape[0], "N": shape[1], **row}
    def evaluate(expression):
        if isinstance(expression, int):
            return expression
        if isinstance(expression, str):
            return values[expression]
        op, lhs, rhs = expression
        a,b = evaluate(lhs),evaluate(rhs)
        if op == "mul":
            return a*b
        if op == "ceildiv":
            return -(-a // b)
        raise ValueError(f"unknown instance mapping operation {op}")
    return tuple(evaluate(axis) for axis in mapping)
