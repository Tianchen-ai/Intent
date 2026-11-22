"""Evaluate the logical dimensions published by a compiled entry."""

from math import prod

from .views import inspect


def arguments(signature, values):
    result = []
    for slot, value in zip(signature, values, strict=True):
        if slot["direction"] == "Scalar":
            result.append(None)
        else:
            v = inspect(value)
            result.append(
                {
                    "shape": list(v.shape),
                    "strides": list(v.strides),
                }
            )
    return result


def extents(descriptor, facts):
    if "extents" not in descriptor:
        return facts[descriptor["slot"]]["shape"]
    shape = []
    for slot, axis, divide, constant in descriptor["extents"]:
        if slot < 0:
            shape.append(constant)
        else:
            value = facts[slot]["shape"][axis]
            if value % divide:
                raise ValueError("logical extent is not divisible")
            shape.append(value // divide)
    return tuple(shape)


def evaluate(descriptor, facts):
    dims = extents(descriptor, facts)
    inner = descriptor["inner"]
    columns = prod(dims[i] for i in inner) if isinstance(inner,list) else dims[inner]
    return prod(dims[axis] for axis in descriptor["outer"]), columns


def bind_views(descriptor, facts, values):
    if "views" not in descriptor:
        return values
    from math import gcd
    import functools
    from .views import adapter, Signature

    shape = extents(descriptor, facts)
    active = all(shape)
    result = []
    written = {}
    write_counts = {}
    for view in descriptor["views"]:
        i = view["source"]
        src = values[i]
        fact = facts[i]
        if "coordinates" not in view:
            if fact is not None:
                raise ValueError("expected a scalar capture in the projected ABI")
            result.append(src)
            continue
        rows = view["coordinates"]
        offsets = [axis[-1] for axis in rows]
        stride = [
            sum(axis[j] * s for axis, s in zip(rows, fact["strides"]))
            for j in range(len(shape))
        ]
        ranges = []
        for axis, n, origin in zip(rows, fact["shape"], offsets, strict=True):
            delta = [
                a * (extent - 1) for a, extent in zip(axis[:-1], shape, strict=True)
            ]
            lo = origin + sum(min(0, d) for d in delta)
            hi = origin + sum(max(0, d) for d in delta)
            if active and (lo < 0 or hi >= n):
                raise ValueError("projected members outside the source extent")
            ranges.append((lo, hi, functools.reduce(gcd, axis[:-1], 0)))
        offset = sum(s * v for s, v in zip(fact["strides"], offsets)) if active else 0
        if "axis" in view:
            axis = view["axis"]
            bound_shape, bound_strides = (shape[axis],), (stride[axis],)
        else:
            bound_shape, bound_strides = shape, stride
        a = adapter(src).project(src, bound_shape, bound_strides, offset)
        if descriptor["public_abi"][i]["direction"] != "In" and active:
            Signature.require_writable(inspect(a))
            for prev in written.setdefault(i, []):

                def disjoint(x, y):
                    g = gcd(x[2], y[2])
                    return x[1] < y[0] or x[0] > y[1] or (g and (x[0] - y[0]) % g != 0)

                if not any(disjoint(x, y) for x, y in zip(ranges, prev)):
                    raise ValueError("overlapping projected output domains")
            written[i].append(ranges)
            write_counts[i]=write_counts.get(i,0)+prod(bound_shape)
        result.append(a)
    for i, maps in written.items():
        if write_counts[i] != prod(facts[i]["shape"]):
            raise ValueError("indexed outputs do not cover the public view")
    return tuple(result)
