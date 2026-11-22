"""Capture a partition of keys and the additive integer reduction it feeds."""
import ast


def capture(c, call):
    keys, weight = call.args
    options = {kw.arg: kw.value for kw in call.keywords}
    if (set(options) != {"size", "invalid", "combine", "init"}
            or ast.unparse(options["combine"]) != c.namespace+".add"
            or ast.literal_eval(options["invalid"]) != "skip"
            or ast.literal_eval(weight) != 1 or ast.literal_eval(options["init"]) != 0):
        raise ValueError("counting requires unit contributions to in-range bins and zero identity")
    convert=(isinstance(keys,ast.Call) and ast.unparse(keys.func)==c.namespace+".cast")
    if convert and (len(keys.args)!=2 or ast.unparse(keys.args[1])!=c.namespace+".i32"):
        raise ValueError("expected an explicit conversion to i32 bin indices")
    chunk = keys.args[0] if convert else keys
    if (not isinstance(chunk, ast.Call) or ast.unparse(chunk.func) != c.namespace+".chunk"
            or len(chunk.args)!=3 or ast.unparse(chunk.args[1])!=c.row
            or ast.unparse(options["size"])!=c.owner+".dim(1)"):
        raise ValueError("keys must cover the partition and counts the output bin axis")
    source, _, size=chunk.args
    dtype=next(a["dtype"] for a in c.args if a["name"]==source.id)
    if (dtype=="f32")!=convert or dtype not in ("f32","i64","i32"):
        raise ValueError("count integer keys at source width; floating keys require a truncation")
    span=ast.literal_eval(size)
    if type(span)!=int or span<=0:raise ValueError("expected a positive partition span")
    return c.emit("count_keys",[c.values[source.id],("%row","index")],"tensor<?xi32>",f"{{span = {span} : i64}}")
