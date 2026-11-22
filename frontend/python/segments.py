"""Capture the bounds and scalar expression in an ordered member fold."""
import ast


def bounds(c, call):
    if len(call.args)!=2 or call.keywords or not all(isinstance(v,ast.Name) for v in call.args):
        raise ValueError("segment takes its offsets view and outer index")
    offsets,row=(v.id for v in call.args)
    view=next((v for v in c.args if v["name"]==offsets),None)
    if row!=c.row or not view or view["direction"]!="In" or view["dtype"]!="i32" or len(view["dims"])!=1:
        raise ValueError("segment requires a read-only i32 offset vector indexed by its owner")
    return [c.values[offsets],("%row","index")]


def fold(c, call):
    if len(call.args)!=2 or call.keywords or not isinstance(call.args[1],ast.Lambda):
        raise ValueError("ordered_sum takes a segment and a pure member expression")
    domain,body=call.args
    if not isinstance(domain,ast.Call) or ast.unparse(domain.func)!=c.namespace+".segment":
        raise ValueError("ordered_sum requires an offset-defined domain")
    keys=bounds(c,domain)
    if len(body.args.args)!=1:raise ValueError("expected one absolute member coordinate")
    member=body.args.args[0].arg
    outer=c.ops
    c.ops=[]
    def expression(node):
        if isinstance(node,ast.Name) and node.id==member:return ("%member","index")
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Add)):
            a,b=expression(node.left),expression(node.right)
            if a[1]!=b[1] or a[1]!="f32":raise ValueError("fold arithmetic needs two f32 members")
            return c.emit("mul" if isinstance(node.op,ast.Mult) else "add",[a,b],"f32")
        if not isinstance(node,ast.Subscript) or not isinstance(node.value,ast.Name):raise ValueError("expected a member load")
        name=node.value.id
        slot=next(a for a in c.args if a["name"]==name)
        coords=node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
        if slot["direction"]!="In" or len(coords)!=len(slot["dims"]):raise ValueError("invalid fold capture")
        indices=[]
        for index in coords:
            if isinstance(index,ast.Slice) and ast.unparse(index)==":" and len(c.axes)==2 and slot["dims"][-1]==c.axes[-1]:
                indices.append(("%feature","index"))
            else:indices.append(expression(index))
        if any(x[1] not in ("index","i32","i64") for x in indices):raise ValueError("expected an integral coordinate")
        return c.emit("member_load",[c.values[name],*indices],slot["dtype"])
    try:
        term=expression(body.body)
        if term[1]!="f32":raise ValueError("ordered sum requires f32 terms")
        c.emit("term",[term]);region='\n'.join(c.ops)
    finally:c.ops=outer
    return c.emit("ordered_sum",keys,"tensor<?xf32>","({^bb0(%member:index,%feature:index):\n"+region+"\n})")
