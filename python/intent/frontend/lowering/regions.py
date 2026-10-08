from __future__ import annotations

from intent.frontend.semantics import OperationKind

from .scope import lowering_scope


def pure_region(lowerer, argument_types, node, body, *, purpose="structured callable"):
    """Build one pure region and finish its result schema in its own scope."""
    region = lowerer.make_region(
        lowerer.location(node), argument_types, isolated_from_above=True,
    )
    with lowering_scope(lowerer, current_block=region.blocks[0]):
        results = tuple(body(tuple(region.blocks[0].arguments)))
        if region.effects:
            lowerer.error(node, f"{purpose} region must be pure")
        lowerer.emit(OperationKind.YIELD, lowerer.location(node), operands=results)
    return region, tuple(value.type for value in results)
