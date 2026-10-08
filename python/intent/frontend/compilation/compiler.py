from __future__ import annotations

from dataclasses import dataclass

from intent.api import HelperDefinition
from intent.api import KernelDefinition
from intent.frontend.mlir import FunctionKind
from intent.frontend.mlir import MlirBuilder
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import ValueType

from ..diagnostics.errors import FrontendError
from ..lowering.ast.context import FunctionLowerer
from ..lowering.ast.model import ConstexprBinding
from ..source.signature import lower_helper_parameters
from ..source.signature import lower_kernel_signature
from ..source.unit import SourceUnit


@dataclass(frozen=True, slots=True)
class HelperKey:
    definition: HelperDefinition[object, object]
    argument_types: tuple[ValueType, ...]


class FrontendCompiler:
    def __init__(
        self,
        definition: KernelDefinition[object, object],
        constexprs: dict[str, object],
    ) -> None:
        self.definition = definition
        self.source = SourceUnit.from_definition(definition)
        self.signature = lower_kernel_signature(definition, self.source, constexprs)
        self.builder = MlirBuilder(definition.__name__, self.source.location(self.source.function))
        self.active_helpers: set[HelperKey] = set()

    def lower(self) -> str:
        function = self.builder.function(
            self.definition.__name__,
            FunctionKind.KERNEL,
            self.signature.parameters,
            (),
            self.source.location(self.source.function),
        )
        lowerer = FunctionLowerer(
            compiler=self,
            definition=self.definition,
            source=self.source,
            function=function,
            constexpr_values=self.signature.constexpr_values,
        )
        lowerer.lower()
        return self.builder.emit_module()

    def lower_helper_inline(
        self,
        caller: FunctionLowerer,
        definition: HelperDefinition[object, object],
        arguments: tuple[object, ...],
        call_location: object,
    ) -> tuple[MlirValue, ...]:
        argument_types = tuple(
            argument.ir_value.type
            if isinstance(argument, ConstexprBinding)
            else argument.type
            for argument in arguments
        )
        key = HelperKey(definition, argument_types)
        if key in self.active_helpers:
            raise FrontendError(
                "recursive @intent.fn requires an explicit recursive IR contract and is unsupported",
                call_location,
            )
        source = SourceUnit.from_definition(definition)
        parameters = lower_helper_parameters(source, argument_types)
        self.active_helpers.add(key)
        try:
            return caller.lower_inline_helper(
                definition=definition,
                source=source,
                parameters=parameters,
                arguments=arguments,
            )
        finally:
            self.active_helpers.remove(key)

def lower_to_mlir(
    definition: KernelDefinition[object, object],
    *,
    constexprs: dict[str, object] | None = None,
) -> str:
    """Capture a typed kernel as Intent MLIR without native Python bindings.

    The compiler's KIR stage verifies and normalizes this module before any
    target analysis or lowering. This function does not require a device.
    """
    if not isinstance(definition, KernelDefinition):
        raise TypeError("lower_to_mlir expects an @intent.kernel definition")
    return FrontendCompiler(definition, dict(constexprs or {})).lower()
