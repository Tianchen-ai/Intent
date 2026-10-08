from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

from intent.api import Definition

from ..diagnostics.errors import FrontendError
from ..diagnostics.locations import Location
from ..diagnostics.locations import SourceSpan


@dataclass(frozen=True, slots=True)
class SourceUnit:
    definition: Definition[object, object]
    tree: ast.Module
    function: ast.FunctionDef
    filename: str
    first_line: int
    bindings: dict[str, object]

    @classmethod
    def from_definition(
        cls,
        definition: Definition[object, object],
    ) -> SourceUnit:
        source_text = definition.source_text()
        tree = ast.parse(source_text, filename=definition.source.filename)
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        location = Location(
            SourceSpan(
                definition.source.filename,
                definition.source.first_line,
                0,
                definition.source.first_line,
                0,
            )
        )
        if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
            raise FrontendError(
                "Intent definition source must contain exactly one synchronous function",
                location,
            )
        function = functions[0]
        if function.name != definition.__name__:
            raise FrontendError("definition source name does not match decorated function", location)
        if ast.get_docstring(function, clean=False) is not None:
            function.body = function.body[1:]
        closure = inspect.getclosurevars(definition.python_function)
        bindings = dict(definition.python_function.__globals__)
        bindings.update(closure.globals)
        bindings.update(closure.nonlocals)
        bindings.update(closure.builtins)
        return cls(
            definition=definition,
            tree=tree,
            function=function,
            filename=definition.source.filename,
            first_line=definition.source.first_line,
            bindings=bindings,
        )

    def location(self, node: ast.AST) -> Location:
        start_line = self.first_line + getattr(node, "lineno", 1) - 1
        end_line = self.first_line + getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1
        start_column = getattr(node, "col_offset", 0)
        end_column = getattr(node, "end_col_offset", start_column)
        return Location(
            SourceSpan(
                self.filename,
                start_line,
                start_column,
                end_line,
                end_column,
            )
        )
