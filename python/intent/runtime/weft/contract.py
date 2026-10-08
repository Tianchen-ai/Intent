"""Pure host/task facts; external Weft native artifacts have their own reader."""
from __future__ import annotations

from dataclasses import dataclass
import json

from ..contract import CPUImplementation, integer_field, name_field, names_field, object_field, sequence_field
from ..native import NativeABI


@dataclass(frozen=True, slots=True)
class WeftCandidate:
    entry: str
    values: tuple[int, ...]
    requires_matrix_i8_i32: bool
    implementations: tuple[CPUImplementation, ...]

    def metadata(self) -> dict:
        return {"entry": self.entry, "values": list(self.values),
                "requires_matrix_i8_i32": self.requires_matrix_i8_i32,
                "implementations": [value.metadata() for value in self.implementations]}


@dataclass(frozen=True, slots=True)
class TaskABI:
    symbol: str
    arguments: str
    shape_parameters: tuple[str, ...]

    @classmethod
    def read(cls, value) -> TaskABI:
        abi = object_field(value, "Weft task ABI")
        parameters = names_field(abi["shape_parameters"], "Weft shape parameters")
        arguments = sequence_field(abi["arguments"], "Weft task arguments")
        names = set()
        for argument in arguments:
            argument = object_field(argument, "Weft task argument")
            name = name_field(argument["name"], "Weft task argument name")
            if name in names:
                raise ValueError("Weft task argument names must be distinct")
            names.add(name)
            for field in ("c_type", "encoding"):
                name_field(argument[field], "Weft task argument " + field)
            for extent in sequence_field(argument["shape"], "Weft task argument shape"):
                # Weft may bind a logical shape symbol at compilation time;
                # such a symbol is intentionally absent from runtime parameters.
                name_field(extent, "Weft task argument extent")
            for field in ("storage_bytes", "record_elements", "alignment"):
                integer_field(argument[field], "Weft task argument " + field, minimum=1)
            integer_field(argument["interleave_rows"], "Weft interleave rows")
            integer_field(argument["alias_set"], "Weft alias set", minimum=-1)
            if type(argument["writable"]) is not bool:
                raise TypeError("Weft task argument writable must be a boolean")
        # This is an external ABI record compared as a whole with Weft's native
        # output, not a second encoding/layout interpretation in Intent runtime.
        return cls(name_field(abi["symbol"], "Weft task symbol"),
                   json.dumps(arguments, sort_keys=True), parameters)

    def verify_native(self, kernel: dict) -> None:
        if json.dumps(kernel["arguments"], sort_keys=True) != self.arguments:
            raise ValueError(f"Weft artifact {self.symbol} arguments disagree with the CPU task ABI")
        if tuple(kernel["shape_parameters"]) != self.shape_parameters:
            raise ValueError(f"Weft artifact {self.symbol} shape_parameters disagree with the CPU task ABI")


@dataclass(frozen=True, slots=True)
class WeftTask:
    cpu_entry: str
    abi: TaskABI


@dataclass(frozen=True, slots=True)
class WeftFacts:
    host_source: str
    candidates: tuple[WeftCandidate, ...]
    tasks: tuple[WeftTask, ...]

    @classmethod
    def read(cls, source: str, metadata: dict, abi: NativeABI) -> WeftFacts:
        host = name_field(metadata["host_source"], "Weft host source")
        if metadata["kind"] != "weft-generation":
            raise ValueError("Weft source metadata must describe a weft-generation")
        candidates, entries = [], set()
        for raw in sequence_field(metadata["candidates"], "Weft candidates"):
            candidate = object_field(raw, "Weft candidate")
            entry = name_field(candidate["entry"], "Weft CPU entry")
            if entry in entries:
                raise ValueError("Weft CPU candidate entries must be distinct")
            entries.add(entry)
            values = tuple(integer_field(value, "CPU configuration value", minimum=-(1 << 63))
                           for value in sequence_field(candidate["values"], "CPU configuration values"))
            required = candidate["requires_matrix_i8_i32"]
            if type(required) is not bool:
                raise TypeError("Weft matrix requirement must be a boolean")
            implementations = tuple(CPUImplementation.read(value) for value in
                                    sequence_field(candidate["implementations"], "CPU implementations"))
            candidates.append(WeftCandidate(entry, values, required, implementations))
        if not candidates:
            raise ValueError("Weft generation requires at least one candidate")
        tasks, symbols = [], set()
        for raw in sequence_field(metadata["tasks"], "Weft tasks"):
            task = object_field(raw, "Weft task")
            entry = name_field(task["cpu_entry"], "Weft task CPU entry")
            task_abi = TaskABI.read(task["abi"])
            if entry not in entries or task_abi.symbol in symbols:
                raise ValueError("Weft task must bind a declared CPU candidate and distinct device symbol")
            symbols.add(task_abi.symbol)
            tasks.append(WeftTask(entry, task_abi))
        return cls(host, tuple(candidates), tuple(tasks))
