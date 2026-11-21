"""Compilation artifacts and bound launches have independent lifetimes."""

import ast
import json
from threading import RLock
from .signature import signature


class Kernel:
    def __init__(self, compiler, definition, backend):
        self.compiler = compiler
        self.source = definition.source
        self.root = None
        self.args = signature(ast.parse(self.source).body[0])
        self.binding = compiler.runtime("kir", "binding.views").Signature(self.args)
        self.target = compiler.runtime(backend, "binding.target").Target()
        self.backend = backend
        self.artifacts = {}
        self.lock = RLock()

    def prepare(self, *inputs, outputs=None):
        values, result, device = self.binding.bind(inputs, outputs)
        with self.lock:
            key, shape = self.target.specialize(device, values)
            artifact = self.artifacts.get(key)
            if artifact is None:
                options = self.target.options(key)
                if self.root is None:
                    self.root = self.compiler.root_ir(self.source)
                with self.compiler.derive(
                    {"exits": {self.backend: {"options": options}}}
                ) as target:
                    data = json.loads(target.compile(self.root, exit=self.backend))
                if data["abi"] != self.args:
                    raise ValueError("compiled ABI differs from author signature")
                artifact = self.target.materialize(data)
                self.artifacts[key] = artifact
        return artifact.bind(values, result, device, *shape)

    def run(self, *inputs):
        call = self.prepare(*inputs)
        call.launch()
        return call.result()

    def __call__(self, *args):
        if len(args) != len(self.args):
            raise TypeError("expected all arguments in signature order")
        ins, outs = [], []
        for arg, v in zip(self.args, args):
            (outs if arg["direction"] == "Out" else ins).append(v)
        self.prepare(*ins, outputs=outs).launch()
