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
        self.domain = compiler.runtime("kir", "binding.domain")
        self.target = compiler.runtime(backend, "binding.target").Target()
        self.backend = backend
        self.artifacts = {}
        self.lock = RLock()

    def prepare(self, *inputs, outputs=None):
        values, result, device = self.binding.bind(inputs, outputs)
        with self.lock:
            facts = self.domain.arguments(self.args, values)
            key = self.target.profile(device, values)
            group = self.artifacts.setdefault(key, [])
            artifact = next((a for a in group if a.accepts(self.domain.evaluate(a.data["domain"],facts))), None)
            if artifact is None:
                options = dict(self.target.options(key), arguments=facts)
                if self.root is None:
                    self.root = self.compiler.root_ir(self.source)
                with self.compiler.derive(
                    {"exits": {self.backend: {"options": options}}}
                ) as target:
                    data = json.loads(target.compile(self.root, exit=self.backend))
                if data["domain"].get("public_abi", data["abi"]) != self.args:
                    raise ValueError("compiled ABI differs from author signature")
                artifact = self.target.materialize(data)
                group.append(artifact)
        shape = self.domain.evaluate(artifact.data["domain"],facts)
        views = self.domain.bind_views(artifact.data["domain"], facts, values)
        call = artifact.bind(views, result, device, shape, key)
        # A prepared call may outlive its JIT object. Pin its installed module namespace.
        call.compiler_owner = self.compiler
        return call

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
