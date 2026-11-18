"""The three-kernel backward pass for grouped-query attention."""

from dataclasses import dataclass

from intent.examples.kernels.backward.attention import (
    attention_backward_delta, attention_backward_dkdv, attention_backward_dq,
)
from intent.runtime import PreparedSequence


@dataclass
class AttentionBackward:
    delta: object
    dkdv: object
    dq: object

    @classmethod
    def from_jit(cls, jit, *, head_group, causal):
        options = {"HEAD_GROUP": head_group, "CAUSAL": causal}
        return cls(jit(attention_backward_delta),
                   jit(attention_backward_dkdv, constexprs=options),
                   jit(attention_backward_dq, constexprs=options))

    def prepare(self, query, key, value, output, grad_output, lse, scale, *,
                outputs=None, workspace=None):
        """Bind dQ, dK and dV in that order. LSE is the natural logarithm."""
        delta = self.delta.prepare(output, grad_output,
                                   outputs=None if workspace is None else (workspace,))
        common = (query, key, value, grad_output, lse, delta.result(), scale)
        dq_outputs = dkdv_outputs = None
        if outputs is not None:
            grad_q, grad_k, grad_v = outputs
            dq_outputs, dkdv_outputs = (grad_q,), (grad_k, grad_v)
        dkdv = self.dkdv.prepare(*common, outputs=dkdv_outputs)
        dq = self.dq.prepare(*common, outputs=dq_outputs)
        return PreparedSequence((delta, dkdv, dq), outputs=(dq.result(), *dkdv.result()))

    def run(self, *arguments, **bindings):
        call = self.prepare(*arguments, **bindings)
        call.launch()
        return call.result()
