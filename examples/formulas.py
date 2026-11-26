"""Numerical examples from the handbook. Run on a CUDA GPU with Triton."""
import torch
import intentdsl as I
from intentdsl import examples as ops


def check(op, args, expected):
    y = I.jit(op, backend="triton").run(*args)
    torch.testing.assert_close(y, expected, rtol=0.002, atol=0.002)
    print(expected.shape, "passed")


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda")
    f = x.half()
    g = torch.rand(128, device="cuda")
    b = torch.rand_like(g)
    eps = 1e-5
    check(ops.relu, (f,), f.relu())
    check(ops.softmax, (f,), f.float().softmax(-1).half())
    check(ops.rms_norm, (x, g, 1 / 128, eps), x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * g)
    check(ops.layer_norm, (x, g, b, 1 / 128, eps), torch.nn.functional.layer_norm(x, (128,), g, b, eps))
    check(ops.transpose, (f,), f.T)
    check(ops.cumsum, (x,), x.cumsum(1))
    ids = torch.arange(63, -1, -1, device="cuda")
    table = x.bfloat16()
    check(ops.embedding, (table, ids), table[ids])
    check(ops.gemm, (f, f.T.contiguous()), f @ f.T)

    # All codes are zero, so the signed 6-bit value is -32. Each scale is one.
    records = torch.zeros(64, 2, 210, dtype=torch.uint8, device="cuda")
    records[:, :, 192:208] = 1
    records[:, :, 208:210] = torch.tensor([1.], dtype=torch.float16, device="cuda").view(torch.uint8)
    activation = torch.ones(2, 256, device="cuda")
    check(ops.q6k_gemv, (records, activation), torch.full((64,), -32. * 512, device="cuda"))
