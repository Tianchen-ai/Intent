import intent
import intent.language as I


TOKENS = 2048
HIDDEN = 4096
VOCABULARY = 32768
CHUNK_SIZE = 1024


@intent.kernel
def linear_logits_chunk(
    hidden: I.In[I.bf16, ("M", "K")],
    weight: I.In[I.bf16, ("V", "K")],
    logits: I.Out[I.bf16, ("M", "V")],
):
    M, K = hidden.shape
    V = weight.shape[0]
    rows = I.domain(0, M)
    vocabulary = I.domain(0, V)
    reduction = I.domain(0, K)
    value = I.matmul(
        hidden[rows, reduction],
        weight[vocabulary, reduction],
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    logits[rows, vocabulary] = I.cast(value, I.bf16)


@intent.kernel
def cross_entropy_probability_chunk(
    logits: I.InOut[I.bf16, ("M", "V")],
    target: I.In[I.i64, ("M",)],
    loss: I.Out[I.f32, ("M",)],
):
    M, V = logits.shape
    vocabulary = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(logits[row, vocabulary], I.f32)
        maximum = I.reduce.max(values, axis=0)
        exponentials = I.exp(values - maximum)
        denominator = I.reduce.sum(exponentials, axis=0)
        label = target[row]
        I.assume_in_bounds(label, logits, axis=1)
        target_logit = I.cast(I.gather(logits, index=(row, label)), I.f32)
        loss[row] = maximum + I.log(denominator) - target_logit
        logits[row, vocabulary] = I.cast(exponentials / denominator, I.bf16)


@intent.kernel
def cross_entropy_mean(
    loss: I.In[I.f32, ("M",)],
    mean_loss: I.Out[I.f32, (1,)],
):
    rows = I.domain(0, loss.shape[0])
    for singleton in I.parallel(I.domain(0, 1)):
        mean_loss[singleton] = (
            I.reduce.sum(loss[rows], axis=0)
            / I.cast(loss.shape[0], I.f32)
        )
