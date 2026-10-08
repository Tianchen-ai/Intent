import intent
import intent.language as I

from kernels.cache.reshape_and_cache import BLOCKS
from kernels.cache.reshape_and_cache import BLOCK_SIZE
from kernels.factorization.cholesky import SIZE as CHOLESKY_SIZE
from kernels.ragged.nested_pool import DOCUMENTS
from kernels.ragged.nested_pool import SENTENCES
from kernels.ragged.nested_pool import TOKENS
from kernels.routing.moe_align import EXPERTS


@intent.kernel
def moe_count_routes_product_domain(
    topk_ids: I.In[I.i32, ("T", "K")],
    expert_counts: I.InOut[I.i32, (EXPERTS,)],
):
    T, K = topk_ids.shape
    tokens = I.domain(0, T)
    choices = I.domain(0, K)
    for token in I.parallel(tokens):
        for choice in I.parallel(choices):
            expert = topk_ids[token, choice]
            I.assume_in_bounds(expert, expert_counts, axis=0)
            I.atomic.add(
                expert_counts,
                index=(expert,),
                value=I.cast(1, I.i32),
                order="relaxed",
            )


@intent.kernel
def adamw_update_moments(
    gradient: I.In[I.f32, ("N",)],
    first_moment: I.InOut[I.f32, ("N",)],
    second_moment: I.InOut[I.f32, ("N",)],
    beta1: I.f32,
    beta2: I.f32,
):
    N = gradient.shape[0]
    for index in I.parallel(I.domain(0, N)):
        gradient_value = gradient[index]
        first_moment[index] = (
            beta1 * first_moment[index] + (1.0 - beta1) * gradient_value
        )
        second_moment[index] = (
            beta2 * second_moment[index]
            + (1.0 - beta2) * gradient_value * gradient_value
        )


@intent.kernel
def adamw_update_parameter(
    parameter: I.InOut[I.f32, ("N",)],
    first_moment: I.In[I.f32, ("N",)],
    second_moment: I.In[I.f32, ("N",)],
    learning_rate: I.f32,
    inverse_bias1: I.f32,
    inverse_bias2: I.f32,
    epsilon: I.f32,
    weight_decay: I.f32,
):
    N = parameter.shape[0]
    for index in I.parallel(I.domain(0, N)):
        corrected_first = first_moment[index] * inverse_bias1
        corrected_second = second_moment[index] * inverse_bias2
        denominator = 1.0 / I.rsqrt(corrected_second) + epsilon
        parameter[index] = (
            parameter[index] * (1.0 - learning_rate * weight_decay)
            - learning_rate
            * corrected_first
            / denominator
        )


@intent.kernel
def reshape_key_cache(
    key: I.In[I.f16, ("T", "H", "D")],
    slot_mapping: I.In[I.i32, ("T",)],
    key_cache: I.InOut[I.f16, (BLOCKS, BLOCK_SIZE, "H", "D")],
):
    T, H, D = key.shape
    heads = I.domain(0, H)
    dimensions = I.domain(0, D)
    for token in I.parallel(I.domain(0, T)):
        slot = slot_mapping[token]
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        I.assume_in_bounds(block, key_cache, axis=0)
        key_cache[block, offset, heads, dimensions] = key[token, heads, dimensions]


@intent.kernel
def reshape_value_cache(
    value: I.In[I.f16, ("T", "H", "D")],
    slot_mapping: I.In[I.i32, ("T",)],
    value_cache: I.InOut[I.f16, (BLOCKS, BLOCK_SIZE, "H", "D")],
):
    T, H, D = value.shape
    heads = I.domain(0, H)
    dimensions = I.domain(0, D)
    for token in I.parallel(I.domain(0, T)):
        slot = slot_mapping[token]
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        I.assume_in_bounds(block, value_cache, axis=0)
        value_cache[block, offset, heads, dimensions] = value[token, heads, dimensions]


@intent.kernel
def nested_jagged_mean_pool_identity(
    document_offsets: I.In[I.i32, (DOCUMENTS + 1,)],
    sentence_offsets: I.In[I.i32, (SENTENCES + 1,)],
    document_indices: I.In[I.i32, (SENTENCES,)],
    token_indices: I.In[I.i32, (TOKENS,)],
    values: I.In[I.f32, (TOKENS, "D")],
    sentence_means: I.Out[I.f32, (SENTENCES, "D")],
    document_means: I.Out[I.f32, (DOCUMENTS, "D")],
):
    D = values.shape[1]
    document_domain = I.domain(0, document_means.shape[0])
    sentence_domain = I.domain(0, sentence_means.shape[0])
    token_domain = I.domain(0, values.shape[0])
    documents = I.ragged(
        outer=document_domain,
        members=sentence_domain,
        offsets=document_offsets,
        indices=document_indices,
    )
    sentences = I.ragged(
        outer=sentence_domain,
        members=token_domain,
        offsets=sentence_offsets,
        indices=token_indices,
    )
    for document in I.parallel(documents.outer):
        for feature in I.parallel(I.domain(0, D)):
            document_sum = I.cast(0.0, I.f32)
            document_tokens = I.cast(0, I.i32)
            for sentence in documents[document]:
                sentence_sum = I.cast(0.0, I.f32)
                sentence_tokens = I.cast(0, I.i32)
                for token in sentences[sentence]:
                    sentence_sum = sentence_sum + values[token, feature]
                    sentence_tokens = sentence_tokens + 1
                sentence_means[sentence, feature] = sentence_sum / I.maximum(
                    I.cast(sentence_tokens, I.f32),
                    1.0,
                )
                document_sum = document_sum + sentence_sum
                document_tokens = document_tokens + sentence_tokens
            document_means[document, feature] = document_sum / I.maximum(
                I.cast(document_tokens, I.f32),
                1.0,
            )


@intent.kernel
def nested_sentence_pool(
    sentence_offsets: I.In[I.i32, (SENTENCES + 1,)],
    values: I.In[I.f32, (TOKENS, "D")],
    sentence_sums: I.Out[I.f32, (SENTENCES, "D")],
    sentence_means: I.Out[I.f32, (SENTENCES, "D")],
):
    D = values.shape[1]
    sentence_domain = I.domain(0, sentence_means.shape[0])
    sentences = I.ragged(
        outer=sentence_domain,
        members=I.domain(0, values.shape[0]),
        offsets=sentence_offsets,
    )
    for sentence in I.parallel(sentences.outer):
        for feature in I.parallel(I.domain(0, D)):
            total = I.cast(0.0, I.f32)
            count = I.cast(0, I.i32)
            for token in sentences[sentence]:
                total = total + values[token, feature]
                count = count + 1
            sentence_sums[sentence, feature] = total
            sentence_means[sentence, feature] = total / I.maximum(
                I.cast(count, I.f32),
                1.0,
            )


@intent.kernel
def nested_document_pool(
    document_offsets: I.In[I.i32, (DOCUMENTS + 1,)],
    sentence_offsets: I.In[I.i32, (SENTENCES + 1,)],
    sentence_sums: I.In[I.f32, (SENTENCES, "D")],
    document_means: I.Out[I.f32, (DOCUMENTS, "D")],
):
    D = sentence_sums.shape[1]
    document_domain = I.domain(0, document_means.shape[0])
    documents = I.ragged(
        outer=document_domain,
        members=I.domain(0, sentence_sums.shape[0]),
        offsets=document_offsets,
    )
    for document in I.parallel(documents.outer):
        for feature in I.parallel(I.domain(0, D)):
            total = I.cast(0.0, I.f32)
            count = I.cast(0, I.i32)
            for sentence in documents[document]:
                next_sentence = sentence + 1
                I.assume_in_bounds(sentence, sentence_offsets, axis=0)
                I.assume_in_bounds(next_sentence, sentence_offsets, axis=0)
                total = total + sentence_sums[sentence, feature]
                count = count + (
                    sentence_offsets[next_sentence] - sentence_offsets[sentence]
                )
            document_means[document, feature] = total / I.maximum(
                I.cast(count, I.f32),
                1.0,
            )


@intent.kernel
def batched_cholesky_right_looking(
    matrices: I.InOut[I.f32, ("B", CHOLESKY_SIZE, CHOLESKY_SIZE)],
):
    B = matrices.shape[0]
    for batch in I.parallel(I.domain(0, B)):
        for column in range(CHOLESKY_SIZE):
            factor = 1.0 / I.rsqrt(matrices[batch, column, column])
            matrices[batch, column, column] = factor
            for row in range(column + 1, CHOLESKY_SIZE):
                matrices[batch, row, column] = (
                    matrices[batch, row, column] / factor
                )
            for trailing_column in range(column + 1, CHOLESKY_SIZE):
                for trailing_row in range(trailing_column, CHOLESKY_SIZE):
                    matrices[batch, trailing_row, trailing_column] = (
                        matrices[batch, trailing_row, trailing_column]
                        - matrices[batch, trailing_row, column]
                        * matrices[batch, trailing_column, column]
                    )
            for upper in range(column + 1, CHOLESKY_SIZE):
                matrices[batch, column, upper] = 0.0


@intent.kernel
def adafactor_apply_scalar_product(
    gradient: I.In[I.f32, ("M", "N")],
    row_state: I.In[I.f32, ("M",)],
    column_state: I.In[I.f32, ("N",)],
    row_mean: I.In[I.f32, (1,)],
    parameter: I.InOut[I.f32, ("M", "N")],
    learning_rate: I.f32,
    epsilon: I.f32,
):
    M, N = parameter.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for row in I.parallel(rows):
        for column in I.parallel(columns):
            variance = (
                row_state[row]
                * column_state[column]
                / I.maximum(row_mean[0], epsilon)
            )
            update = gradient[row, column] * I.rsqrt(variance + epsilon)
            parameter[row, column] = parameter[row, column] - learning_rate * update


@intent.kernel
def matrix_transpose_product_domains(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("N", "M")],
):
    M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for row in I.parallel(rows):
        for column in I.parallel(columns):
            output[column, row] = x[row, column]


@intent.kernel
def ordered_prefix_nested(
    x: I.In[I.f32, ("B", "M", "N")],
    output: I.Out[I.f32, ("B", "M", "N")],
):
    B, M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for batch in I.parallel(I.domain(0, B)):
        prefix = I.cast(0.0, I.f32)
        for row in rows:
            for column in columns:
                prefix = prefix + x[batch, row, column]
                output[batch, row, column] = prefix
