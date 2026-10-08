import intent
import intent.language as I


DOCUMENTS = 256
SENTENCES = 4096
TOKENS = 65536
FEATURES = 128


@intent.kernel
def nested_jagged_mean_pool(
    document_offsets: I.In[I.i32, (DOCUMENTS + 1,)],
    sentence_offsets: I.In[I.i32, (SENTENCES + 1,)],
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
    )
    sentences = I.ragged(
        outer=sentence_domain,
        members=token_domain,
        offsets=sentence_offsets,
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
                denominator = I.maximum(I.cast(sentence_tokens, I.f32), 1.0)
                sentence_means[sentence, feature] = sentence_sum / denominator
                document_sum = document_sum + sentence_sum
                document_tokens = document_tokens + sentence_tokens
            document_denominator = I.maximum(
                I.cast(document_tokens, I.f32),
                1.0,
            )
            document_means[document, feature] = document_sum / document_denominator
