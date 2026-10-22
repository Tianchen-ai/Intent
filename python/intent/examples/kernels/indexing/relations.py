import intent
import intent.language as I


OFFSET_ROWS = 4096
OFFSET_FEATURES = 1024
GQA_QUERY_HEADS = 32
GQA_KEY_HEADS = 8
GQA_TOKENS = 4096
GQA_HEAD_GROUP = GQA_QUERY_HEADS // GQA_KEY_HEADS
LOOKUP_ROWS = 65536
LOOKUP_ENTRIES = 65536
LOOKUP_FEATURES = 128
INDEX_SELECT_SOURCE_ROWS = 65536
INDEX_SELECT_ROWS = 32768
INDEX_SELECT_FEATURES = 4096
SCALED_ADD_DESTINATION_ROWS = 65536
SCALED_ADD_SOURCE_ROWS = 32768
SCALED_ADD_INNER_ROWS = 1
SCALED_ADD_FEATURES = 4096


@intent.kernel
def shifted_row_copy(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    features = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        shifted_row = ((row + 1) % -M) % M
        output[row, features] = x[shifted_row, features]


@intent.kernel
def roll_rows_forward(
    x: I.In[I.f16, ("T",)],
    output: I.Out[I.f16, ("T",)],
    ROW_WIDTH: I.Constexpr[int],
):
    T = x.shape[0]
    elements = I.domain(0, T)
    source = (I.indices(elements) + ROW_WIDTH) % T
    output[elements] = x[source]


@intent.kernel
def grouped_query_head_add(
    query: I.In[I.f16, ("HQ", "N")],
    key: I.In[I.f16, ("HK", "N")],
    output: I.Out[I.f16, ("HQ", "N")],
):
    HQ, N = query.shape
    tokens = I.domain(0, N)
    for query_head in I.parallel(I.domain(0, HQ)):
        key_head = (
            (query_head - GQA_QUERY_HEADS) // GQA_HEAD_GROUP
            + GQA_KEY_HEADS
        )
        output[query_head, tokens] = (
            query[query_head, tokens] + key[key_head, tokens]
        )


@intent.kernel
def scalar_table_lookup(
    labels: I.In[I.i32, ("M",)],
    table: I.In[I.f32, ("C", "D")],
    output: I.Out[I.f32, ("M", "D")],
):
    M = labels.shape[0]
    D = table.shape[1]
    features = I.domain(0, D)
    for row in I.parallel(I.domain(0, M)):
        label = labels[row]
        I.assume_in_bounds(label, table, axis=0)
        output[row, features] = table[label, features]


@intent.kernel
def index_select_rows(
    source: I.In[I.f16, ("C", "D")],
    indices: I.In[I.i64, ("M",)],
    output: I.Out[I.f16, ("M", "D")],
):
    M = indices.shape[0]
    D = source.shape[1]
    features = I.domain(0, D)
    for row in I.parallel(I.domain(0, M)):
        source_row = indices[row]
        I.assume_in_bounds(source_row, source, axis=0)
        output[row, features] = source[source_row, features]


@intent.kernel
def scaled_index_add_unique(
    destination: I.InOut[I.f16, ("C", "R", "D")],
    indices: I.In[I.i64, ("M",)],
    source: I.In[I.f16, ("M", "R", "D")],
    scaling: I.In[I.f16, ("D",)],
    alpha: I.f32,
):
    M, R, D = source.shape
    inner_rows = I.domain(0, R)
    features = I.domain(0, D)
    for source_row in I.parallel(I.domain(0, M)):
        destination_row = indices[source_row]
        I.assume_in_bounds(destination_row, destination, axis=0)
        updated = I.cast(
            I.cast(destination[destination_row, inner_rows, features], I.f32)
            + alpha
            * I.cast(scaling[features], I.f32)[None, :]
            * I.cast(source[source_row, inner_rows, features], I.f32),
            I.f16,
        )
        I.scatter_unique(
            destination,
            index=(destination_row, inner_rows, features),
            value=updated,
        )
