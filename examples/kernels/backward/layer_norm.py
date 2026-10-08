import intent
import intent.language as I


ROWS = 4096
FEATURES = 4096
PARTIAL_GROUPS = 128
MATRIX_STRIDES = I.constraints(strides=(None, 1))
VECTOR_STRIDES = I.constraints(strides=(1,))


@intent.kernel
def layer_norm_backward_rows(
    x: I.In[I.bf16, ("M", "N"), MATRIX_STRIDES],
    dy: I.In[I.bf16, ("M", "N"), MATRIX_STRIDES],
    weight: I.In[I.bf16, ("N",), VECTOR_STRIDES],
    mean: I.In[I.f32, ("M",), VECTOR_STRIDES],
    rstd: I.In[I.f32, ("M",), VECTOR_STRIDES],
    dx: I.Out[I.bf16, ("M", "N"), MATRIX_STRIDES],
    dw_partial: I.InOut[I.bf16, (PARTIAL_GROUPS, "N"), MATRIX_STRIDES],
    db_partial: I.InOut[I.bf16, (PARTIAL_GROUPS, "N"), MATRIX_STRIDES],
    inverse_features: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        x_values = I.cast(x[row, columns], I.f32)
        dy_values = I.cast(dy[row, columns], I.f32)
        weight_values = I.cast(weight[columns], I.f32)
        row_mean = mean[row]
        row_rstd = rstd[row]
        normalized = (x_values - row_mean) * row_rstd
        weighted_dy = weight_values * dy_values
        mean_weighted_dy = (
            I.reduce.sum(weighted_dy, axis=0) * inverse_features
        )
        mean_weighted_dy_normalized = (
            I.reduce.sum(weighted_dy * normalized, axis=0)
            * inverse_features
        )
        dx[row, columns] = I.cast(
            (
                weighted_dy
                - mean_weighted_dy
                - normalized * mean_weighted_dy_normalized
            )
            * row_rstd,
            I.bf16,
        )
        group = row % PARTIAL_GROUPS
        dw_value = dy_values * normalized
        db_value = dy_values
        I.scatter_reduce(
            dw_partial,
            index=(group, columns),
            value=I.cast(dw_value, I.bf16),
            combine=I.add,
        )
        I.scatter_reduce(
            db_partial,
            index=(group, columns),
            value=I.cast(db_value, I.bf16),
            combine=I.add,
        )


@intent.kernel
def layer_norm_backward_reduce(
    dw_partial: I.In[I.bf16, (PARTIAL_GROUPS, "N"), MATRIX_STRIDES],
    db_partial: I.In[I.bf16, (PARTIAL_GROUPS, "N"), MATRIX_STRIDES],
    dw: I.Out[I.f32, ("N",), VECTOR_STRIDES],
    db: I.Out[I.f32, ("N",), VECTOR_STRIDES],
):
    G, N = dw_partial.shape
    rows = I.domain(0, G)
    features = I.domain(0, N)
    dw_value = I.reduce.sum(
        I.cast(dw_partial[rows, features], I.f32),
        axis=0,
    )
    db_value = I.reduce.sum(
        I.cast(db_partial[rows, features], I.f32),
        axis=0,
    )
    dw[features] = dw_value
    db[features] = db_value
