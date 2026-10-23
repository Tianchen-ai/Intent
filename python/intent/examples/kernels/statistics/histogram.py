import intent
import intent.language as I


SAMPLES = 8 * 1024 * 1024
BINS = 256


@intent.fn
def count_bins(samples, bins):
    bin_index = I.cast(samples, I.i32)
    valid = (bin_index >= 0) & (bin_index < bins)
    return I.histogram(bin_index, bins=bins, valid=valid, count_dtype=I.i32)


@intent.kernel
def histogram_256(
    samples: I.In[I.f32, ("N",)],
    histogram: I.Out[I.i32, (BINS,)],
):
    N = samples.shape[0]
    bin_extent = histogram.shape[0]
    sample_axis = I.domain(0, N)
    bins = I.domain(0, bin_extent)
    counts = count_bins(samples[sample_axis], bin_extent)
    histogram[bins] = counts


@intent.kernel
def histogram_partials(
    samples: I.In[I.f32, ("N",)],
    partials: I.Out[I.i32, ("P", BINS)],
    CHUNK: I.Constexpr[int],
):
    bins = I.domain(0, BINS)
    for partition in I.parallel(I.domain(0, partials.shape[0])):
        members = I.domain(partition * CHUNK,
                           I.minimum((partition + 1) * CHUNK, samples.shape[0]))
        partials[partition, bins] = count_bins(samples[members], BINS)


@intent.kernel
def histogram_merge(
    partials: I.In[I.i32, ("P", BINS)],
    histogram: I.Out[I.i32, (BINS,)],
):
    partitions = I.domain(0, partials.shape[0])
    bins = I.domain(0, BINS)
    histogram[bins] = I.reduce.sum(partials[partitions, bins], axis=0)
