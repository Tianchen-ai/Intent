#include <cub/device/device_histogram.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <cuda_runtime.h>
#include <cstdint>

struct TruncateKey {
  __host__ __device__ int operator()(float value) const {
    // PTX cvt.rzi.s32.f32 maps NaN to zero and saturates overflows.
    // Overflowed integers and negative keys fall outside all admitted bins.
    if (value != value)
      return 0;
    return value > -1.0f && value < 2147483648.0f ? static_cast<int>(value) : -1;
  }
};

extern "C" int intent_histogram(void *scratch, size_t *bytes, const float *samples,
                               int *counts, int bins, int64_t size, cudaStream_t stream) {
  auto input = thrust::make_transform_iterator(samples, TruncateKey{});
  return cub::DeviceHistogram::HistogramEven(scratch, *bytes, input, counts,
                                            bins + 1, 0, bins, size, stream);
}
