// acvram fused dequantization kernels.
//
// Two formats, because the two GPUs in the target rig cannot run the same one:
//
//   NVFP4     E2M1 elements + E4M3 scale every 16       RTX 5090   sm_120
//   INT4      uint4 elements + fp16 scale/zero per 128  RTX 3080Ti sm_86
//
// Each format gets two entry points:
//
//   *_dequant   materialise the full matrix in a compute dtype, then let
//               cuBLAS do the GEMM. This is the prefill path: the dequant
//               cost is amortised over the whole batch and cuBLAS beats any
//               hand-written GEMM.
//
//   *_gemv      fused dequantize-and-multiply for batch 1..8. This is the
//               decode path, and it is purely memory bound: the point is to
//               read the 4-bit weights straight out of global memory and
//               never write a 16-bit copy of them anywhere.
//
// Three things make the GEMV fast, and all three matter:
//
//   * 8-byte vector loads. One `uint2` carries 16 packed weights, which is
//     exactly one NVFP4 scale block and a whole number of INT4 groups -- so a
//     thread never straddles a scale boundary and the scale is read once per
//     load rather than once per weight.
//   * Several output rows per block. The activation slice is read once and
//     reused across ROWS rows, cutting activation traffic by that factor.
//     Weight traffic is irreducible; activation traffic is not.
//   * Split-K when the matrix is short. A 4096-row GEMV launches 4096/ROWS
//     blocks; on a 170-SM card with 4 rows per block that is fine, but the
//     small projections in a GQA attention block would leave most of the
//     device idle, so those split the reduction instead.
//
// The numerics must match the reference implementations in
// acvram/quant/nvfp4.py and acvram/quant/int4.py exactly; tests/test_kernels.py
// checks that against the PyTorch path.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int WARP = 32;
constexpr int ROWS_PER_BLOCK = 4;
constexpr int WEIGHTS_PER_LOAD = 16;      // one uint2

// E2M1 magnitudes, indexed by the 3-bit magnitude field.
__constant__ float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float e4m3_to_float(unsigned char bits) {
    __nv_fp8_storage_t s = static_cast<__nv_fp8_storage_t>(bits);
    __half_raw h = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half *>(&h));
}

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}
template <> __device__ __forceinline__ __nv_bfloat16
from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// nibble j of a uint2: bytes are little-endian, so nibble j sits at bit 4*j
// of the low word for j < 8 and of the high word beyond that.
__device__ __forceinline__ unsigned int nibble(const uint2 &p, int j) {
    const unsigned int w = (j < 8) ? p.x : p.y;
    return (w >> ((j & 7) * 4)) & 0xFu;
}

__device__ __forceinline__ float warp_reduce(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Reduce ROWS accumulators across the block. shared must hold
// ROWS * (blockDim.x / WARP) floats.
template <int ROWS>
__device__ __forceinline__ void block_reduce_rows(float (&acc)[ROWS],
                                                  float *shared, int nwarps) {
    const int lane = threadIdx.x % WARP;
    const int wid = threadIdx.x / WARP;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float v = warp_reduce(acc[r]);
        if (lane == 0) shared[r * nwarps + wid] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float v = 0.f;
        if (threadIdx.x < static_cast<unsigned>(nwarps))
            v = shared[r * nwarps + threadIdx.x];
        v = warp_reduce(v);
        if (threadIdx.x == 0) acc[r] = v;
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// NVFP4
// -------------------------------------------------------------------------

template <typename T>
__global__ void nvfp4_dequant_kernel(
    const unsigned char *__restrict__ qw,     // [M, K/2]
    const unsigned char *__restrict__ bscale, // [M, K/16] raw E4M3 bytes
    const float gscale,
    T *__restrict__ out,                      // [M, K]
    int M, int K) {
    const long row = blockIdx.x;
    if (row >= M) return;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const uint2 *qrow = reinterpret_cast<const uint2 *>(qw + row * (long)(K >> 1));
    const unsigned char *srow = bscale + row * (long)nloads;
    T *orow = out + row * (long)K;

    for (int i = threadIdx.x; i < nloads; i += blockDim.x) {
        const uint2 p = qrow[i];
        const float s = e4m3_to_float(srow[i]) * gscale;
        const int base = i * WEIGHTS_PER_LOAD;
        #pragma unroll
        for (int j = 0; j < WEIGHTS_PER_LOAD; ++j) {
            const unsigned int c = nibble(p, j);
            float v = kE2M1[c & 7u] * s;
            if (c & 8u) v = -v;
            orow[base + j] = from_float<T>(v);
        }
    }
}

// y[n, m] = sum_k W[m, k] * x[n, k]
template <int ROWS>
__global__ void nvfp4_gemv_kernel(
    const unsigned char *__restrict__ qw,
    const unsigned char *__restrict__ bscale,
    const float gscale,
    const float *__restrict__ x,              // [N, K]
    float *__restrict__ y,                    // [N, M]
    int M, int K, int N, int k_splits) {
    extern __shared__ float smem[];
    const int nwarps = (blockDim.x + WARP - 1) / WARP;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const int split = blockIdx.y;
    const int per_split = (nloads + k_splits - 1) / k_splits;
    const int lo = split * per_split;
    const int hi = min(nloads, lo + per_split);
    const long half_k = K >> 1;

    for (int n = 0; n < N; ++n) {
        const float *xn = x + (long)n * K;
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            // Read the activation slice once and reuse it across ROWS rows.
            float xs[WEIGHTS_PER_LOAD];
            const float4 *x4 = reinterpret_cast<const float4 *>(
                xn + (long)i * WEIGHTS_PER_LOAD);
            #pragma unroll
            for (int c = 0; c < WEIGHTS_PER_LOAD / 4; ++c) {
                const float4 v = x4[c];
                xs[c * 4 + 0] = v.x; xs[c * 4 + 1] = v.y;
                xs[c * 4 + 2] = v.z; xs[c * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                const uint2 p = reinterpret_cast<const uint2 *>(
                    qw + (long)row * half_k)[i];
                const float s = e4m3_to_float(bscale[(long)row * nloads + i])
                                * gscale;
                float part = 0.f;
                #pragma unroll
                for (int j = 0; j < WEIGHTS_PER_LOAD; ++j) {
                    const unsigned int c = nibble(p, j);
                    const float v = kE2M1[c & 7u];
                    part += (c & 8u) ? -v * xs[j] : v * xs[j];
                }
                acc[r] += part * s;
            }
        }

        block_reduce_rows<ROWS>(acc, smem, nwarps);
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                if (k_splits == 1) y[(long)n * M + row] = acc[r];
                else atomicAdd(&y[(long)n * M + row], acc[r]);
            }
        }
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// INT4 group-wise affine
// -------------------------------------------------------------------------

__device__ __forceinline__ float group_zero(
    const unsigned char *__restrict__ zeros, int g) {
    const unsigned char packed = zeros[g >> 1];
    return static_cast<float>((g & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F));
}

template <typename T>
__global__ void int4_dequant_kernel(
    const unsigned char *__restrict__ qw,     // [M, K/2]
    const __half *__restrict__ scales,        // [M, ng]
    const unsigned char *__restrict__ zeros,  // [M, ceil(ng/2)]
    T *__restrict__ out,                      // [M, K]
    int M, int K, int group) {
    const long row = blockIdx.x;
    if (row >= M) return;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const int ng = K / group;
    const int zbytes = (ng + 1) >> 1;
    const uint2 *qrow = reinterpret_cast<const uint2 *>(qw + row * (long)(K >> 1));
    const __half *srow = scales + row * (long)ng;
    const unsigned char *zrow = zeros + row * (long)zbytes;
    T *orow = out + row * (long)K;

    for (int i = threadIdx.x; i < nloads; i += blockDim.x) {
        const uint2 p = qrow[i];
        const int base = i * WEIGHTS_PER_LOAD;
        const int g = base / group;           // group is a multiple of 16
        const float s = __half2float(srow[g]);
        const float z = group_zero(zrow, g);
        #pragma unroll
        for (int j = 0; j < WEIGHTS_PER_LOAD; ++j) {
            const float v = (static_cast<float>(nibble(p, j)) - z) * s;
            orow[base + j] = from_float<T>(v);
        }
    }
}

template <int ROWS>
__global__ void int4_gemv_kernel(
    const unsigned char *__restrict__ qw,
    const __half *__restrict__ scales,
    const unsigned char *__restrict__ zeros,
    const float *__restrict__ x,
    float *__restrict__ y,
    int M, int K, int N, int group, int k_splits) {
    extern __shared__ float smem[];
    const int nwarps = (blockDim.x + WARP - 1) / WARP;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const int ng = K / group;
    const int zbytes = (ng + 1) >> 1;
    const int split = blockIdx.y;
    const int per_split = (nloads + k_splits - 1) / k_splits;
    const int lo = split * per_split;
    const int hi = min(nloads, lo + per_split);
    const long half_k = K >> 1;

    for (int n = 0; n < N; ++n) {
        const float *xn = x + (long)n * K;
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            float xs[WEIGHTS_PER_LOAD];
            const float4 *x4 = reinterpret_cast<const float4 *>(
                xn + (long)i * WEIGHTS_PER_LOAD);
            #pragma unroll
            for (int c = 0; c < WEIGHTS_PER_LOAD / 4; ++c) {
                const float4 v = x4[c];
                xs[c * 4 + 0] = v.x; xs[c * 4 + 1] = v.y;
                xs[c * 4 + 2] = v.z; xs[c * 4 + 3] = v.w;
            }
            const int g = (i * WEIGHTS_PER_LOAD) / group;
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                const uint2 p = reinterpret_cast<const uint2 *>(
                    qw + (long)row * half_k)[i];
                const float s = __half2float(scales[(long)row * ng + g]);
                const float z = group_zero(zeros + (long)row * zbytes, g);
                float part = 0.f;
                #pragma unroll
                for (int j = 0; j < WEIGHTS_PER_LOAD; ++j)
                    part += (static_cast<float>(nibble(p, j)) - z) * xs[j];
                acc[r] += part * s;
            }
        }

        block_reduce_rows<ROWS>(acc, smem, nwarps);
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                if (k_splits == 1) y[(long)n * M + row] = acc[r];
                else atomicAdd(&y[(long)n * M + row], acc[r]);
            }
        }
        __syncthreads();
    }
}

int threads_for(int K) {
    const int nloads = K / WEIGHTS_PER_LOAD;
    int t = 256;
    while (t > 64 && t > nloads) t >>= 1;
    return t;
}

// Enough blocks to fill the device, without splitting when it is unnecessary.
int splits_for(int M, int K, int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 1;
    const int row_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int want = prop.multiProcessorCount * 2;
    if (row_blocks >= want) return 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    int splits = (want + row_blocks - 1) / row_blocks;
    splits = min(splits, max(1, nloads / 256));
    return max(1, splits);
}

}  // namespace

// -------------------------------------------------------------------------
// bindings
// -------------------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must live on a CUDA device")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor nvfp4_dequant(torch::Tensor qweight, torch::Tensor block_scale,
                            double global_scale, int64_t K,
                            c10::ScalarType dtype) {
    CHECK_CUDA(qweight); CHECK_CUDA(block_scale);
    CHECK_CONTIG(qweight); CHECK_CONTIG(block_scale);
    TORCH_CHECK(K % 16 == 0, "NVFP4 needs K divisible by 16, got ", K);
    const int M = qweight.size(0);
    auto out = torch::empty({M, K}, qweight.options().dtype(dtype));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = threads_for(K);
    const auto *qw = qweight.data_ptr<unsigned char>();
    const auto *bs = block_scale.data_ptr<unsigned char>();

    if (dtype == torch::kBFloat16) {
        nvfp4_dequant_kernel<__nv_bfloat16><<<M, threads, 0, stream>>>(
            qw, bs, (float)global_scale,
            reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), M, (int)K);
    } else if (dtype == torch::kHalf) {
        nvfp4_dequant_kernel<__half><<<M, threads, 0, stream>>>(
            qw, bs, (float)global_scale,
            reinterpret_cast<__half *>(out.data_ptr()), M, (int)K);
    } else if (dtype == torch::kFloat) {
        nvfp4_dequant_kernel<float><<<M, threads, 0, stream>>>(
            qw, bs, (float)global_scale, out.data_ptr<float>(), M, (int)K);
    } else {
        TORCH_CHECK(false, "nvfp4_dequant: unsupported output dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nvfp4_gemv(torch::Tensor qweight, torch::Tensor block_scale,
                         double global_scale, torch::Tensor x, int64_t K) {
    CHECK_CUDA(qweight); CHECK_CUDA(x);
    CHECK_CONTIG(qweight); CHECK_CONTIG(block_scale);
    TORCH_CHECK(K % 16 == 0, "NVFP4 needs K divisible by 16");
    auto xc = x.dim() == 1 ? x.reshape({1, -1}) : x;
    xc = xc.to(torch::kFloat).contiguous();
    const int M = qweight.size(0);
    const int N = xc.size(0);
    const int threads = threads_for(K);
    const int nwarps = (threads + 31) / 32;
    const int splits = splits_for(M, (int)K, (int)qweight.get_device());
    auto out = splits == 1 ? torch::empty({N, M}, xc.options())
                           : torch::zeros({N, M}, xc.options());
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    auto stream = at::cuda::getCurrentCUDAStream();
    nvfp4_gemv_kernel<ROWS_PER_BLOCK>
        <<<grid, threads, ROWS_PER_BLOCK * nwarps * sizeof(float), stream>>>(
            qweight.data_ptr<unsigned char>(),
            block_scale.data_ptr<unsigned char>(), (float)global_scale,
            xc.data_ptr<float>(), out.data_ptr<float>(), M, (int)K, N, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x.dim() == 1 ? out.squeeze(0) : out;
}

torch::Tensor int4_dequant(torch::Tensor qweight, torch::Tensor scales,
                           torch::Tensor zeros, int64_t K, int64_t group,
                           c10::ScalarType dtype) {
    CHECK_CUDA(qweight); CHECK_CUDA(scales); CHECK_CUDA(zeros);
    CHECK_CONTIG(qweight); CHECK_CONTIG(scales); CHECK_CONTIG(zeros);
    TORCH_CHECK(K % group == 0, "K must be divisible by the group size");
    TORCH_CHECK(group % 16 == 0, "group size must be a multiple of 16");
    const int M = qweight.size(0);
    auto out = torch::empty({M, K}, qweight.options().dtype(dtype));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = threads_for(K);
    const auto *qw = qweight.data_ptr<unsigned char>();
    const auto *sc = reinterpret_cast<const __half *>(scales.data_ptr());
    const auto *zr = zeros.data_ptr<unsigned char>();

    if (dtype == torch::kHalf) {
        int4_dequant_kernel<__half><<<M, threads, 0, stream>>>(
            qw, sc, zr, reinterpret_cast<__half *>(out.data_ptr()),
            M, (int)K, (int)group);
    } else if (dtype == torch::kBFloat16) {
        int4_dequant_kernel<__nv_bfloat16><<<M, threads, 0, stream>>>(
            qw, sc, zr, reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()),
            M, (int)K, (int)group);
    } else if (dtype == torch::kFloat) {
        int4_dequant_kernel<float><<<M, threads, 0, stream>>>(
            qw, sc, zr, out.data_ptr<float>(), M, (int)K, (int)group);
    } else {
        TORCH_CHECK(false, "int4_dequant: unsupported output dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int4_gemv(torch::Tensor qweight, torch::Tensor scales,
                        torch::Tensor zeros, torch::Tensor x,
                        int64_t K, int64_t group) {
    CHECK_CUDA(qweight); CHECK_CUDA(x);
    TORCH_CHECK(K % group == 0, "K must be divisible by the group size");
    TORCH_CHECK(group % 16 == 0, "group size must be a multiple of 16");
    auto xc = x.dim() == 1 ? x.reshape({1, -1}) : x;
    xc = xc.to(torch::kFloat).contiguous();
    const int M = qweight.size(0);
    const int N = xc.size(0);
    const int threads = threads_for(K);
    const int nwarps = (threads + 31) / 32;
    const int splits = splits_for(M, (int)K, (int)qweight.get_device());
    auto out = splits == 1 ? torch::empty({N, M}, xc.options())
                           : torch::zeros({N, M}, xc.options());
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_gemv_kernel<ROWS_PER_BLOCK>
        <<<grid, threads, ROWS_PER_BLOCK * nwarps * sizeof(float), stream>>>(
            qweight.data_ptr<unsigned char>(),
            reinterpret_cast<const __half *>(scales.data_ptr()),
            zeros.data_ptr<unsigned char>(), xc.data_ptr<float>(),
            out.data_ptr<float>(), M, (int)K, N, (int)group, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x.dim() == 1 ? out.squeeze(0) : out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_dequant", &nvfp4_dequant, "NVFP4 -> dense dequantization");
    m.def("nvfp4_gemv", &nvfp4_gemv, "fused NVFP4 dequantize + matvec");
    m.def("int4_dequant", &int4_dequant, "INT4 group affine -> dense");
    m.def("int4_gemv", &int4_gemv, "fused INT4 dequantize + matvec");
}
