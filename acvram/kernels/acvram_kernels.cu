// Noyaux acvram de déquantification fusionnée.
//
// Deux formats, parce que les deux GPU de la machine cible ne peuvent pas
// exécuter le même :
//
//   NVFP4  éléments E2M1 + échelle E4M3 tous les 16        RTX 5090    sm_120
//   INT4   éléments uint4 + échelle/zéro fp16 tous les 128 RTX 3080 Ti sm_86
//
// Chaque format reçoit deux points d'entrée :
//
//   *_dequant  matérialise la matrice entière dans un type de calcul, puis
//              laisse cuBLAS faire le produit. C'est le chemin du prefill : le
//              coût de déquantification est amorti sur tout le lot et cuBLAS
//              bat tout produit écrit à la main.
//
//   *_gemv     déquantification et multiplication fusionnées pour un lot de 1
//              à 8. C'est le chemin du décodage, purement limité par la
//              mémoire : il s'agit de lire les poids 4 bits directement depuis
//              la mémoire globale sans jamais en écrire une copie 16 bits.
//
// Trois choses rendent le GEMV rapide, et les trois comptent :
//
//   * Des chargements vectoriels de 8 octets. Un `uint2` porte 16 poids
//     empaquetés, ce qui fait exactement un bloc d'échelle NVFP4 et un nombre
//     entier de groupes INT4 : un fil ne chevauche donc jamais une frontière
//     d'échelle, et l'échelle est lue une fois par chargement au lieu d'une
//     fois par poids.
//   * Plusieurs lignes de sortie par bloc. La tranche d'activation est lue une
//     fois et réutilisée sur ROWS lignes, divisant d'autant le trafic
//     d'activation. Le trafic de poids est incompressible ; celui des
//     activations ne l'est pas.
//   * Une découpe sur K quand la matrice est courte. Un GEMV de 4096 lignes
//     lance 4096/ROWS blocs, ce qui va bien sur une carte à 170 multiprocesseurs
//     avec 4 lignes par bloc ; mais les petites projections d'un bloc
//     d'attention à requêtes groupées laisseraient l'essentiel de la carte
//     oisive, elles découpent donc la réduction à la place.
//
// La numérique doit correspondre exactement aux implémentations de référence
// d'acvram/quant/nvfp4.py et d'acvram/quant/int4.py ; tests/test_improvements.py
// le vérifie face au chemin PyTorch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace {

constexpr int WARP = 32;
constexpr int ROWS_PER_BLOCK = 4;
constexpr int WEIGHTS_PER_LOAD = 16;      // one uint2

// Magnitudes E2M1, indexées par le champ de magnitude sur 3 bits.
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

// Quartet j d'un uint2 : les octets sont en petit-boutien, donc le quartet j se
// trouve au bit 4*j du mot bas pour j < 8, et du mot haut au-delà.
__device__ __forceinline__ unsigned int nibble(const uint2 &p, int j) {
    const unsigned int w = (j < 8) ? p.x : p.y;
    return (w >> ((j & 7) * 4)) & 0xFu;
}

// Un octet contient deux poids E2M1 ; Blackwell les convertit en half2 en
// une instruction. Sur les architectures sans elle, l'intrinseque retombe sur
// une emulation arithmetique — dans les deux cas, aucun acces memoire : la
// table en __constant__ qu'elle remplace se serialisait des que les fils d'un
// warp lisaient des entrees differentes, c'est-a-dire toujours.
__device__ __forceinline__ float2 e2m1_pair(unsigned char byte) {
    const __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1);
    const __half2 hv = *reinterpret_cast<const __half2 *>(&h2);
    return __half22float2(hv);
}

__device__ __forceinline__ float warp_reduce(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Réduit ROWS accumulateurs sur tout le bloc. `shared` doit contenir
// ROWS * (blockDim.x / WARP) flottants.
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

        // Deux blocs de 16 par itération : un uint4 charge 32 poids d'un
        // coup, soit une transaction de 16 octets par fil — la 5090 n'attei-
        // gnait qu'un quart de sa bande passante avec des lectures de 8.
        // Le découpage se fait en paires de blocs, jamais au milieu d'une.
        const int npairs = nloads >> 1;
        const int per_split_p = (npairs + k_splits - 1) / k_splits;
        const int lo_p = split * per_split_p;
        const int hi_p = min(npairs, lo_p + per_split_p);
        for (int i = lo_p + threadIdx.x; i < hi_p; i += blockDim.x) {
            float xs[2 * WEIGHTS_PER_LOAD];
            const float4 *x4 = reinterpret_cast<const float4 *>(
                xn + (long)i * 2 * WEIGHTS_PER_LOAD);
            #pragma unroll
            for (int c = 0; c < 2 * WEIGHTS_PER_LOAD / 4; ++c) {
                const float4 v = x4[c];
                xs[c * 4 + 0] = v.x; xs[c * 4 + 1] = v.y;
                xs[c * 4 + 2] = v.z; xs[c * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                const uint4 p4 = reinterpret_cast<const uint4 *>(
                    qw + (long)row * half_k)[i];
                const float s0 = e4m3_to_float(bscale[(long)row * nloads + 2 * i])
                                 * gscale;
                const float s1 = e4m3_to_float(bscale[(long)row * nloads + 2 * i + 1])
                                 * gscale;
                const unsigned int words[4] = {p4.x, p4.y, p4.z, p4.w};
                float part0 = 0.f, part1 = 0.f;
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const unsigned int w0 = words[b >> 2];
                    const float2 v0 = e2m1_pair((w0 >> ((b & 3) * 8)) & 0xFFu);
                    part0 += v0.x * xs[2 * b] + v0.y * xs[2 * b + 1];
                    const unsigned int w1 = words[2 + (b >> 2)];
                    const float2 v1 = e2m1_pair((w1 >> ((b & 3) * 8)) & 0xFFu);
                    part1 += v1.x * xs[WEIGHTS_PER_LOAD + 2 * b]
                           + v1.y * xs[WEIGHTS_PER_LOAD + 2 * b + 1];
                }
                acc[r] += part0 * s0 + part1 * s1;
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
        const int g = base / group;           // le groupe est un multiple de 16
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


// -------------------------------------------------------------------------
// INT8 affine par groupes (zeros pleins, non empaquetes)
// -------------------------------------------------------------------------

template <typename T>
__global__ void int8_dequant_kernel(
    const unsigned char *__restrict__ qw,     // [M, K]
    const __half *__restrict__ scales,        // [M, ng]
    const unsigned char *__restrict__ zeros,  // [M, ng]
    T *__restrict__ out,                      // [M, K]
    int M, int K, int group) {
    const long row = blockIdx.x;
    if (row >= M) return;
    const int nloads = K / WEIGHTS_PER_LOAD;  // 16 octets = un uint4
    const int ng = K / group;
    const uint4 *qrow = reinterpret_cast<const uint4 *>(qw + row * (long)K);
    const __half *srow = scales + row * (long)ng;
    const unsigned char *zrow = zeros + row * (long)ng;
    T *orow = out + row * (long)K;

    for (int i = threadIdx.x; i < nloads; i += blockDim.x) {
        const uint4 p = qrow[i];
        const int base = i * WEIGHTS_PER_LOAD;
        const int g = base / group;
        const float s = __half2float(srow[g]);
        const float z = static_cast<float>(zrow[g]);
        const unsigned int words[4] = {p.x, p.y, p.z, p.w};
        #pragma unroll
        for (int j = 0; j < WEIGHTS_PER_LOAD; ++j) {
            const unsigned int b = (words[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
            orow[base + j] = from_float<T>((static_cast<float>(b) - z) * s);
        }
    }
}

template <int ROWS>
__global__ void int8_gemv_kernel(
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
    const int split = blockIdx.y;
    const int per_split = (nloads + k_splits - 1) / k_splits;
    const int lo = split * per_split;
    const int hi = min(nloads, lo + per_split);

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
                const uint4 p = reinterpret_cast<const uint4 *>(
                    qw + (long)row * K)[i];
                const float s = __half2float(scales[(long)row * ng + g]);
                const float z = static_cast<float>(zeros[(long)row * ng + g]);
                const unsigned int words[4] = {p.x, p.y, p.z, p.w};
                float part = 0.f;
                #pragma unroll
                for (int j = 0; j < WEIGHTS_PER_LOAD; ++j) {
                    const unsigned int b = (words[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
                    part += (static_cast<float>(b) - z) * xs[j];
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
// GEMV groupés : une passe pour tous les experts actifs d'une couche MoE.
//
// Les poids des E experts d'une projection sont empilés en un tenseur
// contigu ; chaque tranche z de la grille traite une paire (jeton, expert
// actif) : `expert_ids[z]` choisit la matrice, `token_ids[z]` la ligne
// d'activation. Une couche MoE coûte ainsi trois lancements au lieu de
// trois par expert actif — la boucle Python par expert lançait un millier
// de petits noyaux par jeton décodé.
// -------------------------------------------------------------------------

template <int ROWS>
__global__ void nvfp4_gemv_grouped_kernel(
    const unsigned char *__restrict__ qw,     // [E, M, K/2] empile
    const unsigned char *__restrict__ bscale, // [E, M, K/16]
    const float *__restrict__ gscales,        // [E]
    const int *__restrict__ expert_ids,       // [G]
    const int *__restrict__ token_ids,        // [G]
    const float *__restrict__ x,              // [T, K]
    float *__restrict__ y,                    // [G, M]
    int M, int K, int k_splits) {
    extern __shared__ float smem[];
    const int nwarps = (blockDim.x + WARP - 1) / WARP;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    const int g = blockIdx.z;
    const int e = expert_ids[g];
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const unsigned char *qe = qw + (long)e * M * half_k;
    const unsigned char *be = bscale + (long)e * M * nloads;
    const float gscale = gscales[e];
    const float *xn = x + (long)token_ids[g] * K;
    const int split = blockIdx.y;

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

    const int npairs = nloads >> 1;
    const int per_split_p = (npairs + k_splits - 1) / k_splits;
    const int lo_p = split * per_split_p;
    const int hi_p = min(npairs, lo_p + per_split_p);
    for (int i = lo_p + threadIdx.x; i < hi_p; i += blockDim.x) {
        float xs[2 * WEIGHTS_PER_LOAD];
        const float4 *x4 = reinterpret_cast<const float4 *>(
            xn + (long)i * 2 * WEIGHTS_PER_LOAD);
        #pragma unroll
        for (int c = 0; c < 2 * WEIGHTS_PER_LOAD / 4; ++c) {
            const float4 v = x4[c];
            xs[c * 4 + 0] = v.x; xs[c * 4 + 1] = v.y;
            xs[c * 4 + 2] = v.z; xs[c * 4 + 3] = v.w;
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row >= M) continue;
            const uint4 p4 = reinterpret_cast<const uint4 *>(
                qe + (long)row * half_k)[i];
            const float s0 = e4m3_to_float(be[(long)row * nloads + 2 * i])
                             * gscale;
            const float s1 = e4m3_to_float(be[(long)row * nloads + 2 * i + 1])
                             * gscale;
            const unsigned int words[4] = {p4.x, p4.y, p4.z, p4.w};
            float part0 = 0.f, part1 = 0.f;
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned int w0 = words[b >> 2];
                const float2 v0 = e2m1_pair((w0 >> ((b & 3) * 8)) & 0xFFu);
                part0 += v0.x * xs[2 * b] + v0.y * xs[2 * b + 1];
                const unsigned int w1 = words[2 + (b >> 2)];
                const float2 v1 = e2m1_pair((w1 >> ((b & 3) * 8)) & 0xFFu);
                part1 += v1.x * xs[WEIGHTS_PER_LOAD + 2 * b]
                       + v1.y * xs[WEIGHTS_PER_LOAD + 2 * b + 1];
            }
            acc[r] += part0 * s0 + part1 * s1;
        }
    }

    block_reduce_rows<ROWS>(acc, smem, nwarps);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row >= M) continue;
            if (k_splits == 1) y[(long)g * M + row] = acc[r];
            else atomicAdd(&y[(long)g * M + row], acc[r]);
        }
    }
}

template <int ROWS>
__global__ void int4_gemv_grouped_kernel(
    const unsigned char *__restrict__ qw,     // [E, M, K/2]
    const __half *__restrict__ scales,        // [E, M, ng]
    const unsigned char *__restrict__ zeros,  // [E, M, zb]
    const int *__restrict__ expert_ids,
    const int *__restrict__ token_ids,
    const float *__restrict__ x,
    float *__restrict__ y,
    int M, int K, int group, int k_splits) {
    extern __shared__ float smem[];
    const int nwarps = (blockDim.x + WARP - 1) / WARP;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    const int g = blockIdx.z;
    const int e = expert_ids[g];
    const int nloads = K / WEIGHTS_PER_LOAD;
    const int ng = K / group;
    const int zbytes = (ng + 1) >> 1;
    const long half_k = (long)K >> 1;
    const unsigned char *qe = qw + (long)e * M * half_k;
    const __half *se = scales + (long)e * M * ng;
    const unsigned char *ze = zeros + (long)e * M * zbytes;
    const float *xn = x + (long)token_ids[g] * K;
    const int split = blockIdx.y;
    const int per_split = (nloads + k_splits - 1) / k_splits;
    const int lo = split * per_split;
    const int hi = min(nloads, lo + per_split);

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
        const int gq = (i * WEIGHTS_PER_LOAD) / group;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row >= M) continue;
            const uint2 p = reinterpret_cast<const uint2 *>(
                qe + (long)row * half_k)[i];
            const float sc = __half2float(se[(long)row * ng + gq]);
            const float z = group_zero(ze + (long)row * zbytes, gq);
            float part = 0.f;
            #pragma unroll
            for (int j = 0; j < WEIGHTS_PER_LOAD; ++j)
                part += (static_cast<float>(nibble(p, j)) - z) * xs[j];
            acc[r] += part * sc;
        }
    }

    block_reduce_rows<ROWS>(acc, smem, nwarps);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row >= M) continue;
            if (k_splits == 1) y[(long)g * M + row] = acc[r];
            else atomicAdd(&y[(long)g * M + row], acc[r]);
        }
    }
}

int threads_for(int K) {
    const int nloads = K / WEIGHTS_PER_LOAD;
    int t = 256;
    while (t > 64 && t > nloads) t >>= 1;
    return t;
}

// Le GEMV NVFP4 marche par paires de blocs (32 poids) : moitié moins
// d'iterations que de blocs. Dimensionner ses fils sur les blocs simples en
// laissait 96 sur 256 sans travail pour K = 5120.
int threads_for_pairs(int K) {
    const int npairs = K / (2 * WEIGHTS_PER_LOAD);
    int t = 256;
    while (t > 64 && t > npairs) t >>= 1;
    return t;
}

// Assez de blocs pour occuper la carte, sans découper quand c'est inutile.
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

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " doit resider sur un peripherique CUDA")

// Le flux courant appartient au *peripherique courant*, pas a celui des
// tenseurs. Sans ce garde, un appel visant la seconde carte lance son noyau
// sur la premiere et lit des adresses qui n'existent pas chez elle : une
// machine a un seul GPU ne le voit jamais, une machine a deux GPU plante des
// le premier appel. Il est donc obligatoire en tete de chaque point d'entree.
#define ACVRAM_DEVICE_GUARD(x) const at::cuda::CUDAGuard acvram_guard((x).device())
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " doit etre contigu")

torch::Tensor nvfp4_dequant(torch::Tensor qweight, torch::Tensor block_scale,
                            double global_scale, int64_t K,
                            c10::ScalarType dtype) {
    CHECK_CUDA(qweight); CHECK_CUDA(block_scale);
    ACVRAM_DEVICE_GUARD(qweight);
    CHECK_CONTIG(qweight); CHECK_CONTIG(block_scale);
    TORCH_CHECK(K % 16 == 0, "le NVFP4 exige K divisible par 16, recu ", K);
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
        TORCH_CHECK(false, "nvfp4_dequant : type de sortie non gere");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nvfp4_gemv(torch::Tensor qweight, torch::Tensor block_scale,
                         double global_scale, torch::Tensor x, int64_t K) {
    CHECK_CUDA(qweight); CHECK_CUDA(x);
    ACVRAM_DEVICE_GUARD(qweight);
    CHECK_CONTIG(qweight); CHECK_CONTIG(block_scale);
    TORCH_CHECK(K % 16 == 0, "le NVFP4 exige K divisible par 16");
    auto xc = x.dim() == 1 ? x.reshape({1, -1}) : x;
    xc = xc.to(torch::kFloat).contiguous();
    const int M = qweight.size(0);
    const int N = xc.size(0);
    const int threads = threads_for_pairs(K);
    const int nwarps = (threads + 31) / 32;
    const int splits = splits_for(M, (int)K, (int)qweight.get_device());
    auto out = splits == 1 ? torch::empty({N, M}, xc.options())
                           : torch::zeros({N, M}, xc.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    // Huit lignes par bloc ont ete essayees pour reutiliser davantage la
    // tranche d'activation : la pression de registres l'emporte, mesure plus
    // lent sur toutes les formes. Quatre lignes restent l'optimum ici.
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
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
    ACVRAM_DEVICE_GUARD(qweight);
    CHECK_CONTIG(qweight); CHECK_CONTIG(scales); CHECK_CONTIG(zeros);
    TORCH_CHECK(K % group == 0, "K doit etre divisible par la taille de groupe");
    TORCH_CHECK(group % 16 == 0, "la taille de groupe doit etre un multiple de 16");
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
        TORCH_CHECK(false, "int4_dequant : type de sortie non gere");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int4_gemv(torch::Tensor qweight, torch::Tensor scales,
                        torch::Tensor zeros, torch::Tensor x,
                        int64_t K, int64_t group) {
    CHECK_CUDA(qweight); CHECK_CUDA(x);
    ACVRAM_DEVICE_GUARD(qweight);
    TORCH_CHECK(K % group == 0, "K doit etre divisible par la taille de groupe");
    TORCH_CHECK(group % 16 == 0, "la taille de groupe doit etre un multiple de 16");
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


torch::Tensor int8_dequant(torch::Tensor qweight, torch::Tensor scales,
                           torch::Tensor zeros, int64_t group,
                           c10::ScalarType dtype) {
    CHECK_CUDA(qweight); CHECK_CUDA(scales); CHECK_CUDA(zeros);
    ACVRAM_DEVICE_GUARD(qweight);
    CHECK_CONTIG(qweight); CHECK_CONTIG(scales); CHECK_CONTIG(zeros);
    const int M = qweight.size(0);
    const int K = qweight.size(1);
    TORCH_CHECK(K % group == 0, "K doit etre divisible par la taille de groupe");
    TORCH_CHECK(group % 16 == 0, "la taille de groupe doit etre un multiple de 16");
    auto out = torch::empty({M, K}, qweight.options().dtype(dtype));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = threads_for(K);
    const auto *qw = qweight.data_ptr<unsigned char>();
    const auto *sc = reinterpret_cast<const __half *>(scales.data_ptr());
    const auto *zr = zeros.data_ptr<unsigned char>();
    if (dtype == torch::kHalf) {
        int8_dequant_kernel<__half><<<M, threads, 0, stream>>>(
            qw, sc, zr, reinterpret_cast<__half *>(out.data_ptr()),
            M, K, (int)group);
    } else if (dtype == torch::kBFloat16) {
        int8_dequant_kernel<__nv_bfloat16><<<M, threads, 0, stream>>>(
            qw, sc, zr, reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()),
            M, K, (int)group);
    } else if (dtype == torch::kFloat) {
        int8_dequant_kernel<float><<<M, threads, 0, stream>>>(
            qw, sc, zr, out.data_ptr<float>(), M, K, (int)group);
    } else {
        TORCH_CHECK(false, "int8_dequant : type de sortie non gere");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int8_gemv(torch::Tensor qweight, torch::Tensor scales,
                        torch::Tensor zeros, torch::Tensor x, int64_t group) {
    CHECK_CUDA(qweight); CHECK_CUDA(x);
    ACVRAM_DEVICE_GUARD(qweight);
    CHECK_CONTIG(qweight); CHECK_CONTIG(scales); CHECK_CONTIG(zeros);
    const int K = qweight.size(1);
    TORCH_CHECK(K % group == 0, "K doit etre divisible par la taille de groupe");
    TORCH_CHECK(group % 16 == 0, "la taille de groupe doit etre un multiple de 16");
    auto xc = x.dim() == 1 ? x.reshape({1, -1}) : x;
    xc = xc.to(torch::kFloat).contiguous();
    const int M = qweight.size(0);
    const int N = xc.size(0);
    const int threads = threads_for(K);
    const int nwarps = (threads + 31) / 32;
    const int splits = splits_for(M, K, (int)qweight.get_device());
    auto out = splits == 1 ? torch::empty({N, M}, xc.options())
                           : torch::zeros({N, M}, xc.options());
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    auto stream = at::cuda::getCurrentCUDAStream();
    int8_gemv_kernel<ROWS_PER_BLOCK>
        <<<grid, threads, ROWS_PER_BLOCK * nwarps * sizeof(float), stream>>>(
            qweight.data_ptr<unsigned char>(),
            reinterpret_cast<const __half *>(scales.data_ptr()),
            zeros.data_ptr<unsigned char>(), xc.data_ptr<float>(),
            out.data_ptr<float>(), M, K, N, (int)group, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x.dim() == 1 ? out.squeeze(0) : out;
}

torch::Tensor nvfp4_gemv_grouped(torch::Tensor qw, torch::Tensor bscale,
                                 torch::Tensor gscales,
                                 torch::Tensor expert_ids,
                                 torch::Tensor token_ids,
                                 torch::Tensor x, int64_t K) {
    CHECK_CUDA(qw); CHECK_CUDA(x);
    ACVRAM_DEVICE_GUARD(qw);
    CHECK_CONTIG(qw); CHECK_CONTIG(bscale); CHECK_CONTIG(x);
    TORCH_CHECK(K % 32 == 0, "le chemin groupe exige K divisible par 32");
    const int M = qw.size(1);
    const int G = expert_ids.size(0);
    auto xc = x.to(torch::kFloat).contiguous();
    const int threads = threads_for_pairs((int)K);
    const int nwarps = (threads + 31) / 32;
    // G tranches occupent deja la grille : pas de decoupage en profondeur.
    auto out = torch::empty({G, M}, xc.options());
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, G);
    auto stream = at::cuda::getCurrentCUDAStream();
    nvfp4_gemv_grouped_kernel<ROWS_PER_BLOCK>
        <<<grid, threads, ROWS_PER_BLOCK * nwarps * sizeof(float), stream>>>(
            qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(),
            gscales.data_ptr<float>(), expert_ids.data_ptr<int>(),
            token_ids.data_ptr<int>(), xc.data_ptr<float>(),
            out.data_ptr<float>(), M, (int)K, 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int4_gemv_grouped(torch::Tensor qw, torch::Tensor scales,
                                torch::Tensor zeros,
                                torch::Tensor expert_ids,
                                torch::Tensor token_ids,
                                torch::Tensor x, int64_t K, int64_t group) {
    CHECK_CUDA(qw); CHECK_CUDA(x);
    ACVRAM_DEVICE_GUARD(qw);
    CHECK_CONTIG(qw); CHECK_CONTIG(scales); CHECK_CONTIG(zeros);
    TORCH_CHECK(K % group == 0, "K doit etre divisible par la taille de groupe");
    const int M = qw.size(1);
    const int G = expert_ids.size(0);
    auto xc = x.to(torch::kFloat).contiguous();
    const int threads = threads_for((int)K);
    const int nwarps = (threads + 31) / 32;
    auto out = torch::empty({G, M}, xc.options());
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, G);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_gemv_grouped_kernel<ROWS_PER_BLOCK>
        <<<grid, threads, ROWS_PER_BLOCK * nwarps * sizeof(float), stream>>>(
            qw.data_ptr<unsigned char>(),
            reinterpret_cast<const __half *>(scales.data_ptr()),
            zeros.data_ptr<unsigned char>(), expert_ids.data_ptr<int>(),
            token_ids.data_ptr<int>(), xc.data_ptr<float>(),
            out.data_ptr<float>(), M, (int)K, (int)group, 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_dequant", &nvfp4_dequant, "NVFP4 -> matrice dense");
    m.def("nvfp4_gemv", &nvfp4_gemv, "NVFP4 : dequantification + produit fusionnes");
    m.def("int4_dequant", &int4_dequant, "INT4 affine par groupes -> matrice dense");
    m.def("int4_gemv", &int4_gemv, "INT4 : dequantification + produit fusionnes");
    m.def("int8_dequant", &int8_dequant, "INT8 affine par groupes -> matrice dense");
    m.def("int8_gemv", &int8_gemv, "INT8 : dequantification + produit fusionnes");
    m.def("nvfp4_gemv_grouped", &nvfp4_gemv_grouped,
          "NVFP4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
    m.def("int4_gemv_grouped", &int4_gemv_grouped,
          "INT4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
}
