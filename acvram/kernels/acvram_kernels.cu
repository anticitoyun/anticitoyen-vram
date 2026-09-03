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
#include <cstdlib>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <mma.h>

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

// Une ligne par bloc ; chaque fil déquantifie 16 poids (un uint2) et les
// écrit d'un coup (deux uint4 en 16 bits, quatre en fp32) : les écritures
// scalaires coûtaient 8 transactions par warp au lieu d'une. ``gscale_rows``
// (optionnel) donne une échelle par groupe de ``rows_per_group`` lignes —
// la pile d'experts d'un MoE, dont l'échelle globale diffère par expert —
// ce qui épargne une passe de multiplication sur la sortie.
template <typename T>
__global__ void nvfp4_dequant_kernel(
    const unsigned char *__restrict__ qw,     // [M, K/2]
    const unsigned char *__restrict__ bscale, // [M, K/16] raw E4M3 bytes
    const float gscale,
    const float *__restrict__ gscale_rows,    // nullptr ou [M / rows_per_group]
    const int rows_per_group,
    T *__restrict__ out,                      // [M, K]
    int M, int K) {
    const long row = blockIdx.x;
    if (row >= M) return;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const uint2 *qrow = reinterpret_cast<const uint2 *>(qw + row * (long)(K >> 1));
    const unsigned char *srow = bscale + row * (long)nloads;
    T *orow = out + row * (long)K;
    const float g = gscale_rows ? gscale_rows[row / rows_per_group] : gscale;

    for (int i = threadIdx.x; i < nloads; i += blockDim.x) {
        const uint2 p = qrow[i];
        const float s = e4m3_to_float(srow[i]) * g;
        const int base = i * WEIGHTS_PER_LOAD;
        __align__(16) T tmp[WEIGHTS_PER_LOAD];
        #pragma unroll
        for (int j = 0; j < WEIGHTS_PER_LOAD; ++j) {
            const unsigned int c = nibble(p, j);
            float v = kE2M1[c & 7u] * s;
            if (c & 8u) v = -v;
            tmp[j] = from_float<T>(v);
        }
        const uint4 *src = reinterpret_cast<const uint4 *>(tmp);
        uint4 *dst = reinterpret_cast<uint4 *>(orow + base);
        #pragma unroll
        for (int q = 0; q < (int)(WEIGHTS_PER_LOAD * sizeof(T) / 16); ++q)
            dst[q] = src[q];
    }
}

// y[n, m] = sum_k W[m, k] * x[n, k]

// Chargement de n activations (n multiple de 8) en float, depuis float ou
// bf16 : lire l'activation en bf16 épargne la conversion préalable et la
// moitié du trafic L1/L2 sur le vecteur partagé par tous les blocs.
template <typename XT, int NX>
__device__ __forceinline__ void load_xs(const XT *__restrict__ p, float *xs) {
    if constexpr (sizeof(XT) == 4) {
        const float4 *x4 = reinterpret_cast<const float4 *>(p);
        #pragma unroll
        for (int c = 0; c < NX / 4; ++c) {
            const float4 v = x4[c];
            xs[c * 4 + 0] = v.x; xs[c * 4 + 1] = v.y;
            xs[c * 4 + 2] = v.z; xs[c * 4 + 3] = v.w;
        }
    } else {
        const uint4 *x8 = reinterpret_cast<const uint4 *>(p);
        #pragma unroll
        for (int c = 0; c < NX / 8; ++c) {
            const uint4 v = x8[c];
            const unsigned int w[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162 *>(&w[j]);
                const float2 f = __bfloat1622float2(b2);
                xs[c * 8 + 2 * j] = f.x; xs[c * 8 + 2 * j + 1] = f.y;
            }
        }
    }
}
template <typename YT>
__device__ __forceinline__ void store_y(YT *p, float v) {
    if constexpr (sizeof(YT) == 4) *p = v; else *p = __float2bfloat16(v);
}

template <int ROWS, typename XT, typename YT>
__global__ void nvfp4_gemv_kernel(
    const unsigned char *__restrict__ qw,
    const unsigned char *__restrict__ bscale,
    const float gscale,
    const XT *__restrict__ x,                 // [N, K]
    YT *__restrict__ y,                       // [N, M]
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
        const XT *xn = x + (long)n * K;
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
            load_xs<XT, 2 * WEIGHTS_PER_LOAD>(xn + (long)i * 2 * WEIGHTS_PER_LOAD, xs);
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
                if (k_splits == 1) store_y(y + (long)n * M + row, acc[r]);
                else atomicAdd(reinterpret_cast<float *>(y) + (long)n * M + row, acc[r]);
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

template <int ROWS, typename XT = float, typename YT = float>
__global__ void int4_gemv_kernel(
    const unsigned char *__restrict__ qw,
    const __half *__restrict__ scales,
    const unsigned char *__restrict__ zeros,
    const XT *__restrict__ x,
    YT *__restrict__ y,
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
        const XT *xn = x + (long)n * K;
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
                if (k_splits == 1) store_y(y + (long)n * M + row, acc[r]);
                else atomicAdd(reinterpret_cast<float *>(y) + (long)n * M + row, acc[r]);
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

template <int ROWS, typename XT, typename YT>
__global__ void int8_gemv_kernel(
    const unsigned char *__restrict__ qw,
    const __half *__restrict__ scales,
    const unsigned char *__restrict__ zeros,
    const XT *__restrict__ x,
    YT *__restrict__ y,
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
        const XT *xn = x + (long)n * K;
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            float xs[WEIGHTS_PER_LOAD];
            load_xs<XT, WEIGHTS_PER_LOAD>(xn + (long)i * WEIGHTS_PER_LOAD, xs);
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
                if (k_splits == 1) store_y(y + (long)n * M + row, acc[r]);
                else atomicAdd(reinterpret_cast<float *>(y) + (long)n * M + row, acc[r]);
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

// -------------------------------------------------------------------------
// Attention paginée fusionnée (décodage, un jeton par séquence).
//
// Le chemin PyTorch relisait le cache KV INT8, le matérialisait en bf16 puis
// le redonnait à SDPA : plusieurs passes mémoire sur tout le contexte, à
// chaque couche, à chaque jeton. Ici les blocs INT8 sont lus une fois,
// déquantifiés en registres, et l'attention se calcule en ligne (softmax
// incrémental, à la flash-decoding). Le contexte est découpé en tranches de
// PA_CHUNK positions traitées par des blocs indépendants ; un second noyau
// combine les tranches par log-somme-exp. Les formes ne dépendent que du
// godet de blocs : le chemin se rejoue tel quel dans un graphe CUDA.
// -------------------------------------------------------------------------

constexpr int PA_CHUNK = 512;
constexpr int PA_WARPS = 4;

template <int D>
__global__ void paged_attn_partial_kernel(
    const float *__restrict__ q,          // [B*QL, HQ, D]
    const signed char *__restrict__ kc,   // [NB, 16, HKV, D]
    const __half *__restrict__ ks,        // [NB, 16, HKV]
    const signed char *__restrict__ vc,
    const __half *__restrict__ vs,
    const long *__restrict__ tables,      // [B, N]
    const long *__restrict__ seq_lens,    // [B]  longueur TOTALE (dernier jeton inclus)
    float *__restrict__ part,             // [B*QL, HQ, C, D]  acc non normalisé
    float *__restrict__ part_m,           // [B*QL, HQ, C]
    float *__restrict__ part_l,           // [B*QL, HQ, C]
    int HQ, int HKV, int N, int C, int QL, float scale, int window) {
    // La vérification spéculative pose QL positions de requête par séquence :
    // la requête qi (0..QL-1) ne voit que les seq_len-QL+1+qi premières
    // positions — causalité oblige. QL=1 redonne le décodage ordinaire.
    const int bq = blockIdx.x;            // b*QL + qi
    const int b = bq / QL;
    const int qi = bq % QL;
    const int h = blockIdx.y;
    const int c = blockIdx.z;
    const int hkv = h / (HQ / HKV);
    const long slen = seq_lens[b] - (QL - 1) + qi;
    const long lo = window > 0 ? max(0L, slen - (long)window) : 0L;   // fenêtre glissante
    const long start = max((long)c * PA_CHUNK, lo);
    const long out_off = ((long)bq * HQ + h) * C + c;

    const int lane = threadIdx.x % WARP;
    const int wid = threadIdx.x / WARP;
    constexpr int PER_LANE = D / WARP;

    if (start >= slen || start >= (long)(c + 1) * PA_CHUNK) {
        if (threadIdx.x == 0) {
            part_m[out_off] = -INFINITY;
            part_l[out_off] = 0.f;
        }
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            part[out_off * D + d] = 0.f;
        return;
    }

    __shared__ float sq[D];
    __shared__ float sm[PA_WARPS], sl[PA_WARPS], scorr[PA_WARPS];
    __shared__ float sacc[PA_WARPS][D];
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        sq[d] = q[((long)bq * HQ + h) * D + d] * scale;
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float acc[PER_LANE];
    #pragma unroll
    for (int i = 0; i < PER_LANE; ++i) acc[i] = 0.f;

    const long end = min(slen, (long)(c + 1) * PA_CHUNK);
    for (long t = start + wid; t < end; t += PA_WARPS) {
        const long blk = tables[(long)b * N + (t >> 4)];
        const long cell = (blk * 16 + (t & 15)) * HKV + hkv;
        const signed char *kp = kc + cell * D;

        float partial = 0.f;
        #pragma unroll
        for (int i = 0; i < PER_LANE; ++i)
            partial += sq[lane * PER_LANE + i]
                       * static_cast<float>(kp[lane * PER_LANE + i]);
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, off);
        const float score = __shfl_sync(0xffffffffu, partial, 0)
                            * __half2float(ks[cell]);

        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float pr = __expf(score - m_new);
        const signed char *vp = vc + cell * D;
        const float pv = __half2float(vs[cell]) * pr;
        #pragma unroll
        for (int i = 0; i < PER_LANE; ++i)
            acc[i] = acc[i] * corr
                     + pv * static_cast<float>(vp[lane * PER_LANE + i]);
        l = l * corr + pr;
        m = m_new;
    }

    if (lane == 0) { sm[wid] = m; sl[wid] = l; }
    #pragma unroll
    for (int i = 0; i < PER_LANE; ++i)
        sacc[wid][lane * PER_LANE + i] = acc[i];
    __syncthreads();

    if (threadIdx.x == 0) {
        float mg = -INFINITY;
        #pragma unroll
        for (int w = 0; w < PA_WARPS; ++w) mg = fmaxf(mg, sm[w]);
        float lg = 0.f;
        #pragma unroll
        for (int w = 0; w < PA_WARPS; ++w) {
            const float cw = (sm[w] > -INFINITY) ? __expf(sm[w] - mg) : 0.f;
            scorr[w] = cw;
            lg += sl[w] * cw;
        }
        part_m[out_off] = mg;
        part_l[out_off] = lg;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = 0.f;
        #pragma unroll
        for (int w = 0; w < PA_WARPS; ++w) a += sacc[w][d] * scorr[w];
        part[out_off * D + d] = a;
    }
}

template <int D>
__global__ void paged_attn_reduce_kernel(
    const float *__restrict__ part,       // [B, HQ, C, D]
    const float *__restrict__ part_m,
    const float *__restrict__ part_l,
    float *__restrict__ out,              // [B, HQ, D]
    int HQ, int C) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long base = (long)b * HQ + h;
    __shared__ float s_m, s_l;
    __shared__ float s_corr[256];         // C <= 256 tranches (2 M de contexte)

    if (threadIdx.x == 0) {
        float mg = -INFINITY;
        for (int c = 0; c < C; ++c) mg = fmaxf(mg, part_m[base * C + c]);
        float lg = 0.f;
        for (int c = 0; c < C; ++c) {
            const float mc = part_m[base * C + c];
            const float cw = (mc > -INFINITY) ? __expf(mc - mg) : 0.f;
            s_corr[c] = cw;
            lg += part_l[base * C + c] * cw;
        }
        s_m = mg;
        s_l = fmaxf(lg, 1e-20f);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = 0.f;
        for (int c = 0; c < C; ++c)
            a += part[(base * C + c) * D + d] * s_corr[c];
        out[base * D + d] = a / s_l;
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
                            c10::ScalarType dtype,
                            c10::optional<torch::Tensor> gscale_rows,
                            int64_t rows_per_group) {
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
    const float *gr = nullptr;
    if (gscale_rows.has_value()) {
        CHECK_CUDA(*gscale_rows); CHECK_CONTIG(*gscale_rows);
        TORCH_CHECK(gscale_rows->scalar_type() == torch::kFloat, "gscale_rows doit etre fp32");
        TORCH_CHECK(rows_per_group > 0 && gscale_rows->numel() * rows_per_group >= M,
                    "gscale_rows trop court");
        gr = gscale_rows->data_ptr<float>();
    }
    const int rpg = (int)std::max<int64_t>(rows_per_group, 1);

    if (dtype == torch::kBFloat16) {
        nvfp4_dequant_kernel<__nv_bfloat16><<<M, threads, 0, stream>>>(
            qw, bs, (float)global_scale, gr, rpg,
            reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), M, (int)K);
    } else if (dtype == torch::kHalf) {
        nvfp4_dequant_kernel<__half><<<M, threads, 0, stream>>>(
            qw, bs, (float)global_scale, gr, rpg,
            reinterpret_cast<__half *>(out.data_ptr()), M, (int)K);
    } else if (dtype == torch::kFloat) {
        nvfp4_dequant_kernel<float><<<M, threads, 0, stream>>>(
            qw, bs, (float)global_scale, gr, rpg, out.data_ptr<float>(), M, (int)K);
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
    const int M = qweight.size(0);
    const int threads = threads_for_pairs(K);
    const int nwarps = (threads + 31) / 32;
    const int splits = splits_for(M, (int)K, (int)qweight.get_device());
    // Activation bf16 lue telle quelle, sortie bf16 : zéro conversion autour
    // du noyau (le décodage en enchaîne des centaines par pas).
    const bool bf = xc.scalar_type() == torch::kBFloat16 && splits == 1;
    xc = (bf ? xc : xc.to(torch::kFloat)).contiguous();
    const int N = xc.size(0);
    auto out = splits == 1 ? torch::empty({N, M}, xc.options())
                           : torch::zeros({N, M}, xc.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    // Huit lignes par bloc ont ete essayees pour reutiliser davantage la
    // tranche d'activation : la pression de registres l'emporte, mesure plus
    // lent sur toutes les formes. Quatre lignes restent l'optimum ici.
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    const size_t shm = ROWS_PER_BLOCK * nwarps * sizeof(float);
    if (bf) {
        nvfp4_gemv_kernel<ROWS_PER_BLOCK, __nv_bfloat16, __nv_bfloat16>
            <<<grid, threads, shm, stream>>>(
                qweight.data_ptr<unsigned char>(),
                block_scale.data_ptr<unsigned char>(), (float)global_scale,
                reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()),
                reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), M, (int)K, N, 1);
    } else {
        nvfp4_gemv_kernel<ROWS_PER_BLOCK, float, float>
            <<<grid, threads, shm, stream>>>(
                qweight.data_ptr<unsigned char>(),
                block_scale.data_ptr<unsigned char>(), (float)global_scale,
                xc.data_ptr<float>(), out.data_ptr<float>(), M, (int)K, N, splits);
    }
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
    int4_gemv_kernel<ROWS_PER_BLOCK, float, float>
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
    const int M = qweight.size(0);
    const int threads = threads_for(K);
    const int nwarps = (threads + 31) / 32;
    const int splits = splits_for(M, K, (int)qweight.get_device());
    const bool bf = xc.scalar_type() == torch::kBFloat16 && splits == 1;
    xc = (bf ? xc : xc.to(torch::kFloat)).contiguous();
    const int N = xc.size(0);
    auto out = splits == 1 ? torch::empty({N, M}, xc.options())
                           : torch::zeros({N, M}, xc.options());
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t shm = ROWS_PER_BLOCK * nwarps * sizeof(float);
    if (bf) {
        int8_gemv_kernel<ROWS_PER_BLOCK, __nv_bfloat16, __nv_bfloat16>
            <<<grid, threads, shm, stream>>>(
                qweight.data_ptr<unsigned char>(),
                reinterpret_cast<const __half *>(scales.data_ptr()),
                zeros.data_ptr<unsigned char>(),
                reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()),
                reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), M, K, N, (int)group, 1);
    } else {
        int8_gemv_kernel<ROWS_PER_BLOCK, float, float>
            <<<grid, threads, shm, stream>>>(
                qweight.data_ptr<unsigned char>(),
                reinterpret_cast<const __half *>(scales.data_ptr()),
                zeros.data_ptr<unsigned char>(), xc.data_ptr<float>(),
                out.data_ptr<float>(), M, K, N, (int)group, splits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x.dim() == 1 ? out.squeeze(0) : out;
}


// Variante « un warp par ligne » du GEMV groupé, pour les projections MoE
// (K de 1 à 8k) : l'activation est copiée une fois en mémoire partagée,
// chaque warp balaie une ligne par uint4 (32 poids) par voie et réduit par
// shuffles — ni mémoire partagée par ligne, ni __syncthreads dans la
// boucle. Le noyau à 4 lignes par bloc restait latent sur ces formes
// (2 chargements par fil puis une réduction de bloc).

constexpr int XSH_GRP = 32;
constexpr int XSH_PAS = XSH_GRP + 1;

__device__ __forceinline__ int xsh_taille(int K) { return K + (K >> 5); }

// Produit d'une ligne NVFP4 par l'activation en mémoire partagée, un warp
// par ligne, deux uint4 en vol par voie (déroulage ×2 : les deux
// chargements partent avant le premier calcul).
__device__ __forceinline__ float nvfp4_row_dot_warp(
        const uint4 *__restrict__ wrow, const unsigned char *__restrict__ brow,
        float gscale, const float *__restrict__ xs_sh, int npairs, int lane) {
    float acc = 0.f;
    int i = lane;
    for (; i + WARP < npairs; i += 2 * WARP) {
        const uint4 pA = wrow[i], pB = wrow[i + WARP];
        const unsigned char bA0 = brow[2 * i], bA1 = brow[2 * i + 1];
        const unsigned char bB0 = brow[2 * (i + WARP)], bB1 = brow[2 * (i + WARP) + 1];
        const unsigned int wA[4] = {pA.x, pA.y, pA.z, pA.w};
        const unsigned int wB[4] = {pB.x, pB.y, pB.z, pB.w};
        const float *xA = xs_sh + (long)i * XSH_PAS;
        const float *xB = xs_sh + (long)(i + WARP) * XSH_PAS;
        float a0 = 0.f, a1 = 0.f, c0 = 0.f, c1 = 0.f;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float2 v0 = e2m1_pair((wA[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
            a0 += v0.x * xA[2 * b] + v0.y * xA[2 * b + 1];
            const float2 v1 = e2m1_pair((wA[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
            a1 += v1.x * xA[WEIGHTS_PER_LOAD + 2 * b] + v1.y * xA[WEIGHTS_PER_LOAD + 2 * b + 1];
            const float2 u0 = e2m1_pair((wB[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
            c0 += u0.x * xB[2 * b] + u0.y * xB[2 * b + 1];
            const float2 u1 = e2m1_pair((wB[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
            c1 += u1.x * xB[WEIGHTS_PER_LOAD + 2 * b] + u1.y * xB[WEIGHTS_PER_LOAD + 2 * b + 1];
        }
        acc += (a0 * e4m3_to_float(bA0) + a1 * e4m3_to_float(bA1)
              + c0 * e4m3_to_float(bB0) + c1 * e4m3_to_float(bB1)) * gscale;
    }
    for (; i < npairs; i += WARP) {
        const uint4 p4 = wrow[i];
        const float s0 = e4m3_to_float(brow[2 * i]) * gscale;
        const float s1 = e4m3_to_float(brow[2 * i + 1]) * gscale;
        const float *xp = xs_sh + (long)i * XSH_PAS;
        const unsigned int words[4] = {p4.x, p4.y, p4.z, p4.w};
        float part0 = 0.f, part1 = 0.f;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float2 v0 = e2m1_pair((words[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
            part0 += v0.x * xp[2 * b] + v0.y * xp[2 * b + 1];
            const float2 v1 = e2m1_pair((words[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
            part1 += v1.x * xp[WEIGHTS_PER_LOAD + 2 * b] + v1.y * xp[WEIGHTS_PER_LOAD + 2 * b + 1];
        }
        acc += part0 * s0 + part1 * s1;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// Le produit ligne-activation fait lire à la voie « l » les 32 flottants
// [l*32, l*32+32) : rangés à plat, ils tombent tous dans la même banque de
// mémoire partagée et le warp sérialise en 32 accès. Un flottant de bourrage
// tous les 32 décale chaque voie d'une banque — l'accès redevient un seul
// cycle. Indice décalé de « i » : i + (i >> 5).
template <typename XT>
__device__ __forceinline__ void charger_x_sh(const XT *__restrict__ xn, float *xs_sh, int K) {
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const int d = i + (i >> 5);
        if constexpr (sizeof(XT) == 4) xs_sh[d] = xn[i];
        else xs_sh[d] = __bfloat162float(xn[i]);
    }
    __syncthreads();
}

constexpr int GW_WARPS = 8;
// RPW lignes par warp, en boucle : l'activation n'est chargée qu'une fois
// par bloc pour GW_WARPS*RPW lignes (sinon son trafic L2 égale celui des
// poids sur ces petites projections).
template <typename XT, int RPW>
__global__ void nvfp4_gemv_grouped_warp_kernel(
    const unsigned char *__restrict__ qw, const unsigned char *__restrict__ bscale,
    const float *__restrict__ gscales, const int *__restrict__ expert_ids,
    const int *__restrict__ token_ids, const XT *__restrict__ x,
    float *__restrict__ y, int M, int K) {
    extern __shared__ float xs_sh[];
    const int g = blockIdx.y, e = expert_ids[g];
    charger_x_sh<XT>(x + (long)token_ids[g] * K, xs_sh, K);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const float gscale = gscales[e];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        const float acc = nvfp4_row_dot_warp(
            reinterpret_cast<const uint4 *>(qw + ((long)e * M + row) * half_k),
            bscale + ((long)e * M + row) * nloads, gscale, xs_sh, nloads >> 1, lane);
        if (lane == 0) y[(long)g * M + row] = acc;
    }
}

__device__ __forceinline__ float acv_act(float v, int act) {
    // 0 : SiLU (Qwen, DeepSeek…) ; 1 : GELU-tanh (Gemma 4)
    if (act == 1) {
        const float c = 0.7978845608028654f;
        return 0.5f * v * (1.f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
    return v / (1.f + __expf(-v));
}

// gate et up fusionnés : mêmes (expert, ligne), même activation ; la sortie
// est déjà act(gate) · up — trois lancements et deux tenseurs en moins.
template <typename XT, int RPW>
__global__ void nvfp4_gemv_grouped_gateup_kernel(
    const unsigned char *__restrict__ qg, const unsigned char *__restrict__ bg,
    const float *__restrict__ gsg,
    const unsigned char *__restrict__ qu, const unsigned char *__restrict__ bu,
    const float *__restrict__ gsu,
    const int *__restrict__ expert_ids, const int *__restrict__ token_ids,
    const XT *__restrict__ x, float *__restrict__ y, int M, int K, int act) {
    extern __shared__ float xs_sh[];
    const int g = blockIdx.y, e = expert_ids[g];
    charger_x_sh<XT>(x + (long)token_ids[g] * K, xs_sh, K);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        const long off = (long)e * M + row;
        const float ag = nvfp4_row_dot_warp(
            reinterpret_cast<const uint4 *>(qg + off * half_k), bg + off * nloads,
            gsg[e], xs_sh, nloads >> 1, lane);
        const float au = nvfp4_row_dot_warp(
            reinterpret_cast<const uint4 *>(qu + off * half_k), bu + off * nloads,
            gsu[e], xs_sh, nloads >> 1, lane);
        if (lane == 0) y[(long)g * M + row] = acv_act(ag, act) * au;
    }
}

torch::Tensor nvfp4_gemv_grouped_gateup(
        torch::Tensor qg, torch::Tensor bg, torch::Tensor gsg,
        torch::Tensor qu, torch::Tensor bu, torch::Tensor gsu,
        torch::Tensor expert_ids, torch::Tensor token_ids,
        torch::Tensor x, int64_t K, int64_t act) {
    CHECK_CUDA(qg); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qg);
    CHECK_CONTIG(qg); CHECK_CONTIG(qu); CHECK_CONTIG(bg); CHECK_CONTIG(bu);
    TORCH_CHECK(K % 32 == 0 && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024,
                "gate-up fusionne : K multiple de 32 et <= 11904");
    const int M = qg.size(1), G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 1;
    dim3 grid((M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
    const size_t shm = (size_t)(K + K / 32) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GU_LAUNCH(XT, PX) do { if (rpw == 1) GU_L(XT, 1, PX); else if (rpw == 2) GU_L(XT, 2, PX); else GU_L(XT, 4, PX); } while (0)
    #define GU_L(XT, R, PX) nvfp4_gemv_grouped_gateup_kernel<XT, R><<<grid, GW_WARPS * WARP, shm, stream>>>( \
        qg.data_ptr<unsigned char>(), bg.data_ptr<unsigned char>(), gsg.data_ptr<float>(), \
        qu.data_ptr<unsigned char>(), bu.data_ptr<unsigned char>(), gsu.data_ptr<float>(), \
        expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K, (int)act)
    if (bf) { GU_LAUNCH(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { GU_LAUNCH(float, xc.data_ptr<float>()); }
    #undef GU_LAUNCH
    #undef GU_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// --------------------------------------------------------------------------
// GEMM groupée NVFP4 : les experts d'une couche MoE, poids lus en 4 bits.
//
// Le chemin de prefill matérialisait la pile d'experts en bf16 (trois passes
// de plusieurs gigaoctets par couche) avant d'appeler torch._grouped_mm. Ici
// les poids ne sortent jamais des 4 bits : chaque bloc déquantifie une tuile
// 64x64 en mémoire partagée et la consomme immédiatement en tensor cores.
//
// Les jetons arrivent triés par expert. L'hôte découpe chaque expert en
// tuiles de BT jetons et transmet, par tuile, (expert, premier jeton, compte).
// --------------------------------------------------------------------------
namespace wmma = nvcuda::wmma;

constexpr int GG_BM = 64;      // lignes de sortie par bloc (4 warps x 16)
constexpr int GG_BT = 16;      // jetons par tuile (fragments de 16)
constexpr int GG_TT = GG_BT / 16;
constexpr int GG_KB = 64;      // profondeur traitée par itération
constexpr int GG_LDX = GG_KB + 8;
constexpr int GG_LDW = GG_KB + 8;

__device__ __forceinline__ float fp4_val(unsigned char nib) {
    const float m = kE2M1[nib & 7];
    return (nib & 8) ? -m : m;
}

__global__ void nvfp4_gemm_grouped_kernel(
    const unsigned char *__restrict__ qw, const unsigned char *__restrict__ bscale,
    const float *__restrict__ gscales, const __nv_bfloat16 *__restrict__ x,
    const int *__restrict__ tile_e, const int *__restrict__ tile_t0,
    const int *__restrict__ tile_n, __nv_bfloat16 *__restrict__ y,
    int M, int K) {
    __shared__ __nv_bfloat16 xs[GG_BT * GG_LDX];
    __shared__ __nv_bfloat16 ws[GG_BM * GG_LDW];
    __shared__ float ys[GG_BT * GG_BM];

    const int tile = blockIdx.y;
    const int e = tile_e[tile], t0 = tile_t0[tile], nt = tile_n[tile];
    const int row0 = blockIdx.x * GG_BM;
    const int tid = threadIdx.x, warp = tid >> 5;
    const float gscale = gscales[e];
    const long half_k = (long)K >> 1;
    const int nblk = K >> 4;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[GG_TT];
    #pragma unroll
    for (int j = 0; j < GG_TT; ++j) wmma::fill_fragment(acc[j], 0.f);

    const int lrow = tid >> 1, moitie = tid & 1;   // 128 fils : 2 par ligne
    const int wcol0 = moitie * 32;
    const int grow_ok = (row0 + lrow) < M;
    const long grow = (long)e * M + min(row0 + lrow, M - 1);

    for (int k0 = 0; k0 < K; k0 += GG_KB) {
        // --- activations [BT, KB] ---------------------------------------
        for (int i = tid; i < GG_BT * GG_KB; i += blockDim.x) {
            const int j = i / GG_KB, c = i - j * GG_KB;
            xs[j * GG_LDX + c] = (j < nt) ? x[(long)(t0 + j) * K + k0 + c]
                                          : __float2bfloat16(0.f);
        }
        // --- poids [BM, KB] déquantifiés ---------------------------------
        {
            const uint4 pk = *reinterpret_cast<const uint4 *>(
                qw + grow * half_k + ((k0 + wcol0) >> 1));
            const unsigned char *sc = bscale + grow * nblk + ((k0 + wcol0) >> 4);
            const float s0 = e4m3_to_float(sc[0]) * gscale;
            const float s1 = e4m3_to_float(sc[1]) * gscale;
            const unsigned char *o = reinterpret_cast<const unsigned char *>(&pk);
            __nv_bfloat16 *dst = ws + lrow * GG_LDW + wcol0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                const unsigned char b = o[i >> 1];
                const unsigned char nib = (i & 1) ? (b >> 4) : (b & 0xF);
                const float v = grow_ok ? fp4_val(nib) * (i < 16 ? s0 : s1) : 0.f;
                dst[i] = __float2bfloat16(v);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GG_KB; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(b, ws + (warp * 16) * GG_LDW + kk, GG_LDW);
            #pragma unroll
            for (int j = 0; j < GG_TT; ++j) {
                wmma::load_matrix_sync(a, xs + (j * 16) * GG_LDX + kk, GG_LDX);
                wmma::mma_sync(acc[j], a, b, acc[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < GG_TT; ++j)
        wmma::store_matrix_sync(ys + (j * 16) * GG_BM + warp * 16, acc[j], GG_BM,
                                wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < GG_BT * GG_BM; i += blockDim.x) {
        const int j = i / GG_BM, n = i - j * GG_BM;
        if (j < nt && row0 + n < M)
            y[(long)(t0 + j) * M + row0 + n] = __float2bfloat16(ys[i]);
    }
}

torch::Tensor nvfp4_gemm_grouped(torch::Tensor qw, torch::Tensor bscale,
                                 torch::Tensor gscales, torch::Tensor x,
                                 torch::Tensor tile_e, torch::Tensor tile_t0,
                                 torch::Tensor tile_n, int64_t K) {
    CHECK_CUDA(qw); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qw);
    CHECK_CONTIG(qw); CHECK_CONTIG(bscale); CHECK_CONTIG(x);
    TORCH_CHECK(K % GG_KB == 0, "GEMM groupee : K multiple de 64");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "GEMM groupee : activations bf16");
    const int M = qw.size(1), G = x.size(0), T = tile_e.size(0);
    auto y = torch::zeros({G, M}, x.options());
    if (T == 0) return y;
    dim3 grid((M + GG_BM - 1) / GG_BM, T);
    auto stream = at::cuda::getCurrentCUDAStream();
    nvfp4_gemm_grouped_kernel<<<grid, 128, 0, stream>>>(
        qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(),
        gscales.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
        tile_e.data_ptr<int>(), tile_t0.data_ptr<int>(), tile_n.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), M, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
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
    auto stream = at::cuda::getCurrentCUDAStream();
    static const bool ancien = std::getenv("ACVRAM_GROUPED_OLD") != nullptr;
    if (!ancien && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024) {
        const bool bf = x.scalar_type() == torch::kBFloat16;
        auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
        auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
        static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 1;
        dim3 grid((M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
        const size_t shm = (size_t)(K + K / 32) * sizeof(float);
        #define GWK(XT, PX) do { \
            if (rpw == 1) nvfp4_gemv_grouped_warp_kernel<XT, 1><<<grid, GW_WARPS * WARP, shm, stream>>>(qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(), gscales.data_ptr<float>(), expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K); \
            else if (rpw == 2) nvfp4_gemv_grouped_warp_kernel<XT, 2><<<grid, GW_WARPS * WARP, shm, stream>>>(qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(), gscales.data_ptr<float>(), expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K); \
            else nvfp4_gemv_grouped_warp_kernel<XT, 4><<<grid, GW_WARPS * WARP, shm, stream>>>(qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(), gscales.data_ptr<float>(), expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K); \
        } while (0)
        if (bf) { GWK(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
        else { GWK(float, xc.data_ptr<float>()); }
        #undef GWK
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    if (false) {
        auto xc = x; auto out = x; dim3 grid(1); const size_t shm = 0; const bool bf = false;
        if (bf) {
            nvfp4_gemv_grouped_warp_kernel<__nv_bfloat16, 1><<<grid, GW_WARPS * WARP, shm, stream>>>(
                qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(),
                gscales.data_ptr<float>(), expert_ids.data_ptr<int>(),
                token_ids.data_ptr<int>(),
                reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()),
                out.data_ptr<float>(), M, (int)K);
        } else {
            nvfp4_gemv_grouped_warp_kernel<float, 1><<<grid, GW_WARPS * WARP, shm, stream>>>(
                qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(),
                gscales.data_ptr<float>(), expert_ids.data_ptr<int>(),
                token_ids.data_ptr<int>(), xc.data_ptr<float>(),
                out.data_ptr<float>(), M, (int)K);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    auto xc = x.to(torch::kFloat).contiguous();
    const int threads = threads_for_pairs((int)K);
    const int nwarps = (threads + 31) / 32;
    // G tranches occupent deja la grille : pas de decoupage en profondeur.
    auto out = torch::empty({G, M}, xc.options());
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, G);
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

torch::Tensor paged_attention(torch::Tensor q, torch::Tensor kc,
                              torch::Tensor ks, torch::Tensor vc,
                              torch::Tensor vs, torch::Tensor tables,
                              torch::Tensor seq_lens, int64_t hkv,
                              double scale, int64_t q_len, int64_t window) {
    CHECK_CUDA(q); CHECK_CUDA(kc); CHECK_CUDA(tables);
    ACVRAM_DEVICE_GUARD(q);
    CHECK_CONTIG(q); CHECK_CONTIG(kc); CHECK_CONTIG(vc);
    const int BQ = q.size(0);             // B * q_len lignes de requete
    const int B = BQ / (int)q_len;
    TORCH_CHECK(B * (int)q_len == BQ, "q.size(0) doit etre B*q_len");
    const int HQ = q.size(1);
    const int D = q.size(2);
    const int N = tables.size(1);
    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256 || D == 512,
                "dimension de tete non instanciee : ", D);
    const int C = (N * 16 + PA_CHUNK - 1) / PA_CHUNK;
    TORCH_CHECK(C <= 256, "contexte au-dela de 256 tranches");
    auto opts = q.options();
    auto part = torch::empty({BQ, HQ, C, D}, opts);
    auto pm = torch::empty({BQ, HQ, C}, opts);
    auto pl = torch::empty({BQ, HQ, C}, opts);
    auto out = torch::empty({BQ, HQ, D}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g1(BQ, HQ, C), g2(BQ, HQ);
    const int threads = PA_WARPS * WARP;

    #define PA_LAUNCH(DD) \
        paged_attn_partial_kernel<DD><<<g1, threads, 0, stream>>>( \
            q.data_ptr<float>(), kc.data_ptr<signed char>(), \
            reinterpret_cast<const __half *>(ks.data_ptr()), \
            vc.data_ptr<signed char>(), \
            reinterpret_cast<const __half *>(vs.data_ptr()), \
            tables.data_ptr<long>(), seq_lens.data_ptr<long>(), \
            part.data_ptr<float>(), pm.data_ptr<float>(), \
            pl.data_ptr<float>(), HQ, (int)hkv, N, C, (int)q_len, \
            (float)scale, (int)window); \
        paged_attn_reduce_kernel<DD><<<g2, 128, 0, stream>>>( \
            part.data_ptr<float>(), pm.data_ptr<float>(), \
            pl.data_ptr<float>(), out.data_ptr<float>(), HQ, C)

    if (D == 32) { PA_LAUNCH(32); }
    else if (D == 64) { PA_LAUNCH(64); }
    else if (D == 128) { PA_LAUNCH(128); }
    else if (D == 256) { PA_LAUNCH(256); }
    else { PA_LAUNCH(512); }
    #undef PA_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ============================================================================
// KDA — Kimi Delta Attention, pas de décodage fusionné (une séquence, t = 1).
// Un bloc par tête, D fils (un par canal). Remplace ~35 lancements : les trois
// convolutions causales à état (+SiLU), la L2-normalisation de q et k, les
// portes g1/β, la récurrence delta sur S (décroissance sur l'axe clé), la
// RMSNorm de sortie et la porte sigmoïde g2.
// S[h] est [D_i (sortie), D_j (clé)], j contigu ; le fil i porte la ligne i.
// ============================================================================
template <int D>
__global__ void kda_decode_kernel(
        const __nv_bfloat16 *__restrict__ xq, const __nv_bfloat16 *__restrict__ xk,
        const __nv_bfloat16 *__restrict__ xv,        // [H*D] projections (bf16)
        const __nv_bfloat16 *__restrict__ g1_pre,    // [H*D] f_b(f_a(x))
        const __nv_bfloat16 *__restrict__ g2,        // [H*D] g_b(g_a(x))
        const __nv_bfloat16 *__restrict__ beta_pre,  // [H]
        const float *__restrict__ wq, const float *__restrict__ wk,
        const float *__restrict__ wv,                // [H*D, K] poids conv
        float *__restrict__ cq, float *__restrict__ ck,
        float *__restrict__ cv,                      // [H*D, K-1] états conv
        const float *__restrict__ dt_bias,           // [H*D]
        const float *__restrict__ a,                 // [H]  = -exp(A_log)
        const float *__restrict__ norm_w,            // [D]
        float *__restrict__ S,                       // [H, D, D]
        __nv_bfloat16 *__restrict__ y,               // [H*D] sortie
        int K, float eps) {
    const int h = blockIdx.x, i = threadIdx.x, c = h * D + i;
    __shared__ float sq[D], sk[D], sv[D], se[D], red[32];

    auto conv = [&](const __nv_bfloat16 *x, const float *w, float *st) {
        float acc = 0.f;
        const float xc = __bfloat162float(x[c]);
        const float *wr = w + (size_t)c * K;
        float *sr = st + (size_t)c * (K - 1);
        #pragma unroll 4
        for (int t = 0; t < K - 1; ++t) acc += wr[t] * sr[t];
        acc += wr[K - 1] * xc;
        #pragma unroll 4
        for (int t = 0; t < K - 2; ++t) sr[t] = sr[t + 1];
        sr[K - 2] = xc;
        return acc / (1.f + __expf(-acc));                  // SiLU
    };
    auto block_sum = [&](float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((i & 31) == 0) red[i >> 5] = v;
        __syncthreads();
        float t = 0.f;
        for (int w = 0; w < D / 32; ++w) t += red[w];
        __syncthreads();
        return t;
    };

    float q = conv(xq, wq, cq), k = conv(xk, wk, ck), v = conv(xv, wv, cv);
    const float nq = block_sum(q * q), nk = block_sum(k * k);
    q *= rsqrtf(nq + eps) * rsqrtf((float)D);
    k *= rsqrtf(nk + eps);
    const float gp = __bfloat162float(g1_pre[c]) + dt_bias[c];
    const float sp = gp > 20.f ? gp : log1pf(__expf(gp));   // softplus
    sq[i] = q; sk[i] = k; sv[i] = v; se[i] = __expf(a[h] * sp);
    __syncthreads();
    const float beta = 1.f / (1.f + __expf(-__bfloat162float(beta_pre[h])));

    float *srow = S + ((size_t)h * D + i) * D;
    float pred = 0.f;
    float row[D];
    #pragma unroll
    for (int j = 0; j < D; ++j) { row[j] = srow[j] * se[j]; pred += row[j] * sk[j]; }
    const float d = beta * (v - pred);
    float o = 0.f;
    #pragma unroll
    for (int j = 0; j < D; ++j) { row[j] += d * sk[j]; o += row[j] * sq[j]; srow[j] = row[j]; }

    const float ms = block_sum(o * o) / (float)D;
    const float n = o * rsqrtf(ms + eps) * norm_w[i];
    y[c] = __float2bfloat16(n / (1.f + __expf(-__bfloat162float(g2[c]))));
}

torch::Tensor kda_decode(torch::Tensor xq, torch::Tensor xk, torch::Tensor xv,
                         torch::Tensor g1_pre, torch::Tensor g2,
                         torch::Tensor beta_pre, torch::Tensor wq,
                         torch::Tensor wk, torch::Tensor wv, torch::Tensor cq,
                         torch::Tensor ck, torch::Tensor cv,
                         torch::Tensor dt_bias, torch::Tensor a,
                         torch::Tensor norm_w, torch::Tensor S, double eps) {
    CHECK_CUDA(S); ACVRAM_DEVICE_GUARD(S);
    CHECK_CONTIG(S); CHECK_CONTIG(cq); CHECK_CONTIG(ck); CHECK_CONTIG(cv);
    TORCH_CHECK(xq.scalar_type() == torch::kBFloat16, "KDA : projections bf16 attendues");
    const int H = S.size(0), D = S.size(1), K = wq.size(1);
    TORCH_CHECK(D == 128 || D == 64, "KDA : dimension de tete non instanciee : ", D);
    auto y = torch::empty({1, H * D}, xq.options().dtype(torch::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    #define BF(t) reinterpret_cast<const __nv_bfloat16 *>((t).data_ptr())
    #define KDA_LAUNCH(DD) kda_decode_kernel<DD><<<H, DD, 0, stream>>>( \
        BF(xq), BF(xk), BF(xv), BF(g1_pre), BF(g2), BF(beta_pre), \
        wq.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), \
        cq.data_ptr<float>(), ck.data_ptr<float>(), cv.data_ptr<float>(), \
        dt_bias.data_ptr<float>(), a.data_ptr<float>(), norm_w.data_ptr<float>(), \
        S.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), \
        K, (float)eps)
    if (D == 128) { KDA_LAUNCH(128); } else { KDA_LAUNCH(64); }
    #undef KDA_LAUNCH
    #undef BF
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ============================================================================
// MLA — attention latente absorbée, pas de décodage (une séquence, t = 1).
// Noyau 1 : scores[h, r] = q_eff[h]·cache[r] pour r ≤ len (−inf au-delà),
//           un fil par ligne, grille (H, L/128).
// Noyau 2 : softmax sur L puis o_lat[h, :] = Σ_r p_r · cache[r, :rank],
//           un bloc par tête, fils sur la dimension latente.
// ============================================================================
__global__ void mla_scores_kernel(const float *__restrict__ q,     // [H, W]
                                  const __nv_bfloat16 *__restrict__ cache,  // [L, W]
                                  const long *__restrict__ len,    // scalaire
                                  float *__restrict__ scores,      // [H, L]
                                  int L, int W, float scale) {
    const int h = blockIdx.x, r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= L) return;
    float s = -INFINITY;
    if (r <= (int)*len) {
        const float *qh = q + (size_t)h * W;
        const __nv_bfloat16 *kr = cache + (size_t)r * W;
        float acc = 0.f;
        for (int d = 0; d < W; d += 2) {
            const float2 kk = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(kr + d));
            acc += qh[d] * kk.x + qh[d + 1] * kk.y;
        }
        s = acc * scale;
    }
    scores[(size_t)h * L + r] = s;
}

__global__ void mla_reduce_kernel(const float *__restrict__ scores,  // [H, L]
                                  const __nv_bfloat16 *__restrict__ cache,  // [L, W]
                                  float *__restrict__ o,             // [H, R]
                                  int L, int W, int R) {
    const int h = blockIdx.x, tid = threadIdx.x, T = blockDim.x;
    extern __shared__ float sh[];                 // [L] probabilités
    __shared__ float red[32];
    const float *sc = scores + (size_t)h * L;
    float m = -INFINITY;
    for (int r = tid; r < L; r += T) m = fmaxf(m, sc[r]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    m = -INFINITY;
    for (int w = 0; w < T / 32; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float l = 0.f;
    for (int r = tid; r < L; r += T) { const float p = __expf(sc[r] - m); sh[r] = p; l += p; }
    for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if ((tid & 31) == 0) red[tid >> 5] = l;
    __syncthreads();
    l = 0.f;
    for (int w = 0; w < T / 32; ++w) l += red[w];
    const float inv = 1.f / l;
    __syncthreads();
    for (int d = tid; d < R; d += T) {
        float acc = 0.f;
        for (int r = 0; r < L; ++r) {
            const float p = sh[r];
            if (p != 0.f) acc += p * __bfloat162float(cache[(size_t)r * W + d]);
        }
        o[(size_t)h * R + d] = acc * inv;
    }
}

torch::Tensor mla_decode(torch::Tensor q_eff, torch::Tensor cache,
                         torch::Tensor len, torch::Tensor scores,
                         int64_t L, int64_t rank, double scale) {
    CHECK_CUDA(q_eff); ACVRAM_DEVICE_GUARD(q_eff);
    CHECK_CONTIG(q_eff); CHECK_CONTIG(cache); CHECK_CONTIG(scores);
    const int H = q_eff.size(0), W = q_eff.size(1);
    TORCH_CHECK(W % 2 == 0, "MLA : largeur paire attendue");
    TORCH_CHECK(scores.size(0) == H && scores.size(1) >= L, "MLA : scores trop petit");
    auto o = torch::empty({H, (long)rank}, q_eff.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g1(H, ((int)L + 127) / 128);
    mla_scores_kernel<<<g1, 128, 0, stream>>>(
        q_eff.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(cache.data_ptr()),
        len.data_ptr<long>(), scores.data_ptr<float>(), (int)L, W, (float)scale);
    const size_t shm = (size_t)L * sizeof(float);
    if (shm > 48 * 1024) {
        static size_t autorise = 0;
        if (shm > autorise) {
            cudaFuncSetAttribute(mla_reduce_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            autorise = shm;
        }
    }
    mla_reduce_kernel<<<H, 256, shm, stream>>>(
        scores.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(cache.data_ptr()),
        o.data_ptr<float>(), (int)L, W, (int)rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}


// RMSNorm fusionnée (bf16 -> bf16, variance en fp32) : un bloc par ligne.
// Même arithmétique que la version torch : x normalisé arrondi en bf16, puis
// produit bf16 par le poids — bit-identique.
__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16 *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ w,
                                    __nv_bfloat16 *__restrict__ y,
                                    int H, float eps) {
    __shared__ float red[32];
    const __nv_bfloat16 *xr = x + (size_t)blockIdx.x * H;
    __nv_bfloat16 *yr = y + (size_t)blockIdx.x * H;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]); ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    ss = 0.f;
    for (int k = 0; k < (int)(blockDim.x >> 5); ++k) ss += red[k];
    const float rs = rsqrtf(ss / (float)H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float n = __bfloat162float(__float2bfloat16(__bfloat162float(xr[i]) * rs));
        yr[i] = __float2bfloat16(n * __bfloat162float(w[i]));
    }
}

torch::Tensor rmsnorm_bf16(torch::Tensor x, torch::Tensor w, double eps) {
    CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(x);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16,
                "rmsnorm_bf16 : bf16 attendu");
    auto xc = x.contiguous();
    const int H = xc.size(-1);
    const long R = xc.numel() / H;
    auto y = torch::empty_like(xc);
    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_bf16_kernel<<<(unsigned)R, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(w.contiguous().data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), H, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


// ============================================================================
// Routage MoE en un lancement : scores (softmax ou sigmoïde), biais de
// sélection, top-k par argmax itéré, poids renormalisés et mis à l'échelle.
// Un bloc par jeton, E <= 1024 experts en mémoire partagée. Remplace
// sigmoid/softmax + add + topk + gather + sum + div + mul (7 lancements).
// ============================================================================
__global__ void moe_route_kernel(const float *__restrict__ logits,   // [T, E]
                                 const float *__restrict__ bias,     // [E] ou nul
                                 float *__restrict__ topw,           // [T, k]
                                 int *__restrict__ topi,             // [T, k]
                                 int E, int k, int sigmoid, int renorm,
                                 float scale) {
    __shared__ float probs[1024];
    __shared__ float sel[1024];
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    __shared__ int choix[32];
    const int t = blockIdx.x, tid = threadIdx.x, T = blockDim.x;
    const float *lg = logits + (long)t * E;
    if (sigmoid) {
        for (int i = tid; i < E; i += T) probs[i] = 1.f / (1.f + __expf(-lg[i]));
    } else {
        float m = -INFINITY;
        for (int i = tid; i < E; i += T) m = fmaxf(m, lg[i]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if ((tid & 31) == 0) red_v[tid >> 5] = m;
        __syncthreads();
        m = -INFINITY;
        for (int w = 0; w < T / 32; ++w) m = fmaxf(m, red_v[w]);
        __syncthreads();
        float sm = 0.f;
        for (int i = tid; i < E; i += T) { const float e = __expf(lg[i] - m); probs[i] = e; sm += e; }
        for (int o = 16; o > 0; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
        if ((tid & 31) == 0) red_v[tid >> 5] = sm;
        __syncthreads();
        sm = 0.f;
        for (int w = 0; w < T / 32; ++w) sm += red_v[w];
        __syncthreads();
        for (int i = tid; i < E; i += T) probs[i] /= sm;
    }
    __syncthreads();
    for (int i = tid; i < E; i += T) sel[i] = probs[i] + (bias ? bias[i] : 0.f);
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY; int bi = 0x7fffffff;
        for (int i = tid; i < E; i += T) {
            const float v = sel[i];
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { red_v[tid >> 5] = bv; red_i[tid >> 5] = bi; }
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < T / 32; ++w)
                if (red_v[w] > red_v[0] || (red_v[w] == red_v[0] && red_i[w] < red_i[0])) {
                    red_v[0] = red_v[w]; red_i[0] = red_i[w];
                }
            choix[j] = red_i[0];
            sel[red_i[0]] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float somme = 0.f;
        for (int j = 0; j < k; ++j) somme += probs[choix[j]];
        const float f = (renorm ? 1.f / somme : 1.f) * scale;
        for (int j = 0; j < k; ++j) {
            topw[(long)t * k + j] = probs[choix[j]] * f;
            topi[(long)t * k + j] = choix[j];
        }
    }
}

std::vector<torch::Tensor> moe_route(torch::Tensor logits, torch::Tensor bias,
                                     int64_t k, bool sigmoid, bool renorm,
                                     double scale) {
    CHECK_CUDA(logits); ACVRAM_DEVICE_GUARD(logits);
    auto lg = logits.to(torch::kFloat).contiguous();
    const int T = lg.size(0), E = lg.size(1);
    TORCH_CHECK(E <= 1024 && k <= 32, "moe_route : E <= 1024 et k <= 32");
    auto topw = torch::empty({T, k}, lg.options());
    auto topi = torch::empty({T, k}, lg.options().dtype(torch::kInt32));
    const float *bp = bias.defined() && bias.numel() > 0
                      ? bias.to(torch::kFloat).contiguous().data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    moe_route_kernel<<<T, 256, 0, stream>>>(lg.data_ptr<float>(), bp,
                                            topw.data_ptr<float>(), topi.data_ptr<int>(),
                                            E, (int)k, sigmoid ? 1 : 0, renorm ? 1 : 0,
                                            (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {topw, topi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_dequant", &nvfp4_dequant, "NVFP4 -> matrice dense",
          py::arg("qweight"), py::arg("block_scale"), py::arg("global_scale"),
          py::arg("K"), py::arg("dtype"), py::arg("gscale_rows") = py::none(),
          py::arg("rows_per_group") = 1);
    m.def("nvfp4_gemv", &nvfp4_gemv, "NVFP4 : dequantification + produit fusionnes");
    m.def("int4_dequant", &int4_dequant, "INT4 affine par groupes -> matrice dense");
    m.def("int4_gemv", &int4_gemv, "INT4 : dequantification + produit fusionnes");
    m.def("int8_dequant", &int8_dequant, "INT8 affine par groupes -> matrice dense");
    m.def("int8_gemv", &int8_gemv, "INT8 : dequantification + produit fusionnes");
    m.def("nvfp4_gemv_grouped", &nvfp4_gemv_grouped,
          "NVFP4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
    m.def("nvfp4_gemv_grouped_gateup", &nvfp4_gemv_grouped_gateup,
          py::arg("qg"), py::arg("bg"), py::arg("gsg"), py::arg("qu"), py::arg("bu"),
          py::arg("gsu"), py::arg("expert_ids"), py::arg("token_ids"), py::arg("x"),
          py::arg("K"), py::arg("act") = 0,
          "MoE NVFP4 : gate et up fusionnes, sortie act(gate)*up (act 0=SiLU, 1=GELU-tanh)");
    m.def("nvfp4_gemm_grouped", &nvfp4_gemm_grouped,
          "MoE NVFP4 : GEMM groupee sur tuiles de jetons, poids lus en 4 bits");
    m.def("int4_gemv_grouped", &int4_gemv_grouped,
          "INT4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
    m.def("moe_route", &moe_route, "routage MoE : scores, biais, top-k, poids");
    m.def("rmsnorm_bf16", &rmsnorm_bf16, "RMSNorm bf16 fusionnee (variance fp32)");
    m.def("kda_decode", &kda_decode, "KDA : pas de decodage fusionne (une sequence)");
    m.def("mla_decode", &mla_decode, "MLA absorbee : scores + softmax + lecture latente");
    m.def("paged_attention", &paged_attention,
          "attention de decodage fusionnee sur cache KV int8 pagine");
}
