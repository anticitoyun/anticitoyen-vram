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
#include <tuple>
#include <array>
#include <map>
#include <cstdlib>
#include <string>
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

// Les conversions FP8 et FP4 n'ont d'instruction materielle qu'a partir de
// sm_89 (e4m3) et sm_100 (e2m1). En dessous, l'intrinseque du toolkit part en
// emulation logicielle : mesure sur une RTX 3080 Ti (sm_86), nvfp4_gemv
// tombait a 144 Go/s pour un plafond de carte a 770. Les deux versions
// arithmetiques ci-dessous tiennent entierement en registres.
__device__ __forceinline__ float e4m3_to_float(unsigned char bits) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    __nv_fp8_storage_t s = static_cast<__nv_fp8_storage_t>(bits);
    __half_raw h = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half *>(&h));
#else
    const unsigned int u = bits;
    const unsigned int e = (u >> 3) & 0xFu, m = u & 7u;
    float v;
    if (e == 0u) {
        v = (float)m * 0.001953125f;              // 2^-9, domaine sous-normal
    } else if (e == 15u && m == 7u) {
        v = __int_as_float(0x7fc00000);           // E4M3FN : seul motif NaN
    } else {
        const unsigned int w = ((e + 120u) << 23) | (m << 20);
        v = __int_as_float((int)w);
    }
    return (u & 0x80u) ? -v : v;
#endif
}

template <typename QT> __device__ __forceinline__ float to_float_q(QT v);
template <> __device__ __forceinline__ float to_float_q<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float_q<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
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
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
    const __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1);
    const __half2 hv = *reinterpret_cast<const __half2 *>(&h2);
    return __half22float2(hv);
#else
    // Les huit magnitudes E2M1 (0, 0,5, 1, 1,5, 2, 3, 4, 6) sont toutes des
    // multiples d'un demi : 0, 1, 2, 3, 4, 6, 8, 12 tiennent chacune sur un
    // quartet, donc la table entiere tient dans une constante 32 bits, lue par
    // decalage. Une table en memoire constante serait serialisee huit fois par
    // warp, l'index differant d'un fil a l'autre.
    const unsigned int L = 0xC8643210u;
    const unsigned int a = byte & 0xFu, b = (byte >> 4) & 0xFu;
    float x = 0.5f * (float)((L >> (4u * (a & 7u))) & 0xFu);
    float y = 0.5f * (float)((L >> (4u * (b & 7u))) & 0xFu);
    if (a & 8u) x = -x;
    if (b & 8u) y = -y;
    return make_float2(x, y);
#endif
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
            float v = 0.5f * (float)((0xC8643210u >> (4u * (c & 7u))) & 0xFu) * s;
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
    } else if constexpr (NX == 4) {
        const uint2 v = *reinterpret_cast<const uint2 *>(p);
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&v.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&v.y));
        xs[0] = f0.x; xs[1] = f0.y; xs[2] = f1.x; xs[3] = f1.y;
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

// NV activations par lecture de poids : la matrice quantifiée est le tenseur
// cher, l'activation tient en registres. Boucler sur N à l'extérieur relisait
// tout le poids par jeton, ce qui rendait la vérification spéculative et les
// lots plus coûteux qu'autant de pas séparés.
template <int ROWS, int NV, typename XT, typename YT>
__global__ void nvfp4_gemv_kernel(
    const unsigned char *__restrict__ qw,
    const unsigned char *__restrict__ bscale,
    const float gscale_unique,
    const XT *__restrict__ x,                 // [N, K]
    YT *__restrict__ y,                       // [N, M]
    int M, int K, int N, int k_splits,
    // Échelle globale par ligne de sortie, ou nullptr. Sert aux projections
    // empilées : q, k et v partagent leur entrée mais pas leur échelle
    // globale, et trois GEMV coûtent trois fois la latence d'une seule. Les
    // segments étant alignés sur ROWS, une lecture par bloc suffit.
    const float *__restrict__ gscale_rows) {
    extern __shared__ float smem[];
    const int nwarps = (blockDim.x + WARP - 1) / WARP;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    const float gscale = gscale_rows ? gscale_rows[row0] : gscale_unique;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const int split = blockIdx.y;
    const long half_k = K >> 1;

    float acc[ROWS][NV];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
        #pragma unroll
        for (int n = 0; n < NV; ++n) acc[r][n] = 0.f;

    // Deux blocs de 16 par itération : un uint4 charge 32 poids d'un coup,
    // soit une transaction de 16 octets par fil — la 5090 n'atteignait qu'un
    // quart de sa bande passante avec des lectures de 8. Le découpage se fait
    // en paires de blocs, jamais au milieu d'une.
    const int npairs = nloads >> 1;
    const int per_split_p = (npairs + k_splits - 1) / k_splits;
    const int lo_p = split * per_split_p;
    const int hi_p = min(npairs, lo_p + per_split_p);

    // Double tampon sur la dimension K. La boucle lisait le poids puis le
    // consommait aussitot : chaque iteration payait la latence DRAM en entier,
    // et le fil restait bloque sur `long_scoreboard`. Les lectures de
    // l'iteration suivante sont maintenant emises AVANT le calcul de la
    // courante, qui les recouvre.
    //
    // Mesure du 9/09/2026 sur Qwen2.5-Coder-14B en decodage : notre GEMV
    // atteignait 770 Go/s contre 991 pour cuBLAS en bf16, soit 78 %. L'ecart
    // etait pire sur les PETITES matrices (0,62 sur o_proj) que sur les
    // grandes (0,85 sur gate/up) — signature d'une latence mal masquee, pas
    // d'une mauvaise coalescence, qui penaliserait uniformement.
    //
    // L'arithmetique est inchangee : memes valeurs, meme ordre
    // d'accumulation. Seul le moment des lectures change.
    uint4 pre_p4[ROWS];
    float pre_s0[ROWS], pre_s1[ROWS];
    int i = lo_p + threadIdx.x;
    bool vif = i < hi_p;
    if (vif) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row >= M) continue;
            pre_p4[r] = reinterpret_cast<const uint4 *>(qw + (long)row * half_k)[i];
            pre_s0[r] = e4m3_to_float(bscale[(long)row * nloads + 2 * i]) * gscale;
            pre_s1[r] = e4m3_to_float(bscale[(long)row * nloads + 2 * i + 1]) * gscale;
        }
    }
    for (; i < hi_p; i += blockDim.x) {
        // ce qui a ete precharge sert maintenant
        uint4 cur_p4[ROWS];
        float cur_s0[ROWS], cur_s1[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            cur_p4[r] = pre_p4[r]; cur_s0[r] = pre_s0[r]; cur_s1[r] = pre_s1[r];
        }
        // les lectures suivantes partent avant le calcul, jamais apres
        const int isuiv = i + blockDim.x;
        if (isuiv < hi_p) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                pre_p4[r] = reinterpret_cast<const uint4 *>(qw + (long)row * half_k)[isuiv];
                pre_s0[r] = e4m3_to_float(bscale[(long)row * nloads + 2 * isuiv]) * gscale;
                pre_s1[r] = e4m3_to_float(bscale[(long)row * nloads + 2 * isuiv + 1]) * gscale;
            }
        }
        float xs[NV][2 * WEIGHTS_PER_LOAD];
        #pragma unroll
        for (int n = 0; n < NV; ++n)
            load_xs<XT, 2 * WEIGHTS_PER_LOAD>(
                x + (long)n * K + (long)i * 2 * WEIGHTS_PER_LOAD, xs[n]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row >= M) continue;
            const uint4 p4 = cur_p4[r];
            const float s0 = cur_s0[r];
            const float s1 = cur_s1[r];
            const unsigned int words[4] = {p4.x, p4.y, p4.z, p4.w};
            float p0[NV], p1[NV];
            #pragma unroll
            for (int n = 0; n < NV; ++n) { p0[n] = 0.f; p1[n] = 0.f; }
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const float2 v0 = e2m1_pair((words[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
                const float2 v1 = e2m1_pair((words[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
                #pragma unroll
                for (int n = 0; n < NV; ++n) {
                    p0[n] += v0.x * xs[n][2 * b] + v0.y * xs[n][2 * b + 1];
                    p1[n] += v1.x * xs[n][WEIGHTS_PER_LOAD + 2 * b]
                           + v1.y * xs[n][WEIGHTS_PER_LOAD + 2 * b + 1];
                }
            }
            #pragma unroll
            for (int n = 0; n < NV; ++n) acc[r][n] += p0[n] * s0 + p1[n] * s1;
        }
    }

    #pragma unroll
    for (int n = 0; n < NV; ++n) {
        float a[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) a[r] = acc[r][n];
        block_reduce_rows<ROWS>(a, smem, nwarps);
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                if (k_splits == 1) store_y(y + (long)n * M + row, a[r]);
                else atomicAdd(reinterpret_cast<float *>(y) + (long)n * M + row, a[r]);
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

// N activations par lecture de poids. Le noyau d'origine bouclait sur les
// lignes d'activation à l'extérieur et relisait la matrice entière pour
// chacune : un pas de vérification spéculative à cinq jetons coûtait cinq
// fois le trafic d'un pas simple, et la spéculation ne pouvait jamais payer.
// Ici le poids est lu une fois et sert aux N activations, gardées en
// registres.
// Poste F, fusion (3b) (verdict-lancements-b1-17-09 : add_norm x 2 par couche,
// 0,235 ms, 97 lancements a b=1) : la NORME D'ENTREE est faite DANS le GEMV.
// REFUTE, temoin nomme (Laure, verdict-f3b-norme-fusee-17-09, 6dbb1bb) :
// exact (bit a bit, capture ok, -47 lancements) mais +0,31 ms par pas — la
// norme est recalculee par CHACUN des ~1 024 blocs du GEMV (K lectures +
// deux passes synchronisees, +4,4 us par lancement de 7,8 us ; b=8 : 7,5 ->
// 12,0 ms). Structurel : la somme des carres de la ligne n'existe qu'apres
// tous les blocs du GEMV precedent (pas d'epilogue possible), et un bloc
// dedie qui normalise et diffuse EST add_norm. Reste opt-in
// (ACVRAM_NORME_FUSEE=1), jamais defaut ; le gain vise etait 4 % du pas.
// Quand `nw` est fourni, chaque bloc recalcule la ligne normalisee dans sa
// memoire partagee — x_in = bf16(res + mult*x) si `res` (le flux residuel,
// ecrit une fois dans `xout` par le bloc (0,0)), puis rs = rsqrtf(sum(x_in^2)/K
// + eps) et xs = bf16(bf16(x_in*rs) * w) — l'arithmetique EXACTE de
// rmsnorm_bf16_kernel, une somme fp32 dont seul l'ordre differe (rs peut
// bouger d'un ulp fp32 : au plus 1 ulp bf16 sur quelques xs). Recalculer
// sum(x^2) par bloc coute K lectures de L2 par bloc (4 Kio a K=2048) contre
// un lancement et une ecriture-relecture de la ligne normalisee.
template <int ROWS, int NV, typename XT, typename YT>
__global__ void int8_gemv_kernel(
    const unsigned char *__restrict__ qw,
    const __half *__restrict__ scales,
    const unsigned char *__restrict__ zeros,
    const XT *__restrict__ x,
    YT *__restrict__ y,
    int M, int K, int N, int group, int k_splits,
    const __nv_bfloat16 *__restrict__ res = nullptr,   // flux residuel [N, K] ou nul
    const __nv_bfloat16 *__restrict__ nw = nullptr,    // poids RMSNorm [K] : active la norme
    __nv_bfloat16 *__restrict__ xout = nullptr,        // res + mult*x, ecrit par le bloc (0,0)
    float eps = 0.f, float mult = 1.f) {
    extern __shared__ float smem[];
    const int nwarps = (blockDim.x + WARP - 1) / WARP;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    if (nw != nullptr) {
        // xs apres la zone de reduction (ROWS*nwarps floats, arrondie a 8)
        __nv_bfloat16 *xs = reinterpret_cast<__nv_bfloat16 *>(smem + ((ROWS * nwarps + 7) & ~7));
        const __nv_bfloat16 *xb = reinterpret_cast<const __nv_bfloat16 *>(x);
        for (int n = 0; n < NV; ++n) {
            float ss = 0.f;
            for (int i = threadIdx.x; i < K; i += blockDim.x) {
                float v = __bfloat162float(xb[(long)n * K + i]);
                if (res != nullptr)
                    v = __bfloat162float(__float2bfloat16(__bfloat162float(res[(long)n * K + i]) + mult * v));
                xs[(long)n * K + i] = __float2bfloat16(v);
                if (xout != nullptr && blockIdx.x == 0 && blockIdx.y == 0)
                    xout[(long)n * K + i] = __float2bfloat16(v);
                ss += v * v;
            }
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = ss;
            __syncthreads();
            ss = 0.f;
            for (int k = 0; k < nwarps; ++k) ss += smem[k];
            const float rs = rsqrtf(ss / (float)K + eps);
            __syncthreads();                       // smem[] relu par tous avant reecriture
            for (int i = threadIdx.x; i < K; i += blockDim.x) {
                const float nv = __bfloat162float(__float2bfloat16(__bfloat162float(xs[(long)n * K + i]) * rs));
                xs[(long)n * K + i] = __float2bfloat16(nv * __bfloat162float(nw[i]));
            }
            __syncthreads();
        }
        x = reinterpret_cast<const XT *>(xs);
    }
    const int nloads = K / WEIGHTS_PER_LOAD;
    const int ng = K / group;
    const int split = blockIdx.y;
    const int per_split = (nloads + k_splits - 1) / k_splits;
    const int lo = split * per_split;
    const int hi = min(nloads, lo + per_split);

    float acc[ROWS][NV];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
        #pragma unroll
        for (int n = 0; n < NV; ++n) acc[r][n] = 0.f;

    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        // Les poids des ROWS lignes sont lus une fois (le tenseur cher) et
        // servent aux NV activations. Les activations sont chargées par mot
        // de 4 poids et non par uint4 de 16 : NV x 4 flottants en registres
        // au lieu de NV x 16, ce qui permet NV = 12 (un lot de 12 séquences en
        // UNE passe sur les poids, là où NV <= 8 en imposait deux — profil du
        // 14/09 : int8_gemv<4,8> + <4,4> à chaque couche, poids lus 2x).
        // L'ordre d'accumulation (j croissant dans `part`) est celui d'avant.
        const int g = (i * WEIGHTS_PER_LOAD) / group;
        uint4 p[ROWS]; float s[ROWS];
        float part[ROWS][NV];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = min(row0 + r, M - 1);
            p[r] = reinterpret_cast<const uint4 *>(qw + (long)row * K)[i];
            s[r] = __half2float(scales[(long)row * ng + g]);
            #pragma unroll
            for (int n = 0; n < NV; ++n) part[r][n] = 0.f;
        }
        float z[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
            z[r] = static_cast<float>(zeros[(long)min(row0 + r, M - 1) * ng + g]);
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            float x4[NV][4];
            #pragma unroll
            for (int n = 0; n < NV; ++n)
                load_xs<XT, 4>(x + (long)n * K + (long)i * WEIGHTS_PER_LOAD + 4 * w, x4[n]);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const unsigned int word = (w == 0) ? p[r].x : (w == 1) ? p[r].y : (w == 2) ? p[r].z : p[r].w;
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const float v = static_cast<float>((word >> (jj * 8)) & 0xFFu) - z[r];
                    #pragma unroll
                    for (int n = 0; n < NV; ++n) part[r][n] += v * x4[n][jj];
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (row0 + r >= M) continue;
            #pragma unroll
            for (int n = 0; n < NV; ++n) acc[r][n] += part[r][n] * s[r];
        }
    }

    #pragma unroll
    for (int n = 0; n < NV; ++n) {
        float a[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) a[r] = acc[r][n];
        block_reduce_rows<ROWS>(a, smem, nwarps);
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                if (k_splits == 1) store_y(y + (long)n * M + row, a[r]);
                else atomicAdd(reinterpret_cast<float *>(y) + (long)n * M + row, a[r]);
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

// CONTROLE POSITIF DU HARNAIS DE MESURE (Oceane, 10/09/2026).
// Un instrument ne rend un resultat que s'il pouvait en rendre un autre. On lui
// donne donc une difference CONNUE D'AVANCE et grande : K iterations de FMA
// CHAINEES — chacune depend de la precedente, donc ni eliminees par le
// compilateur ni recouvertes par l'ordonnanceur.
//   - le temps doit devenir lineaire en K des que le travail depasse le plancher
//   - LE COUDE CHIFFRE LE PLANCHER, sans aucune hypothese sur sa cause
// C'est ce qui manquait quand un montage a rendu ~49,5 us pour un noyau qui
// ecrit trois flottants et sort : nous n'avions aucun moyen de savoir que
// c'etait le plancher de l'instrument et non le cout du noyau.
// COMPTEUR DE PARTICIPATION. La colonne « grille » d'un script Python est
// RECONSTRUITE depuis le parametre demande : elle dit ce qui a ete demande,
// jamais ce qui a tourne. Si le noyau borne ou recalcule son decoupage, elle
// afficherait 2048 pendant que 32 blocs travaillent — et corroborerait la
// fausse refutation au lieu de la denoncer. Ici chaque bloc s'annonce, et le
// compte se relit cote hote. Aucune chronometrie.
__device__ unsigned long long acvram_pa_participants = 0ULL;

// TEMOIN DU TAMPON. Le remede a une fuite doit se verifier autrement que par
// la lecture du code : ceci rend les octets effectivement retenus par les
// tampons partiels. Apres un balayage de longueurs, il doit se stabiliser au
// pire cas vu et ne plus bouger — la ou l'ancien cache croissait a chaque
// nouvelle forme.
unsigned long long acvram_pa_tampon_octets = 0ULL;

unsigned long long paged_attn_participants(bool remettre_a_zero) {
    unsigned long long n = 0ULL;
    cudaMemcpyFromSymbol(&n, acvram_pa_participants, sizeof(n));
    if (remettre_a_zero) {
        const unsigned long long z = 0ULL;
        cudaMemcpyToSymbol(acvram_pa_participants, &z, sizeof(z));
    }
    return n;
}

__global__ void banc_fma_kernel(float *__restrict__ sortie, int K) {
    float a = (float)(threadIdx.x + 1) * 1e-3f;
    const float b = 1.0000001f, c = 1e-7f;
    #pragma unroll 1
    for (int i = 0; i < K; ++i) a = fmaf(a, b, c);   // chainee : a depend de a
    if (threadIdx.x == 0)
        sortie[blockIdx.z * gridDim.y * gridDim.x
               + blockIdx.y * gridDim.x + blockIdx.x] = a;
}

torch::Tensor banc_fma(int64_t gx, int64_t gy, int64_t gz,
                       int64_t threads, int64_t K) {
    auto opt = torch::TensorOptions().dtype(torch::kFloat)
                   .device(torch::kCUDA, c10::cuda::current_device());
    auto out = torch::empty({(long)(gx * gy * gz)}, opt);
    dim3 g((unsigned)gx, (unsigned)gy, (unsigned)gz);
    banc_fma_kernel<<<g, (unsigned)threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<float>(), (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// EMPREINTE DU SOURCE, posee par le lanceur Python (-DACVRAM_SRC_HASH).
// Elle doit se retrouver DANS le binaire : c'est le seul controle qui prouve
// que le .so compile bien ce fichier-ci. Comparer les horodatages ne prouve
// rien — ccache reecrit le .so avec un contenu ancien, donc sa date est bonne
// et son contenu perime. Un ENTIER, pas une chaine : l'echappement des
// guillemets ne survivait pas jusqu'a nvcc et le controle refusait TOUT.
#ifndef ACVRAM_SRC_HASH
#define ACVRAM_SRC_HASH 0ULL
#endif
extern "C" __attribute__((used)) const unsigned long long acvram_src_hash
    = ACVRAM_SRC_HASH;

constexpr int PA_CHUNK = 512;
constexpr int PA_WARPS = 4;

// CANAL (C5-b, chantier-c5b-19-09) : les clés portent en plus une échelle
// E4M3 par (bloc, tête, canal) — sc [NB, HKV, D] — qui se replie dans q UNE
// fois par bloc de 16 jetons (qs = q·scale ⊙ sc_bloc, PER_LANE valeurs par
// voie, en registres), puis le produit scalaire int8 n'a plus qu'un facteur
// par jeton (ks, comme aujourd'hui : 0 octet de plus par jeton, 128 o E4M3
// de plus par bloc). Le bloc COURANT d'une séquence (tampon_de[bloc] >= 0) est
// lu en bf16 dans la réserve `tampon` [R, 16, HKV, D], sans échelle. Hors
// CANAL les trois pointeurs sont nuls et le noyau est celui d'avant.
template <int D, typename QT, typename OT, bool CANAL>
__global__ void paged_attn_partial_kernel(
    const QT *__restrict__ q,             // [B*QL, HQ, D]
    const signed char *__restrict__ kc,   // [NB, 16, HKV, D]
    const __half *__restrict__ ks,        // [NB, 16, HKV]
    const signed char *__restrict__ vc,
    const __half *__restrict__ vs,
    const unsigned char *__restrict__ sc, // CANAL : [NB, HKV, D] E4M3
    const __nv_bfloat16 *__restrict__ tampon,   // CANAL : [R, 16, HKV, D]
    const int *__restrict__ tampon_de,    // CANAL : [NB] ligne du bloc courant ou < 0
    const long *__restrict__ tables,      // [B, N]
    const long *__restrict__ seq_lens,    // [B]  longueur TOTALE (dernier jeton inclus)
    float *__restrict__ part,             // [B*QL, HQ, C, D]  acc non normalisé
    float *__restrict__ part_m,           // [B*QL, HQ, C]
    float *__restrict__ part_l,           // [B*QL, HQ, C]
    OT *__restrict__ sortie,              // [B*QL, HQ, D] si C == 1, sinon nul
    int HQ, int HKV, int N, int C, int QL, float scale, int window,
    int chunk,            // le noyau et le lanceur DOIVENT decouper pareil
    int etape,            // BISECTION : sortir plus ou moins tot du noyau.
    bool compter) {       // marqueur de participation : instrument payant
                          // 0 indices · 1 +chargement de q · 2 +boucle
                          // principale · 3 tout (comportement normal).
                          // Les sorties neutres sont ECRITES a chaque etape,
                          // sinon le compilateur supprime le noyau entier et
                          // l'on mesure un lancement vide.

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
    const long start = max((long)c * chunk, lo);
    const long out_off = ((long)bq * HQ + h) * C + c;

    const int lane = threadIdx.x % WARP;
    const int wid = threadIdx.x / WARP;
    constexpr int PER_LANE = D / WARP;

    if (start >= slen || start >= (long)(c + 1) * chunk) {
        if (threadIdx.x == 0) {
            part_m[out_off] = -INFINITY;
            part_l[out_off] = 0.f;
        }
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            part[out_off * D + d] = 0.f;
        return;
    }

    if (etape < 1) {
        // MARQUEUR DE PARTICIPATION : un temps plat ne distingue pas « noyau
        // insensible au parallelisme » de « grille inerte ». On observe donc
        // QUI A TOURNE : chaque bloc marque sa case, et le compte des cases
        // marquees se relit cote hote. participants == grille -> les blocs
        // tournent ; participants < grille -> le lancement est en cause et
        // rien n'a jamais ete teste.
        // L'ATOMIQUE EST UN INSTRUMENT, ET IL SE PAIE. Tous les blocs frappent
        // LA MEME adresse : le L2 les serialise, et le cout croit avec le
        // nombre de blocs — exactement la forme que nous attribuions au
        // « plancher du lancement ». banc_fma, a travail par bloc constant,
        // rend 7,1 us de 170 a 4080 blocs sans une marche : le lancement ne
        // coute rien sur cette plage. Donc ce que l'etape 0 mesurait en plus
        // (15,36 us a 1504 blocs) etait en partie CE COMPTEUR.
        // On le rend donc coupable : pose ACVRAM_PA_SANS_COMPTEUR=1 et la
        // participation n'est plus observee, mais le temps est celui du noyau
        // seul. Un instrument qui ne peut pas etre eteint ne peut pas etre
        // disculpe.
        if (threadIdx.x == 0) {
            part_m[out_off] = 1.f; part_l[out_off] = 0.f;
            if (compter)
            atomicAdd(&acvram_pa_participants, 1ULL);   // ce bloc a tourne
        }
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            part[out_off * D + d] = 0.f;
        return;
    }

    __shared__ float sq[D];
    __shared__ float sm[PA_WARPS], sl[PA_WARPS], scorr[PA_WARPS];
    __shared__ float sacc[PA_WARPS][D];
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        sq[d] = to_float_q<QT>(q[((long)bq * HQ + h) * D + d]) * scale;
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float acc[PER_LANE];
    #pragma unroll
    if (etape < 2) {
        // q est charge et synchronise ; on sort avant la boucle principale.
        if (threadIdx.x == 0) { part_m[out_off] = sq[0]; part_l[out_off] = 0.f; }
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            part[out_off * D + d] = sq[d];   // depend de sq : rien n'est supprime
        return;
    }

    for (int i = 0; i < PER_LANE; ++i) acc[i] = 0.f;

    // CANAL : q ⊙ sc du bloc en cours, rechargé quand le warp change de bloc
    // (PA_WARPS divise 16 : le warp voit 16/PA_WARPS jetons par bloc).
    float qs[PER_LANE];
    long bloc_qs = -1;
    #pragma unroll
    for (int i = 0; i < PER_LANE; ++i) qs[i] = 0.f;

    const long end = min(slen, (long)(c + 1) * chunk);
    for (long t = start + wid; t < end; t += PA_WARPS) {
        const long blk = tables[(long)b * N + (t >> 4)];
        const long cell = (blk * 16 + (t & 15)) * HKV + hkv;
        const signed char *kp = kc + cell * D;

        float partial = 0.f;
        float score;
        if (CANAL) {
            const int rang = tampon_de[blk];          // uniforme dans le warp
            if (rang >= 0) {
                // bloc courant : K bf16 exact dans la réserve, pas d'échelle
                const __nv_bfloat16 *kb = tampon
                    + (((long)rang * 16 + (t & 15)) * HKV + hkv) * D;
                #pragma unroll
                for (int i = 0; i < PER_LANE; ++i)
                    partial += sq[lane * PER_LANE + i]
                               * __bfloat162float(kb[lane * PER_LANE + i]);
                #pragma unroll
                for (int off = WARP / 2; off > 0; off >>= 1)
                    partial += __shfl_down_sync(0xffffffffu, partial, off);
                score = __shfl_sync(0xffffffffu, partial, 0);
            } else {
                if (blk != bloc_qs) {
                    const unsigned char *sp = sc + (blk * HKV + hkv) * D;
                    #pragma unroll
                    for (int i = 0; i < PER_LANE; ++i)
                        qs[i] = sq[lane * PER_LANE + i]
                                * e4m3_to_float(sp[lane * PER_LANE + i]);
                    bloc_qs = blk;
                }
                #pragma unroll
                for (int i = 0; i < PER_LANE; ++i)
                    partial += qs[i] * static_cast<float>(kp[lane * PER_LANE + i]);
                #pragma unroll
                for (int off = WARP / 2; off > 0; off >>= 1)
                    partial += __shfl_down_sync(0xffffffffu, partial, off);
                score = __shfl_sync(0xffffffffu, partial, 0) * __half2float(ks[cell]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < PER_LANE; ++i)
                partial += sq[lane * PER_LANE + i]
                           * static_cast<float>(kp[lane * PER_LANE + i]);
            #pragma unroll
            for (int off = WARP / 2; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            score = __shfl_sync(0xffffffffu, partial, 0) * __half2float(ks[cell]);
        }

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

    if (etape < 3) {
        // La boucle principale a tourne ; on sort AVANT la reduction
        // inter-warps. On ecrit depuis sm/sq pour que rien ne soit supprime :
        // sans dependance aux resultats, le compilateur retirerait la boucle
        // et l'on mesurerait un noyau vide.
        if (threadIdx.x == 0) { part_m[out_off] = sm[0]; part_l[out_off] = sl[0]; }
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            part[out_off * D + d] = sq[d];
        return;
    }

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
    // Une seule tranche (contexte plus court que PA_CHUNK) : la réduction se
    // résume à diviser par la somme des poids, autant l'écrire ici et éviter
    // un second lancement par couche.
    if (C == 1 && sortie != nullptr) {
        const float lg = part_l[out_off];
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float a = 0.f;
            #pragma unroll
            for (int w = 0; w < PA_WARPS; ++w) a += sacc[w][d] * scorr[w];
            sortie[out_off * D + d] = from_float<OT>(a / lg);
        }
        return;
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = 0.f;
        #pragma unroll
        for (int w = 0; w < PA_WARPS; ++w) a += sacc[w][d] * scorr[w];
        part[out_off * D + d] = a;
    }
}

template <int D, typename OT>
__global__ void paged_attn_reduce_kernel(
    const float *__restrict__ part,       // [B, HQ, C, D]
    const float *__restrict__ part_m,
    const float *__restrict__ part_l,
    OT *__restrict__ out,                 // [B, HQ, D]
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
        out[base * D + d] = from_float<OT>(a / s_l);
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
                         double global_scale, torch::Tensor x, int64_t K,
                         c10::optional<torch::Tensor> global_scale_rows) {
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
    const float *gs_rows = nullptr;
    torch::Tensor gsr_c;
    if (global_scale_rows.has_value() && global_scale_rows->defined()) {
        gsr_c = global_scale_rows->contiguous();
        TORCH_CHECK(gsr_c.scalar_type() == torch::kFloat && gsr_c.numel() == M,
                    "global_scale_rows doit etre un float32 de M elements");
        gs_rows = gsr_c.data_ptr<float>();
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    // Huit lignes par bloc ont ete essayees pour reutiliser davantage la
    // tranche d'activation : la pression de registres l'emporte, mesure plus
    // lent sur toutes les formes. Quatre lignes restent l'optimum ici.
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    const size_t shm = ROWS_PER_BLOCK * nwarps * sizeof(float);
    #define F4G(NV, XT, YT, PX, PY, SP) nvfp4_gemv_kernel<ROWS_PER_BLOCK, NV, XT, YT> \
        <<<grid, threads, shm, stream>>>( \
            qweight.data_ptr<unsigned char>(), \
            block_scale.data_ptr<unsigned char>(), (float)global_scale, \
            PX, PY, M, (int)K, n_, SP, gs_rows)
    #define F4G_N(XT, YT, PX, PY, SP) do { \
        switch (n_) { \
        case 1: F4G(1, XT, YT, PX, PY, SP); break; \
        case 2: F4G(2, XT, YT, PX, PY, SP); break; \
        case 3: F4G(3, XT, YT, PX, PY, SP); break; \
        case 4: F4G(4, XT, YT, PX, PY, SP); break; \
        case 5: F4G(5, XT, YT, PX, PY, SP); break; \
        case 6: F4G(6, XT, YT, PX, PY, SP); break; \
        case 7: F4G(7, XT, YT, PX, PY, SP); break; \
        default: F4G(8, XT, YT, PX, PY, SP); break; } } while (0)
    for (int base = 0; base < N; base += 8) {
        const int n_ = min(8, N - base);
        if (bf) {
            F4G_N(__nv_bfloat16, __nv_bfloat16,
                  reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()) + (long)base * K,
                  reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()) + (long)base * M, 1);
        } else {
            F4G_N(float, float, xc.data_ptr<float>() + (long)base * K,
                  out.data_ptr<float>() + (long)base * M, splits);
        }
    }
    #undef F4G
    #undef F4G_N
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


// Variante « un warp par ligne » du GEMV INT8 pour un petit lot (bead z5q).
// Le noyau à 4 lignes par bloc, à K = 2048, donne 128 fils et UN chargement
// de 16 octets par fil et par ligne, puis douze réductions de bloc par la
// mémoire partagée (une par activation) : la latence du chargement n'est pas
// recouverte et la réduction domine (profil du 14/09 : 2,62 ms pour 0,86 Go,
// ~31 % de la borne). Ici chaque voie d'un warp charge K/512 uint4 d'un
// coup (4 en vol à K = 2048), les activations sont copiées une fois par bloc
// en mémoire partagée, et la réduction se fait par shuffles — ni shared ni
// __syncthreads dans la boucle. Même granularité d'échelle (par uint4 de 16
// poids, un groupe) que le noyau à blocs ; l'ordre de réduction diffère.
constexpr int I8W_WARPS = 8;                     // warps par bloc
constexpr int I8W_RPW = 4;                       // lignes par warp : 32 lignes par bloc
// Compte qui a tranché (14/09) : avec une ligne par warp, chaque bloc de 8
// lignes recopie les 48 Ko d'activations (12 x 2048 bf16) — 640 blocs x
// 48 Ko = 31 Mo de trafic L2 pour 10 Mo de poids ; le noyau à 4 lignes par
// bloc en relisait 61 Mo. Le GEMV « borné par les poids » était borné par la
// relecture de x. Quatre lignes par warp divisent ce trafic par quatre.

template <int NV>
__global__ void __launch_bounds__(I8W_WARPS * WARP) int8_gemv_warp_kernel(
    const unsigned char *__restrict__ qw, const __half *__restrict__ scales,
    const unsigned char *__restrict__ zeros, const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y, int M, int K, int group) {
    extern __shared__ __align__(16) unsigned char i8w_smem[];
    __nv_bfloat16 *xs = reinterpret_cast<__nv_bfloat16 *>(i8w_smem);   // [NV][K]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // activations : NV x K bf16, copiées par uint4
    {
        const uint4 *src = reinterpret_cast<const uint4 *>(x);
        uint4 *dst = reinterpret_cast<uint4 *>(xs);
        const int n16 = NV * K / 8;
        for (int i = tid; i < n16; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    const int row0 = (blockIdx.x * I8W_WARPS + warp) * I8W_RPW;
    if (row0 >= M) return;
    const int ng = K / group;
    const int nchunks = K / (WARP * WEIGHTS_PER_LOAD);              // 512 poids par tour de warp
    constexpr int MAXC = 4;                                         // K <= 2048 par ce chemin
    // tous les chargements d'abord : RPW x nchunks uint4 en vol par voie
    uint4 p[I8W_RPW][MAXC];
    #pragma unroll
    for (int r = 0; r < I8W_RPW; ++r) {
        const uint4 *qrow = reinterpret_cast<const uint4 *>(qw + (long)min(row0 + r, M - 1) * K);
        #pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (c < nchunks) p[r][c] = qrow[c * WARP + lane];
    }
    #pragma unroll
    for (int r = 0; r < I8W_RPW; ++r) {
        const int row = row0 + r;
        if (row >= M) break;
        const __half *srow = scales + (long)row * ng;
        const unsigned char *zrow = zeros + (long)row * ng;
        float acc[NV];
        #pragma unroll
        for (int n = 0; n < NV; ++n) acc[n] = 0.f;
        #pragma unroll
        for (int c = 0; c < MAXC; ++c) {
            if (c >= nchunks) break;
            const int col0 = (c * WARP + lane) * WEIGHTS_PER_LOAD;
            const int g = col0 / group;
            const float sc = __half2float(srow[g]);
            const float z = static_cast<float>(zrow[g]);
            const unsigned int words[4] = {p[r][c].x, p[r][c].y, p[r][c].z, p[r][c].w};
            float part[NV];
            #pragma unroll
            for (int n = 0; n < NV; ++n) part[n] = 0.f;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                float xv[NV][4];
                #pragma unroll
                for (int n = 0; n < NV; ++n) {
                    const uint2 u = *reinterpret_cast<const uint2 *>(xs + (long)n * K + col0 + 4 * w);
                    const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.x));
                    const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.y));
                    xv[n][0] = f0.x; xv[n][1] = f0.y; xv[n][2] = f1.x; xv[n][3] = f1.y;
                }
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const float v = static_cast<float>((words[w] >> (jj * 8)) & 0xFFu) - z;
                    #pragma unroll
                    for (int n = 0; n < NV; ++n) part[n] += v * xv[n][jj];
                }
            }
            #pragma unroll
            for (int n = 0; n < NV; ++n) acc[n] += part[n] * sc;
        }
        #pragma unroll
        for (int n = 0; n < NV; ++n) {
            float a = acc[n];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0) y[(long)n * M + row] = __float2bfloat16(a);
        }
    }
}

// Fusion (3b) : y = W . rmsnorm(res + mult*x) et xout = res + mult*x, en un
// lancement (le noyau de base, prologue de norme active ; x bf16, N <= 8,
// N*K*2 <= 32 Kio de memoire partagee — sinon l'appelant fait add_norm).
std::vector<torch::Tensor> int8_gemv_norme(torch::Tensor qweight, torch::Tensor scales,
                                           torch::Tensor zeros, torch::Tensor x, int64_t group,
                                           torch::Tensor res, torch::Tensor nw, double eps, double mult) {
    CHECK_CUDA(qweight); CHECK_CUDA(x); CHECK_CUDA(res); CHECK_CUDA(nw);
    ACVRAM_DEVICE_GUARD(qweight);
    CHECK_CONTIG(qweight); CHECK_CONTIG(scales); CHECK_CONTIG(zeros);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && res.scalar_type() == torch::kBFloat16
                && nw.scalar_type() == torch::kBFloat16, "int8_gemv_norme : bf16 attendu");
    const int K = qweight.size(1);
    TORCH_CHECK(K % group == 0 && group % 16 == 0, "int8_gemv_norme : groupe");
    auto xc = (x.dim() == 1 ? x.reshape({1, -1}) : x).contiguous();
    auto rc = (res.dim() == 1 ? res.reshape({1, -1}) : res).contiguous();
    TORCH_CHECK(xc.size(1) == K && rc.sizes() == xc.sizes() && nw.numel() == K,
                "int8_gemv_norme : x [N, K], res [N, K], w [K]");
    const int M = qweight.size(0);
    const int N = xc.size(0);
    TORCH_CHECK(N >= 1 && N <= 8, "int8_gemv_norme : N <= 8");
    const int threads = threads_for(K);
    const int nwarps = (threads + 31) / 32;
    const int splits = splits_for(M, K, (int)qweight.get_device());
    auto out = splits == 1 ? torch::empty({N, M}, xc.options())
                           : torch::zeros({N, M}, xc.options().dtype(torch::kFloat));
    auto xout = torch::empty_like(xc);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    const size_t shm = ((ROWS_PER_BLOCK * nwarps + 7) & ~7) * sizeof(float) + (size_t)N * K * sizeof(__nv_bfloat16);
    TORCH_CHECK(shm <= 48 * 1024, "int8_gemv_norme : memoire partagee ", shm, " > 48 Kio");
    auto wc = nw.contiguous();
    #define I8N(NV, YT, PY) int8_gemv_kernel<ROWS_PER_BLOCK, NV, __nv_bfloat16, YT> \
        <<<grid, threads, shm, stream>>>( \
            qweight.data_ptr<unsigned char>(), \
            reinterpret_cast<const __half *>(scales.data_ptr()), \
            zeros.data_ptr<unsigned char>(), \
            reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()), PY, M, K, N, (int)group, splits, \
            reinterpret_cast<const __nv_bfloat16 *>(rc.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16 *>(wc.data_ptr()), \
            reinterpret_cast<__nv_bfloat16 *>(xout.data_ptr()), (float)eps, (float)mult)
    #define I8N_N(YT, PY) do { switch (N) { \
        case 1: I8N(1, YT, PY); break; case 2: I8N(2, YT, PY); break; \
        case 3: I8N(3, YT, PY); break; case 4: I8N(4, YT, PY); break; \
        case 5: I8N(5, YT, PY); break; case 6: I8N(6, YT, PY); break; \
        case 7: I8N(7, YT, PY); break; default: I8N(8, YT, PY); break; } } while (0)
    if (splits == 1) I8N_N(__nv_bfloat16, reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()));
    else I8N_N(float, out.data_ptr<float>());
    #undef I8N_N
    #undef I8N
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (splits != 1) out = out.to(torch::kBFloat16);
    return {x.dim() == 1 ? out.squeeze(0) : out, x.dim() == 1 ? xout.squeeze(0) : xout};
}

torch::Tensor int8_gemv(torch::Tensor qweight, torch::Tensor scales,
                        torch::Tensor zeros, torch::Tensor x, int64_t group, bool sortie_fp32) {
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
    // sortie_fp32 (tete lm_head, sage-duel-verdict par. 14 (ii)) : x lu en bf16,
    // accumulation et SORTIE fp32 — le meme noyau, les memes produits (un bf16
    // converti est exact en fp32) et le meme ordre de sommes que le chemin
    // x.to(float32) : logits egaux au bit, sans la conversion de h ni le
    // double trafic de x en fp32 dans chaque bloc.
    const bool bf_y32 = bf && sortie_fp32;
    xc = (bf ? xc : xc.to(torch::kFloat)).contiguous();
    const int N = xc.size(0);
    auto opt_y = bf_y32 ? xc.options().dtype(torch::kFloat) : xc.options();
    auto out = splits == 1 ? torch::empty({N, M}, opt_y)
                           : torch::zeros({N, M}, opt_y);
    auto stream = at::cuda::getCurrentCUDAStream();
    // Variante « un warp par ligne » (bead z5q), COUPÉE par défaut : mesurée
    // le 14/09 à 2,79 ms (1 ligne/warp) puis 1,69 ms sur q/k/v seuls (4
    // lignes/warp) contre ~1,29 pour le noyau à blocs. À N = 12 une ligne
    // touche 48 Ko d'activations pour 2 Ko de poids : ce GEMV est un GEMM
    // étroit, et le noyau à blocs lit x depuis le L1 mieux que la copie en
    // shared. ACVRAM_INT8_GEMV_WARP=1 pour la remesurer.
    static const bool warp_ok = std::getenv("ACVRAM_INT8_GEMV_WARP") != nullptr
                                && std::string(std::getenv("ACVRAM_INT8_GEMV_WARP")) == "1";
    const size_t shm_x = (size_t)N * K * sizeof(__nv_bfloat16);
    if (bf && !bf_y32 && warp_ok && N <= 12 && K % (WARP * WEIGHTS_PER_LOAD) == 0 && K <= 2048 && shm_x <= 96 * 1024) {
        dim3 gridw((M + I8W_WARPS * I8W_RPW - 1) / (I8W_WARPS * I8W_RPW));
        #define I8W(NV) do { \
            if (shm_x > 48 * 1024) cudaFuncSetAttribute(int8_gemv_warp_kernel<NV>, \
                                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_x); \
            int8_gemv_warp_kernel<NV><<<gridw, I8W_WARPS * WARP, shm_x, stream>>>( \
                qweight.data_ptr<unsigned char>(), reinterpret_cast<const __half *>(scales.data_ptr()), \
                zeros.data_ptr<unsigned char>(), reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()), \
                reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), M, K, (int)group); } while (0)
        switch (N) {
            case 1: I8W(1); break; case 2: I8W(2); break; case 3: I8W(3); break;
            case 4: I8W(4); break; case 5: I8W(5); break; case 6: I8W(6); break;
            case 7: I8W(7); break; case 8: I8W(8); break; case 9: I8W(9); break;
            case 10: I8W(10); break; case 11: I8W(11); break; default: I8W(12); break;
        }
        #undef I8W
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return x.dim() == 1 ? out.squeeze(0) : out;
    }
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
    const size_t shm = ROWS_PER_BLOCK * nwarps * sizeof(float);
    // NV : nombre d'activations traitées par lecture de poids, arrondi à la
    // puissance de deux supérieure (les lignes en trop sont ignorées).
    #define I8G(NV, XT, YT, PX, PY, SP) int8_gemv_kernel<ROWS_PER_BLOCK, NV, XT, YT> \
        <<<grid, threads, shm, stream>>>( \
            qweight.data_ptr<unsigned char>(), \
            reinterpret_cast<const __half *>(scales.data_ptr()), \
            zeros.data_ptr<unsigned char>(), PX, PY, M, K, N, (int)group, SP)
    #define I8G_N(XT, YT, PX, PY, SP) do { \
        switch (N) { \
        case 1: I8G(1, XT, YT, PX, PY, SP); break; \
        case 2: I8G(2, XT, YT, PX, PY, SP); break; \
        case 3: I8G(3, XT, YT, PX, PY, SP); break; \
        case 4: I8G(4, XT, YT, PX, PY, SP); break; \
        case 5: I8G(5, XT, YT, PX, PY, SP); break; \
        case 6: I8G(6, XT, YT, PX, PY, SP); break; \
        case 7: I8G(7, XT, YT, PX, PY, SP); break; \
        case 8: I8G(8, XT, YT, PX, PY, SP); break; \
        case 9: I8G(9, XT, YT, PX, PY, SP); break; \
        case 10: I8G(10, XT, YT, PX, PY, SP); break; \
        case 11: I8G(11, XT, YT, PX, PY, SP); break; \
        case 12: I8G(12, XT, YT, PX, PY, SP); break; \
        case 13: I8G(13, XT, YT, PX, PY, SP); break; \
        case 14: I8G(14, XT, YT, PX, PY, SP); break; \
        case 15: I8G(15, XT, YT, PX, PY, SP); break; \
        default: I8G(16, XT, YT, PX, PY, SP); break; } } while (0)
    // Tranches de 16 activations : un lot de 12 séquences passe en UNE lecture
    // des poids (NV=12), et le godet 16 des graphes CUDA (12 vraies lignes +
    // 4 fantômes) aussi (NV=16). Avant le 14/09 la tranche était de 12 : le
    // godet 16 lançait <4,12> puis <4,4>, et le second lancement RELISAIT
    // tous les poids — 1,25 Go par pas de décodage b=12, 19 % des octets DRAM
    // du pas, vus sous ncu en rejeu (revue/instr-par-octet-14-09.md).
    // ACVRAM_INT8_TRANCHE=12 rend l'ancien découpage (témoin A/B).
    static const int tranche = [] {
        const char *e = std::getenv("ACVRAM_INT8_TRANCHE");
        return (e && std::string(e) == "12") ? 12 : 16;
    }();
    const int Ntot = N;
    for (int base = 0; base < Ntot; base += tranche) {
        const int N = min(tranche, Ntot - base);     // masque volontaire pour I8G_N
        if (bf_y32) {
            I8G_N(__nv_bfloat16, float,
                  reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()) + (long)base * K,
                  out.data_ptr<float>() + (long)base * M, 1);
        } else if (bf) {
            I8G_N(__nv_bfloat16, __nv_bfloat16,
                  reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()) + (long)base * K,
                  reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()) + (long)base * M, 1);
        } else {
            I8G_N(float, float, xc.data_ptr<float>() + (long)base * K,
                  out.data_ptr<float>() + (long)base * M, splits);
        }
    }
    #undef I8G
    #undef I8G_N
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

#ifndef GW_WARPS
#define GW_WARPS 8
#endif
// RPW lignes par warp, en boucle : l'activation n'est chargée qu'une fois
// par bloc pour GW_WARPS*RPW lignes (sinon son trafic L2 égale celui des
// poids sur ces petites projections). Défaut 4 depuis sage-rpw-defaut-18-09
// (Laure, Coder b=12 in situ ABAB : 1 262 t/s nu contre 1 114 à rpw=1 ;
// ACVRAM_GROUPED_RPW=1|2 témoins).
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
    // Creneau fantome du remplissage godet (bucket_batch) : e < 0, pose par
    // le masque cote hote. Sortie a zero (ignoree par l'appelant), aucune
    // lecture de poids -- c'est tout le point (bead pds, 14/09) : un expert
    // qu'aucun jeton REEL ne demande ne traverse jamais le bus.
    if (e < 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)g * M + row] = 0.f;
        }
        return;
    }
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
    // Creneau fantome (e < 0, bead pds 14/09) : sortie a zero, aucune
    // lecture de poids -- meme garde que la variante table et la variante
    // simple (nvfp4_gemv_grouped_warp_kernel).
    if (e < 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)g * M + row] = 0.f;
        }
        return;
    }
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

// x en registres (défini plus bas) : aiguillage depuis les wrappers v1
static bool xreg_demande();
static bool xreg_demande_gateup();
static bool xreg_possible(int64_t K);
torch::Tensor nvfp4_gemv_grouped_xreg(torch::Tensor qw, torch::Tensor bscale, torch::Tensor gscales,
                                      torch::Tensor expert_ids, torch::Tensor token_ids, torch::Tensor x, int64_t K);
torch::Tensor nvfp4_gemv_grouped_gateup_xreg(torch::Tensor qg, torch::Tensor bg, torch::Tensor gsg,
                                             torch::Tensor qu, torch::Tensor bu, torch::Tensor gsu,
                                             torch::Tensor expert_ids, torch::Tensor token_ids,
                                             torch::Tensor x, int64_t K, int64_t act);

torch::Tensor nvfp4_gemv_grouped_gateup(
        torch::Tensor qg, torch::Tensor bg, torch::Tensor gsg,
        torch::Tensor qu, torch::Tensor bu, torch::Tensor gsu,
        torch::Tensor expert_ids, torch::Tensor token_ids,
        torch::Tensor x, int64_t K, int64_t act) {
    CHECK_CUDA(qg); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qg);
    CHECK_CONTIG(qg); CHECK_CONTIG(qu); CHECK_CONTIG(bg); CHECK_CONTIG(bu);
    TORCH_CHECK(K % 32 == 0 && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024,
                "gate-up fusionne : K multiple de 32 et <= 11904");
    if (xreg_demande_gateup() && xreg_possible(K))
        return nvfp4_gemv_grouped_gateup_xreg(qg, bg, gsg, qu, bu, gsu, expert_ids, token_ids, x, K, act);
    const int M = qg.size(1), G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
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

// ---------------------------------------------------------------------------
// GEMV groupée sur la DISPOSITION MARLIN — forme (b) de sage-p1-disposition-
// unique-18-09 (chiffrage : revue/p1-disposition-unique-chiffrage-18-09) :
// une seule disposition des experts en VRAM, celle de la GEMM du préfill
// (marlin_port.preparer_pile : gptq_marlin_repack, tuiles 16 k × 64 n de 128
// mots int32 ; échelles S0E5M3 permutées). La voie t du warp lit le uint4 t
// de la tuile = les mots {t·4 + w}, w = 0..3 (warp du repack) : colonne
// n = w·16 + t/4 et n + 8, k = (t%4)·2 + {0, 1, 8, 9} ; dans le mot, octet 0
// = (n, k0),(n, k8) ; octet 1 = (n+8, k0),(n+8, k8) ; octet 2 = (n, k1),(n, k9) ;
// octet 3 = (n+8, k1),(n+8, k9) (pack_idx {0,2,4,6,1,3,5,7} du repack, bas
// d'abord). Les échelles de ses 8 colonnes sont CONTIGUËS dans la ligne de
// la tuile k : octets 8·(t/4) + {0, 2, 1, 3, 4, 6, 5, 7} pour (w, n / n+8)
// (permuter_echelles : p = 8·(n%8) + n/8, puis [0,2,1,3] par 4 —
// traiter_echelles_nvfp4). Un warp lit donc une tuile entière en 512 o
// contigus, tous utiles. Bloc = 8 warps sur UNE tuile de colonnes, chaque
// warp un huitième des tuiles k ; réduction : 4 voies par colonne
// (shuffles), puis les 8 warps en mémoire partagée — ordre fixe, sortie
// déterministe, mais PAS identique au bit à v1 (autre ordre fp32) : juge
// fp32 par ligne (≤ 2⁻⁷·max|y|), tests/test_gemv_marlin.py.
constexpr int MB_WARPS = 8;
constexpr int MB_TN = 64;                 // colonnes par tuile
constexpr int MB_TK = 16;                 // k par tuile (= groupe d'échelle)

// Échelle S0E5M3 (traiter_echelles_nvfp4) : octet = exposant half (5 bits)
// << 3 | 3 bits hauts de mantisse de half(s·facteur)·2⁷ ; 0 = échelle nulle.
// s·facteur = (1 + m/8)·2^(e−22) → bits fp32 = (e+105) << 23 | m << 20
// = (b << 20) + 0x34800000. Le facteur se retire par l'échelle globale
// (traiter_echelle_globale : g·2^119/facteur → g/facteur = g_marlin·2⁻¹¹⁹).
// L'octet est pris à sa place par PRMT (bits 16-23, zéro ailleurs) puis un
// seul IMAD (×16 + C) : deux instructions par échelle. L'octet 0 (échelle
// que la pile a annulée : s·facteur < 2⁻⁶) donne 2⁻²²/facteur au lieu de 0 —
// ≤ 2⁻²²·6·|x| par poids, invisible au juge 2⁻⁷ ; on s'épargne le SEL.
__device__ __forceinline__ float s0e5m3_octet(unsigned int mot, int octet) {   // octet < 4, constant
    return __uint_as_float(__byte_perm(mot, 0u, 0x4044u | ((unsigned)octet << 8)) * 16u + 0x34800000u);
}
__device__ __forceinline__ unsigned int octet_de(unsigned int mot, int octet) {   // octet < 4, constant
    return __byte_perm(mot, 0u, 0x4440u | (unsigned)octet);
}

// Une tuile pour une voie : 4 mots (8 colonnes × 4 k), 8 octets d'échelle,
// x aux k {k0, k1} (x01) et {k8, k9} (x89). acc[w][h] : colonne w·16 + t/4 + 8h.
__device__ __forceinline__ void mb_tuile(const uint4 p, const uint2 sc,
                                         const float2 x01, const float2 x89, float (&acc)[4][2]) {
    const unsigned int w[4] = {p.x, p.y, p.z, p.w};
    const unsigned int s[2] = {sc.x, sc.y};
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float2 b0 = e2m1_pair(octet_de(w[q], 0));
        const float2 b1 = e2m1_pair(octet_de(w[q], 1));
        const float2 b2 = e2m1_pair(octet_de(w[q], 2));
        const float2 b3 = e2m1_pair(octet_de(w[q], 3));
        float pn = b0.x * x01.x + b0.y * x89.x;          // (n, k0), (n, k8)
        pn += b2.x * x01.y + b2.y * x89.y;               // (n, k1), (n, k9)
        float pn8 = b1.x * x01.x + b1.y * x89.x;         // (n+8, k0), (n+8, k8)
        pn8 += b3.x * x01.y + b3.y * x89.y;              // (n+8, k1), (n+8, k9)
        const int i0 = 4 * (q >> 1) + (q & 1);           // octet d'échelle de n ; n+8 : i0 + 2
        acc[q][0] += pn * s0e5m3_octet(s[i0 >> 2], i0 & 3);
        acc[q][1] += pn8 * s0e5m3_octet(s[(i0 + 2) >> 2], (i0 + 2) & 3);
    }
}

// NW = 1 : une projection (down) ; NW = 2 : gate et up fusionnés, sortie
// act(gate)·up. Grille (N/64, G, S), 256 fils, shared = K flottants + NW·8·64.
// S = split-K (poste b=1, verdict-profil-b1-b-18-09) : à b=1 la grille ne
// fait que 96 blocs pour 170 SM et chaque warp enchaîne 16 tuiles × 2
// projections (gate/up 0,922 ms contre 0,713 en v1, +29 %), alors que le
// down (256 blocs, 6 tuiles par warp) égale v1. Le bloc z ne somme que les
// tuiles [z·KT/S, (z+1)·KT/S) ; S > 1 : dépôt partiel dans `part`
// [S][NW][G][N], puis le DERNIER bloc arrivé (compteur par colonne de
// tuile, __threadfence) somme les S partiels dans l'ordre z = 0..S−1 —
// sortie déterministe quel que soit l'ordre d'arrivée — applique les
// échelles globales et l'activation, et remet le compteur à zéro (réutilisable
// sous graphe sans memset). S = 1 : chemin d'origine, `part`/`cpt` nuls.
template <typename XT, int NW>
__global__ void __launch_bounds__(MB_WARPS * WARP) nvfp4_gemv_marlin_kernel(
    const uint4 *__restrict__ q0, const unsigned char *__restrict__ s0, const float *__restrict__ g0,
    const uint4 *__restrict__ q1, const unsigned char *__restrict__ s1, const float *__restrict__ g1,
    const int *__restrict__ expert_ids, const int *__restrict__ token_ids,
    const XT *__restrict__ x, float *__restrict__ y, int N, int K, int act,
    float *__restrict__ part, unsigned int *__restrict__ cpt,
    const __nv_bfloat16 *__restrict__ xsc, int ld_sc, int ldn) {
    extern __shared__ float xs[];
    float *red = xs + K;
    __shared__ bool dernier;
    const int g = blockIdx.y, e = expert_ids[g], nt = blockIdx.x;
    const int S = gridDim.z, z = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // creneau fantome (e < 0, bead pds) : zero, aucun octet de poids lu
    if (e < 0) {
        if (z == 0 && threadIdx.x < MB_TN) y[(long)g * N + nt * MB_TN + threadIdx.x] = 0.f;
        return;
    }
    const XT *xn = x + (long)token_ids[g] * K;
    const int KT = K / MB_TK, NT = N / MB_TN;
    // Pièce 82 : largeur de ligne STOCKÉE (colonnes) des poids et des échelles — N partout, 2N quand gate et up
    // sont lues dans w13 (tuiles de gate puis d up sur chaque ligne de tuiles). La sortie reste en N.
    const int LN = ldn / MB_TN;
    const int kt0 = z * KT / S, kt1 = (z + 1) * KT / S;     // tuiles de ce bloc
    // Pièce 47 : l'échelle AWQ par expert (x / s[e]) se faisait en torch devant
    // le GEMV — un gather, une division et un cast par projection et par
    // couche, 8 lancements et 0,473 ms/pas à b=12 (oceane-piece42-glue-22-09).
    // Ici elle est lue au même endroit que le chargement en mémoire partagée,
    // sans lancement. MESURÉ (pièce 47, 22/09) : la glue tombe de 0,964 à
    // 0,492 ms/pas mais le GEMV monte de 3,965 à 4,334 — la table [E, K] est
    // relue par CHACUN des N/64 blocs de la grille, comme x lui-même. Gain net
    // 0,21 ms/pas sur le mur, pas 0,47 : l'octet relu mange la moitié du
    // lancement épargné.
    // AU BIT contre `(x.to(bf16) / table).to(dtype)` : PyTorch promeut en
    // float, divise et arrondit UNE fois en bf16 ; une entrée fp32 (le down)
    // est arrondie en bf16 AVANT la division, comme `act.to(torch.bfloat16)`.
    const __nv_bfloat16 *sce = xsc ? xsc + (long)e * ld_sc : nullptr;
    for (int i = threadIdx.x + kt0 * MB_TK; i < kt1 * MB_TK; i += blockDim.x) {
        float v;
        if constexpr (sizeof(XT) == 4) v = sce ? __bfloat162float(__float2bfloat16_rn(xn[i])) : xn[i];
        else v = __bfloat162float(xn[i]);
        if (sce) v = __bfloat162float(__float2bfloat16_rn(v / __bfloat162float(sce[i])));
        xs[i] = v;
    }
    __syncthreads();
    const long bw = (long)e * KT * LN * (MB_TK * MB_TN / 32) + (long)nt * (MB_TK * MB_TN / 32) + lane;
    const int tr = (lane & 3) * 2, c = lane >> 2;
    const long bs = (long)e * KT * ldn + (long)nt * MB_TN + 8 * c;
    float acc[NW][4][2] = {};
    int kt = kt0 + warp;
    // deux tuiles en vol par voie (les chargements des deux partent avant le calcul)
    for (; kt + MB_WARPS < kt1; kt += 2 * MB_WARPS) {
        const int k2 = kt + MB_WARPS;
        const uint4 pa = q0[bw + (long)kt * LN * 32], pb = q0[bw + (long)k2 * LN * 32];
        const uint2 sa = *reinterpret_cast<const uint2 *>(s0 + bs + (long)kt * ldn);
        const uint2 sb = *reinterpret_cast<const uint2 *>(s0 + bs + (long)k2 * ldn);
        uint4 pa1, pb1; uint2 sa1, sb1;
        if constexpr (NW == 2) {
            pa1 = q1[bw + (long)kt * LN * 32]; pb1 = q1[bw + (long)k2 * LN * 32];
            sa1 = *reinterpret_cast<const uint2 *>(s1 + bs + (long)kt * ldn);
            sb1 = *reinterpret_cast<const uint2 *>(s1 + bs + (long)k2 * ldn);
        }
        const float2 xa01 = *reinterpret_cast<const float2 *>(xs + kt * MB_TK + tr);
        const float2 xa89 = *reinterpret_cast<const float2 *>(xs + kt * MB_TK + tr + 8);
        const float2 xb01 = *reinterpret_cast<const float2 *>(xs + k2 * MB_TK + tr);
        const float2 xb89 = *reinterpret_cast<const float2 *>(xs + k2 * MB_TK + tr + 8);
        mb_tuile(pa, sa, xa01, xa89, acc[0]);
        mb_tuile(pb, sb, xb01, xb89, acc[0]);
        if constexpr (NW == 2) {
            mb_tuile(pa1, sa1, xa01, xa89, acc[1]);
            mb_tuile(pb1, sb1, xb01, xb89, acc[1]);
        }
    }
    for (; kt < kt1; kt += MB_WARPS) {
        const uint4 pa = q0[bw + (long)kt * LN * 32];
        const uint2 sa = *reinterpret_cast<const uint2 *>(s0 + bs + (long)kt * ldn);
        const float2 xa01 = *reinterpret_cast<const float2 *>(xs + kt * MB_TK + tr);
        const float2 xa89 = *reinterpret_cast<const float2 *>(xs + kt * MB_TK + tr + 8);
        mb_tuile(pa, sa, xa01, xa89, acc[0]);
        if constexpr (NW == 2) {
            const uint4 pa1 = q1[bw + (long)kt * LN * 32];
            const uint2 sa1 = *reinterpret_cast<const uint2 *>(s1 + bs + (long)kt * ldn);
            mb_tuile(pa1, sa1, xa01, xa89, acc[1]);
        }
    }
    // 4 voies (t%4) par colonne → une ; puis dépôt par warp
    #pragma unroll
    for (int n = 0; n < NW; ++n)
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                float v = acc[n][q][h];
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                if ((lane & 3) == 0) red[(n * MB_WARPS + warp) * MB_TN + q * 16 + c + 8 * h] = v;
            }
    __syncthreads();
    if (threadIdx.x < MB_TN) {
        const int col = threadIdx.x;
        float t0 = 0.f, t1 = 0.f;
        #pragma unroll
        for (int w = 0; w < MB_WARPS; ++w) t0 += red[w * MB_TN + col];
        if constexpr (NW == 2) {
            #pragma unroll
            for (int w = 0; w < MB_WARPS; ++w) t1 += red[(MB_WARPS + w) * MB_TN + col];
        }
        if (S > 1) {
            // dépôt partiel, visible aux autres blocs avant l'incrément du compteur
            part[((long)(z * NW) * gridDim.y + g) * N + nt * MB_TN + col] = t0;
            if constexpr (NW == 2) part[((long)(z * NW + 1) * gridDim.y + g) * N + nt * MB_TN + col] = t1;
            __threadfence();
        } else {
            t0 *= g0[e] * 0x1p-119f;
            if constexpr (NW == 2) t0 = acv_act(t0, act) * (t1 * (g1[e] * 0x1p-119f));
            y[(long)g * N + nt * MB_TN + col] = t0;
        }
    }
    if (S == 1) return;
    __syncthreads();
    if (threadIdx.x == 0) dernier = (atomicAdd(cpt + g * NT + nt, 1u) == (unsigned)(S - 1));
    __syncthreads();
    if (!dernier) return;
    __threadfence();
    if (threadIdx.x < MB_TN) {
        const int col = threadIdx.x;
        const volatile float *vp = part;
        float t0 = 0.f, t1 = 0.f;
        for (int zz = 0; zz < S; ++zz) {                 // ordre fixe : somme déterministe
            t0 += vp[((long)(zz * NW) * gridDim.y + g) * N + nt * MB_TN + col];
            if constexpr (NW == 2) t1 += vp[((long)(zz * NW + 1) * gridDim.y + g) * N + nt * MB_TN + col];
        }
        t0 *= g0[e] * 0x1p-119f;
        if constexpr (NW == 2) t0 = acv_act(t0, act) * (t1 * (g1[e] * 0x1p-119f));
        y[(long)g * N + nt * MB_TN + col] = t0;
    }
    if (threadIdx.x == 0) cpt[g * NT + nt] = 0u;         // prêt pour le rejeu suivant
}

// Split-K par lot — OPT-IN (sage, 19/09, verdict-splitk-b1-19-09) :
// ACVRAM_GEMV_SPLITK=0 (défaut) : S = 1, noyau d'avant, sortie inchangée ;
// =1 : S doublé tant que la grille reste sous MB_BLOCS_MIN et que chaque bloc
// garde ≥ MB_WARPS tuiles k (b=1 gate/up Coder : 96 blocs → S = 4 ; b=2 → 2 ;
// b ≥ 4 → 1) — mesuré b=1 +3,1 % t/s, PPL décodage +0,0042 contre un témoin
// graphes/eager de 0,0026 : non tranché, donc pas au défaut ; =n ≥ 2 : S forcé
// (diagnostic). Lu une fois par processus (comme GROUPED_RPW).
constexpr int MB_BLOCS_MIN = 384;
// Tampons du split-K, statiques par appareil : `cpt` est remis à zéro par le
// dernier bloc de chaque colonne, donc zéroté UNE fois à l'allocation (pas de
// memset par appel : +3 µs mesurés au banc, plus que le gain à b=1) ; `part`
// est un brouillon réutilisé. Adresses fixes : compatibles avec la capture de
// graphe (le tenseur vit aussi longtemps que le processus).
static std::pair<float *, unsigned int *> mb_tampons(const torch::Tensor &ref, long n_part, long n_cpt) {
    static std::vector<torch::Tensor> parts(64), cpts(64);
    const int d = ref.device().index() < 0 ? 0 : ref.device().index();
    if (!parts[d].defined() || parts[d].numel() < n_part)
        parts[d] = torch::empty({std::max(n_part, 1L << 20)}, ref.options().dtype(torch::kFloat));
    if (!cpts[d].defined() || cpts[d].numel() < n_cpt)
        cpts[d] = torch::zeros({std::max(n_cpt, 1L << 16)}, ref.options().dtype(torch::kInt));
    return {parts[d].data_ptr<float>(), reinterpret_cast<unsigned int *>(cpts[d].data_ptr<int>())};
}
static int mb_splitk(int NT, int G, int KT) {
    static const int mode = std::getenv("ACVRAM_GEMV_SPLITK") ? atoi(std::getenv("ACVRAM_GEMV_SPLITK")) : 1;
    if (mode <= 0) return 1;
    if (mode >= 2) return mode;
    int S = 1;
    while (NT * G * S < MB_BLOCS_MIN && KT / (2 * S) >= MB_WARPS && 2 * S <= 8) S *= 2;
    return S;
}

static void mb_verifier(const torch::Tensor &w, const torch::Tensor &s, const torch::Tensor &g, int64_t K, int64_t N) {
    CHECK_CUDA(w); CHECK_CONTIG(w); CHECK_CONTIG(s); CHECK_CONTIG(g);
    TORCH_CHECK(w.scalar_type() == torch::kInt, "disposition Marlin : poids int32 repackés");
    TORCH_CHECK(s.scalar_type() == torch::kByte || s.scalar_type() == torch::kFloat8_e4m3fn,
                "disposition Marlin : échelles S0E5M3 (uint8 ou float8_e4m3fn)");
    TORCH_CHECK(g.scalar_type() == torch::kFloat, "échelle globale Marlin fp32");
    TORCH_CHECK(K % 64 == 0 && N % 64 == 0, "disposition Marlin : K et N multiples de 64");
    TORCH_CHECK(w.dim() == 3 && w.size(1) == K / 16 && w.size(2) == N * 2, "poids Marlin [E, K/16, N·2]");
    TORCH_CHECK(s.dim() == 3 && s.size(1) == K / 16 && s.size(2) == N, "échelles Marlin [E, K/16, N]");
    TORCH_CHECK((size_t)(K + 2 * MB_WARPS * MB_TN) * sizeof(float) <= 48 * 1024, "K ≤ 11264");
}

// x [T, K] bf16 ou fp32 ; sortie [G, N] fp32, ligne g = paire (expert_ids[g], token_ids[g]).

// ---------------------------------------------------------------------------
// Pièce 61 (23/09) : GEMV Marlin PAR CRÉNEAU D EXPERT. Le noyau par paire
// (ci-dessous) charge la tuile de poids d un expert une fois PAR PAIRE
// (expert, jeton) : à b=12, 96 paires par couche pour 33 experts distincts
// (mesuré, revue/gaelle-experts-par-paire-23-09), 12,3 Go/pas de tuiles pour
// 4,29 Go/pas de poids distincts — borné par le L2, 71 % du plancher HBM.
// Ici un bloc = (tuile N, créneau) ; un créneau = un expert et jusqu à TPB de
// ses paires : la tuile est lue une fois et appliquée aux TPB lignes de x
// tenues en mémoire partagée. AU BIT contre le noyau par paire : pour chaque
// paire, même ordre de tuiles par warp (deux en vol), mêmes fma dans le même
// accumulateur, même réduction inter-warps, même épilogue — seule la SOURCE de
// x change (partagée [j][K] au lieu de [K]). Sortie y[G, N] par paire,
// inchangée. Grille STATIQUE (N/64, G) sous capture : le nombre de créneaux
// réels varie à chaque pas (≤ G), les créneaux vides (slot_e = -2) sortent
// avant tout chargement, les paires fantômes (e < 0) sont regroupées dans des
// créneaux slot_e = -1 qui écrivent des zéros comme le noyau par paire.
// Pas de split-K : les créneaux servent les godets ≥ 2, le noyau par paire
// reste servi au godet 1 (b=1, split-K utile).
// ---------------------------------------------------------------------------
template <typename XT, int NW, int TPB>
__global__ void __launch_bounds__(MB_WARPS * WARP) nvfp4_gemv_marlin_slots_kernel(
    const uint4 *__restrict__ q0, const unsigned char *__restrict__ s0, const float *__restrict__ g0,
    const uint4 *__restrict__ q1, const unsigned char *__restrict__ s1, const float *__restrict__ g1,
    const int *__restrict__ slot_e, const int *__restrict__ slot_pair, const int *__restrict__ token_ids,
    const XT *__restrict__ x, float *__restrict__ y, int N, int K, int act,
    const __nv_bfloat16 *__restrict__ xsc, int ld_sc) {
    extern __shared__ float xs[];                        // [TPB][K]
    float *red = xs + (long)TPB * K;                     // [TPB][NW][MB_WARPS][MB_TN]
    const int sl = blockIdx.y, e = slot_e[sl], nt = blockIdx.x;
    if (e == -2) return;                                 // créneau vide : rien à écrire
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int gp[TPB];
    int nj = 0;
    #pragma unroll
    for (int j = 0; j < TPB; ++j) { gp[j] = slot_pair[sl * TPB + j]; if (gp[j] >= 0) nj = j + 1; }
    if (e < 0) {                                         // créneau de fantômes : zéros, aucun octet de poids lu
        if (threadIdx.x < MB_TN)
            for (int j = 0; j < nj; ++j) y[(long)gp[j] * N + nt * MB_TN + threadIdx.x] = 0.f;
        return;
    }
    const int KT = K / MB_TK, NT = N / MB_TN;
    const __nv_bfloat16 *sce = xsc ? xsc + (long)e * ld_sc : nullptr;
    for (int j = 0; j < nj; ++j) {                       // même chargement et même arrondi que le noyau par paire
        const XT *xn = x + (long)token_ids[gp[j]] * K;
        float *xj = xs + (long)j * K;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v;
            if constexpr (sizeof(XT) == 4) v = sce ? __bfloat162float(__float2bfloat16_rn(xn[i])) : xn[i];
            else v = __bfloat162float(xn[i]);
            if (sce) v = __bfloat162float(__float2bfloat16_rn(v / __bfloat162float(sce[i])));
            xj[i] = v;
        }
    }
    __syncthreads();
    const long bw = (long)e * KT * NT * (MB_TK * MB_TN / 32) + (long)nt * (MB_TK * MB_TN / 32) + lane;
    const int tr = (lane & 3) * 2, c = lane >> 2;
    const long bs = (long)e * KT * N + (long)nt * MB_TN + 8 * c;
    float acc[TPB][NW][4][2] = {};
    int kt = warp;
    for (; kt + MB_WARPS < KT; kt += 2 * MB_WARPS) {
        const int k2 = kt + MB_WARPS;
        const uint4 pa = q0[bw + (long)kt * NT * 32], pb = q0[bw + (long)k2 * NT * 32];
        const uint2 sa = *reinterpret_cast<const uint2 *>(s0 + bs + (long)kt * N);
        const uint2 sb = *reinterpret_cast<const uint2 *>(s0 + bs + (long)k2 * N);
        uint4 pa1, pb1; uint2 sa1, sb1;
        if constexpr (NW == 2) {
            pa1 = q1[bw + (long)kt * NT * 32]; pb1 = q1[bw + (long)k2 * NT * 32];
            sa1 = *reinterpret_cast<const uint2 *>(s1 + bs + (long)kt * N);
            sb1 = *reinterpret_cast<const uint2 *>(s1 + bs + (long)k2 * N);
        }
        #pragma unroll
        for (int j = 0; j < TPB; ++j) {
            if (j < nj) {
                const float *xj = xs + (long)j * K;
                const float2 xa01 = *reinterpret_cast<const float2 *>(xj + kt * MB_TK + tr);
                const float2 xa89 = *reinterpret_cast<const float2 *>(xj + kt * MB_TK + tr + 8);
                const float2 xb01 = *reinterpret_cast<const float2 *>(xj + k2 * MB_TK + tr);
                const float2 xb89 = *reinterpret_cast<const float2 *>(xj + k2 * MB_TK + tr + 8);
                mb_tuile(pa, sa, xa01, xa89, acc[j][0]);
                mb_tuile(pb, sb, xb01, xb89, acc[j][0]);
                if constexpr (NW == 2) {
                    mb_tuile(pa1, sa1, xa01, xa89, acc[j][1]);
                    mb_tuile(pb1, sb1, xb01, xb89, acc[j][1]);
                }
            }
        }
    }
    for (; kt < KT; kt += MB_WARPS) {
        const uint4 pa = q0[bw + (long)kt * NT * 32];
        const uint2 sa = *reinterpret_cast<const uint2 *>(s0 + bs + (long)kt * N);
        uint4 pa1; uint2 sa1;
        if constexpr (NW == 2) {
            pa1 = q1[bw + (long)kt * NT * 32];
            sa1 = *reinterpret_cast<const uint2 *>(s1 + bs + (long)kt * N);
        }
        #pragma unroll
        for (int j = 0; j < TPB; ++j) {
            if (j < nj) {
                const float *xj = xs + (long)j * K;
                const float2 xa01 = *reinterpret_cast<const float2 *>(xj + kt * MB_TK + tr);
                const float2 xa89 = *reinterpret_cast<const float2 *>(xj + kt * MB_TK + tr + 8);
                mb_tuile(pa, sa, xa01, xa89, acc[j][0]);
                if constexpr (NW == 2) mb_tuile(pa1, sa1, xa01, xa89, acc[j][1]);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < TPB; ++j) {
        if (j < nj) {
            #pragma unroll
            for (int n = 0; n < NW; ++n)
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float v = acc[j][n][q][h];
                        v += __shfl_xor_sync(0xffffffffu, v, 1);
                        v += __shfl_xor_sync(0xffffffffu, v, 2);
                        if ((lane & 3) == 0) red[(((long)j * NW + n) * MB_WARPS + warp) * MB_TN + q * 16 + c + 8 * h] = v;
                    }
        }
    }
    __syncthreads();
    if (threadIdx.x < MB_TN) {
        const int col = threadIdx.x;
        for (int j = 0; j < nj; ++j) {
            float t0 = 0.f, t1 = 0.f;
            #pragma unroll
            for (int w = 0; w < MB_WARPS; ++w) t0 += red[(((long)j * NW) * MB_WARPS + w) * MB_TN + col];
            if constexpr (NW == 2) {
                #pragma unroll
                for (int w = 0; w < MB_WARPS; ++w) t1 += red[(((long)j * NW + 1) * MB_WARPS + w) * MB_TN + col];
            }
            t0 *= g0[e] * 0x1p-119f;
            if constexpr (NW == 2) t0 = acv_act(t0, act) * (t1 * (g1[e] * 0x1p-119f));
            y[(long)gp[j] * N + nt * MB_TN + col] = t0;
        }
    }
}

// Constructeur des créneaux : tri par comptage sur E + 1 experts (les paires
// fantômes e < 0 forment le groupe E, écrit slot_e = -1), rang par expert,
// créneau = slot_start[e] + rang / TPB. Un bloc, G ≤ 1024 paires, E ≤ 1024.
// Sorties de taille FIXE (G créneaux, G × TPB entrées) : capturable sous graphe.
// L ordre des paires à l intérieur d un créneau n influe sur aucune sortie.
__global__ void moe_slots_kernel(const int *__restrict__ expert_ids, int G, int E, int TPB,
                                 int *__restrict__ slot_e, int *__restrict__ slot_pair) {
    __shared__ int hist[1025];
    __shared__ int start[1025];
    __shared__ int cursor[1025];
    for (int i = threadIdx.x; i <= E; i += blockDim.x) { hist[i] = 0; cursor[i] = 0; }
    for (int i = threadIdx.x; i < G; i += blockDim.x) slot_e[i] = -2;
    for (int i = threadIdx.x; i < G * TPB; i += blockDim.x) slot_pair[i] = -1;
    __syncthreads();
    for (int i = threadIdx.x; i < G; i += blockDim.x) {
        const int e = expert_ids[i];
        atomicAdd(&hist[e < 0 ? E : e], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {                                     // balayage exclusif des créneaux par expert
        int acc = 0;
        for (int e = 0; e <= E; ++e) { start[e] = acc; acc += (hist[e] + TPB - 1) / TPB; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G; i += blockDim.x) {
        const int e = expert_ids[i];
        const int ei = e < 0 ? E : e;
        const int r = atomicAdd(&cursor[ei], 1);
        const int sl = start[ei] + r / TPB;
        slot_pair[sl * TPB + (r % TPB)] = i;
        slot_e[sl] = e < 0 ? -1 : e;
    }
}

std::tuple<torch::Tensor, torch::Tensor> moe_slots(torch::Tensor expert_ids, int64_t E, int64_t tpb) {
    CHECK_CUDA(expert_ids); ACVRAM_DEVICE_GUARD(expert_ids);
    TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32 && expert_ids.dim() == 1, "expert_ids : int32 [G]");
    TORCH_CHECK(tpb == 4 || tpb == 8, "TPB ∈ {4, 8}");
    const int G = (int)expert_ids.size(0);
    TORCH_CHECK(G >= 1 && G <= 1024 && E >= 1 && E <= 1024, "G, E ≤ 1024");
    auto slot_e = torch::empty({G}, expert_ids.options());
    auto slot_pair = torch::empty({G * tpb}, expert_ids.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    moe_slots_kernel<<<1, 256, 0, stream>>>(expert_ids.data_ptr<int>(), G, (int)E, (int)tpb,
                                            slot_e.data_ptr<int>(), slot_pair.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(slot_e, slot_pair);
}

template <typename XT, int NW, int TPB>
static void mb_slots_lancer(const torch::Tensor &w0, const torch::Tensor &s0, const torch::Tensor &g0,
                            const torch::Tensor *w1, const torch::Tensor *s1, const torch::Tensor *g1,
                            const torch::Tensor &slot_e, const torch::Tensor &slot_pair, const torch::Tensor &token_ids,
                            const XT *px, float *py, int N, int K, int act, const __nv_bfloat16 *psc, int ldsc,
                            cudaStream_t stream) {
    const size_t shm = (size_t)TPB * (K + NW * MB_WARPS * MB_TN) * sizeof(float);
    TORCH_CHECK(shm <= 227 * 1024, "créneaux : mémoire partagée > 227 Ko (K trop grand pour ce TPB)");
    static size_t attribut = 0;                          // plafond déjà déclaré pour cette instanciation
    if (shm > 48 * 1024 && shm > attribut) {
        const cudaError_t rc = cudaFuncSetAttribute(nvfp4_gemv_marlin_slots_kernel<XT, NW, TPB>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        TORCH_CHECK(rc == cudaSuccess, "créneaux : cudaFuncSetAttribute(", (long)shm, " o) : ", cudaGetErrorString(rc));
        attribut = shm;
    }
    dim3 grid(N / MB_TN, (int)slot_e.size(0), 1);
    nvfp4_gemv_marlin_slots_kernel<XT, NW, TPB><<<grid, MB_WARPS * WARP, shm, stream>>>(
        reinterpret_cast<const uint4 *>(w0.data_ptr()), static_cast<const unsigned char *>(s0.data_ptr()), g0.data_ptr<float>(),
        w1 ? reinterpret_cast<const uint4 *>(w1->data_ptr()) : nullptr, s1 ? static_cast<const unsigned char *>(s1->data_ptr()) : nullptr,
        g1 ? g1->data_ptr<float>() : nullptr,
        slot_e.data_ptr<int>(), slot_pair.data_ptr<int>(), token_ids.data_ptr<int>(), px, py, N, K, act, psc, ldsc);
}

static torch::Tensor mb_slots_commun(const torch::Tensor &w0, const torch::Tensor &s0, const torch::Tensor &g0,
                                     const torch::Tensor *w1, const torch::Tensor *s1, const torch::Tensor *g1,
                                     torch::Tensor slot_e, torch::Tensor slot_pair, torch::Tensor token_ids,
                                     torch::Tensor x, int64_t K, int64_t N, int64_t act, int64_t tpb,
                                     c10::optional<torch::Tensor> xscale) {
    mb_verifier(w0, s0, g0, K, N); if (w1) mb_verifier(*w1, *s1, *g1, K, N);
    CHECK_CUDA(x); CHECK_CUDA(slot_e); CHECK_CUDA(slot_pair); CHECK_CUDA(token_ids); ACVRAM_DEVICE_GUARD(w0);
    TORCH_CHECK(tpb == 4 || tpb == 8, "TPB ∈ {4, 8}");
    TORCH_CHECK(slot_pair.size(0) == slot_e.size(0) * tpb, "slot_pair [G × TPB]");
    const int G = (int)slot_e.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    TORCH_CHECK(xc.size(-1) == K, "x : dernière dimension K");
    auto out = torch::empty({G, N}, xc.options().dtype(torch::kFloat));
    const __nv_bfloat16 *psc = nullptr; int ldsc = 0;
    if (xscale.has_value() && xscale->defined()) {
        const torch::Tensor &sc = *xscale;
        CHECK_CUDA(sc);
        TORCH_CHECK(sc.scalar_type() == torch::kBFloat16, "échelle AWQ : bf16 attendu (table de moe.py)");
        TORCH_CHECK(sc.dim() == 2 && sc.size(1) >= K, "échelle AWQ [E, ≥ K]");
        TORCH_CHECK(sc.stride(1) == 1, "échelle AWQ : lignes contiguës");
        psc = reinterpret_cast<const __nv_bfloat16 *>(sc.data_ptr());
        ldsc = (int)sc.stride(0);
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    #define MB_SL(XT, PX, NW, TPB) mb_slots_lancer<XT, NW, TPB>(w0, s0, g0, w1, s1, g1, slot_e, slot_pair, token_ids, PX, \
                                                                 out.data_ptr<float>(), (int)N, (int)K, (int)act, psc, ldsc, stream)
    #define MB_SL2(XT, PX) do { if (w1) { if (tpb == 4) MB_SL(XT, PX, 2, 4); else MB_SL(XT, PX, 2, 8); } \
                                else { if (tpb == 4) MB_SL(XT, PX, 1, 4); else MB_SL(XT, PX, 1, 8); } } while (0)
    if (bf) { MB_SL2(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { MB_SL2(float, xc.data_ptr<float>()); }
    #undef MB_SL2
    #undef MB_SL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nvfp4_gemv_marlin_slots(torch::Tensor w, torch::Tensor s, torch::Tensor g,
                                      torch::Tensor slot_e, torch::Tensor slot_pair, torch::Tensor token_ids,
                                      torch::Tensor x, int64_t K, int64_t N, int64_t tpb,
                                      c10::optional<torch::Tensor> xscale) {
    return mb_slots_commun(w, s, g, nullptr, nullptr, nullptr, slot_e, slot_pair, token_ids, x, K, N, 0, tpb, xscale);
}

torch::Tensor nvfp4_gemv_marlin_gateup_slots(torch::Tensor wg, torch::Tensor sg, torch::Tensor gg,
                                             torch::Tensor wu, torch::Tensor su, torch::Tensor gu,
                                             torch::Tensor slot_e, torch::Tensor slot_pair, torch::Tensor token_ids,
                                             torch::Tensor x, int64_t K, int64_t N, int64_t act, int64_t tpb,
                                             c10::optional<torch::Tensor> xscale) {
    return mb_slots_commun(wg, sg, gg, &wu, &su, &gu, slot_e, slot_pair, token_ids, x, K, N, act, tpb, xscale);
}

torch::Tensor nvfp4_gemv_marlin(torch::Tensor w, torch::Tensor s, torch::Tensor g,
                                torch::Tensor expert_ids, torch::Tensor token_ids,
                                torch::Tensor x, int64_t K, int64_t N,
                                c10::optional<torch::Tensor> xscale) {
    mb_verifier(w, s, g, K, N); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(w);
    const int G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    TORCH_CHECK(xc.size(-1) == K, "x : dernière dimension K");
    auto out = torch::empty({G, N}, xc.options().dtype(torch::kFloat));
    const int S = mb_splitk(N / MB_TN, G, K / MB_TK);
    dim3 grid(N / MB_TN, G, S);
    float *part = nullptr; unsigned int *cpt = nullptr;
    if (S > 1) std::tie(part, cpt) = mb_tampons(out, (long)S * G * N, (long)G * (N / MB_TN));
    const size_t shm = (size_t)(K + MB_WARPS * MB_TN) * sizeof(float);
    const __nv_bfloat16 *psc = nullptr; int ldsc = 0;
    if (xscale.has_value() && xscale->defined()) {
        const torch::Tensor &sc = *xscale;
        CHECK_CUDA(sc);
        TORCH_CHECK(sc.scalar_type() == torch::kBFloat16, "échelle AWQ : bf16 attendu (table de moe.py)");
        TORCH_CHECK(sc.dim() == 2 && sc.size(1) >= K, "échelle AWQ [E, ≥ K]");
        TORCH_CHECK(sc.stride(1) == 1, "échelle AWQ : lignes contiguës");
        psc = reinterpret_cast<const __nv_bfloat16 *>(sc.data_ptr());
        ldsc = (int)sc.stride(0);
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    #define MB_L(XT, PX) nvfp4_gemv_marlin_kernel<XT, 1><<<grid, MB_WARPS * WARP, shm, stream>>>( \
        reinterpret_cast<const uint4 *>(w.data_ptr()), static_cast<const unsigned char *>(s.data_ptr()), g.data_ptr<float>(), \
        nullptr, nullptr, nullptr, expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), (int)N, (int)K, 0, \
        part, cpt, psc, ldsc, (int)N)
    if (bf) { MB_L(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { MB_L(float, xc.data_ptr<float>()); }
    #undef MB_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Le S que prendrait un appel (K, N, G) : testable sans lancer le noyau
// (tests/test_gemv_marlin.py : 1 au défaut quel que soit le lot).
int64_t nvfp4_gemv_marlin_splitk(int64_t K, int64_t N, int64_t G) {
    return mb_splitk((int)(N / MB_TN), (int)G, (int)(K / MB_TK));
}

torch::Tensor nvfp4_gemv_marlin_gateup(torch::Tensor wg, torch::Tensor sg, torch::Tensor gg,
                                       torch::Tensor wu, torch::Tensor su, torch::Tensor gu,
                                       torch::Tensor expert_ids, torch::Tensor token_ids,
                                       torch::Tensor x, int64_t K, int64_t N, int64_t act,
                                       c10::optional<torch::Tensor> xscale) {
    mb_verifier(wg, sg, gg, K, N); mb_verifier(wu, su, gu, K, N); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(wg);
    const int G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    TORCH_CHECK(xc.size(-1) == K, "x : dernière dimension K");
    auto out = torch::empty({G, N}, xc.options().dtype(torch::kFloat));
    const int S = mb_splitk(N / MB_TN, G, K / MB_TK);
    dim3 grid(N / MB_TN, G, S);
    float *part = nullptr; unsigned int *cpt = nullptr;
    if (S > 1) std::tie(part, cpt) = mb_tampons(out, (long)S * 2 * G * N, (long)G * (N / MB_TN));
    const size_t shm = (size_t)(K + 2 * MB_WARPS * MB_TN) * sizeof(float);
    const __nv_bfloat16 *psc = nullptr; int ldsc = 0;
    if (xscale.has_value() && xscale->defined()) {
        const torch::Tensor &sc = *xscale;
        CHECK_CUDA(sc);
        TORCH_CHECK(sc.scalar_type() == torch::kBFloat16, "échelle AWQ : bf16 attendu (table de moe.py)");
        TORCH_CHECK(sc.dim() == 2 && sc.size(1) >= K, "échelle AWQ [E, ≥ K]");
        TORCH_CHECK(sc.stride(1) == 1, "échelle AWQ : lignes contiguës");
        psc = reinterpret_cast<const __nv_bfloat16 *>(sc.data_ptr());
        ldsc = (int)sc.stride(0);
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    #define MB_L(XT, PX) nvfp4_gemv_marlin_kernel<XT, 2><<<grid, MB_WARPS * WARP, shm, stream>>>( \
        reinterpret_cast<const uint4 *>(wg.data_ptr()), static_cast<const unsigned char *>(sg.data_ptr()), gg.data_ptr<float>(), \
        reinterpret_cast<const uint4 *>(wu.data_ptr()), static_cast<const unsigned char *>(su.data_ptr()), gu.data_ptr<float>(), \
        expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), (int)N, (int)K, (int)act, \
        part, cpt, psc, ldsc, (int)N)
    if (bf) { MB_L(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { MB_L(float, xc.data_ptr<float>()); }
    #undef MB_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Pièce 82 : le même GEMV gate·up, lisant gate et up dans la pile w13 (disposition unique : les piles gate et up
// sont rendues). w13 [E, K/16, 4N] int32 = sur chaque ligne de tuiles, les N/64 tuiles de gate puis celles d up ;
// s13 [E, K/16, 2N] de même. Le noyau reçoit la largeur stockée 2N (ldn) et deux bases décalées : mêmes
// octets, mêmes expressions, mêmes échelles globales par projection qu avec deux piles — sortie AU BIT du
// chemin séparé (tests/test_moe_w13.py).
torch::Tensor nvfp4_gemv_marlin_w13(torch::Tensor w13, torch::Tensor s13, torch::Tensor gg, torch::Tensor gu,
                                    torch::Tensor expert_ids, torch::Tensor token_ids,
                                    torch::Tensor x, int64_t K, int64_t N, int64_t act,
                                    c10::optional<torch::Tensor> xscale) {
    mb_verifier(w13, s13, gg, K, 2 * N); CHECK_CONTIG(gu); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(w13);
    TORCH_CHECK(gu.scalar_type() == torch::kFloat && gu.numel() == gg.numel(), "w13 : échelle globale d up fp32 [E]");
    const int G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    TORCH_CHECK(xc.size(-1) == K, "x : dernière dimension K");
    auto out = torch::empty({G, N}, xc.options().dtype(torch::kFloat));
    const int S = mb_splitk(N / MB_TN, G, K / MB_TK);
    dim3 grid(N / MB_TN, G, S);
    float *part = nullptr; unsigned int *cpt = nullptr;
    if (S > 1) std::tie(part, cpt) = mb_tampons(out, (long)S * 2 * G * N, (long)G * (N / MB_TN));
    const size_t shm = (size_t)(K + 2 * MB_WARPS * MB_TN) * sizeof(float);
    const __nv_bfloat16 *psc = nullptr; int ldsc = 0;
    if (xscale.has_value() && xscale->defined()) {
        const torch::Tensor &sc = *xscale;
        CHECK_CUDA(sc);
        TORCH_CHECK(sc.scalar_type() == torch::kBFloat16, "échelle AWQ : bf16 attendu (table de moe.py)");
        TORCH_CHECK(sc.dim() == 2 && sc.size(1) >= K, "échelle AWQ [E, ≥ K]");
        TORCH_CHECK(sc.stride(1) == 1, "échelle AWQ : lignes contiguës");
        psc = reinterpret_cast<const __nv_bfloat16 *>(sc.data_ptr());
        ldsc = (int)sc.stride(0);
    }
    const uint4 *qg = reinterpret_cast<const uint4 *>(w13.data_ptr());
    const uint4 *qu = qg + (long)(N / MB_TN) * (MB_TK * MB_TN / 32);   // tuiles d up : après les N/64 de gate
    const unsigned char *sgp = static_cast<const unsigned char *>(s13.data_ptr());
    const unsigned char *sup = sgp + N;                                   // échelles d up : après les N de gate
    auto stream = at::cuda::getCurrentCUDAStream();
    #define MB_L(XT, PX) nvfp4_gemv_marlin_kernel<XT, 2><<<grid, MB_WARPS * WARP, shm, stream>>>( \
        qg, sgp, gg.data_ptr<float>(), qu, sup, gu.data_ptr<float>(), \
        expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), (int)N, (int)K, (int)act, \
        part, cpt, psc, ldsc, (int)(2 * N))
    if (bf) { MB_L(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { MB_L(float, xc.data_ptr<float>()); }
    #undef MB_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---------------------------------------------------------------------------
// GEMV groupée « x en registres » (18/09, sage-gemv-experts-dernier-geste-18-09) :
// la passe ncu (Laure, verdict-ncu-gemv-experts-rpw-18-09) montre la mémoire
// PARTAGÉE dominante — x relu par flottant (32 LDS.32 par uint4 de poids, 128 o
// de shared pour 16 o de poids), mio_throttle + short_scoreboard = 60 % des
// décrochages à rpw=4, issue_active 56 %. Ici chaque voie charge UNE fois sa
// tranche de x (les 32 flottants en face de chacun de ses uint4) en registres
// par LDS.128 — étage à décalage de 4 flottants par 32 (XSH_PAS4 = 36 :
// alignement 16 o gardé, bancs distincts par quart de warp) — puis balaie
// ses RPW lignes sans plus toucher la shared. Même arithmétique, MÊMES
// EXPRESSIONS que nvfp4_row_dot_warp (a0/a1/c0/c1, (…)·gscale, réduction par
// shuffles) : sortie identique au bit (juge : tests/test_gemv_experts_v2.py).
// Réservé à K ≤ 2 048 (≤ 2 uint4 par voie : 64 flottants de registres) ; au-delà,
// le chemin partagé. ACVRAM_GROUPED_XREG=1 pour l'emprunter (0 : témoin).
constexpr int XSH_GRP4 = 4;                 // flottants de décalage par groupe de 32
constexpr int XSH_PAS4 = XSH_GRP + XSH_GRP4;
__host__ __device__ __forceinline__ int xsh_taille4(int K) { return K + (K >> 5) * XSH_GRP4; }

template <typename XT>
__device__ __forceinline__ void charger_x_sh4(const XT *__restrict__ xn, float *xs_sh, int K) {
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const int d = i + (i >> 5) * XSH_GRP4;
        if constexpr (sizeof(XT) == 4) xs_sh[d] = xn[i];
        else xs_sh[d] = __bfloat162float(xn[i]);
    }
    __syncthreads();
}

// Tranche de x d'une voie : NP uint4 (i = lane + 32·t), 32 flottants chacun.
template <int NP>
struct XReg { float4 v[NP][8]; };

template <int NP>
__device__ __forceinline__ void charger_xreg(const float *__restrict__ xs_sh, int npairs, int lane, XReg<NP> &xr) {
    #pragma unroll
    for (int t = 0; t < NP; ++t) {
        const int i = lane + t * WARP;
        const float4 *src = reinterpret_cast<const float4 *>(xs_sh + (long)i * XSH_PAS4);
        #pragma unroll
        for (int q = 0; q < 8; ++q) xr.v[t][q] = (i < npairs) ? src[q] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

__device__ __forceinline__ float xr_at(const float4 *v, int k) {   // k < 32, constant après déroulage
    const float4 f = v[k >> 2];
    return (k & 3) == 0 ? f.x : (k & 3) == 1 ? f.y : (k & 3) == 2 ? f.z : f.w;
}

template <int NP>
__device__ __forceinline__ float nvfp4_row_dot_warp_xreg(
        const uint4 *__restrict__ wrow, const unsigned char *__restrict__ brow,
        float gscale, const XReg<NP> &xr, int npairs, int lane) {
    float acc = 0.f;
    int i = lane;
    #pragma unroll
    for (int t = 0; t + 1 < NP; t += 2) {
        if (i + WARP < npairs) {
            const uint4 pA = wrow[i], pB = wrow[i + WARP];
            const unsigned char bA0 = brow[2 * i], bA1 = brow[2 * i + 1];
            const unsigned char bB0 = brow[2 * (i + WARP)], bB1 = brow[2 * (i + WARP) + 1];
            const unsigned int wA[4] = {pA.x, pA.y, pA.z, pA.w};
            const unsigned int wB[4] = {pB.x, pB.y, pB.z, pB.w};
            const float4 *xA = xr.v[t];
            const float4 *xB = xr.v[t + 1];
            float a0 = 0.f, a1 = 0.f, c0 = 0.f, c1 = 0.f;
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const float2 v0 = e2m1_pair((wA[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
                a0 += v0.x * xr_at(xA, 2 * b) + v0.y * xr_at(xA, 2 * b + 1);
                const float2 v1 = e2m1_pair((wA[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
                a1 += v1.x * xr_at(xA, WEIGHTS_PER_LOAD + 2 * b) + v1.y * xr_at(xA, WEIGHTS_PER_LOAD + 2 * b + 1);
                const float2 u0 = e2m1_pair((wB[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
                c0 += u0.x * xr_at(xB, 2 * b) + u0.y * xr_at(xB, 2 * b + 1);
                const float2 u1 = e2m1_pair((wB[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
                c1 += u1.x * xr_at(xB, WEIGHTS_PER_LOAD + 2 * b) + u1.y * xr_at(xB, WEIGHTS_PER_LOAD + 2 * b + 1);
            }
            acc += (a0 * e4m3_to_float(bA0) + a1 * e4m3_to_float(bA1)
                  + c0 * e4m3_to_float(bB0) + c1 * e4m3_to_float(bB1)) * gscale;
            i += 2 * WARP;
        }
    }
    #pragma unroll
    for (int t = 0; t < NP; ++t) {
        const int ii = lane + t * WARP;
        // la même condition de reste que nvfp4_row_dot_warp : les indices
        // qu'une paire n'a pas pris (i + WARP ≥ npairs) et ii < npairs
        if (ii >= i && ii < npairs) {
            const uint4 p4 = wrow[ii];
            const float s0 = e4m3_to_float(brow[2 * ii]) * gscale;
            const float s1 = e4m3_to_float(brow[2 * ii + 1]) * gscale;
            const float4 *xp = xr.v[t];
            const unsigned int words[4] = {p4.x, p4.y, p4.z, p4.w};
            float part0 = 0.f, part1 = 0.f;
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const float2 v0 = e2m1_pair((words[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
                part0 += v0.x * xr_at(xp, 2 * b) + v0.y * xr_at(xp, 2 * b + 1);
                const float2 v1 = e2m1_pair((words[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
                part1 += v1.x * xr_at(xp, WEIGHTS_PER_LOAD + 2 * b) + v1.y * xr_at(xp, WEIGHTS_PER_LOAD + 2 * b + 1);
            }
            acc += part0 * s0 + part1 * s1;
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

template <typename XT, int RPW, int NP>
__global__ void nvfp4_gemv_grouped_xreg_kernel(
    const unsigned char *__restrict__ qw, const unsigned char *__restrict__ bscale,
    const float *__restrict__ gscales, const int *__restrict__ expert_ids,
    const int *__restrict__ token_ids, const XT *__restrict__ x,
    float *__restrict__ y, int M, int K) {
    extern __shared__ float xs_sh[];
    const int g = blockIdx.y, e = expert_ids[g];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (e < 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)g * M + row] = 0.f;
        }
        return;
    }
    charger_x_sh4<XT>(x + (long)token_ids[g] * K, xs_sh, K);
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD, npairs = nloads >> 1;
    const float gscale = gscales[e];
    XReg<NP> xr;
    charger_xreg<NP>(xs_sh, npairs, lane, xr);
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        const float acc = nvfp4_row_dot_warp_xreg<NP>(
            reinterpret_cast<const uint4 *>(qw + ((long)e * M + row) * half_k),
            bscale + ((long)e * M + row) * nloads, gscale, xr, npairs, lane);
        if (lane == 0) y[(long)g * M + row] = acc;
    }
}

template <typename XT, int RPW, int NP>
__global__ void nvfp4_gemv_grouped_gateup_xreg_kernel(
    const unsigned char *__restrict__ qg, const unsigned char *__restrict__ bg,
    const float *__restrict__ gsg,
    const unsigned char *__restrict__ qu, const unsigned char *__restrict__ bu,
    const float *__restrict__ gsu,
    const int *__restrict__ expert_ids, const int *__restrict__ token_ids,
    const XT *__restrict__ x, float *__restrict__ y, int M, int K, int act) {
    extern __shared__ float xs_sh[];
    const int g = blockIdx.y, e = expert_ids[g];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (e < 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)g * M + row] = 0.f;
        }
        return;
    }
    charger_x_sh4<XT>(x + (long)token_ids[g] * K, xs_sh, K);
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD, npairs = nloads >> 1;
    XReg<NP> xr;
    charger_xreg<NP>(xs_sh, npairs, lane, xr);
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        const long off = (long)e * M + row;
        const float ag = nvfp4_row_dot_warp_xreg<NP>(
            reinterpret_cast<const uint4 *>(qg + off * half_k), bg + off * nloads, gsg[e], xr, npairs, lane);
        const float au = nvfp4_row_dot_warp_xreg<NP>(
            reinterpret_cast<const uint4 *>(qu + off * half_k), bu + off * nloads, gsu[e], xr, npairs, lane);
        if (lane == 0) y[(long)g * M + row] = acv_act(ag, act) * au;
    }
}

// ACVRAM_GROUPED_XREG : 0 (témoin : x relu en shared) | down (DÉFAUT : x en registres
// sur la projection down seule — sage-gemv-experts-clos-18-09 : gateup à 96
// registres tombait à 2 blocs/SM, +19 % ; down 56 registres, −12 %) | 1 (les
// deux, témoin réfuté). Rend 0, 1 (down seul) ou 2 (les deux).
// Défaut « down » depuis verdict-gemv-experts-xreg-down-18-09 (Laure, ABAB
// Coder b=12 : × 0,944 nu, J/jeton × 0,954, bit-exact ; cellule 1 307 t/s nu).
static int xreg_mode() {
    static const int v = [] {
        const char *e = std::getenv("ACVRAM_GROUPED_XREG");
        if (!e || !*e) return 1;                 // défaut : down
        if (std::string(e) == "down") return 1;
        return atoi(e) != 0 ? 2 : 0;
    }();
    return v;
}
static bool xreg_demande() { return xreg_mode() >= 1; }            // down
static bool xreg_demande_gateup() { return xreg_mode() >= 2; }     // gate/up aussi
static bool xreg_possible(int64_t K) { return K % 32 == 0 && K <= 2048; }   // ≤ 2 uint4 par voie

torch::Tensor nvfp4_gemv_grouped_xreg(torch::Tensor qw, torch::Tensor bscale, torch::Tensor gscales,
                                      torch::Tensor expert_ids, torch::Tensor token_ids,
                                      torch::Tensor x, int64_t K) {
    CHECK_CUDA(qw); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qw);
    CHECK_CONTIG(qw); CHECK_CONTIG(bscale); CHECK_CONTIG(x);
    TORCH_CHECK(xreg_possible(K), "xreg : K multiple de 32 et <= 2048");
    const int M = qw.size(1), G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
    const int np = (K / 32 + WARP - 1) / WARP;            // uint4 par voie : 1 (K ≤ 1024) ou 2
    dim3 grid((M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
    const size_t shm = (size_t)xsh_taille4((int)K) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define XR_L(XT, R, P, PX) nvfp4_gemv_grouped_xreg_kernel<XT, R, P><<<grid, GW_WARPS * WARP, shm, stream>>>( \
        qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(), gscales.data_ptr<float>(), \
        expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K)
    #define XR_R(XT, P, PX) do { if (rpw == 1) XR_L(XT, 1, P, PX); else if (rpw == 2) XR_L(XT, 2, P, PX); else XR_L(XT, 4, P, PX); } while (0)
    #define XR_T(XT, PX) do { if (np == 1) XR_R(XT, 1, PX); else XR_R(XT, 2, PX); } while (0)
    if (bf) { XR_T(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { XR_T(float, xc.data_ptr<float>()); }
    #undef XR_T
    #undef XR_R
    #undef XR_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nvfp4_gemv_grouped_gateup_xreg(
        torch::Tensor qg, torch::Tensor bg, torch::Tensor gsg,
        torch::Tensor qu, torch::Tensor bu, torch::Tensor gsu,
        torch::Tensor expert_ids, torch::Tensor token_ids,
        torch::Tensor x, int64_t K, int64_t act) {
    CHECK_CUDA(qg); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qg);
    CHECK_CONTIG(qg); CHECK_CONTIG(qu); CHECK_CONTIG(bg); CHECK_CONTIG(bu);
    TORCH_CHECK(xreg_possible(K), "xreg : K multiple de 32 et <= 2048");
    const int M = qg.size(1), G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
    const int np = (K / 32 + WARP - 1) / WARP;
    dim3 grid((M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
    const size_t shm = (size_t)xsh_taille4((int)K) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GX_L(XT, R, P, PX) nvfp4_gemv_grouped_gateup_xreg_kernel<XT, R, P><<<grid, GW_WARPS * WARP, shm, stream>>>( \
        qg.data_ptr<unsigned char>(), bg.data_ptr<unsigned char>(), gsg.data_ptr<float>(), \
        qu.data_ptr<unsigned char>(), bu.data_ptr<unsigned char>(), gsu.data_ptr<float>(), \
        expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K, (int)act)
    #define GX_R(XT, P, PX) do { if (rpw == 1) GX_L(XT, 1, P, PX); else if (rpw == 2) GX_L(XT, 2, P, PX); else GX_L(XT, 4, P, PX); } while (0)
    #define GX_T(XT, PX) do { if (np == 1) GX_R(XT, 1, PX); else GX_R(XT, 2, PX); } while (0)
    if (bf) { GX_T(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { GX_T(float, xc.data_ptr<float>()); }
    #undef GX_T
    #undef GX_R
    #undef GX_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---------------------------------------------------------------------------
// GEMV groupée v2 (18/09, sage-lecture-profils-coder-17-09 § 2 : « GEMV
// groupée à ≥ 85 % de bande ») : les paires (expert, jeton) arrivent TRIÉES
// par expert (`ordre` = argsort(eid) côté hôte) ; un bloc « meneur » sert
// jusqu'à TPB jetons du même expert d'un coup — les poids de l'expert sont
// lus UNE fois par pas pour ces jetons (v1 : une fois par paire, 1,39 × sur
// Coder b=12 : 96 paires pour 69 experts, relectures hors L2 pour partie), et
// l'activation n'est étagée qu'une fois par bloc pour ses TPB jetons. Même
// arithmétique que v1, dans le même ordre (nvfp4_row_dot_warp dupliquée
// par jeton) : sortie identique au bit — le juge (tests/test_gemv_experts_v2.py).
// Sortie écrite à la place d'ORIGINE de chaque paire (y[ordre[g]]) : les
// consommateurs (act, down, pondération) ne changent pas.
template <int TPB>
__device__ __forceinline__ void nvfp4_row_dot_warp_multi(
        const uint4 *__restrict__ wrow, const unsigned char *__restrict__ brow,
        float gscale, const float *__restrict__ xs_sh, int xsz, int npairs, int lane, int n,
        float *__restrict__ acc) {
    #pragma unroll
    for (int j = 0; j < TPB; ++j) acc[j] = 0.f;
    int i = lane;
    for (; i + WARP < npairs; i += 2 * WARP) {
        const uint4 pA = wrow[i], pB = wrow[i + WARP];
        const unsigned char bA0 = brow[2 * i], bA1 = brow[2 * i + 1];
        const unsigned char bB0 = brow[2 * (i + WARP)], bB1 = brow[2 * (i + WARP) + 1];
        const unsigned int wA[4] = {pA.x, pA.y, pA.z, pA.w};
        const unsigned int wB[4] = {pB.x, pB.y, pB.z, pB.w};
        const float fA0 = e4m3_to_float(bA0), fA1 = e4m3_to_float(bA1);
        const float fB0 = e4m3_to_float(bB0), fB1 = e4m3_to_float(bB1);
        #pragma unroll
        for (int j = 0; j < TPB; ++j) {
            if (j < n) {
                const float *xA = xs_sh + (long)j * xsz + (long)i * XSH_PAS;
                const float *xB = xs_sh + (long)j * xsz + (long)(i + WARP) * XSH_PAS;
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
                acc[j] += (a0 * fA0 + a1 * fA1 + c0 * fB0 + c1 * fB1) * gscale;
            }
        }
    }
    for (; i < npairs; i += WARP) {
        const uint4 p4 = wrow[i];
        const float s0 = e4m3_to_float(brow[2 * i]) * gscale;
        const float s1 = e4m3_to_float(brow[2 * i + 1]) * gscale;
        const unsigned int words[4] = {p4.x, p4.y, p4.z, p4.w};
        #pragma unroll
        for (int j = 0; j < TPB; ++j) {
            if (j < n) {
                const float *xp = xs_sh + (long)j * xsz + (long)i * XSH_PAS;
                float part0 = 0.f, part1 = 0.f;
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const float2 v0 = e2m1_pair((words[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
                    part0 += v0.x * xp[2 * b] + v0.y * xp[2 * b + 1];
                    const float2 v1 = e2m1_pair((words[2 + (b >> 2)] >> ((b & 3) * 8)) & 0xFFu);
                    part1 += v1.x * xp[WEIGHTS_PER_LOAD + 2 * b] + v1.y * xp[WEIGHTS_PER_LOAD + 2 * b + 1];
                }
                acc[j] += part0 * s0 + part1 * s1;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < TPB; ++j)
        for (int o = 16; o > 0; o >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o);
}

// Meneur du sous-segment [g, g+n) : e ≥ 0, g au début d'un segment d'expert
// ou à un multiple de TPB depuis ce début ; n ≤ TPB paires. Rend n (0 : pas
// meneur). Calcul par le fil 0, diffusé par la mémoire partagée.
template <int TPB>
__device__ __forceinline__ int gv2_meneur(const int *__restrict__ eid_s, int G, int g, int e, int *s_n) {
    if (threadIdx.x == 0) {
        int debut = g;
        while (debut > 0 && eid_s[debut - 1] == e) --debut;
        int n = 0;
        if ((g - debut) % TPB == 0) {
            n = 1;
            while (n < TPB && g + n < G && eid_s[g + n] == e) ++n;
        }
        *s_n = n;
    }
    __syncthreads();
    return *s_n;
}

template <typename XT, int TPB>
__device__ __forceinline__ void gv2_charger_x(const XT *__restrict__ x, const int *__restrict__ tok_s,
                                              int g, int n, float *xs_sh, int K, int xsz) {
    for (int j = 0; j < n; ++j) {
        const XT *xn = x + (long)tok_s[g + j] * K;
        float *dst = xs_sh + (long)j * xsz;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            const int d = i + (i >> 5);
            if constexpr (sizeof(XT) == 4) dst[d] = xn[i];
            else dst[d] = __bfloat162float(xn[i]);
        }
    }
    __syncthreads();
}

template <typename XT, int TPB, int RPW>
__global__ void nvfp4_gemv_grouped_v2_kernel(
    const unsigned char *__restrict__ qw, const unsigned char *__restrict__ bscale,
    const float *__restrict__ gscales, const int *__restrict__ eid_s,
    const int *__restrict__ tok_s, const int *__restrict__ ordre,
    const XT *__restrict__ x, float *__restrict__ y, int M, int K, int G) {
    extern __shared__ float xs_sh[];
    __shared__ int s_n;
    const int g = blockIdx.y, e = eid_s[g];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (e < 0) {                     // créneau fantôme : zéro, aucun poids lu
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)ordre[g] * M + row] = 0.f;
        }
        return;
    }
    const int n = gv2_meneur<TPB>(eid_s, G, g, e, &s_n);
    if (n == 0) return;
    const int xsz = xsh_taille(K);
    gv2_charger_x<XT, TPB>(x, tok_s, g, n, xs_sh, K, xsz);
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const float gscale = gscales[e];
    float acc[TPB];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        nvfp4_row_dot_warp_multi<TPB>(
            reinterpret_cast<const uint4 *>(qw + ((long)e * M + row) * half_k),
            bscale + ((long)e * M + row) * nloads, gscale, xs_sh, xsz, nloads >> 1, lane, n, acc);
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < TPB; ++j)
                if (j < n) y[(long)ordre[g + j] * M + row] = acc[j];
        }
    }
}

template <typename XT, int TPB, int RPW>
__global__ void nvfp4_gemv_grouped_gateup_v2_kernel(
    const unsigned char *__restrict__ qg, const unsigned char *__restrict__ bg,
    const float *__restrict__ gsg,
    const unsigned char *__restrict__ qu, const unsigned char *__restrict__ bu,
    const float *__restrict__ gsu,
    const int *__restrict__ eid_s, const int *__restrict__ tok_s, const int *__restrict__ ordre,
    const XT *__restrict__ x, float *__restrict__ y, int M, int K, int G, int act) {
    extern __shared__ float xs_sh[];
    __shared__ int s_n;
    const int g = blockIdx.y, e = eid_s[g];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (e < 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)ordre[g] * M + row] = 0.f;
        }
        return;
    }
    const int n = gv2_meneur<TPB>(eid_s, G, g, e, &s_n);
    if (n == 0) return;
    const int xsz = xsh_taille(K);
    gv2_charger_x<XT, TPB>(x, tok_s, g, n, xs_sh, K, xsz);
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    float ag[TPB], au[TPB];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        const long off = (long)e * M + row;
        nvfp4_row_dot_warp_multi<TPB>(reinterpret_cast<const uint4 *>(qg + off * half_k), bg + off * nloads,
                                      gsg[e], xs_sh, xsz, nloads >> 1, lane, n, ag);
        nvfp4_row_dot_warp_multi<TPB>(reinterpret_cast<const uint4 *>(qu + off * half_k), bu + off * nloads,
                                      gsu[e], xs_sh, xsz, nloads >> 1, lane, n, au);
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < TPB; ++j)
                if (j < n) y[(long)ordre[g + j] * M + row] = acv_act(ag[j], act) * au[j];
        }
    }
}

// TPB : le plus grand de {4, 2, 1} dont l'étage d'activations tient en 48 Kio.
static int gv2_tpb(int64_t K) {
    const size_t un = (size_t)(K + K / 32) * sizeof(float);
    if (4 * un <= 48 * 1024) return 4;
    if (2 * un <= 48 * 1024) return 2;
    return 1;
}

torch::Tensor nvfp4_gemv_grouped_v2(torch::Tensor qw, torch::Tensor bscale, torch::Tensor gscales,
                                    torch::Tensor eid_s, torch::Tensor tok_s, torch::Tensor ordre,
                                    torch::Tensor x, int64_t K) {
    CHECK_CUDA(qw); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qw);
    CHECK_CONTIG(qw); CHECK_CONTIG(bscale); CHECK_CONTIG(x);
    TORCH_CHECK(K % 32 == 0 && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024, "v2 : K multiple de 32 et <= 11904");
    const int M = qw.size(1), G = eid_s.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
    const int tpb = gv2_tpb(K);
    dim3 grid((M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
    const size_t shm = (size_t)tpb * (K + K / 32) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GV2_L(XT, T, R, PX) nvfp4_gemv_grouped_v2_kernel<XT, T, R><<<grid, GW_WARPS * WARP, shm, stream>>>( \
        qw.data_ptr<unsigned char>(), bscale.data_ptr<unsigned char>(), gscales.data_ptr<float>(), \
        eid_s.data_ptr<int>(), tok_s.data_ptr<int>(), ordre.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K, G)
    #define GV2_R(XT, T, PX) do { if (rpw == 1) GV2_L(XT, T, 1, PX); else if (rpw == 2) GV2_L(XT, T, 2, PX); else GV2_L(XT, T, 4, PX); } while (0)
    #define GV2_T(XT, PX) do { if (tpb == 4) GV2_R(XT, 4, PX); else if (tpb == 2) GV2_R(XT, 2, PX); else GV2_R(XT, 1, PX); } while (0)
    if (bf) { GV2_T(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { GV2_T(float, xc.data_ptr<float>()); }
    #undef GV2_T
    #undef GV2_R
    #undef GV2_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nvfp4_gemv_grouped_gateup_v2(
        torch::Tensor qg, torch::Tensor bg, torch::Tensor gsg,
        torch::Tensor qu, torch::Tensor bu, torch::Tensor gsu,
        torch::Tensor eid_s, torch::Tensor tok_s, torch::Tensor ordre,
        torch::Tensor x, int64_t K, int64_t act) {
    CHECK_CUDA(qg); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qg);
    CHECK_CONTIG(qg); CHECK_CONTIG(qu); CHECK_CONTIG(bg); CHECK_CONTIG(bu);
    TORCH_CHECK(K % 32 == 0 && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024, "v2 : K multiple de 32 et <= 11904");
    const int M = qg.size(1), G = eid_s.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
    const int tpb = gv2_tpb(K);
    dim3 grid((M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
    const size_t shm = (size_t)tpb * (K + K / 32) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GU2_L(XT, T, R, PX) nvfp4_gemv_grouped_gateup_v2_kernel<XT, T, R><<<grid, GW_WARPS * WARP, shm, stream>>>( \
        qg.data_ptr<unsigned char>(), bg.data_ptr<unsigned char>(), gsg.data_ptr<float>(), \
        qu.data_ptr<unsigned char>(), bu.data_ptr<unsigned char>(), gsu.data_ptr<float>(), \
        eid_s.data_ptr<int>(), tok_s.data_ptr<int>(), ordre.data_ptr<int>(), PX, out.data_ptr<float>(), M, (int)K, G, (int)act)
    #define GU2_R(XT, T, PX) do { if (rpw == 1) GU2_L(XT, T, 1, PX); else if (rpw == 2) GU2_L(XT, T, 2, PX); else GU2_L(XT, T, 4, PX); } while (0)
    #define GU2_T(XT, PX) do { if (tpb == 4) GU2_R(XT, 4, PX); else if (tpb == 2) GU2_R(XT, 2, PX); else GU2_R(XT, 1, PX); } while (0)
    if (bf) { GU2_T(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { GU2_T(float, xc.data_ptr<float>()); }
    #undef GU2_T
    #undef GU2_R
    #undef GU2_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Pendant décodage de nvfp4_gemm_grouped_mma (patron identique, table.hpp
// bead pds) : chaque expert lu par ADRESSE (`table_*[e]`), résident ou
// épinglé zéro-copie, jamais par une pile contiguë — une couche au
// placement hétérogène (certains experts froids) n'a plus besoin de
// désactiver le chemin groupé (`_try_build_stacks`, model.py).
template <typename XT, int RPW>
__global__ void nvfp4_gemv_grouped_gateup_table_kernel(
    const int64_t *__restrict__ table_qg, const int64_t *__restrict__ table_bg,
    const float *__restrict__ gsg,
    const int64_t *__restrict__ table_qu, const int64_t *__restrict__ table_bu,
    const float *__restrict__ gsu,
    const int *__restrict__ expert_ids, const int *__restrict__ token_ids,
    const XT *__restrict__ x, float *__restrict__ y, int M, int K, int act) {
    extern __shared__ float xs_sh[];
    const int g = blockIdx.y, e = expert_ids[g];
    charger_x_sh<XT>(x + (long)token_ids[g] * K, xs_sh, K);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // Creneau fantome (e < 0, bead pds 14/09) : AVANT tout dereferencement
    // de table -- table_qg[-1] serait un acces hors bornes, pas seulement
    // une lecture de poids gaspillee. Sortie a zero, aucune lecture hote.
    if (e < 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)g * M + row] = 0.f;
        }
        return;
    }
    const unsigned char *qg_e = reinterpret_cast<const unsigned char *>(table_qg[e]);
    const unsigned char *bg_e = reinterpret_cast<const unsigned char *>(table_bg[e]);
    const unsigned char *qu_e = reinterpret_cast<const unsigned char *>(table_qu[e]);
    const unsigned char *bu_e = reinterpret_cast<const unsigned char *>(table_bu[e]);
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        const float ag = nvfp4_row_dot_warp(
            reinterpret_cast<const uint4 *>(qg_e + (long)row * half_k),
            bg_e + (long)row * nloads, gsg[e], xs_sh, nloads >> 1, lane);
        const float au = nvfp4_row_dot_warp(
            reinterpret_cast<const uint4 *>(qu_e + (long)row * half_k),
            bu_e + (long)row * nloads, gsu[e], xs_sh, nloads >> 1, lane);
        if (lane == 0) y[(long)g * M + row] = acv_act(ag, act) * au;
    }
}

torch::Tensor nvfp4_gemv_grouped_gateup_table(
        torch::Tensor table_qg, torch::Tensor table_bg, torch::Tensor gsg,
        torch::Tensor table_qu, torch::Tensor table_bu, torch::Tensor gsu,
        torch::Tensor expert_ids, torch::Tensor token_ids,
        torch::Tensor x, int64_t M, int64_t K, int64_t act) {
    CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(x);
    TORCH_CHECK(K % 32 == 0 && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024,
                "gate-up fusionne : K multiple de 32 et <= 11904");
    const int G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
    dim3 grid(((int)M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
    const size_t shm = (size_t)(K + K / 32) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GUT_LAUNCH(XT, PX) do { if (rpw == 1) GUT_L(XT, 1, PX); else if (rpw == 2) GUT_L(XT, 2, PX); else GUT_L(XT, 4, PX); } while (0)
    #define GUT_L(XT, R, PX) nvfp4_gemv_grouped_gateup_table_kernel<XT, R><<<grid, GW_WARPS * WARP, shm, stream>>>( \
        table_qg.data_ptr<int64_t>(), table_bg.data_ptr<int64_t>(), gsg.data_ptr<float>(), \
        table_qu.data_ptr<int64_t>(), table_bu.data_ptr<int64_t>(), gsu.data_ptr<float>(), \
        expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), (int)M, (int)K, (int)act)
    if (bf) { GUT_LAUNCH(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { GUT_LAUNCH(float, xc.data_ptr<float>()); }
    #undef GUT_LAUNCH
    #undef GUT_L
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
    // Tuile vide (grille FIXE du décodage sous graphes, _tuiles(cnt, bt, t_max) :
    // n=0 au-delà du compte réel, t0 pouvant dépasser la fin de xq) : rien à
    // lire ni à écrire. Sans ce retour, min(r, nt-1) = -1 et les chargements
    // partent à t0-1 : accès mémoire illégal mesuré le 14/09 sur Coder-30B.
    if (nt <= 0) return;
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


// --------------------------------------------------------------------------
// GEMM groupée NVFP4 sur la MMA FP4 native de sm_120 (W4A4).
//
// La GEMM groupée ci-dessus déquantifie les poids en bf16 dans la mémoire
// partagée avant de les consommer en wmma bf16 : c'est un W4A16 logiciel, et
// la conversion est ce qui la fait plafonner au-delà de 48 jetons par expert
// (revue/banc-prefill-moe-12-09.md). Blackwell grand public possède une MMA
// à échelles par bloc — mma.sync m16n8k64 kind::mxf4nvf4 — qui consomme
// directement E2M1 × UE4M3 par bloc de 16, exactement le format de nos
// poids (revue/mma-fp4-native-sm120.md : 128/128 bit-identique, x7,9 en
// débit contre bf16). Le prix : les activations doivent elles aussi être en
// E2M1 par bloc de 16 (W4A4). Elles sont quantifiées à la volée par
// nvfp4_quant_act ; la référence Python du test refait exactement le même
// calcul (amax/6 → E4M3 au plus proche, puis E2M1 au plus proche ; les
// égalités vont vers la magnitude supérieure, c'est ce que fait
// cvt.rn.satfinite.e2m1x2 — mesuré, pas supposé).
//
// Le kind mxf8f6f4 (activations en 8 bits) n'est PAS un repli possible avec
// nos poids : ses échelles sont UE8M0 par bloc de 32, pas E4M3 par bloc de
// 16. Un W4A8 exigerait de reconvertir les poids.
//
// L'asm n'existe que sous une cible famille (sm_120f / sm_120a) ; le repli
// PTX générique compile un stub qui ne doit jamais être atteint.
// --------------------------------------------------------------------------
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define ACVRAM_MMA_FP4 1
#endif

constexpr int GM_BM = 64;        // lignes de sortie par bloc (4 warps x 16)
constexpr int GM_KB = 64;        // profondeur d'une MMA

__device__ __forceinline__ void mma_mxf4nvf4(float d[4], const unsigned a[4], const unsigned b[2],
                                             unsigned sfa, unsigned sfb) {
#ifdef ACVRAM_MMA_FP4
    const unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
#endif
}

// Activations bf16 [G, K] -> E2M1 [G, K/2] + échelles UE4M3 [G, K/16]
// + échelle globale fp32 PAR LIGNE grow [G] (sage-glm-pile-correctif-16-09
// § 7). Un CTA par ligne : la ligne en mémoire partagée, amax de ligne par
// réduction, g_r = amax_r / 2688 (= 6 × 448), puis chaque bloc de 16 :
// s = (amax_blk / amax_r) × 448 → E4M3 (le bloc maximal donne 448 exactement :
// saturation impossible par construction), valeurs / (sdec · g_r) → E2M1. Un bloc n'est flushé à
// zéro que si amax_blk < ~2,2e-6 × amax_r (E4M3 sous 2^-10). L'épilogue de
// la GEMM multiplie par grow[r] × gscales[e]. Sans échelle globale (avant),
// amax/6 < 2^-9 mettait le bloc entier à zéro : 23-31 % des blocs de
// silu(g)·u sur GLM (verdict-glm-saturation-16-09) ; un 2^k fixe saturait
// les queues d'une fenêtre réelle (verdict-reppl-alpha-commun-16-09).
//
// Échelle AWQ par expert fusionnée (awq [E, K] bf16, e_sorted [G]) :
// bf16(bf16(v) / s[e][c]) avant tout, la même arithmétique que la ligne de
// moe_route_pack et que scaler.apply de la boucle. Toutes les divisions en
// __fdiv_rn (--use_fast_math rend une division approchée qui bascule les
// égalités E2M1 : 5 codes sur 5 376 mesurés).
// compteurs (facultatif, ACVRAM_QA_COMPTE=1) : [0] blocs non nuls, [1] flushés
// (amax_blk > 0 et échelle 0), [2] saturés (s > 448 avant satfinite ; 0 par
// construction, le compteur est là pour le prouver).
// Rotation de Hadamard (sage-hadamard-16-09, QuaRot) : x' = x·H par bloc de
// `hadamard` colonnes (H_512 bloc-diagonale sur GLM : K = 2 048 = 4 × 512,
// 1 536 = 3 × 512), poids tournés W·H par le convertisseur, produit
// inchangé, canal aberrant étalé sur tout le bloc E2M1. Ici : FWHT fp32 en
// mémoire partagée (log2 étages papillon, chaque sortie = une somme de deux
// valeurs, reproductible au bit), normalisation par 1/√bloc en fp32 (RN),
// puis arrondi bf16 — l'arithmétique de ChannelScaler.apply (hadamard(x)
// puis / s) que la boucle par expert applique ; la référence Python
// (tests/test_gemm_grouped_mma.py quant_act_ref) refait les mêmes étapes.
constexpr int QA_FILS = 256;
__global__ void __launch_bounds__(QA_FILS) nvfp4_quant_act_kernel(
    const __nv_bfloat16 *__restrict__ x, unsigned char *__restrict__ xq,
    unsigned char *__restrict__ xsf, float *__restrict__ grow, int nblk,
    const __nv_bfloat16 *__restrict__ awq, const int *__restrict__ e_sorted,
    long long *__restrict__ compteurs, int hadamard) {
    extern __shared__ float qa_row[];               // [K] fp32 après rotation et division AWQ
    __shared__ float qa_red[QA_FILS / 32];
    const long row = blockIdx.x;
    const int K = nblk * 16;
    const __nv_bfloat16 *src = x + row * (long)K;
    const __nv_bfloat16 *sc = awq ? awq + (long)e_sorted[row] * K : nullptr;
    float amax = 0.f;
    if (hadamard > 0) {
        for (int i = threadIdx.x; i < K; i += QA_FILS) qa_row[i] = __bfloat162float(src[i]);
        __syncthreads();
        for (int h = 1; h < hadamard; h <<= 1) {
            for (int p = threadIdx.x; p < K / 2; p += QA_FILS) {
                const int g = p / h, j = p - g * h;
                const int i0 = g * 2 * h + j, i1 = i0 + h;
                const float a = qa_row[i0], c = qa_row[i1];
                qa_row[i0] = a + c; qa_row[i1] = a - c;
            }
            __syncthreads();
        }
        const float inv = __fdiv_rn(1.f, __fsqrt_rn((float)hadamard)); // 1/√bloc en fp32 RN (pas le sqrtf approché de fast_math)
        for (int i = threadIdx.x; i < K; i += QA_FILS) {
            float t = __bfloat162float(__float2bfloat16(__fmul_rn(qa_row[i], inv)));
            if (sc) t = __bfloat162float(__float2bfloat16(__fdiv_rn(t, __bfloat162float(sc[i]))));
            qa_row[i] = t;
            amax = fmaxf(amax, fabsf(t));
        }
    } else
    for (int i = threadIdx.x; i < K; i += QA_FILS) {
        float t = __bfloat162float(src[i]);
        if (sc) t = __bfloat162float(__float2bfloat16(__fdiv_rn(t, __bfloat162float(sc[i]))));
        qa_row[i] = t;
        amax = fmaxf(amax, fabsf(t));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) qa_red[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float m = threadIdx.x < QA_FILS / 32 ? qa_red[threadIdx.x] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (threadIdx.x == 0) qa_red[0] = m;
    }
    __syncthreads();
    const float amax_r = qa_red[0];
    // 2688 = 6 × 448 ; toutes les opérations en RN explicite (__fdiv_rn,
    // __fmul_rn) : la référence Python (tests/test_gemm_grouped_mma.py
    // quant_act_ref) refait exactement les mêmes — torch divise un tenseur par
    // un scalaire Python en multipliant par l'inverse (1 ulp d'écart sur 20 %
    // des lignes, t-qa 065960a), la référence divise donc par un tenseur.
    const float g = amax_r > 0.f ? __fdiv_rn(amax_r, 2688.f) : 0.f;
    if (threadIdx.x == 0) grow[row] = g;
    for (int blk = threadIdx.x; blk < nblk; blk += QA_FILS) {
        const float *v = qa_row + blk * 16;
        float amax_b = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) amax_b = fmaxf(amax_b, fabsf(v[j]));
        unsigned char sbits = 0;
        float sdec = 0.f;
        if (amax_b > 0.f && g > 0.f) {
            // (amax_blk / amax_r) × 448 : le bloc maximal donne 1 × 448 = 448
            // exactement (E4M3 0xFE), les autres ≤ 448 — pas de clamp, pas
            // d'arrondi qui dépasse (amax_r / (6 g) pouvait rendre 448 + 1 ulp)
            const float sb = __fmul_rn(__fdiv_rn(amax_b, amax_r), 448.f);
            sbits = (unsigned char)__nv_cvt_float_to_fp8(sb, __NV_SATFINITE, __NV_E4M3);
            sdec = e4m3_to_float(sbits);
            if (compteurs) {
                atomicAdd((unsigned long long *)compteurs, 1ull);
                if (sdec == 0.f) atomicAdd((unsigned long long *)compteurs + 1, 1ull);
                if (sb > 448.f) atomicAdd((unsigned long long *)compteurs + 2, 1ull);
            }
        }
        unsigned long long packed = 0ull;
        if (sdec > 0.f) {
            const float d = __fmul_rn(sdec, g);                     // fp32 RN, comme la référence
            #pragma unroll
            for (int j = 0; j < 16; j += 2) {
                const float2 p = make_float2(__fdiv_rn(v[j], d), __fdiv_rn(v[j + 1], d));
                const __nv_fp4x2_storage_t q = __nv_cvt_float2_to_fp4x2(p, __NV_E2M1, cudaRoundNearest);
                packed |= (unsigned long long)(q & 0xFF) << (4 * j);
            }
        }
        *reinterpret_cast<unsigned long long *>(xq + row * (long)nblk * 8 + blk * 8) = packed;
        xsf[row * (long)nblk + blk] = sbits;
    }
}

// Un bloc = 64 lignes de sortie x BT jetons d'une tuile ; 4 warps de 16
// lignes. Fragments A (jetons) et B (poids) chargés depuis la mémoire
// globale directement dans le layout de la MMA (aucune mémoire partagée :
// plus rien à convertir). Double tampon de registres sur K.
template <int BT>
__global__ void __launch_bounds__(128) nvfp4_gemm_grouped_mma_kernel(
    const float *__restrict__ gscales, const float *__restrict__ grow,
    const unsigned char *__restrict__ xq, const unsigned char *__restrict__ xsf,
    const int *__restrict__ tile_e, const int *__restrict__ tile_t0,
    const int *__restrict__ tile_n, const int64_t *__restrict__ table_qw,
    const int64_t *__restrict__ table_bscale,
    __nv_bfloat16 *__restrict__ y, int M, int K) {
#ifdef ACVRAM_MMA_FP4
    constexpr int MF = BT / 16;
    const int tile = blockIdx.y;
    const int e = tile_e[tile], t0 = tile_t0[tile], nt = tile_n[tile];
    // Tuile vide (grille FIXE du décodage sous graphes, _tuiles(cnt, bt, t_max) :
    // n=0 au-delà du compte réel, t0 pouvant dépasser la fin de xq) : rien à
    // lire ni à écrire. Sans ce retour, min(r, nt-1) = -1 et les chargements
    // partent à t0-1 : accès mémoire illégal mesuré le 14/09 sur Coder-30B.
    if (nt <= 0) return;
    // les piles de l'expert : adresses octet fournies par la table (contrat
    // bead pds : résident = data_ptr() de sa tranche, froid = pointeur device
    // zéro-copie d'un tampon épinglé), jamais recalculées d'après e
    const unsigned char *qw_e = reinterpret_cast<const unsigned char *>(table_qw[e]);
    const unsigned char *bs_e = reinterpret_cast<const unsigned char *>(table_bscale[e]);
    const int row0 = blockIdx.x * GM_BM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const long half_k = (long)K >> 1;
    const int nblk = K >> 4;

    // lignes de poids de ce warp : 8*nf + g, valides si < M
    long woff[2]; unsigned wok[2];
    #pragma unroll
    for (int nf = 0; nf < 2; ++nf) {
        const int r = row0 + warp * 16 + nf * 8 + g;
        wok[nf] = r < M;
        woff[nf] = min(r, M - 1);
    }
    // lignes de jetons : 16*mf + g (+8) ; échelles : ligne (lane>>2) + 8*(lane&1)
    long toff[MF][2]; unsigned tok_ok[MF][2]; long soff[MF]; unsigned sok[MF];
    #pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j = mf * 16 + g + 8 * h;
            tok_ok[mf][h] = j < nt;
            toff[mf][h] = (long)(t0 + min(j, nt - 1));
        }
        const int js = mf * 16 + (lane >> 2) + 8 * (lane & 1);
        sok[mf] = js < nt;
        soff[mf] = (long)(t0 + min(js, nt - 1));
    }

    float acc[MF][2][4];
    #pragma unroll
    for (int mf = 0; mf < MF; ++mf)
        #pragma unroll
        for (int nf = 0; nf < 2; ++nf)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[mf][nf][q] = 0.f;

    unsigned a[2][MF][4], sfa[2][MF], b[2][2][2], sfb[2][2];
    auto charger = [&](int buf, int k0) {
        const long kb = k0 >> 1, ks = k0 >> 4;
        #pragma unroll
        for (int mf = 0; mf < MF; ++mf) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned char *p = xq + toff[mf][h] * half_k + kb + 4 * tq;
                const unsigned lo = *reinterpret_cast<const unsigned *>(p);
                const unsigned hi = *reinterpret_cast<const unsigned *>(p + 16);
                a[buf][mf][h]     = tok_ok[mf][h] ? lo : 0u;
                a[buf][mf][h + 2] = tok_ok[mf][h] ? hi : 0u;
            }
            const unsigned s = *reinterpret_cast<const unsigned *>(xsf + soff[mf] * nblk + ks);
            sfa[buf][mf] = sok[mf] ? s : 0u;
        }
        #pragma unroll
        for (int nf = 0; nf < 2; ++nf) {
            const unsigned char *p = qw_e + woff[nf] * half_k + kb + 4 * tq;
            const unsigned lo = *reinterpret_cast<const unsigned *>(p);
            const unsigned hi = *reinterpret_cast<const unsigned *>(p + 16);
            b[buf][nf][0] = wok[nf] ? lo : 0u;
            b[buf][nf][1] = wok[nf] ? hi : 0u;
            const unsigned s = *reinterpret_cast<const unsigned *>(bs_e + woff[nf] * nblk + ks);
            sfb[buf][nf] = wok[nf] ? s : 0u;
        }
    };

    charger(0, 0);
    for (int k0 = 0; k0 < K; k0 += GM_KB) {
        const int cur = (k0 / GM_KB) & 1;
        if (k0 + GM_KB < K) charger(cur ^ 1, k0 + GM_KB);
        #pragma unroll
        for (int mf = 0; mf < MF; ++mf)
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf)
                mma_mxf4nvf4(acc[mf][nf], a[cur][mf], b[cur][nf], sfa[cur][mf], sfb[cur][nf]);
    }

    const float gscale = gscales[e];
    #pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        #pragma unroll
        for (int nf = 0; nf < 2; ++nf) {
            const int r = row0 + warp * 16 + nf * 8 + 2 * tq;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = mf * 16 + g + 8 * h;
                if (j < nt) {
                    __nv_bfloat16 *dst = y + (long)(t0 + j) * M + r;
                    // échelle globale par ligne d'activation (nvfp4_quant_act, grow[G])
                    const float gs = grow ? gscale * grow[t0 + j] : gscale;
                    if (r < M)     dst[0] = __float2bfloat16(acc[mf][nf][2 * h] * gs);
                    if (r + 1 < M) dst[1] = __float2bfloat16(acc[mf][nf][2 * h + 1] * gs);
                }
            }
        }
    }
#endif
}

// --- variante à étages : tuiles A/B/échelles en mémoire partagée par
// cp.async, S étapes en pipeline, BM=128 lignes et 8 warps par bloc. Le
// profil du 13/09 mettait la variante directe à ~25 % de sa borne mémoire
// (chargements globaux un pas de K en avance, latence non couverte).
// Foulée de ligne 48 octets : 32 utiles + 16 de bourrage, pour que les 32
// lectures de 4 octets d'un warp (8 lignes x 4 quarts) tombent sur 32
// bancs distincts (12g + tq mod 32).
constexpr int GM2_BM = 128;
constexpr int GM2_LD = 48;
constexpr int GM2_FILS = 256;

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
#ifdef ACVRAM_MMA_FP4
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
#endif
}
__device__ __forceinline__ void cp_async4(void *smem, const void *gmem) {
#ifdef ACVRAM_MMA_FP4
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#ifdef ACVRAM_MMA_FP4
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
#ifdef ACVRAM_MMA_FP4
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ void cp_async8(void *smem, const void *gmem) {
#ifdef ACVRAM_MMA_FP4
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem));
#endif
}

// KS : profondeur d'un étage, 64 (une MMA par fragment, foulée 48 o) ou 128
// (deux MMA par fragment et moitié de __syncthreads, foulée 80 o : 20g+tq
// donne 32 bancs distincts). Les échelles d'un étage tiennent en KS/16 octets
// par ligne : un mot de 4 par MMA.
// C17 (chantier-c17-mma2-lit-marlin-19-09, scellé sage-c17-scelle-mesure1-ter) : MARLIN = le
// même noyau lisant la DISPOSITION MARLIN des experts (P1, disposition unique) au lieu de la
// pile naturelle — aucune copie, aucun repack : les tuiles 16 k × 64 n de 512 o (`table_qw` =
// w_marlin [K/16, 2N] int32 par expert) arrivent en shared par les mêmes cp.async coalescés
// (une tuile = 32 voies × 16 o), et le fragment B de la MMA (8 k consécutifs d'une colonne
// par mot) est rassemblé depuis 4 mots de 4 voies (marlin_port._indices : voie t = 4·c + j
// porte les colonnes w·16 + c (+8) aux k 2·j + {0, 1, 8, 9}, quartets dans l'ordre
// _PACK_IDX = 0 2 4 6 1 3 5 7 : positions p0 n0:k, p1 n0:k+8, p2 n8:k, p3 n8:k+8, p4 n0:k+1,
// p5 n0:k+9, p6 n8:k+1, p7 n8:k+9). Les échelles Marlin (`table_bscale` = s_marlin [K/16, N]
// octets « S0E5M3 » = moitié haute de half(s·facteur·2^7) << 1, colonnes permutées
// 8·(o%8) + swap4(o/8) par tuile de 64) sont reconverties en UE4M3 dans le fil :
// champ d'exposant (octet >> 3) − (15 + log2 facteur) (= `decal`), forme sous-normale
// reconstruite quand l'exposant tombe à ≤ 0 ; les échelles que le repack a annulées
// (s·facteur·2^7 < 2, 0,0064 % sur Coder) restent nulles — c'est ce que la GEMV Marlin
// calcule déjà. Même arithmétique MMA, même épilogue (gscales = échelle globale naturelle).
__device__ __forceinline__ unsigned gm2_ue4m3_depuis_marlin(unsigned b, int decal) {
    if (b == 0u) return 0u;
    const int E = (int)(b >> 3) - decal;               // exposant UE4M3 (biais 7) reconstitué
    const unsigned m = b & 7u;
    if (E >= 1) return ((unsigned)E << 3) | m;
    if (E >= -2) return (8u | m) >> (1 - E);             // sous-normal UE4M3 : 2^-6 · 0.M
    return 0u;
}

template <int BT, int S, int KS, bool MARLIN = false>
__global__ void __launch_bounds__(GM2_FILS) nvfp4_gemm_grouped_mma2_kernel(
    const float *__restrict__ gscales, const float *__restrict__ grow,
    const unsigned char *__restrict__ xq, const unsigned char *__restrict__ xsf,
    const int *__restrict__ tile_e, const int *__restrict__ tile_t0,
    const int *__restrict__ tile_n, const int64_t *__restrict__ table_qw,
    const int64_t *__restrict__ table_bscale,
    __nv_bfloat16 *__restrict__ y, int M, int K, int decal) {
#ifdef ACVRAM_MMA_FP4
    constexpr int MF = BT / 16;
    constexpr int LD = KS / 2 + 16;          // 48 ou 80 octets par ligne
    constexpr int NM = KS / GM_KB;           // MMA par fragment et par étage
    constexpr int SB = KS / 16;              // octets d'échelles par ligne et par étage
    constexpr int CH = KS / 32;              // chargements de 16 o par ligne et par étage
    // MARLIN : par étage, (KS/16) tuiles de k × 2 tuiles de n (BM = 128 = 2 × 64), 512 o chacune,
    // foulée 528 (16 o de bourrage : cp.async 16 o aligné ; 132 mots ≡ 4 mod 32 bancs) ; échelles
    // (KS/16) × 2 × 64 o = BM · SB, comme la pile naturelle.
    constexpr int NTK = KS / 16;             // tuiles de 16 k par étage
    constexpr int TUILE = 528;
    extern __shared__ __align__(16) unsigned char gm2_smem[];
    typedef unsigned char (*TA)[BT * LD];
    typedef unsigned char (*TB)[GM2_BM * LD];
    typedef unsigned char (*TSA)[BT * SB];
    typedef unsigned char (*TSB)[GM2_BM * SB];
    TA sA = reinterpret_cast<TA>(gm2_smem);
    TB sB = reinterpret_cast<TB>(gm2_smem + S * BT * LD);
    TSA sSA = reinterpret_cast<TSA>(gm2_smem + S * (BT + GM2_BM) * LD);
    TSB sSB = reinterpret_cast<TSB>(gm2_smem + S * (BT + GM2_BM) * LD + S * BT * SB);

    const int tile = blockIdx.y;
    const int e = tile_e[tile], t0 = tile_t0[tile], nt = tile_n[tile];
    // Tuile vide (grille FIXE du décodage sous graphes, _tuiles(cnt, bt, t_max) :
    // n=0 au-delà du compte réel, t0 pouvant dépasser la fin de xq) : rien à
    // lire ni à écrire. Sans ce retour, min(r, nt-1) = -1 et les chargements
    // partent à t0-1 : accès mémoire illégal mesuré le 14/09 sur Coder-30B.
    if (nt <= 0) return;
    const unsigned char *qw_e = reinterpret_cast<const unsigned char *>(table_qw[e]);
    const unsigned char *bs_e = reinterpret_cast<const unsigned char *>(table_bscale[e]);
    const int row0 = blockIdx.x * GM2_BM;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const long half_k = (long)K >> 1;
    const int nblk = K >> 4;
    const int KT = K / KS;

    auto emettre = [&](int st, int k0) {
        const long kb = k0 >> 1, ks = k0 >> 4;
        for (int c = tid; c < CH * BT; c += GM2_FILS) {
            const int r = c / CH, h = c % CH;
            const long src = (long)(t0 + min(r, nt - 1)) * half_k + kb + 16 * h;
            cp_async16(&sA[st][r * LD + 16 * h], xq + src);
        }
        if constexpr (!MARLIN) {
            for (int c = tid; c < CH * GM2_BM; c += GM2_FILS) {
                const int r = c / CH, h = c % CH;
                const long src = (long)min(row0 + r, M - 1) * half_k + kb + 16 * h;
                cp_async16(&sB[st][r * LD + 16 * h], qw_e + src);
            }
        } else {
            // tuile (kt, nt) de w_marlin [K/16, 2N] int32 : octets kt · 8N + nt · 512 ; les deux
            // tuiles de n du bloc sont row0/64 et +1 (N multiple de 64, BM = 128)
            const int nt0 = row0 >> 6;
            for (int c = tid; c < NTK * 2 * 32; c += GM2_FILS) {
                const int ti = c >> 5, lc = c & 31, ktl = ti >> 1, ntl = ti & 1;
                const long src = ((long)(ks + ktl) * 8 * (long)M) + (long)(nt0 + ntl) * 512 + lc * 16;
                cp_async16(&sB[st][ti * TUILE + lc * 16], qw_e + src);
            }
        }
        if (tid < BT) {
            const unsigned char *src = xsf + (long)(t0 + min(tid, nt - 1)) * nblk + ks;
            if constexpr (SB == 4) cp_async4(&sSA[st][tid * SB], src); else cp_async8(&sSA[st][tid * SB], src);
        } else if (!MARLIN && tid < BT + GM2_BM) {
            const int r = tid - BT;
            const unsigned char *src = bs_e + (long)min(row0 + r, M - 1) * nblk + ks;
            if constexpr (SB == 4) cp_async4(&sSB[st][r * SB], src); else cp_async8(&sSB[st][r * SB], src);
        } else if (MARLIN && tid < BT + NTK * 2 * 4) {
            // échelles Marlin : ligne kt de s_marlin [K/16, N] octets, 64 o par tuile de n ;
            // rangées en shared par (ktl, ntl) : 64 o chacune, 4 morceaux de 16 o
            const int c = tid - BT, ti = c >> 2, part = c & 3, ktl = ti >> 1, ntl = ti & 1;
            const unsigned char *src = bs_e + (long)(ks + ktl) * M + (long)((row0 >> 6) + ntl) * 64 + part * 16;
            cp_async16(&sSB[st][ti * 64 + part * 16], src);
        }
    };

    float acc[MF][2][4];
    #pragma unroll
    for (int mf = 0; mf < MF; ++mf)
        #pragma unroll
        for (int nf = 0; nf < 2; ++nf)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[mf][nf][q] = 0.f;

    #pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < KT) emettre(s, s * KS);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        {
            const int kn = kt + S - 1;
            if (kn < KT) emettre(kn % S, kn * KS);
            cp_async_commit();
        }
        const int st = kt % S;
        #pragma unroll
        for (int m = 0; m < NM; ++m) {
            const int ko = 32 * m;                     // décalage d'octets de la MMA m dans la ligne
            unsigned a[MF][4], sfa[MF], b[2][2], sfb[2];
            #pragma unroll
            for (int mf = 0; mf < MF; ++mf) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = mf * 16 + g + 8 * h;
                    const unsigned char *p = &sA[st][j * LD + ko + 4 * tq];
                    const unsigned lo = *reinterpret_cast<const unsigned *>(p);
                    const unsigned hi = *reinterpret_cast<const unsigned *>(p + 16);
                    a[mf][h] = (j < nt) ? lo : 0u;
                    a[mf][h + 2] = (j < nt) ? hi : 0u;
                }
                const int js = mf * 16 + (lane >> 2) + 8 * (lane & 1);
                const unsigned sv = *reinterpret_cast<const unsigned *>(&sSA[st][js * SB + 4 * m]);
                sfa[mf] = (js < nt) ? sv : 0u;
            }
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                const int r = warp * 16 + nf * 8 + g;
                const bool ok = row0 + r < M;
                if constexpr (!MARLIN) {
                    const unsigned char *p = &sB[st][r * LD + ko + 4 * tq];
                    const unsigned lo = *reinterpret_cast<const unsigned *>(p);
                    const unsigned hi = *reinterpret_cast<const unsigned *>(p + 16);
                    b[nf][0] = ok ? lo : 0u;
                    b[nf][1] = ok ? hi : 0u;
                    const unsigned sv = *reinterpret_cast<const unsigned *>(&sSB[st][r * SB + 4 * m]);
                    sfb[nf] = ok ? sv : 0u;
                } else {
                    // colonne n = row0 + r : tuile de n ntl = r/64, o = r%64 ; dans la tuile Marlin
                    // w = o/16 (mot), h = (o%16)/8 (moitié n0 / n8), c = o%8 (voie t = 4c + j)
                    const int o = r & 63, ntl = r >> 6, w = o >> 4, h = (o >> 3) & 1, c = o & 7;
                    // le mot lo de la MMA m couvre k = 64m + 8tq..+7 : tuile de k ktl = 4m + tq/2,
                    // moitié kb = 8·(tq&1) ; hi = +32 k : ktl + 2
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int ktl = 4 * m + (tq >> 1) + 2 * half, kb = tq & 1;
                        const unsigned char *tuile = &sB[st][(ktl * 2 + ntl) * TUILE];
                        const int pA = kb + 2 * h;             // quartet k (pair) ; +4 = k+1 (impair)
                        unsigned mot = 0u;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int j = (i + g) & 3;         // ordre tourné par g : bancs distincts
                            const unsigned W = *reinterpret_cast<const unsigned *>(tuile + (4 * c + j) * 16 + w * 4);
                            const unsigned nl = (W >> (4 * pA)) & 0xFu, nh = (W >> (4 * (pA + 4))) & 0xFu;
                            mot |= (nl | (nh << 4)) << (8 * j);    // octet j = paire (2j, 2j+1), pair en bas
                        }
                        b[nf][half] = ok ? mot : 0u;
                    }
                    // échelles : 4 tuiles de k de la MMA m, octet permuté 8·(o%8) + swap4(o/8) dans la
                    // tuile de 64, reconverties S0E5M3 → UE4M3
                    const int q = o >> 3, bidx = 8 * (o & 7) + ((q & ~3) | ((q & 1) << 1) | ((q >> 1) & 1));
                    unsigned sv = 0u;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const unsigned by = sSB[st][((4 * m + i) * 2 + ntl) * 64 + bidx];
                        sv |= gm2_ue4m3_depuis_marlin(by, decal) << (8 * i);
                    }
                    sfb[nf] = ok ? sv : 0u;
                }
            }
            #pragma unroll
            for (int mf = 0; mf < MF; ++mf)
                #pragma unroll
                for (int nf = 0; nf < 2; ++nf)
                    mma_mxf4nvf4(acc[mf][nf], a[mf], b[nf], sfa[mf], sfb[nf]);
        }
    }
    cp_async_wait<0>();

    const float gscale = gscales[e];
    #pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        #pragma unroll
        for (int nf = 0; nf < 2; ++nf) {
            const int r = row0 + warp * 16 + nf * 8 + 2 * tq;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = mf * 16 + g + 8 * h;
                if (j < nt) {
                    __nv_bfloat16 *dst = y + (long)(t0 + j) * M + r;
                    // échelle globale par ligne d'activation (nvfp4_quant_act, grow[G])
                    const float gs = grow ? gscale * grow[t0 + j] : gscale;
                    if (r < M)     dst[0] = __float2bfloat16(acc[mf][nf][2 * h] * gs);
                    if (r + 1 < M) dst[1] = __float2bfloat16(acc[mf][nf][2 * h + 1] * gs);
                }
            }
        }
    }
#endif
}

__global__ void nvfp4_mma_sonde_kernel(int *flag) {
#ifdef ACVRAM_MMA_FP4
    flag[0] = 1;
#else
    flag[0] = 0;
#endif
}

// Vrai si le binaire chargé pour cette carte contient l'asm mxf4nvf4 (cible
// famille sm_120f) : le repli PTX générique compile un stub, et rien d'autre
// ne le dirait.
bool nvfp4_gemm_grouped_mma_disponible() {
    static int cache = -1;
    if (cache >= 0) return cache == 1;
    int major = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major != 12) { cache = 0; return false; }
    auto flag = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, dev));
    nvfp4_mma_sonde_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(flag.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cache = flag.item<int>();
    return cache == 1;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> nvfp4_quant_act(
        torch::Tensor x, c10::optional<torch::Tensor> awq, c10::optional<torch::Tensor> e_sorted,
        c10::optional<torch::Tensor> compteurs, int64_t hadamard) {
    CHECK_CUDA(x); CHECK_CONTIG(x); ACVRAM_DEVICE_GUARD(x);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "quant_act : activations bf16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 64 == 0, "quant_act : [G, K] avec K multiple de 64");
    const long G = x.size(0), K = x.size(1);
    TORCH_CHECK(K <= 16384, "quant_act : K <= 16384 (une ligne en memoire partagee)");
    const int nblk = (int)(K >> 4);
    TORCH_CHECK(hadamard >= 0 && (hadamard == 0 || ((hadamard & (hadamard - 1)) == 0 && K % hadamard == 0 && hadamard >= 16)),
                "quant_act : bloc de Hadamard = puissance de 2 >= 16 divisant K, ou 0");
    const __nv_bfloat16 *awq_p = nullptr;
    const int *es_p = nullptr;
    if (awq.has_value() && awq->defined()) {
        TORCH_CHECK(e_sorted.has_value() && e_sorted->defined(), "quant_act : table AWQ sans e_sorted");
        CHECK_CUDA(*awq); CHECK_CONTIG(*awq); CHECK_CUDA(*e_sorted); CHECK_CONTIG(*e_sorted);
        TORCH_CHECK(awq->scalar_type() == torch::kBFloat16 && awq->dim() == 2 && awq->size(1) == K,
                    "quant_act : table AWQ [E, K] bf16 de la largeur de x");
        TORCH_CHECK(e_sorted->scalar_type() == torch::kInt && e_sorted->numel() == G,
                    "quant_act : e_sorted int32 [G]");
        awq_p = reinterpret_cast<const __nv_bfloat16 *>(awq->data_ptr());
        es_p = e_sorted->data_ptr<int>();
    }
    long long *cpt_p = nullptr;
    if (compteurs.has_value() && compteurs->defined()) {
        CHECK_CUDA(*compteurs); CHECK_CONTIG(*compteurs);
        TORCH_CHECK(compteurs->scalar_type() == torch::kLong && compteurs->numel() == 3,
                    "quant_act : compteurs int64 [3] (blocs, flushes, satures)");
        cpt_p = reinterpret_cast<long long *>(compteurs->data_ptr<int64_t>());
    }
    auto opt = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto xq = torch::empty({G, K / 2}, opt);
    auto xsf = torch::empty({G, nblk}, opt);
    auto grow = torch::empty({G}, opt.dtype(torch::kFloat32));
    if (G > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const size_t shm = (size_t)K * sizeof(float);
        if (shm > 48 * 1024)
            cudaFuncSetAttribute(nvfp4_quant_act_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        nvfp4_quant_act_kernel<<<(unsigned)G, QA_FILS, shm, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
            xq.data_ptr<unsigned char>(), xsf.data_ptr<unsigned char>(), grow.data_ptr<float>(),
            nblk, awq_p, es_p, cpt_p, (int)hadamard);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {xq, xsf, grow};
}

// table_qw / table_bscale [E] int64 : adresse octet (device) des piles
// [M, K/2] et [M, K/16] de chaque expert — contrat du bead pds (Océane) :
// résident = data_ptr() de sa tranche, froid = pointeur device zéro-copie
// d'un tampon épinglé ; jamais 0 ; mise à jour hors pas seulement. Le noyau
// ne connaît aucune autre adresse. gscales [E] reste résident, indexé par e.
torch::Tensor nvfp4_gemm_grouped_mma(torch::Tensor table_qw, torch::Tensor table_bscale,
                                     torch::Tensor gscales, torch::Tensor xq,
                                     torch::Tensor xsf, torch::Tensor tile_e,
                                     torch::Tensor tile_t0, torch::Tensor tile_n,
                                     int64_t M, int64_t K, int64_t bt, int64_t etages, int64_t ks,
                                     c10::optional<torch::Tensor> grow, int64_t marlin) {
    CHECK_CUDA(xq); ACVRAM_DEVICE_GUARD(xq);
    // C17 : marlin >= 0 = table_qw/table_bscale pointent la DISPOSITION MARLIN (w_marlin, s_marlin),
    // `marlin` = decal d'exposant des echelles (15 + log2 facteur) ; -1 = pile naturelle
    TORCH_CHECK(marlin < 0 || (etages > 0 && M % 128 == 0),
                "GEMM groupee MMA sur disposition Marlin : etages > 0 et N multiple de 128 (deux tuiles de n par bloc)");
    const float *grow_p = nullptr;
    if (grow.has_value() && grow->defined()) {
        CHECK_CUDA(*grow); CHECK_CONTIG(*grow);
        TORCH_CHECK(grow->scalar_type() == torch::kFloat && grow->numel() == xq.size(0),
                    "GEMM groupee MMA : grow fp32 [G] (une echelle globale par ligne d'activation)");
        grow_p = grow->data_ptr<float>();
    }
    TORCH_CHECK(etages == 0 || etages == 2 || etages == 3 || etages == 4,
                "GEMM groupee MMA : etages dans {0 (direct), 2, 3, 4}");
    TORCH_CHECK(ks == 64 || ks == 128, "GEMM groupee MMA : ks dans {64, 128}");
    TORCH_CHECK(K % ks == 0, "GEMM groupee MMA : K multiple de ks");
    CHECK_CONTIG(xq); CHECK_CONTIG(xsf); CHECK_CONTIG(gscales);
    TORCH_CHECK(K % GM_KB == 0, "GEMM groupee MMA : K multiple de 64");
    TORCH_CHECK(table_qw.scalar_type() == torch::kInt64 && table_qw.is_cuda() && table_qw.is_contiguous(),
                "GEMM groupee MMA : table_qw int64 contigu sur la carte");
    TORCH_CHECK(table_bscale.scalar_type() == torch::kInt64 && table_bscale.is_cuda() && table_bscale.is_contiguous(),
                "GEMM groupee MMA : table_bscale int64 contigu sur la carte");
    TORCH_CHECK(table_qw.numel() == gscales.numel() && table_bscale.numel() == gscales.numel(),
                "GEMM groupee MMA : tables et gscales de meme longueur E");
    TORCH_CHECK(bt == 16 || bt == 32 || bt == 64 || bt == 128,
                "GEMM groupee MMA : bt dans {16, 32, 64, 128} (128 : variante a etages seulement)");
    TORCH_CHECK(bt != 128 || etages != 0, "GEMM groupee MMA : bt=128 exige etages > 0");
    const int G = xq.size(0), T = tile_e.size(0);
    // empty, pas zeros : chaque ligne de xq appartient a exactement une tuile
    // (routage par comptage, _tuiles), le noyau ecrit toutes les lignes et
    // toutes les colonnes M ; le remplissage a zero etait un lancement de plus
    // par GEMM (3 par couche au decodage, compte par test_moe_route_pack).
    auto y = torch::empty({G, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(xq.device()));
    if (T == 0) return y.zero_();
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GM_ARGS gscales.data_ptr<float>(), grow_p, \
        xq.data_ptr<unsigned char>(), xsf.data_ptr<unsigned char>(), \
        tile_e.data_ptr<int>(), tile_t0.data_ptr<int>(), tile_n.data_ptr<int>(), \
        table_qw.data_ptr<int64_t>(), table_bscale.data_ptr<int64_t>(), \
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), (int)M, (int)K
    #define GM_ARGS2 GM_ARGS, (int)marlin
    if (etages == 0) {
        dim3 grid((M + GM_BM - 1) / GM_BM, T);
        #define GM_L(BT) nvfp4_gemm_grouped_mma_kernel<BT><<<grid, 128, 0, stream>>>(GM_ARGS)
        if (bt == 16) GM_L(16); else if (bt == 32) GM_L(32); else GM_L(64);
        #undef GM_L
    } else {
        dim3 grid((M + GM2_BM - 1) / GM2_BM, T);
        const size_t shm = (size_t)etages * ((bt + GM2_BM) * (ks / 2 + 16) + (bt + GM2_BM) * (ks / 16));
        #define GM_L2(BT, S, KS) do { \
            if (marlin < 0) { \
                if (shm > 48 * 1024) cudaFuncSetAttribute(nvfp4_gemm_grouped_mma2_kernel<BT, S, KS, false>, \
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); \
                nvfp4_gemm_grouped_mma2_kernel<BT, S, KS, false><<<grid, GM2_FILS, shm, stream>>>(GM_ARGS2); \
            } else { \
                if (shm > 48 * 1024) cudaFuncSetAttribute(nvfp4_gemm_grouped_mma2_kernel<BT, S, KS, true>, \
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); \
                nvfp4_gemm_grouped_mma2_kernel<BT, S, KS, true><<<grid, GM2_FILS, shm, stream>>>(GM_ARGS2); \
            } } while (0)
        #define GM_LS(S, KS) do { if (bt == 16) GM_L2(16, S, KS); else if (bt == 32) GM_L2(32, S, KS); \
                                  else if (bt == 64) GM_L2(64, S, KS); else GM_L2(128, S, KS); } while (0)
        #define GM_LK(KS) do { if (etages == 2) GM_LS(2, KS); else if (etages == 3) GM_LS(3, KS); else GM_LS(4, KS); } while (0)
        if (ks == 64) GM_LK(64); else GM_LK(128);
        #undef GM_LK
        #undef GM_LS
        #undef GM_L2
    }
    #undef GM_ARGS2
    #undef GM_ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


// --------------------------------------------------------------------------
// RoPE en place : q et k tournés d'un seul lancement.
//
// En PyTorch la même chose coûte, par couche, deux tranches, deux négations,
// deux concaténations et quatre produits — une dizaine de petits noyaux dont
// aucun ne fait de calcul utile au-delà d'une multiplication par jeton.
// Les « d » premières dimensions tournent (RoPE partiel de qwen3-next), le
// reste passe tel quel.
// --------------------------------------------------------------------------
__global__ void rope_inplace_kernel(__nv_bfloat16 *__restrict__ q,
                                    __nv_bfloat16 *__restrict__ k,
                                    const float *__restrict__ cosv,
                                    const float *__restrict__ sinv,
                                    const long *__restrict__ pos,
                                    const __nv_bfloat16 *__restrict__ wq,
                                    const __nv_bfloat16 *__restrict__ wk,
                                    float eps,
                                    int Hq, int Hk, int Dq, int Dk, int d,
                                    long ldq, long ldk) {
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const bool est_q = h < Hq;
    // ldq/ldk : pas entre deux jetons. q et k sont souvent des tranches d'une
    // projection empilée — les traiter en place évite deux copies par couche.
    __nv_bfloat16 *base = est_q ? q + (long)t * ldq + (long)h * Dq
                                : k + (long)t * ldk + (long)(h - Hq) * Dk;
    const int D = est_q ? Dq : Dk;
    const __nv_bfloat16 *w = est_q ? wq : wk;
    // Les tables complètes sont passées telles quelles : indexer ici épargne
    // deux index_select et deux conversions par couche.
    const long ligne = (pos != nullptr) ? pos[t] : (long)t;
    const float *c = cosv + ligne * d, *s = sinv + ligne * d;

    // Normalisation RMS par tête (Qwen3, Gemma), fusionnée : la tête tient
    // dans un bloc, sa somme des carrés ne coûte donc qu'une réduction.
    __shared__ float red[32];
    float inv = 1.f;
    if (w != nullptr) {
        float somme = 0.f;
        for (int i = tid; i < D; i += blockDim.x) {
            const float v = __bfloat162float(base[i]);
            somme += v * v;
        }
        for (int o = 16; o > 0; o >>= 1) somme += __shfl_xor_sync(0xffffffffu, somme, o);
        const int nw = (blockDim.x + 31) >> 5;
        if ((tid & 31) == 0) red[tid >> 5] = somme;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.f;
            for (int i = 0; i < nw; ++i) tot += red[i];
            red[0] = rsqrtf(tot / (float)D + eps);
        }
        __syncthreads();
        inv = red[0];
    }

    const int demi = d >> 1;
    for (int i = tid; i < demi; i += blockDim.x) {
        float a = __bfloat162float(base[i]) * inv;
        float b = __bfloat162float(base[i + demi]) * inv;
        if (w != nullptr) {
            a *= __bfloat162float(w[i]);
            b *= __bfloat162float(w[i + demi]);
        }
        base[i] = __float2bfloat16(a * c[i] - b * s[i]);
        base[i + demi] = __float2bfloat16(b * c[i + demi] + a * s[i + demi]);
    }
    // RoPE partiel : la queue non tournée est seulement normalisée
    if (w != nullptr)
        for (int i = d + tid; i < D; i += blockDim.x)
            base[i] = __float2bfloat16(__bfloat162float(base[i]) * inv
                                       * __bfloat162float(w[i]));
}

void rope_inplace_pos(torch::Tensor q, torch::Tensor k, torch::Tensor cosv,
                      torch::Tensor sinv, c10::optional<torch::Tensor> pos,
                      c10::optional<torch::Tensor> wq,
                      c10::optional<torch::Tensor> wk, double eps) {
    CHECK_CUDA(q); CHECK_CUDA(k); ACVRAM_DEVICE_GUARD(q);
    CHECK_CONTIG(cosv); CHECK_CONTIG(sinv);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "rope : [jetons, tetes, dim]");
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1
                && q.stride(1) == q.size(2) && k.stride(1) == k.size(2),
                "rope : tetes contigues exigees");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16
                && k.scalar_type() == torch::kBFloat16, "rope : bf16");
    TORCH_CHECK(cosv.scalar_type() == torch::kFloat, "rope : cos/sin en fp32");
    const int T = q.size(0), Hq = q.size(1), Dq = q.size(2);
    const int Hk = k.size(1), Dk = k.size(2), d = cosv.size(-1);
    TORCH_CHECK(d % 2 == 0 && d <= Dq && d <= Dk, "rope : dimension tournee invalide");
    const long *ppos = nullptr;
    if (pos.has_value()) {
        TORCH_CHECK(pos->scalar_type() == torch::kLong, "rope : positions int64");
        TORCH_CHECK(pos->numel() == T, "rope : une position par jeton");
        ppos = pos->data_ptr<long>();
    }
    const __nv_bfloat16 *pwq = nullptr, *pwk = nullptr;
    if (wq.has_value())
        pwq = reinterpret_cast<const __nv_bfloat16 *>(wq->contiguous().data_ptr());
    if (wk.has_value())
        pwk = reinterpret_cast<const __nv_bfloat16 *>(wk->contiguous().data_ptr());
    dim3 grid(T, Hq + Hk);
    const int th = std::min(256, (std::max(Dq, Dk) + 31) / 32 * 32);
    rope_inplace_kernel<<<grid, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16 *>(q.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(k.data_ptr()),
        cosv.data_ptr<float>(), sinv.data_ptr<float>(), ppos, pwq, pwk,
        (float)eps, Hq, Hk, Dq, Dk, d, q.stride(0), k.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}




// --------------------------------------------------------------------------
// Écriture du cache KV quantifié en INT8, en un seul lancement.
//
// Le chemin PyTorch enchaînait, par couche et pour chacun de k et v : un abs,
// un amax, deux conversions, une division, un round, un clamp, une conversion
// de sortie, puis la dispersion — une vingtaine de noyaux sur des tenseurs de
// quelques centaines de valeurs, où seule la latence de lancement compte.
// Ici un bloc porte une tête d'un jeton : il calcule son amax, quantifie et
// écrit à l'emplacement voulu.
// --------------------------------------------------------------------------
__global__ void kv_write_int8_kernel(
    const __nv_bfloat16 *__restrict__ k, const __nv_bfloat16 *__restrict__ v,
    const long *__restrict__ slots, signed char *__restrict__ kc,
    signed char *__restrict__ vc, __half *__restrict__ ks, __half *__restrict__ vs,
    int H, int D, int bs, long sk, long sv) {
    __shared__ float red[8];
    const int t = blockIdx.x, h = blockIdx.y;
    const long slot = slots[t];
    if (slot < 0) return;
    const long pos = (slot / bs) * bs + (slot % bs);   // index à plat [bloc, offset]
    #pragma unroll 1
    for (int quel = 0; quel < 2; ++quel) {
        // sk / sv : pas d'un jeton (C15 niveau 3 : k et v sont des tranches de
        // la projection q/k/v empilée, lues en place au lieu d'être recopiées
        // contiguës — deux nœuds de graphe par couche)
        const __nv_bfloat16 *src = (quel ? v : k) + (long)t * (quel ? sv : sk) + (long)h * D;
        float amax = 0.f;
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const int nw = (blockDim.x + 31) >> 5;
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = 0.f;
            for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
            red[0] = fmaxf(m / 127.f, 1e-8f);
        }
        __syncthreads();
        const float sc = red[0], inv = 1.f / sc;
        signed char *dst = (quel ? vc : kc) + (pos * H + h) * D;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            const int q = __float2int_rn(__bfloat162float(src[i]) * inv);
            dst[i] = (signed char)max(-127, min(127, q));
        }
        if (threadIdx.x == 0)
            (quel ? vs : ks)[pos * H + h] = __float2half(sc);
        __syncthreads();
    }
}

void kv_write_int8(torch::Tensor k, torch::Tensor v, torch::Tensor slots,
                   torch::Tensor kc, torch::Tensor vc,
                   torch::Tensor ks, torch::Tensor vs, int64_t bs) {
    CHECK_CUDA(k); ACVRAM_DEVICE_GUARD(k);
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16,
                "cache KV : k et v en bf16");
    TORCH_CHECK(kc.scalar_type() == torch::kChar && vc.scalar_type() == torch::kChar,
                "cache KV : stockage int8");
    TORCH_CHECK(slots.scalar_type() == torch::kLong, "cache KV : emplacements int64");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "cache KV : k et v [T, H, D]");
    // Une tête est lue contiguë ([H, D] à pas (D, 1)) ; le pas du jeton est
    // libre : une tranche de la projection empilée passe sans copie (C15
    // niveau 3). Tout autre agencement est recopié comme avant.
    auto contigu_par_tete = [](const torch::Tensor &x) {
        return x.stride(2) == 1 && x.stride(1) == x.size(2);
    };
    auto kk = contigu_par_tete(k) ? k : k.contiguous();
    auto vv = contigu_par_tete(v) ? v : v.contiguous();
    const int T = kk.size(0), H = kk.size(1), D = kk.size(2);
    TORCH_CHECK(vv.size(0) == T && vv.size(1) == H && vv.size(2) == D, "cache KV : k et v de même forme");
    dim3 grid(T, H);
    const int th = std::min(256, (D + 31) / 32 * 32);
    kv_write_int8_kernel<<<grid, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(kk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(vv.data_ptr()),
        slots.contiguous().data_ptr<long>(),
        reinterpret_cast<signed char *>(kc.data_ptr()),
        reinterpret_cast<signed char *>(vc.data_ptr()),
        reinterpret_cast<__half *>(ks.data_ptr()),
        reinterpret_cast<__half *>(vs.data_ptr()), H, D, (int)bs,
        (long)kk.stride(0), (long)vv.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// --------------------------------------------------------------------------
// C5-b (chantier-c5b-19-09, jumeau : memory/kv_canal.py) — clés int8 à échelle
// PAR CANAL et par tête sur chaque bloc de 16 jetons. Déquantification d'une
// cellule, uniforme : k = code × ks[jeton, tête] (half) × sc[bloc, tête, canal]
// (E4M3). Bloc fermé par canal : sc = E4M3↑(amax_canal × 16 / 127), ks = 2^-4 ;
// bloc par jeton (repli) : sc = 1, ks = amax/127 — le chemin d'aujourd'hui.
// Au décodage les jetons arrivent un par un et l'échelle par canal d'un bloc
// n'est connue qu'à sa fermeture : le bloc COURANT garde ses clés en bf16 dans
// une ligne d'une réserve (tampon [R, 16, HKV, D]), tampon_de[bloc] = ligne
// (ou -1 fermé, -2 par jeton) ; à la 16e écriture la ligne est quantifiée par
// canal et rendue à la pile des lignes libres. Trois lancements par appel,
// tous sur l'appareil (capturables) : rôles → écriture → fermeture. Les
// allocations (pop) n'ont lieu que dans le premier, les restitutions (push)
// que dans le troisième : jamais les deux dans un même noyau, la pile n'a
// donc besoin que d'un atomique sur son sommet.
// V reste par jeton (lu après le softmax : son échelle par canal ne se
// replie pas dans q).
// --------------------------------------------------------------------------
constexpr int KVC_ROLE_RIEN = 0, KVC_ROLE_DIRECT = 1, KVC_ROLE_TAMPON = 2,
              KVC_ROLE_PAR_JETON = 3;
constexpr unsigned char KVC_SC_PAR_JETON = 0x38;   // E4M3 de 1,0
constexpr float KVC_KS_CANAL = 0.0625f;            // 2^-4, exact en half

// Plus petit E4M3 >= x (x >= 0 fini) : arrondi vers le HAUT pour qu'aucun code
// ne sature ; 0 -> 2^-9 (jamais d'échelle nulle) ; >= 448 -> 448 (amax >= 3556 :
// saturation, à borner par le test carte). Arithmétique entière identique à
// kv_canal.e4m3_haut : les deux côtés rendent le même octet.
__device__ __forceinline__ unsigned char e4m3_haut(float x) {
    if (!(x > 0.f)) return 1;
    if (x >= 448.f) return 0x7E;
    const unsigned int u = __float_as_uint(x);
    const int fe = (int)((u >> 23) & 0xFFu) - 127;
    if (fe < -6) {                                   // sous-normal : multiples de 2^-9
        unsigned int b = (unsigned int)floorf(x * 512.f);
        if ((float)b * 0.001953125f < x) b += 1u;    // 8 = 0x08 = 2^-6 : report juste
        return (unsigned char)b;
    }
    unsigned int b = ((unsigned int)(fe + 7) << 3) | ((u >> 20) & 7u);
    if (u & 0xFFFFFu) b += 1u;
    return (unsigned char)(b > 0x7Eu ? 0x7Eu : b);
}

// Une ligne [D] bf16 -> codes int8 + une échelle half (amax/127) : le chemin
// par jeton de kv_write_int8_kernel, en fonction ; blockDim multiple de 32.
__device__ __forceinline__ void kvc_quant_par_jeton(
    const __nv_bfloat16 *__restrict__ src, signed char *__restrict__ dst,
    __half *__restrict__ echelle, float *red, int D) {
    float amax = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const int nw = (blockDim.x + 31) >> 5;
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.f;
        for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
        red[0] = fmaxf(m / 127.f, 1e-8f);
    }
    __syncthreads();
    const float sc = red[0], inv = 1.f / sc;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const int q = __float2int_rn(__bfloat162float(src[i]) * inv);
        dst[i] = (signed char)max(-127, min(127, q));
    }
    if (threadIdx.x == 0) *echelle = __float2half(sc);
    __syncthreads();
}

// Un bloc entier [16, HKV, D] d'une tête, quantifié PAR CANAL : le fil d porte
// le canal d (amax sur les 16 jetons, échelle E4M3 vers le haut, codes en
// division IEEE — --use_fast_math rendrait la division approchée et le jumeau
// torch ne tomberait plus au bit). src : jeton j à src + j*stride ; dst : codes
// de la cellule (bloc, j, tête) à dst + j*H*D ; ks : (bloc, j, tête) = 2^-4.
__device__ __forceinline__ void kvc_quant_bloc_canal(
    const __nv_bfloat16 *__restrict__ src, long stride,
    signed char *__restrict__ dst, __half *__restrict__ ks,
    unsigned char *__restrict__ sc, int H, int D, int bs) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float amax = 0.f;
        for (int j = 0; j < bs; ++j)
            amax = fmaxf(amax, fabsf(__bfloat162float(src[(long)j * stride + d])));
        const unsigned char bits = e4m3_haut(__fdiv_rn(amax * 16.f, 127.f));
        sc[d] = bits;
        const float s = e4m3_to_float(bits);
        for (int j = 0; j < bs; ++j) {
            const float x16 = __bfloat162float(src[(long)j * stride + d]) * 16.f;
            const int q = __float2int_rn(__fdiv_rn(x16, s));
            dst[(long)j * H * D + d] = (signed char)max(-127, min(127, q));
        }
    }
    if (threadIdx.x < bs) ks[(long)threadIdx.x * H] = __float2half(KVC_KS_CANAL);
}

// (1) Rôles : un fil par jeton. Meneur = le jeton au décalage 0 de son bloc
// dans CET appel (les jetons d'une séquence ont des emplacements consécutifs
// au préfill, runner._build_batch) ; run plein (16 jetons) -> DIRECT pour le
// meneur, RIEN pour les 15 autres ; run partiel -> TAMPON, le meneur alloue
// la ligne (ou réemploie celle restée attachée à un bloc abandonné) ; pas de
// meneur dans l'appel -> le bloc est déjà ouvert : sa ligne, ou par jeton.
__global__ void kv_canal_plan_kernel(
    const long *__restrict__ slots, int T, int bs,
    int *__restrict__ tampon_de, int *__restrict__ libres, int *__restrict__ sommet,
    unsigned char *__restrict__ sc, int H, int D,
    int *__restrict__ role, int *__restrict__ rang) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const long slot = slots[t];
    role[t] = KVC_ROLE_RIEN; rang[t] = -1;
    if (slot < 0) return;
    const long blk = slot / bs, off = slot % bs, base = slot - off;
    const int t0 = t - (int)off;
    const bool meneur_ici = (t0 >= 0 && slots[t0] == base);
    if (meneur_ici) {
        const bool plein = (t0 + bs - 1 < T && slots[t0 + bs - 1] == base + bs - 1);
        if (plein) {
            if (t == t0) { role[t] = KVC_ROLE_DIRECT; rang[t] = tampon_de[blk]; tampon_de[blk] = -1; }
            return;
        }
        if (t == t0) {
            int r = tampon_de[blk];
            if (r < 0) {
                const int idx = atomicSub(sommet, 1) - 1;
                if (idx >= 0) {
                    r = libres[idx];
                } else {
                    atomicAdd(sommet, 1);
                    r = -2;                              // réserve épuisée : par jeton
                    for (int i = 0; i < H * D; ++i) sc[blk * H * D + i] = KVC_SC_PAR_JETON;
                }
                tampon_de[blk] = r;
            }
        }
        role[t] = KVC_ROLE_TAMPON;
        return;
    }
    const int r = tampon_de[blk];
    if (r >= 0) { role[t] = KVC_ROLE_TAMPON; return; }
    if (r == -1) {                                       // bloc jamais ouvert par ce chemin
        tampon_de[blk] = -2;
        for (int i = 0; i < H * D; ++i) sc[blk * H * D + i] = KVC_SC_PAR_JETON;
    }
    role[t] = KVC_ROLE_PAR_JETON;
}

// (2) Écriture : un bloc par (jeton, tête). V par jeton toujours ; K selon le
// rôle — DIRECT : les 16 jetons du run (à partir de ce jeton) par canal ;
// TAMPON : la ligne bf16 du bloc (ou par jeton si le meneur n'a pas eu de
// ligne : tampon_de[blk] < 0, relu ici, après le noyau des rôles).
__global__ void kv_canal_write_kernel(
    const __nv_bfloat16 *__restrict__ k, const __nv_bfloat16 *__restrict__ v,
    const long *__restrict__ slots, const int *__restrict__ role,
    signed char *__restrict__ kc, signed char *__restrict__ vc,
    __half *__restrict__ ks, __half *__restrict__ vs,
    unsigned char *__restrict__ sc, __nv_bfloat16 *__restrict__ tampon,
    const int *__restrict__ tampon_de, int H, int D, int bs) {
    __shared__ float red[8];
    const int t = blockIdx.x, h = blockIdx.y;
    const long slot = slots[t];
    if (slot < 0) return;
    const long blk = slot / bs, off = slot % bs;
    const long cell = blk * bs + off;
    kvc_quant_par_jeton(v + ((long)t * H + h) * D, vc + (cell * H + h) * D,
                        vs + cell * H + h, red, D);
    const int ro = role[t];
    if (ro == KVC_ROLE_RIEN) return;
    const __nv_bfloat16 *src = k + ((long)t * H + h) * D;
    if (ro == KVC_ROLE_DIRECT) {
        kvc_quant_bloc_canal(src, (long)H * D, kc + (blk * bs * H + h) * D,
                             ks + blk * bs * H + h, sc + (blk * H + h) * D, H, D, bs);
        return;
    }
    const int r = (ro == KVC_ROLE_TAMPON) ? tampon_de[blk] : -2;
    if (r >= 0) {
        __nv_bfloat16 *dst = tampon + (((long)r * bs + off) * H + h) * D;
        for (int i = threadIdx.x; i < D; i += blockDim.x) dst[i] = src[i];
        return;
    }
    kvc_quant_par_jeton(src, kc + (cell * H + h) * D, ks + cell * H + h, red, D);
}

// (3) Fermeture : un bloc par (jeton, tête). Le 16e jeton d'un bloc TAMPON
// quantifie la ligne par canal et la rend ; un meneur DIRECT rend la ligne
// périmée que le bloc gardait. Aucune allocation ici (voir l'en-tête).
__global__ void kv_canal_close_kernel(
    const long *__restrict__ slots, const int *__restrict__ role,
    const int *__restrict__ rang, signed char *__restrict__ kc,
    __half *__restrict__ ks, unsigned char *__restrict__ sc,
    const __nv_bfloat16 *__restrict__ tampon, int *__restrict__ tampon_de,
    int *__restrict__ libres, int *__restrict__ sommet, int H, int D, int bs) {
    const int t = blockIdx.x, h = blockIdx.y;
    const long slot = slots[t];
    if (slot < 0) return;
    const int ro = role[t];
    const long blk = slot / bs, off = slot % bs;
    if (ro == KVC_ROLE_DIRECT) {
        if (h == 0 && threadIdx.x == 0 && rang[t] >= 0)
            libres[atomicAdd(sommet, 1)] = rang[t];
        return;
    }
    if (ro != KVC_ROLE_TAMPON || off != bs - 1) return;
    const int r = tampon_de[blk];
    if (r < 0) return;
    kvc_quant_bloc_canal(tampon + (((long)r * bs) * H + h) * D, (long)H * D,
                         kc + (blk * bs * H + h) * D, ks + blk * bs * H + h,
                         sc + (blk * H + h) * D, H, D, bs);
    __syncthreads();
    if (h == 0 && threadIdx.x == 0) {
        tampon_de[blk] = -1;
        libres[atomicAdd(sommet, 1)] = r;
    }
}

void kv_write_int8_canal(torch::Tensor k, torch::Tensor v, torch::Tensor slots,
                         torch::Tensor kc, torch::Tensor vc,
                         torch::Tensor ks, torch::Tensor vs,
                         torch::Tensor sc, torch::Tensor tampon,
                         torch::Tensor tampon_de, torch::Tensor libres,
                         torch::Tensor sommet, int64_t bs) {
    CHECK_CUDA(k); ACVRAM_DEVICE_GUARD(k);
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16,
                "cache KV canal : k et v en bf16");
    TORCH_CHECK(kc.scalar_type() == torch::kChar && vc.scalar_type() == torch::kChar,
                "cache KV canal : stockage int8");
    TORCH_CHECK(slots.scalar_type() == torch::kLong, "cache KV canal : emplacements int64");
    TORCH_CHECK(sc.scalar_type() == torch::kByte && tampon.scalar_type() == torch::kBFloat16
                && tampon_de.scalar_type() == torch::kInt && libres.scalar_type() == torch::kInt
                && sommet.scalar_type() == torch::kInt, "cache KV canal : sc uint8, tampon bf16, indices int32");
    CHECK_CONTIG(sc); CHECK_CONTIG(tampon); CHECK_CONTIG(tampon_de); CHECK_CONTIG(libres);
    auto kk = k.contiguous(), vv = v.contiguous(), sl = slots.contiguous();
    const int T = kk.size(0), H = kk.size(1), D = kk.size(2);
    TORCH_CHECK(bs <= 32 && D % 32 == 0, "cache KV canal : bloc <= 32 et D multiple de 32");
    if (T == 0) return;
    auto opt = torch::TensorOptions().dtype(torch::kInt).device(k.device());
    auto plan = torch::empty({2, T}, opt);          // rôles, lignes périmées
    int *role = plan.data_ptr<int>(), *rang = role + T;
    auto stream = at::cuda::getCurrentCUDAStream();
    kv_canal_plan_kernel<<<(T + 127) / 128, 128, 0, stream>>>(
        sl.data_ptr<long>(), T, (int)bs, tampon_de.data_ptr<int>(),
        libres.data_ptr<int>(), sommet.data_ptr<int>(),
        sc.data_ptr<unsigned char>(), H, D, role, rang);
    dim3 grid(T, H);
    const int th = std::min(256, (D + 31) / 32 * 32);
    kv_canal_write_kernel<<<grid, th, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(kk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(vv.data_ptr()),
        sl.data_ptr<long>(), role,
        reinterpret_cast<signed char *>(kc.data_ptr()),
        reinterpret_cast<signed char *>(vc.data_ptr()),
        reinterpret_cast<__half *>(ks.data_ptr()),
        reinterpret_cast<__half *>(vs.data_ptr()),
        sc.data_ptr<unsigned char>(),
        reinterpret_cast<__nv_bfloat16 *>(tampon.data_ptr()),
        tampon_de.data_ptr<int>(), H, D, (int)bs);
    kv_canal_close_kernel<<<grid, th, 0, stream>>>(
        sl.data_ptr<long>(), role, rang,
        reinterpret_cast<signed char *>(kc.data_ptr()),
        reinterpret_cast<__half *>(ks.data_ptr()),
        sc.data_ptr<unsigned char>(),
        reinterpret_cast<const __nv_bfloat16 *>(tampon.data_ptr()),
        tampon_de.data_ptr<int>(), libres.data_ptr<int>(), sommet.data_ptr<int>(),
        H, D, (int)bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__device__ __forceinline__ float moe_vers_f32(float v) { return v; }
__device__ __forceinline__ float moe_vers_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

// Réduction pondérée du MoE : les top_k lignes d'un jeton, multipliées par
// leur poids de routage et sommées. Le chemin PyTorch demandait une
// multiplication, une réduction et une conversion — trois lancements par
// couche pour quelques kilooctets.
// Pièce 63 (23/09) : `d` en fp32 (GEMV) ou en bf16 (GEMM Marlin du chemin tensor) — la conversion
// bf16 → fp32 est exacte et faite en registre, même ordre de somme : au bit contre `d.float()` + fp32.
template <typename DT>
__global__ void moe_reduce_kernel(const DT *__restrict__ d,
                                  const float *__restrict__ topw,
                                  __nv_bfloat16 *__restrict__ y,
                                  int M, int k) {
    const int t = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;
    float s = 0.f;
    for (int e = 0; e < k; ++e)
        s += topw[t * k + e] * moe_vers_f32(d[((long)t * k + e) * M + col]);
    y[(long)t * M + col] = __float2bfloat16(s);
}

torch::Tensor moe_reduce(torch::Tensor d, torch::Tensor topw, int64_t k) {
    CHECK_CUDA(d); ACVRAM_DEVICE_GUARD(d);
    CHECK_CONTIG(d); CHECK_CONTIG(topw);
    TORCH_CHECK(d.scalar_type() == torch::kFloat || d.scalar_type() == torch::kBFloat16,
                "moe_reduce : sorties fp32 ou bf16");
    const int M = d.size(1), T = d.size(0) / (int)k;
    auto y = torch::empty({T, M}, d.options().dtype(torch::kBFloat16));
    dim3 grid((M + 255) / 256, T);
    if (d.scalar_type() == torch::kFloat)
        moe_reduce_kernel<float><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            d.data_ptr<float>(), topw.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), M, (int)k);
    else
        moe_reduce_kernel<__nv_bfloat16><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(d.data_ptr()), topw.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), M, (int)k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Pièce 63 (23/09) : aligneur du chemin tensor en UN lancement, un bloc — remplace `eid.clamp` +
// les deux noyaux `moe_align_block_size` de vLLM (0,7 + 2,6 + 0,8 µs par couche mesurés en service).
// Entrées : eid [G] int32 (expert de chaque paire, −1 = fantôme → expert 0 sur une ligne nulle, poids 0
// au reduce). Sorties (tampons fixes, capturables) : sorted_ids [P] (paires groupées par expert,
// rembourrées par bloc à la sentinelle G, puis G jusqu à P), expert_ids [P/bloc] (−1 au-delà de
// num_post), num_post [1]. Déterministe : la place d une paire est off[e] + son rang parmi les paires
// de même expert (aucun atomique de dispersion) — l ordre à l intérieur d un bloc n influe pas sur la
// sortie Marlin (chaque ligne accumule seule sur K), mais un tri stable rend les tampons comparables.
__global__ void moe_aligner_petit_kernel(const int *__restrict__ eid, int G, int E, int bloc, int P,
                                         int *__restrict__ sorted_ids, int *__restrict__ expert_ids,
                                         int *__restrict__ num_post) {
    // v2 (pièce 63/2) : préfixe par balayage de warp (plus de boucle sérielle sur E), eid en mémoire
    // partagée, rembourrage écrit sur les seules positions libres (3 barrières au lieu de 5).
    extern __shared__ int sh_aligneur[];
    int *cnt = sh_aligneur;           // [E] paires par expert
    int *off = sh_aligneur + E;       // [E] début (rembourré) de chaque expert
    int *es = sh_aligneur + 2 * E;    // [G] expert de chaque paire, fantômes ramenés à 0
    for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < G; i += blockDim.x) { const int e = max(eid[i], 0); es[i] = e; atomicAdd(&cnt[e], 1); }
    __syncthreads();
    if (threadIdx.x < 32) {                       // warp 0 : préfixe exclusif des tailles rembourrées
        const int par_lane = (E + 31) / 32;
        const int e0 = threadIdx.x * par_lane;
        int local = 0;
        for (int e = e0; e < min(e0 + par_lane, E); ++e) local += ((cnt[e] + bloc - 1) / bloc) * bloc;
        int inclus = local;
        for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, inclus, d); if ((int)threadIdx.x >= d) inclus += v; }
        int acc = inclus - local;
        for (int e = e0; e < min(e0 + par_lane, E); ++e) { off[e] = acc; acc += ((cnt[e] + bloc - 1) / bloc) * bloc; }
        if (threadIdx.x == 31) *num_post = inclus;
    }
    __syncthreads();
    const int total = *num_post;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {          // par expert : experts par bloc et rembourrage
        const int fin = off[e] + ((cnt[e] + bloc - 1) / bloc) * bloc;
        for (int p = off[e]; p < fin; p += bloc) expert_ids[p / bloc] = e;
        for (int p = off[e] + cnt[e]; p < fin; ++p) sorted_ids[p] = G;
    }
    for (int p = total + threadIdx.x; p < P; p += blockDim.x) sorted_ids[p] = G;
    for (int b = total / bloc + threadIdx.x; b < P / bloc; b += blockDim.x) expert_ids[b] = -1;
    for (int i = threadIdx.x; i < G; i += blockDim.x) {          // dispersion stable : rang parmi le même expert
        const int e = es[i];
        int rang = 0;
        for (int j = 0; j < i; ++j) rang += (es[j] == e);
        sorted_ids[off[e] + rang] = i;
    }
}

void moe_aligner_petit(torch::Tensor eid, int64_t E, int64_t bloc,
                       torch::Tensor sorted_ids, torch::Tensor expert_ids, torch::Tensor num_post) {
    CHECK_CUDA(eid); ACVRAM_DEVICE_GUARD(eid);
    CHECK_CONTIG(eid); CHECK_CONTIG(sorted_ids); CHECK_CONTIG(expert_ids);
    TORCH_CHECK(eid.scalar_type() == torch::kInt && sorted_ids.scalar_type() == torch::kInt
                && expert_ids.scalar_type() == torch::kInt && num_post.scalar_type() == torch::kInt,
                "moe_aligner_petit : int32 partout");
    const int G = (int)eid.numel(), P = (int)sorted_ids.numel();
    TORCH_CHECK(P % bloc == 0 && P >= G + (E - 1) * (bloc - 1) && expert_ids.numel() * bloc >= P,
                "moe_aligner_petit : tampons trop petits");
    TORCH_CHECK(E <= 4096, "moe_aligner_petit : E > 4096");
    const size_t shm = (2 * (size_t)E + (size_t)G) * sizeof(int);
    moe_aligner_petit_kernel<<<1, 256, shm, at::cuda::getCurrentCUDAStream()>>>(
        eid.data_ptr<int>(), G, (int)E, (int)bloc, P,
        sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(), num_post.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --------------------------------------------------------------------------
// Glue du prefill MoE (jetons triés par expert), deux lancements au lieu de
// douze. Le profil du 13/09 (revue/mma-fp4-native-sm120.md) donnait 25 % du
// pas à cette glue : conversions bf16→fp32 de [G, M] entiers, produit,
// rembourrage, permutation inverse, somme, reconversion — chacune une passe
// sur 25 à 50 Mo. Ici tout se fait en registres, en fp32, depuis les vues
// bf16 [:, :m] à foulée Mp, sans intermédiaire.
// --------------------------------------------------------------------------

// act(g) * u pour les colonnes < m, zéro au-delà jusqu'à Kd (rembourrage de
// l'entrée de down_proj). act 0 = SiLU, 1 = GELU-tanh — le même calcul fp32
// que F.silu / F.gelu(approximate="tanh") sur g.to(float32).
__global__ void moe_act_kernel(const __nv_bfloat16 *__restrict__ g,
                               const __nv_bfloat16 *__restrict__ u,
                               __nv_bfloat16 *__restrict__ out,
                               long n, int Mp, int m, int Kd, int act,
                               const __nv_bfloat16 *__restrict__ awq, const int *__restrict__ e_sorted,
                               const float *__restrict__ gsg, const float *__restrict__ gsu) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long r = i / Kd;
    const int c = (int)(i - r * Kd);
    float v = 0.f;
    if (c < m) {
        float x = __bfloat162float(g[r * Mp + c]);
        float y = __bfloat162float(u[r * Mp + c]);
        if (gsg) {
            // Pièce 71 bis : gate·up en UNE GEMM Marlin (w13, échelle globale 1) — les échelles globales par
            // expert de gate et d up, distinctes (12/12 sur l alias officiel), s appliquent ici, avant l activation.
            const int e = e_sorted ? max(e_sorted[r], 0) : 0;
            x *= gsg[e]; y *= gsu[e];
        }
        float a;
        if (act == 1) {
            // tanh en double : sous --use_fast_math, tanhf est l'approximation
            // MUFU et 0,2 % des sorties bf16 différaient de F.gelu d'un ulp ou
            // deux. Le noyau est borné par la mémoire, le double ne coûte rien.
            // même suite d'opérations fp32 que F.gelu(approximate="tanh")
            const float kb = 0.7978845608028654f, kk = 0.044715f;
            const float inner = kb * (x + kk * x * x * x);
            a = 0.5f * x * (1.0f + (float)tanh((double)inner));
        } else {
            a = x / (1.f + expf(-x));
        }
        v = a * y;
        if (awq) {
            // échelle AWQ par expert de down_proj : même suite qu'en boucle,
            // bf16(act) / bf16(s[e]) puis arrondi bf16 (ChannelScaler.apply)
            const float s = __bfloat162float(awq[(long)e_sorted[r] * Kd + c]);
            v = __bfloat162float(__float2bfloat16(v)) / s;
        }
    }
    out[i] = __float2bfloat16(v);
}

torch::Tensor moe_act(torch::Tensor g, torch::Tensor u, int64_t m, int64_t Kd, int64_t act,
                      c10::optional<torch::Tensor> awq, c10::optional<torch::Tensor> e_sorted,
                      c10::optional<torch::Tensor> gs_gate, c10::optional<torch::Tensor> gs_up) {
    CHECK_CUDA(g); CHECK_CUDA(u); ACVRAM_DEVICE_GUARD(g);
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16 && u.scalar_type() == torch::kBFloat16,
                "moe_act : g et u en bf16");
    TORCH_CHECK(g.dim() == 2 && u.sizes() == g.sizes() && g.stride(1) == 1 && u.stride(0) == g.stride(0),
                "moe_act : g et u [G, .] de meme forme et meme foulee");
    TORCH_CHECK(m <= g.size(1) && m <= Kd, "moe_act : m <= largeur et m <= Kd");
    const long G = g.size(0);
    auto out = torch::empty({G, Kd}, g.options());
    const long n = G * Kd;
    if (n == 0) return out;
    const int th = 256;
    moe_act_kernel<<<(unsigned)((n + th - 1) / th), th, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(g.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(u.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()),
        n, (int)g.stride(0), (int)m, (int)Kd, (int)act,
        awq.has_value() ? reinterpret_cast<const __nv_bfloat16 *>(awq->data_ptr()) : nullptr,
        e_sorted.has_value() ? e_sorted->data_ptr<int>() : nullptr,
        gs_gate.has_value() ? gs_gate->data_ptr<float>() : nullptr,
        gs_up.has_value() ? gs_up->data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// y[t, c] = somme_j topw[t*k + j] * d[inv[t*k + j], c] : d est en ordre trié
// par expert (foulée Mp, m colonnes utiles), inv donne la ligne de chaque
// emplacement (jeton, j). Somme en fp32 dans l'ordre j = 0..k-1, sortie bf16.
__global__ void moe_reduce_trie_kernel(const __nv_bfloat16 *__restrict__ d, int Mp,
                                       const float *__restrict__ topw,
                                       const int *__restrict__ inv,
                                       __nv_bfloat16 *__restrict__ y, int m, int k) {
    const int t = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= m) return;
    float s = 0.f;
    for (int j = 0; j < k; ++j) {
        const long row = inv[t * k + j];
        s += topw[t * k + j] * __bfloat162float(d[row * Mp + col]);
    }
    y[(long)t * m + col] = __float2bfloat16(s);
}

torch::Tensor moe_reduce_trie(torch::Tensor d, torch::Tensor topw, torch::Tensor inv,
                              int64_t m, int64_t k) {
    CHECK_CUDA(d); ACVRAM_DEVICE_GUARD(d); CHECK_CONTIG(topw); CHECK_CONTIG(inv);
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 && d.dim() == 2 && d.stride(1) == 1,
                "moe_reduce_trie : d bf16 [G, .] a foulee de ligne");
    TORCH_CHECK(topw.scalar_type() == torch::kFloat && inv.scalar_type() == torch::kInt,
                "moe_reduce_trie : topw fp32, inv int32");
    TORCH_CHECK(topw.numel() == inv.numel() && topw.numel() % k == 0, "moe_reduce_trie : t*k emplacements");
    TORCH_CHECK(m <= d.size(1), "moe_reduce_trie : m <= largeur de d");
    const int T = (int)(topw.numel() / k);
    auto y = torch::empty({T, m}, d.options());
    if (T == 0) return y;
    dim3 grid((unsigned)((m + 255) / 256), T);
    moe_reduce_trie_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(d.data_ptr()), (int)d.stride(0),
        topw.data_ptr<float>(), inv.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), (int)m, (int)k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


// -------------------------------------------------------------------------
// Routage + rassemblement (route+pack) du MoE en UN lancement — frontend du
// chemin MMA au decodage (sage-reprise-15-09-b.md § 2). Remplace, par couche,
// ~40 lancements torch (where, argsort, scatter_add_, _tuiles : searchsorted /
// cumsum / arange / clamp, index gather, pad, inv) mesures a 22 us sous ncu.
//
// Contrat : exactement ce que rendait le torch (bit a bit) :
//   ordre  = argsort STABLE de flat_e (fantomes topi<0 -> expert 0, poids 0)
//   inv    = permutation inverse (inv[ordre[p]] = p)
//   tw     = poids fp32 dans l'ordre plat, 0 pour les fantomes
//   cnt    = histogramme des experts [E]
//   tuiles = MoEBlock._tuiles(cnt, bt, t_max) : (tile_e, t0, n) sur une grille
//            FIXE de t_max tuiles, n=0 au-dela (jamais e hors [0,E)) —
//            tile_e = DERNIER expert dont base <= slot (searchsorted right - 1)
//   xs     = x[jeton(ordre[p])] en bf16, rembourre a Hpad de zeros
// G = t*k <= 1024 (un bloc recompte tout en shared : 96 paires a b=12, 128 en
// godet 16 ; ce n'est pas un noyau de prefill). Grille : G blocs, chaque bloc
// recalcule le tri par comptage (trivial) et copie SA ligne ; le bloc 0 ecrit
// en plus cnt, inv, tw et les tuiles.
constexpr int RP_GMAX = 1024;
constexpr int RP_FILS = 256;
__device__ __forceinline__ float rp_to_float(float v) { return v; }
__device__ __forceinline__ float rp_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename XT, typename IT>
__global__ void __launch_bounds__(RP_FILS) moe_route_pack_kernel(
    const IT *__restrict__ topi, const float *__restrict__ topw,
    const XT *__restrict__ x, int H, int Hpad,
    int G, int t, int k, int E, int bt, int t_max,
    int *__restrict__ ordre, int *__restrict__ inv, float *__restrict__ tw,
    int *__restrict__ cnt, int *__restrict__ tile_e, int *__restrict__ tile_t0,
    int *__restrict__ tile_n, __nv_bfloat16 *__restrict__ xs,
    const __nv_bfloat16 *__restrict__ awq, int *__restrict__ e_sorted,
    const __nv_bfloat16 *__restrict__ awq2, __nv_bfloat16 *__restrict__ xs2) {
    __shared__ int s_e[RP_GMAX];
    __shared__ int s_cnt[1024];          // E <= 1024
    __shared__ int s_start[1024];
    __shared__ int s_base[1024];
    __shared__ int s_pos[RP_GMAX];
    const int tid = threadIdx.x;
    for (int e = tid; e < E; e += RP_FILS) s_cnt[e] = 0;
    __syncthreads();
    for (int f = tid; f < G; f += RP_FILS) {
        const int e = (int)topi[f];
        s_e[f] = e < 0 ? 0 : e;
    }
    __syncthreads();
    // comptage (l'ordre des atomiques n'importe pas : on ne lit que le total)
    for (int f = tid; f < G; f += RP_FILS) atomicAdd(&s_cnt[s_e[f]], 1);
    __syncthreads();
    // starts = cumsum exclusif de cnt ; base = cumsum exclusif de ceil(cnt/bt)
    if (tid == 0) {
        int acc = 0, accb = 0;
        for (int e = 0; e < E; ++e) {
            s_start[e] = acc; acc += s_cnt[e];
            s_base[e] = accb; accb += (s_cnt[e] + bt - 1) / bt;
        }
    }
    __syncthreads();
    // position triee de chaque paire : start[e] + rang parmi les paires de meme
    // expert d'indice plat inferieur (= argsort stable)
    for (int f = tid; f < G; f += RP_FILS) {
        const int e = s_e[f];
        int rang = 0;
        for (int g = 0; g < f; ++g) rang += (s_e[g] == e);
        s_pos[f] = s_start[e] + rang;
    }
    __syncthreads();
    // ce bloc copie la ligne triee p = blockIdx.x : trouver f tel que pos[f] == p
    const int p = blockIdx.x;
    int f_src = -1;
    for (int f = tid; f < G; f += RP_FILS) if (s_pos[f] == p) f_src = f;
    __shared__ int s_f;
    if (f_src >= 0) s_f = f_src;
    __syncthreads();
    const int f = s_f;
    const int jeton = f / k;
    const int e_p = s_e[f];
    // ligne bf16 : H elements de x, puis des zeros jusqu'a Hpad ; echelle AWQ
    // par expert : bf16(x) / bf16(s[e]) en fp32 puis bf16 = ce que fait
    // scaler.apply (x / s en bf16) dans la boucle par expert
    const XT *src = x + (long)jeton * H;
    __nv_bfloat16 *dst = xs + (long)p * Hpad;
    const __nv_bfloat16 *sc = awq ? awq + (long)e_p * Hpad : nullptr;
    // gate et up a echelles distinctes (awq2) : seconde ligne xs2, meme x
    const __nv_bfloat16 *sc2 = awq2 ? awq2 + (long)e_p * Hpad : nullptr;
    __nv_bfloat16 *dst2 = xs2 ? xs2 + (long)p * Hpad : nullptr;
    for (int i = tid; i < Hpad; i += RP_FILS) {
        const float x0 = i < H ? rp_to_float(src[i]) : 0.f;
        float v = x0;
        if (sc) v = __bfloat162float(__float2bfloat16(v)) / __bfloat162float(sc[i]);
        dst[i] = __float2bfloat16(v);
        if (dst2) {
            float v2 = x0;
            if (sc2) v2 = __bfloat162float(__float2bfloat16(v2)) / __bfloat162float(sc2[i]);
            dst2[i] = __float2bfloat16(v2);
        }
    }
    if (tid == 0) { ordre[p] = f; e_sorted[p] = e_p; }
    if (blockIdx.x == 0) {
        for (int g = tid; g < G; g += RP_FILS) {
            inv[g] = s_pos[g];
            tw[g] = topi[g] < 0 ? 0.f : topw[g];
        }
        for (int e = tid; e < E; e += RP_FILS) cnt[e] = s_cnt[e];
        for (int slot = tid; slot < t_max; slot += RP_FILS) {
            // dernier expert dont base <= slot, borne a [0, E-1]
            int e = 0;
            for (int q = 0; q < E; ++q) if (s_base[q] <= slot) e = q;
            const int idx = slot - s_base[e];
            int n = s_cnt[e] - idx * bt;
            n = n < 0 ? 0 : (n > bt ? bt : n);
            tile_e[slot] = e;
            tile_t0[slot] = s_start[e] + idx * bt;
            tile_n[slot] = n;
        }
    }
}

std::vector<torch::Tensor> moe_route_pack(torch::Tensor topi, torch::Tensor topw,
                                          torch::Tensor x, int64_t E, int64_t bt,
                                          int64_t t_max, int64_t Hpad,
                                          c10::optional<torch::Tensor> awq,
                                          c10::optional<torch::Tensor> awq2) {
    CHECK_CUDA(topi); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(x);
    CHECK_CONTIG(topi); CHECK_CONTIG(topw); CHECK_CONTIG(x);
    TORCH_CHECK(topi.dim() == 2 && topw.sizes() == topi.sizes(), "moe_route_pack : topi/topw [t, k]");
    TORCH_CHECK(topi.scalar_type() == torch::kInt || topi.scalar_type() == torch::kLong, "moe_route_pack : topi int");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == topi.size(0), "moe_route_pack : x [t, H]");
    TORCH_CHECK(E <= 1024 && bt >= 1 && Hpad >= x.size(1), "moe_route_pack : E <= 1024, bt >= 1, Hpad >= H");
    const int t = (int)topi.size(0), k = (int)topi.size(1), G = t * k;
    TORCH_CHECK(G >= 1 && G <= RP_GMAX, "moe_route_pack : 1 <= t*k <= 1024 (decodage)");
    // Pas de conversion de type (chaque .to() est un lancement de plus par couche —
    // le test compte les lancements) : topi int32 ou int64, x bf16 ou fp32 en direct.
    TORCH_CHECK(topw.scalar_type() == torch::kFloat, "moe_route_pack : topw fp32 (tel que sorti du routage)");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat, "moe_route_pack : x bf16 ou fp32");
    auto oi = torch::empty({G}, topi.options().dtype(torch::kInt));
    auto ordre = torch::empty({G}, oi.options()), inv = torch::empty({G}, oi.options());
    auto tw = torch::empty({G}, topw.options());
    auto cnt = torch::empty({E}, oi.options());
    auto te = torch::empty({t_max}, oi.options()), t0 = torch::empty({t_max}, oi.options()),
         tn = torch::empty({t_max}, oi.options());
    auto xs = torch::empty({G, Hpad}, x.options().dtype(torch::kBFloat16));
    auto es = torch::empty({G}, oi.options());
    const __nv_bfloat16 *pawq = nullptr;
    if (awq.has_value()) {
        TORCH_CHECK(awq->scalar_type() == torch::kBFloat16 && awq->dim() == 2 && awq->size(0) == E && awq->size(1) == Hpad,
                    "moe_route_pack : table AWQ [E, Hpad] bf16");
        CHECK_CONTIG(*awq);
        pawq = reinterpret_cast<const __nv_bfloat16 *>(awq->data_ptr());
    }
    // awq2 : echelle d'up distincte de celle de gate -> seconde ligne xs2 ;
    // sinon xs2 EST xs (un seul rassemblement, une seule quantification)
    const __nv_bfloat16 *pawq2 = nullptr; __nv_bfloat16 *pxs2 = nullptr;
    torch::Tensor xs2 = xs;
    if (awq2.has_value()) {
        TORCH_CHECK(awq2->scalar_type() == torch::kBFloat16 && awq2->dim() == 2 && awq2->size(0) == E && awq2->size(1) == Hpad,
                    "moe_route_pack : table AWQ up [E, Hpad] bf16");
        CHECK_CONTIG(*awq2);
        pawq2 = reinterpret_cast<const __nv_bfloat16 *>(awq2->data_ptr());
        xs2 = torch::empty({G, Hpad}, xs.options());
        pxs2 = reinterpret_cast<__nv_bfloat16 *>(xs2.data_ptr());
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    #define RP_LANCE(XT, IT, PX) moe_route_pack_kernel<XT, IT><<<G, RP_FILS, 0, stream>>>( \
            topi.data_ptr<IT>(), topw.data_ptr<float>(), PX, (int)x.size(1), (int)Hpad, \
            G, t, k, (int)E, (int)bt, (int)t_max, ordre.data_ptr<int>(), inv.data_ptr<int>(), \
            tw.data_ptr<float>(), cnt.data_ptr<int>(), te.data_ptr<int>(), t0.data_ptr<int>(), \
            tn.data_ptr<int>(), reinterpret_cast<__nv_bfloat16 *>(xs.data_ptr()), pawq, es.data_ptr<int>(), \
            pawq2, pxs2)
    const bool bf = x.scalar_type() == torch::kBFloat16, i64 = topi.scalar_type() == torch::kLong;
    if (bf && i64) RP_LANCE(__nv_bfloat16, int64_t, reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()));
    else if (bf) RP_LANCE(__nv_bfloat16, int, reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()));
    else if (i64) RP_LANCE(float, int64_t, x.data_ptr<float>());
    else RP_LANCE(float, int, x.data_ptr<float>());
    #undef RP_LANCE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {xs, ordre, inv, tw, cnt, te, t0, tn, es, xs2};
}


// -------------------------------------------------------------------------
// MoE FUSIONNÉ au décodage (port de b12x, plan revue/p1-flashinfer-b12x-15-09.md,
// décisions de Sage sage-reprise-15-09-b § 4) : un CTA = (tranche
// d'intermédiaire de TN colonnes, tuile de 16 jetons d'un expert), WARPS =
// TN/16 warps (un warp = 16 lignes de poids = un bloc de quantification).
// Phase 1 : gate et up de la tranche dans la même boucle K (fragments A
// partagés, poids lus une fois), pipeline cp.async S étages × KS ; épilogue :
// g, u arrondis en bf16 (comme la GEMM groupée), SiLU(g)·u en fp32 (formule de
// moe_act), arrondi bf16, quantifié E2M1 bloc 16 + UE4M3 (formule de
// nvfp4_quant_act) DANS la mémoire partagée — bit-identique au chemin B jusque
// là (tests/test_moe_fused.py). Phase 2 : FC2 balaie les M_out/128 tuiles de
// sortie avec la tranche (K = TN) ; SPLIT-K SÉRIEL, sans atomique sur les
// données : chaque CTA écrit son partiel fp32 [16 × 128] dans ws, incrémente
// le compteur de sa (tuile, tuile N) ; le DERNIER arrivé somme les NS partiels
// dans l'ordre fixe des tranches (bit-reproductible d'un pas à l'autre),
// applique gs_down et écrit d [G, M_out] bf16 — que moe_reduce_trie réduit par
// jeton comme sur le chemin B. L'intermédiaire ne touche jamais la globale.
template <int S, int KS, int TN, bool ATOM>
__global__ void __launch_bounds__(TN * 2) nvfp4_moe_fused_kernel(
    const float *__restrict__ gs_g, const float *__restrict__ gs_u, const float *__restrict__ gs_d,
    const unsigned char *__restrict__ xq, const unsigned char *__restrict__ xsf,
    const int *__restrict__ tile_e, const int *__restrict__ tile_t0, const int *__restrict__ tile_n,
    const int64_t *__restrict__ tq_g, const int64_t *__restrict__ tb_g,
    const int64_t *__restrict__ tq_u, const int64_t *__restrict__ tb_u,
    const int64_t *__restrict__ tq_d, const int64_t *__restrict__ tb_d,
    float *__restrict__ ws, int *__restrict__ compteurs,
    __nv_bfloat16 *__restrict__ d, const int *__restrict__ ordre, const float *__restrict__ tw,
    float *__restrict__ y32, int k_top, int K, int I, int M_out, int act,
    const float *__restrict__ grow) {
#ifdef ACVRAM_MMA_FP4
    constexpr int WARPS = TN / 16, FILS = WARPS * 32;
    static_assert(TN == 64 || TN == 128, "TN : 64 ou 128");
    constexpr int LD = KS / 2 + 16, NM = KS / GM_KB, SB = KS / 16, CH = KS / 32;
    constexpr int A_OCT = 16 * LD, B_OCT = TN * LD, SA_OCT = 16 * SB, SB_OCT = TN * SB;
    constexpr int ETAGE = A_OCT + 2 * B_OCT + SA_OCT + 2 * SB_OCT;
    constexpr int LD2 = TN / 2 + 16, SB2 = TN / 16, NM2 = TN / GM_KB;
    constexpr int B2_OCT = 128 * LD2 + 128 * SB2, A2_OCT = 16 * LD2 + 16 * SB2;
    constexpr int NF2 = 128 / WARPS / 8;               // tuiles n8 par warp en phase 2
    constexpr int S2 = 3;
    extern __shared__ __align__(16) unsigned char smem_mf[];
    unsigned char *smem = smem_mf;
    const int tile = blockIdx.y, sl = blockIdx.x, NS = gridDim.x;
    const int e = tile_e[tile], t0 = tile_t0[tile], nt = tile_n[tile];
    if (nt <= 0) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const long half_k = (long)K >> 1; const int nblk = K >> 4, KT = K / KS;
    const unsigned char *qg = reinterpret_cast<const unsigned char *>(tq_g[e]);
    const unsigned char *bg = reinterpret_cast<const unsigned char *>(tb_g[e]);
    const unsigned char *qu = reinterpret_cast<const unsigned char *>(tq_u[e]);
    const unsigned char *bu = reinterpret_cast<const unsigned char *>(tb_u[e]);
    const int row0 = sl * TN;
    auto sA  = [&](int st) { return smem + st * ETAGE; };
    auto sBg = [&](int st) { return smem + st * ETAGE + A_OCT; };
    auto sBu = [&](int st) { return smem + st * ETAGE + A_OCT + B_OCT; };
    auto sSA = [&](int st) { return smem + st * ETAGE + A_OCT + 2 * B_OCT; };
    auto sSBg = [&](int st) { return smem + st * ETAGE + A_OCT + 2 * B_OCT + SA_OCT; };
    auto sSBu = [&](int st) { return smem + st * ETAGE + A_OCT + 2 * B_OCT + SA_OCT + SB_OCT; };
    auto emettre = [&](int st, int k0) {
        const long kb = k0 >> 1; const int ks = k0 >> 4;
        for (int c = tid; c < CH * 16; c += FILS) {
            const int r = c / CH, h = c % CH;
            cp_async16(sA(st) + r * LD + 16 * h, xq + (long)(t0 + min(r, nt - 1)) * half_k + kb + 16 * h);
        }
        for (int c = tid; c < CH * TN; c += FILS) {
            const int r = c / CH, h = c % CH;
            const long src = (long)(row0 + r) * half_k + kb + 16 * h;
            cp_async16(sBg(st) + r * LD + 16 * h, qg + src);
            cp_async16(sBu(st) + r * LD + 16 * h, qu + src);
        }
        if (tid < 16) {
            const unsigned char *src = xsf + (long)(t0 + min(tid, nt - 1)) * nblk + ks;
            if constexpr (SB == 4) cp_async4(sSA(st) + tid * SB, src); else cp_async8(sSA(st) + tid * SB, src);
        } else if (tid < 16 + TN) {
            const int r = tid - 16;
            const unsigned char *sg = bg + (long)(row0 + r) * nblk + ks, *su = bu + (long)(row0 + r) * nblk + ks;
            if constexpr (SB == 4) { cp_async4(sSBg(st) + r * SB, sg); cp_async4(sSBu(st) + r * SB, su); }
            else { cp_async8(sSBg(st) + r * SB, sg); cp_async8(sSBu(st) + r * SB, su); }
        }
    };
    float accg[2][4], accu[2][4];
    #pragma unroll
    for (int nf = 0; nf < 2; ++nf)
        #pragma unroll
        for (int q = 0; q < 4; ++q) { accg[nf][q] = 0.f; accu[nf][q] = 0.f; }
    #pragma unroll
    for (int s = 0; s < S - 1; ++s) { if (s < KT) emettre(s, s * KS); cp_async_commit(); }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        { const int kn = kt + S - 1; if (kn < KT) emettre(kn % S, kn * KS); cp_async_commit(); }
        const int st = kt % S;
        #pragma unroll
        for (int m = 0; m < NM; ++m) {
            const int ko = 32 * m;
            unsigned a[4], sfa, bgf[2][2], buf[2][2], sfbg[2], sfbu[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = g + 8 * h;
                const unsigned char *pa = sA(st) + j * LD + ko + 4 * tq;
                const unsigned lo = *reinterpret_cast<const unsigned *>(pa), hi = *reinterpret_cast<const unsigned *>(pa + 16);
                a[h] = (j < nt) ? lo : 0u; a[h + 2] = (j < nt) ? hi : 0u;
            }
            { const int js = (lane >> 2) + 8 * (lane & 1);
              const unsigned sv = *reinterpret_cast<const unsigned *>(sSA(st) + js * SB + 4 * m);
              sfa = (js < nt) ? sv : 0u; }
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                const int r = warp * 16 + nf * 8 + g;
                const unsigned char *pg_ = sBg(st) + r * LD + ko + 4 * tq, *pu_ = sBu(st) + r * LD + ko + 4 * tq;
                bgf[nf][0] = *reinterpret_cast<const unsigned *>(pg_); bgf[nf][1] = *reinterpret_cast<const unsigned *>(pg_ + 16);
                buf[nf][0] = *reinterpret_cast<const unsigned *>(pu_); buf[nf][1] = *reinterpret_cast<const unsigned *>(pu_ + 16);
                sfbg[nf] = *reinterpret_cast<const unsigned *>(sSBg(st) + r * SB + 4 * m);
                sfbu[nf] = *reinterpret_cast<const unsigned *>(sSBu(st) + r * SB + 4 * m);
            }
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                mma_mxf4nvf4(accg[nf], a, bgf[nf], sfa, sfbg[nf]);
                mma_mxf4nvf4(accu[nf], a, buf[nf], sfa, sfbu[nf]);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    // --- épilogue FC1 : bf16(g·gs), bf16(u·gs) -> act bf16 -> E2M1 bloc 16 (= ce warp) en shared
    unsigned char *A2 = smem, *SA2 = smem + 16 * LD2;
    const float gsg0 = gs_g[e], gsu0 = gs_u[e];
    float v[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        // échelle globale par ligne d'activation (grow de nvfp4_quant_act) ;
        // l'activation, elle, est requantifiée ci-dessous sans échelle globale
        // (chemin témoin, plancher E4M3 non corrigé — ACVRAM_MOE_DECODE_FUSED=0)
        const float gr = grow ? grow[t0 + min(g + 8 * h, nt - 1)] : 1.f;
        const float gsg = gsg0 * gr, gsu = gsu0 * gr;
        #pragma unroll
        for (int nf = 0; nf < 2; ++nf)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const float gg = __bfloat162float(__float2bfloat16(accg[nf][2 * h + c] * gsg));
                const float uu = __bfloat162float(__float2bfloat16(accu[nf][2 * h + c] * gsu));
                float a_;
                if (act == 1) {
                    const float kb = 0.7978845608028654f, kk = 0.044715f;
                    const float inner = kb * (gg + kk * gg * gg * gg);
                    a_ = 0.5f * gg * (1.0f + (float)tanh((double)inner));
                } else {
                    a_ = gg / (1.f + expf(-gg));
                }
                v[h][nf * 2 + c] = __bfloat162float(__float2bfloat16(a_ * uu));
            }
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(v[h][i]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        unsigned char sbits = 0; float sdec = 0.f;
        if (amax > 0.f) {
            const float sc = fminf(__fdiv_rn(amax, 6.f), 448.f);
            sbits = (unsigned char)__nv_cvt_float_to_fp8(sc, __NV_SATFINITE, __NV_E4M3);
            sdec = e4m3_to_float(sbits);
        }
        unsigned char q0 = 0, q1 = 0;
        if (sdec > 0.f) {
            const float2 p0 = make_float2(__fdiv_rn(v[h][0], sdec), __fdiv_rn(v[h][1], sdec));
            const float2 p1 = make_float2(__fdiv_rn(v[h][2], sdec), __fdiv_rn(v[h][3], sdec));
            q0 = (unsigned char)(__nv_cvt_float2_to_fp4x2(p0, __NV_E2M1, cudaRoundNearest) & 0xFF);
            q1 = (unsigned char)(__nv_cvt_float2_to_fp4x2(p1, __NV_E2M1, cudaRoundNearest) & 0xFF);
        }
        const int j = g + 8 * h;
        if (j < nt) {
            A2[j * LD2 + warp * 8 + tq] = q0;
            A2[j * LD2 + warp * 8 + 4 + tq] = q1;
            if (tq == 0) SA2[j * SB2 + warp] = sbits;
        }
    }
    __syncthreads();
    // --- FC2 : partiels [16 x 128] par tuile N, split-K sériel sur les NS tranches
    const unsigned char *qd = reinterpret_cast<const unsigned char *>(tq_d[e]);
    const unsigned char *bd = reinterpret_cast<const unsigned char *>(tb_d[e]);
    const long half_i = (long)I >> 1; const int nblk_i = I >> 4;
    unsigned char *B2 = smem + A2_OCT;
    const int NT2 = M_out / 128;
    auto emettre2 = [&](int st, int n) {
        const long kb = (long)sl * (TN / 2); const int ks = sl * (TN / 16);
        unsigned char *b = B2 + st * B2_OCT, *sb = b + 128 * LD2;
        constexpr int CH2 = TN / 32;
        for (int c = tid; c < CH2 * 128; c += FILS) {
            const int r = c / CH2, h = c % CH2;
            cp_async16(b + r * LD2 + 16 * h, qd + (long)(n * 128 + r) * half_i + kb + 16 * h);
        }
        if (tid < 128) {
            const unsigned char *src = bd + (long)(n * 128 + tid) * nblk_i + ks;
            if constexpr (SB2 == 4) cp_async4(sb + tid * SB2, src); else cp_async8(sb + tid * SB2, src);
        }
    };
    unsigned a2[NM2][4], sfa2[NM2];
    #pragma unroll
    for (int m = 0; m < NM2; ++m) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j = g + 8 * h;
            const unsigned char *pa = A2 + j * LD2 + 32 * m + 4 * tq;
            const unsigned lo = *reinterpret_cast<const unsigned *>(pa), hi = *reinterpret_cast<const unsigned *>(pa + 16);
            a2[m][h] = (j < nt) ? lo : 0u; a2[m][h + 2] = (j < nt) ? hi : 0u;
        }
        const int js = (lane >> 2) + 8 * (lane & 1);
        const unsigned sv = *reinterpret_cast<const unsigned *>(SA2 + js * SB2 + 4 * m);
        sfa2[m] = (js < nt) ? sv : 0u;
    }
    const float gsd = gs_d[e];
    __shared__ int s_dernier;
    // partiel de ce CTA : ws[(tile*NS + sl)][16][M_out] fp32 — écrit tuile N par
    // tuile N (magasins coalescés), UNE clôture + UN compteur par CTA à la fin
    // (la version « clôture par tuile N » coûtait 16 __threadfence + 3
    // __syncthreads par CTA : 133 µs/couche mesurés le 15/09 contre 83 en B)
    float *part = ws + ((long)tile * NS + sl) * (16L * M_out);
    // témoin ATOMIQUES (sage-reprise-15-09-b § 4 : si le sériel coûte > 10 µs) :
    // y32[jeton, col] += partiel · gs_d · poids, pas de ws, pas de reduce_trie
    float w_[2]; int tok_[2];
    if constexpr (ATOM) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j = g + 8 * h;
            const int f = (j < nt) ? ordre[t0 + j] : 0;
            w_[h] = (j < nt) ? tw[f] * gsd : 0.f; tok_[h] = f / k_top;
        }
    }
    #pragma unroll
    for (int s = 0; s < S2 - 1; ++s) { if (s < NT2) emettre2(s, s); cp_async_commit(); }
    for (int n = 0; n < NT2; ++n) {
        cp_async_wait<S2 - 2>();
        __syncthreads();
        { const int nn = n + S2 - 1; if (nn < NT2) emettre2(nn % S2, nn); cp_async_commit(); }
        const unsigned char *b = B2 + (n % S2) * B2_OCT, *sb = b + 128 * LD2;
        float acc2[NF2][4];
        #pragma unroll
        for (int nf = 0; nf < NF2; ++nf)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc2[nf][q] = 0.f;
        #pragma unroll
        for (int m = 0; m < NM2; ++m)
            #pragma unroll
            for (int nf = 0; nf < NF2; ++nf) {
                const int r = warp * (128 / WARPS) + nf * 8 + g;
                const unsigned char *pb = b + r * LD2 + 32 * m + 4 * tq;
                unsigned bb[2] = { *reinterpret_cast<const unsigned *>(pb), *reinterpret_cast<const unsigned *>(pb + 16) };
                const unsigned sfb = *reinterpret_cast<const unsigned *>(sb + r * SB2 + 4 * m);
                mma_mxf4nvf4(acc2[nf], a2[m], bb, sfa2[m], sfb);
            }
        #pragma unroll
        for (int nf = 0; nf < NF2; ++nf) {
            const int col = n * 128 + warp * (128 / WARPS) + nf * 8 + 2 * tq;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                if constexpr (ATOM) {
                    if (w_[h] != 0.f) {
                        float *dst = y32 + (long)tok_[h] * M_out + col;
                        atomicAdd(dst, acc2[nf][2 * h] * w_[h]); atomicAdd(dst + 1, acc2[nf][2 * h + 1] * w_[h]);
                    }
                } else {
                    *reinterpret_cast<float2 *>(part + (g + 8 * h) * M_out + col) = make_float2(acc2[nf][2 * h], acc2[nf][2 * h + 1]);
                }
            }
        }
    }
    cp_async_wait<0>();
    if constexpr (!ATOM) {
        __threadfence();
        __syncthreads();
        if (tid == 0) s_dernier = (atomicAdd(&compteurs[tile], 1) == NS - 1);
        __syncthreads();
        if (s_dernier) {
            __threadfence();
            const float *base = ws + (long)tile * NS * (16L * M_out);
            for (int n = 0; n < NT2; ++n) {
                #pragma unroll
                for (int nf = 0; nf < NF2; ++nf) {
                    const int col = n * 128 + warp * (128 / WARPS) + nf * 8 + 2 * tq;
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int j = g + 8 * h;
                        float s0 = 0.f, s1 = 0.f;
                        for (int q = 0; q < NS; ++q) {
                            const float2 pv = *reinterpret_cast<const float2 *>(base + (long)q * (16L * M_out) + j * M_out + col);
                            s0 += pv.x; s1 += pv.y;
                        }
                        if (j < nt) {
                            __nv_bfloat16 *dst = d + (long)(t0 + j) * M_out + col;
                            dst[0] = __float2bfloat16(s0 * gsd); dst[1] = __float2bfloat16(s1 * gsd);
                        }
                    }
                }
            }
            if (tid == 0) compteurs[tile] = 0;
        }
    }
    cp_async_wait<0>();
#endif
}

__global__ void f32_vers_bf16_kernel(const float *__restrict__ x, __nv_bfloat16 *__restrict__ y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

torch::Tensor nvfp4_moe_fused(torch::Tensor tq_g, torch::Tensor tb_g, torch::Tensor gs_g,
                              torch::Tensor tq_u, torch::Tensor tb_u, torch::Tensor gs_u,
                              torch::Tensor tq_d, torch::Tensor tb_d, torch::Tensor gs_d,
                              torch::Tensor xq, torch::Tensor xsf,
                              torch::Tensor tile_e, torch::Tensor tile_t0, torch::Tensor tile_n,
                              torch::Tensor ws, torch::Tensor compteurs,
                              int64_t K, int64_t I, int64_t M_out, int64_t act, int64_t tn,
                              torch::Tensor ordre, torch::Tensor tw, int64_t k_top, int64_t t, bool atomique,
                              int64_t etages, c10::optional<torch::Tensor> grow) {
    CHECK_CUDA(xq); ACVRAM_DEVICE_GUARD(xq);
    const float *grow_p = nullptr;
    if (grow.has_value() && grow->defined()) {
        CHECK_CUDA(*grow); CHECK_CONTIG(*grow);
        TORCH_CHECK(grow->scalar_type() == torch::kFloat && grow->numel() == xq.size(0), "MoE fusionne : grow fp32 [G]");
        grow_p = grow->data_ptr<float>();
    }
    for (auto &z : {tq_g, tb_g, gs_g, tq_u, tb_u, gs_u, tq_d, tb_d, gs_d, xq, xsf, tile_e, tile_t0, tile_n, ws, compteurs})
        CHECK_CONTIG(z);
    TORCH_CHECK(K % 128 == 0 && I % tn == 0 && M_out % 128 == 0, "MoE fusionne : K multiple de 128, I de tn, M_out de 128");
    TORCH_CHECK(tn == 64 || tn == 128, "MoE fusionne : tn 64 ou 128");
    const int T = tile_e.size(0), G = xq.size(0), NS = (int)(I / tn), NT2 = (int)(M_out / 128);
    TORCH_CHECK(ws.numel() >= (long)T * NS * 16 * M_out && ws.scalar_type() == torch::kFloat,
                "MoE fusionne : ws fp32 de T*NS*16*M_out elements");
    TORCH_CHECK(compteurs.numel() >= (long)T && compteurs.scalar_type() == torch::kInt, "MoE fusionne : compteurs int32 T (a zero)");
    // sériel : d [G, M_out] (reduce_trie ensuite) ; atomique : y32 [t, M_out] fp32 remis a zero ici
    auto d = torch::empty({atomique ? 0 : G, M_out}, xq.options().dtype(torch::kBFloat16));
    auto y32 = atomique ? torch::zeros({t, M_out}, xq.options().dtype(torch::kFloat))
                        : torch::empty({0}, xq.options().dtype(torch::kFloat));
    if (T == 0 || G == 0) return atomique ? y32 : d;
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int KS = 128;
    TORCH_CHECK(etages == 2 || etages == 3, "MoE fusionne : etages 2 ou 3");
    #define MF_LANCE(TN, S) do { \
        constexpr int LD = KS / 2 + 16, SB = KS / 16; \
        constexpr int ETAGE = 16 * LD + 2 * TN * LD + 16 * SB + 2 * TN * SB; \
        constexpr int LD2 = TN / 2 + 16, SB2 = TN / 16; \
        constexpr int SHM2 = 16 * LD2 + 16 * SB2 + 3 * (128 * LD2 + 128 * SB2); \
        constexpr int SHM = (S * ETAGE > SHM2) ? S * ETAGE : SHM2; \
        static bool attr = false; \
        if (!attr) { cudaFuncSetAttribute(nvfp4_moe_fused_kernel<S, KS, TN, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHM); \
                     cudaFuncSetAttribute(nvfp4_moe_fused_kernel<S, KS, TN, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHM); attr = true; } \
        dim3 grid((unsigned)NS, (unsigned)T); \
        auto lance = atomique ? nvfp4_moe_fused_kernel<S, KS, TN, true> : nvfp4_moe_fused_kernel<S, KS, TN, false>; \
        lance<<<grid, TN * 2, SHM, stream>>>( \
            gs_g.data_ptr<float>(), gs_u.data_ptr<float>(), gs_d.data_ptr<float>(), \
            xq.data_ptr<unsigned char>(), xsf.data_ptr<unsigned char>(), \
            tile_e.data_ptr<int>(), tile_t0.data_ptr<int>(), tile_n.data_ptr<int>(), \
            tq_g.data_ptr<int64_t>(), tb_g.data_ptr<int64_t>(), tq_u.data_ptr<int64_t>(), tb_u.data_ptr<int64_t>(), \
            tq_d.data_ptr<int64_t>(), tb_d.data_ptr<int64_t>(), ws.data_ptr<float>(), compteurs.data_ptr<int>(), \
            reinterpret_cast<__nv_bfloat16 *>(d.data_ptr()), ordre.data_ptr<int>(), tw.data_ptr<float>(), \
            y32.data_ptr<float>(), (int)k_top, (int)K, (int)I, (int)M_out, (int)act, grow_p); } while (0)
    if (tn == 64) { if (etages == 2) MF_LANCE(64, 2); else MF_LANCE(64, 3); }
    else          { if (etages == 2) MF_LANCE(128, 2); else MF_LANCE(128, 3); }
    #undef MF_LANCE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!atomique) return d;
    auto y = torch::empty({t, M_out}, xq.options().dtype(torch::kBFloat16));
    const long n = (long)t * M_out;
    f32_vers_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
        y32.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


// -------------------------------------------------------------------------
// GEMM ÉTROIT (M <= 16 jetons) sur tensor cores bf16, poids int8 (groupes) ou
// NVFP4 (blocs de 16) déquantifiés en registres — 1aj marche 2 élargie
// (sage-reprise-15-09-b § 5) : les GEMV int8/nvfp4 à M=12 étaient à ×3 de la
// borne (18,9 Mo de projections lus en 54 µs par couche au lieu de 18 ;
// lm_head 311 Mo en 0,9 ms au lieu de 0,3) parce qu'un fil y porte une seule
// lecture de 16 octets par ligne, sans latence recouverte, et relit x par
// tranche de NV. Ici : CTA = ROWS lignes de poids x tout K en étages cp.async
// (x 16 lignes + poids), mma.sync m16n8k16 bf16, accumulation fp32 par
// groupe (int8 : (w - z) exact en bf16, x s par groupe de 128 ; nvfp4 : nibble
// -> bf16 exact, x échelle UE4M3 par bloc de 16, x échelle globale à la fin).
// Même arithmétique que les GEMV (produits exacts, sommes fp32), ordre des
// sommes différent : déterministe, pas d'atomique.
constexpr int NG_KS = 64;                            // profondeur d'un étage
constexpr int NG_S = 4;
__device__ __forceinline__ void mma_bf16_16816(float c[4], const unsigned a[4], const unsigned b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned bf16x2_pack(float lo, float hi) {
    return (unsigned)__bfloat16_as_ushort(__float2bfloat16(lo)) | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}
__device__ __forceinline__ float e2m1_val(unsigned n) {
    // 0,0.5,1,1.5,2,3,4,6 et leurs opposés
    const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = t[n & 7];
    return (n & 8) ? -v : v;
}
template <int ROWS, bool NVFP4>
__global__ void __launch_bounds__(ROWS * 4) narrow_gemm_kernel(
    const unsigned char *__restrict__ qw,      // int8 [N, K] ou nvfp4 [N, K/2]
    const unsigned char *__restrict__ sc8,     // nvfp4 : échelles UE4M3 [N, K/16]
    const __half *__restrict__ sc16,           // int8 : échelles fp16 [N, K/g]
    const unsigned char *__restrict__ zr,      // int8 : zéros [N, K/g]
    const __nv_bfloat16 *__restrict__ x,       // [M, K] bf16
    __nv_bfloat16 *__restrict__ y,             // [M, N] bf16
    int M, int N, int K, int group, float gscale) {
    constexpr int WARPS = ROWS / 8, FILS = WARPS * 32;
    constexpr int XB = 16 * NG_KS * 2;                                // x : 16 lignes x 64 bf16
    constexpr int WB = NVFP4 ? ROWS * (NG_KS / 2) : ROWS * NG_KS;    // poids d'un étage
    constexpr int SB = NVFP4 ? ROWS * (NG_KS / 16) : 0;               // échelles nvfp4 (4 o par ligne)
    constexpr int XLD = NG_KS * 2 + 16;                               // foulée x (144 o)
    constexpr int WLD = (NVFP4 ? NG_KS / 2 : NG_KS) + 16;             // foulée poids
    constexpr int ETAGE = 16 * XLD + ROWS * WLD + (NVFP4 ? ROWS * 4 : 0);
    extern __shared__ __align__(16) unsigned char smem_ng[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const int row0 = blockIdx.x * ROWS;
    const int KT = K / NG_KS;
    const long wl = NVFP4 ? (K >> 1) : K;                               // octets par ligne de poids
    auto sX = [&](int st) { return smem_ng + st * ETAGE; };
    auto sW = [&](int st) { return smem_ng + st * ETAGE + 16 * XLD; };
    auto sS = [&](int st) { return smem_ng + st * ETAGE + 16 * XLD + ROWS * WLD; };
    auto emettre = [&](int st, int k0) {
        // x : 16 lignes x 128 o (lignes >= M : ligne M-1, masquées à l'écriture)
        for (int c = tid; c < 16 * 8; c += FILS) {
            const int r = c >> 3, h = c & 7;
            cp_async16(sX(st) + r * XLD + 16 * h, reinterpret_cast<const unsigned char *>(x + (long)min(r, M - 1) * K + k0) + 16 * h);
        }
        constexpr int CH = (NVFP4 ? NG_KS / 2 : NG_KS) / 16;            // 16 o par ligne : 2 ou 4
        for (int c = tid; c < ROWS * CH; c += FILS) {
            const int r = c / CH, h = c % CH;
            const int row = min(row0 + r, N - 1);
            cp_async16(sW(st) + r * WLD + 16 * h, qw + (long)row * wl + (NVFP4 ? (k0 >> 1) : k0) + 16 * h);
        }
        if constexpr (NVFP4) {
            if (tid < ROWS) {
                const int row = min(row0 + tid, N - 1);
                cp_async4(sS(st) + tid * 4, sc8 + (long)row * (K >> 4) + (k0 >> 4));
            }
        }
    };
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float accg[4] = {0.f, 0.f, 0.f, 0.f};                       // int8 : accumulateur du groupe courant
    const int n_g = row0 + warp * 8 + g;                        // ma ligne de poids pour B
    const int n0 = row0 + warp * 8 + 2 * tq, n1 = n0 + 1;       // mes deux colonnes de C
    const int ngrp = K / group;
    #pragma unroll
    for (int s = 0; s < NG_S - 1; ++s) { if (s < KT) emettre(s, s * NG_KS); cp_async_commit(); }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<NG_S - 2>();
        __syncthreads();
        { const int kn = kt + NG_S - 1; if (kn < KT) emettre(kn % NG_S, kn * NG_KS); cp_async_commit(); }
        const int st = kt % NG_S;
        const unsigned char *xs = sX(st), *ws = sW(st);
        #pragma unroll
        for (int k16 = 0; k16 < NG_KS / 16; ++k16) {
            const int kk = k16 * 16;
            unsigned a[4];
            a[0] = *reinterpret_cast<const unsigned *>(xs + g * XLD + (kk + 2 * tq) * 2);
            a[1] = *reinterpret_cast<const unsigned *>(xs + (g + 8) * XLD + (kk + 2 * tq) * 2);
            a[2] = *reinterpret_cast<const unsigned *>(xs + g * XLD + (kk + 2 * tq + 8) * 2);
            a[3] = *reinterpret_cast<const unsigned *>(xs + (g + 8) * XLD + (kk + 2 * tq + 8) * 2);
            unsigned b[2];
            if constexpr (NVFP4) {
                // nibbles k = kk+2tq, +1 (octet (kk+2tq)/2) et kk+2tq+8, +9
                const unsigned char *wr = ws + (warp * 8 + g) * WLD + (kk >> 1);
                const unsigned c0 = wr[tq], c1 = wr[tq + 4];
                b[0] = bf16x2_pack(e2m1_val(c0 & 15), e2m1_val(c0 >> 4));
                b[1] = bf16x2_pack(e2m1_val(c1 & 15), e2m1_val(c1 >> 4));
                float cb[4] = {0.f, 0.f, 0.f, 0.f};
                mma_bf16_16816(cb, a, b);
                // échelle UE4M3 du bloc k16 pour mes deux colonnes (lignes de poids n0, n1)
                const unsigned char *se = sS(st);
                const float s0 = e4m3_to_float(se[(warp * 8 + 2 * tq) * 4 + k16]);
                const float s1 = e4m3_to_float(se[(warp * 8 + 2 * tq + 1) * 4 + k16]);
                acc[0] += cb[0] * s0; acc[1] += cb[1] * s1; acc[2] += cb[2] * s0; acc[3] += cb[3] * s1;
            } else {
                const int grp = (kt * NG_KS + kk) / group;
                const float z = (float)zr[(long)min(n_g, N - 1) * ngrp + grp];
                const unsigned char *wr = ws + (warp * 8 + g) * WLD + kk;
                b[0] = bf16x2_pack((float)wr[2 * tq] - z, (float)wr[2 * tq + 1] - z);
                b[1] = bf16x2_pack((float)wr[2 * tq + 8] - z, (float)wr[2 * tq + 9] - z);
                mma_bf16_16816(accg, a, b);
                // fin de groupe : x échelle fp16 de chaque colonne, comme int8_gemv (part * s)
                if (((kt * NG_KS + kk + 16) % group) == 0) {
                    const float s0 = __half2float(sc16[(long)min(n0, N - 1) * ngrp + grp]);
                    const float s1 = __half2float(sc16[(long)min(n1, N - 1) * ngrp + grp]);
                    acc[0] += accg[0] * s0; acc[1] += accg[1] * s1; acc[2] += accg[2] * s0; acc[3] += accg[3] * s1;
                    accg[0] = accg[1] = accg[2] = accg[3] = 0.f;
                }
            }
        }
    }
    cp_async_wait<0>();
    // C : lignes g et g+8 (jetons), colonnes n0, n1
    if (g < M) {
        if (n0 < N) y[(long)g * N + n0] = __float2bfloat16(acc[0] * gscale);
        if (n1 < N) y[(long)g * N + n1] = __float2bfloat16(acc[1] * gscale);
    }
    if (g + 8 < M) {
        if (n0 < N) y[(long)(g + 8) * N + n0] = __float2bfloat16(acc[2] * gscale);
        if (n1 < N) y[(long)(g + 8) * N + n1] = __float2bfloat16(acc[3] * gscale);
    }
}

// x [M <= 16, K] bf16 -> y [M, N] bf16. rows : lignes de poids par CTA (32 pour
// les projections — 160-256 CTA —, 128 pour le lm_head).
torch::Tensor narrow_gemm(torch::Tensor qw, c10::optional<torch::Tensor> sc8, c10::optional<torch::Tensor> sc16,
                          c10::optional<torch::Tensor> zr, torch::Tensor x, int64_t K, int64_t group,
                          double gscale, int64_t rows) {
    CHECK_CUDA(qw); CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(qw); CHECK_CONTIG(qw); CHECK_CONTIG(x);
    const bool nv = sc8.has_value();
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16 && x.size(1) == K, "narrow_gemm : x [M<=16, K] bf16");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "narrow_gemm : x bf16");
    TORCH_CHECK(K % NG_KS == 0, "narrow_gemm : K multiple de 64");
    TORCH_CHECK(rows == 16 || rows == 32 || rows == 128, "narrow_gemm : rows 16, 32 ou 128");
    if (!nv) TORCH_CHECK(group % NG_KS == 0 && sc16.has_value() && zr.has_value(), "narrow_gemm int8 : groupe multiple de 64, echelles et zeros");
    const int M = x.size(0), N = qw.size(0);
    auto y = torch::empty({M, N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    #define NG_LANCE(ROWS, NV) do { \
        constexpr int XLD = NG_KS * 2 + 16, WLD = (NV ? NG_KS / 2 : NG_KS) + 16; \
        constexpr int ETAGE = 16 * XLD + ROWS * WLD + (NV ? ROWS * 4 : 0); \
        constexpr int SHM = NG_S * ETAGE; \
        static bool attr = false; \
        if (!attr && SHM > 48 * 1024) { cudaFuncSetAttribute(narrow_gemm_kernel<ROWS, NV>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHM); attr = true; } \
        narrow_gemm_kernel<ROWS, NV><<<(unsigned)((N + ROWS - 1) / ROWS), ROWS * 4, SHM, stream>>>( \
            qw.data_ptr<unsigned char>(), NV ? sc8->data_ptr<unsigned char>() : nullptr, \
            NV ? nullptr : reinterpret_cast<const __half *>(sc16->data_ptr()), NV ? nullptr : zr->data_ptr<unsigned char>(), \
            reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), \
            M, N, (int)K, (int)group, (float)gscale); } while (0)
    if (nv) { if (rows == 16) NG_LANCE(16, true); else if (rows == 32) NG_LANCE(32, true); else NG_LANCE(128, true); }
    else    { if (rows == 16) NG_LANCE(16, false); else if (rows == 32) NG_LANCE(32, false); else NG_LANCE(128, false); }
    #undef NG_LANCE
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
    if (!ancien && xreg_demande() && xreg_possible(K))
        return nvfp4_gemv_grouped_xreg(qw, bscale, gscales, expert_ids, token_ids, x, K);
    if (!ancien && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024) {
        const bool bf = x.scalar_type() == torch::kBFloat16;
        auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
        auto out = torch::empty({G, M}, xc.options().dtype(torch::kFloat));
        static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
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

// Pendant table de nvfp4_gemv_grouped_warp_kernel (down_proj, une seule
// projection) : même patron table_*[e] que le gate-up ci-dessus.
template <typename XT, int RPW>
__global__ void nvfp4_gemv_grouped_table_kernel(
    const int64_t *__restrict__ table_qw, const int64_t *__restrict__ table_bs,
    const float *__restrict__ gscales, const int *__restrict__ expert_ids,
    const int *__restrict__ token_ids, const XT *__restrict__ x,
    float *__restrict__ y, int M, int K) {
    extern __shared__ float xs_sh[];
    const int g = blockIdx.y, e = expert_ids[g];
    charger_x_sh<XT>(x + (long)token_ids[g] * K, xs_sh, K);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // Creneau fantome (e < 0, bead pds 14/09) : AVANT tout dereferencement
    // de table. Sortie a zero, aucune lecture hote.
    if (e < 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
            if (row >= M) return;
            if (lane == 0) y[(long)g * M + row] = 0.f;
        }
        return;
    }
    const unsigned char *qw_e = reinterpret_cast<const unsigned char *>(table_qw[e]);
    const unsigned char *bs_e = reinterpret_cast<const unsigned char *>(table_bs[e]);
    const long half_k = (long)K >> 1;
    const int nloads = K / WEIGHTS_PER_LOAD;
    const float gscale = gscales[e];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = (blockIdx.x * GW_WARPS + warp) * RPW + r;
        if (row >= M) return;
        const float acc = nvfp4_row_dot_warp(
            reinterpret_cast<const uint4 *>(qw_e + (long)row * half_k),
            bs_e + (long)row * nloads, gscale, xs_sh, nloads >> 1, lane);
        if (lane == 0) y[(long)g * M + row] = acc;
    }
}

torch::Tensor nvfp4_gemv_grouped_table(torch::Tensor table_qw, torch::Tensor table_bs,
                                       torch::Tensor gscales,
                                       torch::Tensor expert_ids, torch::Tensor token_ids,
                                       torch::Tensor x, int64_t M, int64_t K) {
    CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(x);
    TORCH_CHECK(K % 32 == 0 && (size_t)(K + K / 32) * sizeof(float) <= 48 * 1024,
                "chemin table : K multiple de 32 et <= 11904");
    const int G = expert_ids.size(0);
    const bool bf = x.scalar_type() == torch::kBFloat16;
    auto xc = (bf ? x : x.to(torch::kFloat)).contiguous();
    auto out = torch::empty({G, (int)M}, xc.options().dtype(torch::kFloat));
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 4;
    dim3 grid(((int)M + GW_WARPS * rpw - 1) / (GW_WARPS * rpw), G);
    const size_t shm = (size_t)(K + K / 32) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GT_LAUNCH(XT, PX) do { if (rpw == 1) GT_L(XT, 1, PX); else if (rpw == 2) GT_L(XT, 2, PX); else GT_L(XT, 4, PX); } while (0)
    #define GT_L(XT, R, PX) nvfp4_gemv_grouped_table_kernel<XT, R><<<grid, GW_WARPS * WARP, shm, stream>>>( \
        table_qw.data_ptr<int64_t>(), table_bs.data_ptr<int64_t>(), gscales.data_ptr<float>(), \
        expert_ids.data_ptr<int>(), token_ids.data_ptr<int>(), PX, out.data_ptr<float>(), (int)M, (int)K)
    if (bf) { GT_LAUNCH(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr())); }
    else { GT_LAUNCH(float, xc.data_ptr<float>()); }
    #undef GT_LAUNCH
    #undef GT_L
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

// Corps commun des deux points d'entrée : CANAL (C5-b) reçoit en plus sc,
// tampon et tampon_de ; hors CANAL ces trois tenseurs sont ignorés (nuls).
template <bool CANAL>
torch::Tensor paged_attention_gen(torch::Tensor q, torch::Tensor kc,
                                  torch::Tensor ks, torch::Tensor vc,
                                  torch::Tensor vs, torch::Tensor sc,
                                  torch::Tensor tampon, torch::Tensor tampon_de,
                                  torch::Tensor tables,
                                  torch::Tensor seq_lens, int64_t hkv,
                                  double scale, int64_t q_len, int64_t window) {
    CHECK_CUDA(q); CHECK_CUDA(kc); CHECK_CUDA(tables);
    ACVRAM_DEVICE_GUARD(q);
    CHECK_CONTIG(q); CHECK_CONTIG(kc); CHECK_CONTIG(vc);
    const unsigned char *p_sc = nullptr;
    const __nv_bfloat16 *p_tampon = nullptr;
    const int *p_tampon_de = nullptr;
    if (CANAL) {
        CHECK_CUDA(sc); CHECK_CUDA(tampon); CHECK_CUDA(tampon_de);
        CHECK_CONTIG(sc); CHECK_CONTIG(tampon); CHECK_CONTIG(tampon_de);
        TORCH_CHECK(sc.scalar_type() == torch::kByte && tampon.scalar_type() == torch::kBFloat16
                    && tampon_de.scalar_type() == torch::kInt,
                    "attention paginee canal : sc uint8, tampon bf16, tampon_de int32");
        TORCH_CHECK(sc.size(0) == kc.size(0) && tampon_de.size(0) == kc.size(0),
                    "attention paginee canal : sc et tampon_de indexes par bloc");
        p_sc = sc.data_ptr<unsigned char>();
        p_tampon = reinterpret_cast<const __nv_bfloat16 *>(tampon.data_ptr());
        p_tampon_de = tampon_de.data_ptr<int>();
    }
    const int BQ = q.size(0);             // B * q_len lignes de requete
    const int B = BQ / (int)q_len;
    TORCH_CHECK(B * (int)q_len == BQ, "q.size(0) doit etre B*q_len");
    const int HQ = q.size(1);
    const int D = q.size(2);
    const int N = tables.size(1);
    // TAILLE DE TRANCHE A L'EXECUTION. C'etait `constexpr PA_CHUNK` : une
    // constante compilee ne peut porter une formule qui depend du nombre de SM
    // et du contexte reel, et elle oblige a recompiler pour l'explorer — neuf
    // minutes par valeur, avec le risque verifie que la recompilation n'ait pas
    // lieu du tout. Le defaut reproduit exactement l'ancien comportement.
    // TRANCHE ADAPTATIVE. Mesure du 10/09, contexte 3007, un appel isole :
    //     chunk   64  128  256  512  1024  2048
    //     total   25   32   46   79   144   279  us
    // Le temps DOUBLE quand la tranche double : le noyau est serialise sur la
    // longueur de tranche, donc le total vaut le temps d'UNE tranche et
    // decouper davantage le reduit d'autant. 512 etait le pire reglage
    // atteignable parmi ceux mesures.
    // On ne peut pas pour autant poser 64 en constante : le nombre de tranches
    // C = ceil(N*16 / chunk) est borne a 256, donc 64 cesserait d'etre legal
    // vers 16 k jetons. La regle prend LA PLUS PETITE TRANCHE QUI RESTE DANS LA
    // BORNE, et la borne est ensuite VERIFIEE, pas supposee (TORCH_CHECK plus
    // bas). 64 est le plancher parce que c'est la plus petite valeur MESUREE :
    // 32 et 16 pourraient etre meilleurs, ils n'ont pas ete essayes, et on ne
    // reglera pas un defaut par une extrapolation.
    int chunk = PA_CHUNK;
    if (const char *v = std::getenv("ACVRAM_PA_CHUNK")) {
        int demande = atoi(v);
        if (demande >= 16) chunk = demande;   // priorite a la main de l'operateur
    } else {
        const int mini = (N * 16 + 255) / 256;   // en deca, C depasserait 256
        chunk = 64;
        while (chunk < mini) chunk <<= 1;
    }
    // BISECTION : ACVRAM_PA_ETAPE < 3 sort du noyau plus tot. Le resultat est
    // alors FAUX par construction — c'est un instrument de diagnostic, jamais
    // un chemin de production. Defaut 3 = comportement normal.
    // Le compteur de participation coute un atomique par bloc sur une adresse
    // unique : mesurable, donc extinguible.
    bool compter = true;
    if (const char *v = std::getenv("ACVRAM_PA_SANS_COMPTEUR"))
        compter = !(v[0] == '1');
    int etape = 3;
    if (const char *v = std::getenv("ACVRAM_PA_ETAPE")) {
        int d = atoi(v);
        if (d >= 0 && d < 3) etape = d;
    }
    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256 || D == 512,
                "dimension de tete non instanciee : ", D);
    const int C = (N * 16 + chunk - 1) / chunk;
    TORCH_CHECK(C <= 256, "contexte au-dela de 256 tranches (chunk=", chunk,
                ", blocs=", N, ")");
    const bool qbf = q.scalar_type() == torch::kBFloat16;
    auto f32 = q.options().dtype(torch::kFloat);
    // TAMPONS REUTILISES. Le noyau porte 46,5 us de cout FIXE — 94 % de sa
    // duree — qui ne depend ni du travail, ni du nombre de blocs (grille x8 :
    // aucun effet), ni du nombre de noyaux lances. Reste ce qui ENTOURE le
    // lancement. Chaque tampon est ECRIT avant d'etre lu, y compris par la
    // sortie anticipee qui pose -INFINITY et 0.
    static bool sans_cache = [] {
        const char *v = std::getenv("ACVRAM_PAGED_ALLOC");
        return v && v[0] == '1';
    }();
    torch::Tensor part, pm, pl;
    if (sans_cache) {
        part = torch::empty({BQ, HQ, C, D}, f32);
        pm = torch::empty({BQ, HQ, C}, f32);
        pl = torch::empty({BQ, HQ, C}, f32);
    } else {
        // UN SEUL TAMPON, GARDE AU PIRE CAS VU. C'etait un std::map indexe par
        // (BQ, HQ, C, D, device) et JAMAIS VIDE : une entree definitive par
        // forme rencontree. UN `static std::map` JAMAIS VIDE EST UNE FUITE QUI
        // ATTEND SON DECLENCHEUR — celui-ci dormait a 2,2 Mo depuis toujours
        // parce que `chunk` valait 512 et que C ne prenait que 16 valeurs. La
        // tranche adaptative l'a reveille a 135 Mo sans toucher a une ligne de
        // son code : C = ceil(N/4) prend 128 valeurs. Le cache n'etait pas
        // faux, il etait a la merci d'un changement ailleurs.
        // Le noyau recoit des pointeurs bruts et calcule ses index a partir de
        // C : un tampon PLUS GRAND que necessaire convient, seule la capacite
        // compte. On garde donc le maximum vu, et rien de plus. Au pire cas
        // legal (C = 256, HQ = 32, D = 128, BQ = 1) : 4,2 Mo pour `part`,
        // 32 Kio pour les deux autres — a comparer aux 135 Mo cumules.
        // ON N'AGRANDIT JAMAIS UN TAMPON DEJA UTILISE : UN GRAPHE CUDA EN A
        // CAPTURE L'ADRESSE. Un tampon unique agrandi au besoin semblait la
        // bonne reponse — il stabilisait bien la memoire a 1,07 Mo — puis la
        // neuvieme longueur a rendu un acces memoire illegal : la
        // reallocation avait libere l'adresse que des graphes deja captures
        // rejouaient. L'ancien cache par forme ne realloue jamais une forme
        // vue : c'etait sa qualite cachee, et sa fuite venait du NOMBRE de
        // formes, pas du principe.
        // On garde donc un cache, mais indexe sur C ARRONDI A LA PUISSANCE DE
        // 2 : 9 tailles possibles (1..256) au lieu de 128 valeurs distinctes,
        // chacune allouee une fois et jamais deplacee. Cumul au pire :
        // (1+2+...+256) = 511 unites, soit 8,4 Mo contre 135.
        int Cb = 1;
        while (Cb < C) Cb <<= 1;
        static std::map<std::tuple<int, int, int, int, int>,
                        std::array<torch::Tensor, 3>> cache_t;
        static unsigned long long octets_caches = 0ULL;
        auto cle = std::make_tuple(BQ, HQ, Cb, D, (int)q.device().index());
        auto it = cache_t.find(cle);
        if (it == cache_t.end()) {
            it = cache_t.emplace(cle, std::array<torch::Tensor, 3>{
                torch::empty({BQ, HQ, Cb, D}, f32),
                torch::empty({BQ, HQ, Cb}, f32),
                torch::empty({BQ, HQ, Cb}, f32)}).first;
            octets_caches += (unsigned long long)BQ * HQ * Cb * (D + 2) * 4ULL;
        }
        acvram_pa_tampon_octets = octets_caches;
        part = it->second[0]; pm = it->second[1]; pl = it->second[2];
    }
    // `out` reste FRAIS : il est RENDU a l'appelant.
    auto out = torch::empty({BQ, HQ, D}, q.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g1(BQ, HQ, C), g2(BQ, HQ);
    const int threads = PA_WARPS * WARP;

    #define PA_LAUNCH_T(DD, QT, OT, PQ, PO) do { \
        paged_attn_partial_kernel<DD, QT, OT, CANAL><<<g1, threads, 0, stream>>>( \
            PQ, kc.data_ptr<signed char>(), \
            reinterpret_cast<const __half *>(ks.data_ptr()), \
            vc.data_ptr<signed char>(), \
            reinterpret_cast<const __half *>(vs.data_ptr()), \
            p_sc, p_tampon, p_tampon_de, \
            tables.data_ptr<long>(), seq_lens.data_ptr<long>(), \
            part.data_ptr<float>(), pm.data_ptr<float>(), \
            pl.data_ptr<float>(), (C == 1 ? (PO) : nullptr), \
            HQ, (int)hkv, N, C, (int)q_len, (float)scale, (int)window, \
            chunk, etape, compter); \
        if (C > 1) \
        paged_attn_reduce_kernel<DD, OT><<<g2, 128, 0, stream>>>( \
            part.data_ptr<float>(), pm.data_ptr<float>(), \
            pl.data_ptr<float>(), PO, HQ, C); } while (0)
    // q et la sortie gardent le type de l'appelant : les convertir coûtait
    // deux copies par couche, pour un tenseur de quelques kilooctets.
    #define PA_LAUNCH(DD) do { \
        if (qbf) PA_LAUNCH_T(DD, __nv_bfloat16, __nv_bfloat16, \
            reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr()), \
            reinterpret_cast<__nv_bfloat16 *>(out.data_ptr())); \
        else PA_LAUNCH_T(DD, float, float, q.data_ptr<float>(), \
            out.data_ptr<float>()); } while (0)

    if (D == 32) { PA_LAUNCH(32); }
    else if (D == 64) { PA_LAUNCH(64); }
    else if (D == 128) { PA_LAUNCH(128); }
    else if (D == 256) { PA_LAUNCH(256); }
    else { PA_LAUNCH(512); }
    #undef PA_LAUNCH
    #undef PA_LAUNCH_T
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor paged_attention(torch::Tensor q, torch::Tensor kc,
                              torch::Tensor ks, torch::Tensor vc,
                              torch::Tensor vs, torch::Tensor tables,
                              torch::Tensor seq_lens, int64_t hkv,
                              double scale, int64_t q_len, int64_t window) {
    return paged_attention_gen<false>(q, kc, ks, vc, vs, torch::Tensor(), torch::Tensor(),
                                      torch::Tensor(), tables, seq_lens, hkv, scale,
                                      q_len, window);
}

// C5-b : clés par canal (sc E4M3 par bloc, bloc courant bf16 dans la réserve).
torch::Tensor paged_attention_canal(torch::Tensor q, torch::Tensor kc,
                                    torch::Tensor ks, torch::Tensor vc,
                                    torch::Tensor vs, torch::Tensor sc,
                                    torch::Tensor tampon, torch::Tensor tampon_de,
                                    torch::Tensor tables,
                                    torch::Tensor seq_lens, int64_t hkv,
                                    double scale, int64_t q_len, int64_t window) {
    return paged_attention_gen<true>(q, kc, ks, vc, vs, sc, tampon, tampon_de,
                                     tables, seq_lens, hkv, scale, q_len, window);
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


// --- version batchée : les B créneaux du pas en un lancement (bead 6wa,
// spec acvram-memoire/revue/spec-batching-mla-6wa-13-09.md). Les caches ne
// sont pas contigus entre créneaux (un tenseur par créneau, adresses
// stables sur la vie du serveur) : ils arrivent par une table d'adresses
// [B] int64, comme les tables de blocs de paged_attn. Même arithmétique que
// mla_scores_kernel / mla_reduce_kernel : sortie bit-identique par créneau.
__global__ void mla_scores_batch_kernel(const float *__restrict__ q,          // [B, H, W]
                                        const int64_t *__restrict__ cache_ptrs, // [B]
                                        const long *__restrict__ lens,        // [B]
                                        float *__restrict__ scores,           // [B, H, L]
                                        int H, int L, int W, float scale) {
    const int b = blockIdx.x, h = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (r >= L) return;
    const __nv_bfloat16 *cache = reinterpret_cast<const __nv_bfloat16 *>(cache_ptrs[b]);
    float s = -INFINITY;
    if (r <= (int)lens[b]) {
        const float *qh = q + ((size_t)b * H + h) * W;
        const __nv_bfloat16 *kr = cache + (size_t)r * W;
        float acc = 0.f;
        for (int d = 0; d < W; d += 2) {
            const float2 kk = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(kr + d));
            acc += qh[d] * kk.x + qh[d + 1] * kk.y;
        }
        s = acc * scale;
    }
    scores[((size_t)b * H + h) * L + r] = s;
}

__global__ void mla_reduce_batch_kernel(const float *__restrict__ scores,     // [B, H, L]
                                        const int64_t *__restrict__ cache_ptrs,
                                        float *__restrict__ o,                // [B, H, R]
                                        int H, int L, int W, int R) {
    const int b = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, T = blockDim.x;
    extern __shared__ float sh[];
    __shared__ float red[32];
    const __nv_bfloat16 *cache = reinterpret_cast<const __nv_bfloat16 *>(cache_ptrs[b]);
    const float *sc = scores + ((size_t)b * H + h) * L;
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
        o[((size_t)b * H + h) * R + d] = acc * inv;
    }
}

torch::Tensor mla_decode_batch(torch::Tensor q_eff, torch::Tensor cache_ptrs,
                               torch::Tensor lens, torch::Tensor scores,
                               int64_t L, int64_t rank, double scale) {
    CHECK_CUDA(q_eff); ACVRAM_DEVICE_GUARD(q_eff);
    CHECK_CONTIG(q_eff); CHECK_CONTIG(cache_ptrs); CHECK_CONTIG(lens); CHECK_CONTIG(scores);
    TORCH_CHECK(q_eff.dim() == 3 && q_eff.scalar_type() == torch::kFloat, "MLA batch : q_eff [B, H, W] fp32");
    const int B = q_eff.size(0), H = q_eff.size(1), W = q_eff.size(2);
    TORCH_CHECK(W % 2 == 0, "MLA : largeur paire attendue");
    TORCH_CHECK(cache_ptrs.scalar_type() == torch::kInt64 && cache_ptrs.is_cuda() && cache_ptrs.numel() == B,
                "MLA batch : cache_ptrs [B] int64 sur la carte");
    TORCH_CHECK(lens.scalar_type() == torch::kLong && lens.numel() == B, "MLA batch : lens [B] int64");
    // comme mla_decode, le tampon de scores est lu à plat [B*H*L] : seule sa
    // taille compte, pas sa forme
    TORCH_CHECK(scores.scalar_type() == torch::kFloat && scores.numel() >= (long)B * H * L,
                "MLA batch : scores fp32 d'au moins B*H*L elements");
    auto o = torch::empty({B, H, (long)rank}, q_eff.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g1(B, H, ((int)L + 127) / 128);
    mla_scores_batch_kernel<<<g1, 128, 0, stream>>>(
        q_eff.data_ptr<float>(), cache_ptrs.data_ptr<int64_t>(), lens.data_ptr<long>(),
        scores.data_ptr<float>(), H, (int)L, W, (float)scale);
    const size_t shm = (size_t)L * sizeof(float);
    if (shm > 48 * 1024) {
        static size_t autorise = 0;
        if (shm > autorise) {
            cudaFuncSetAttribute(mla_reduce_batch_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            autorise = shm;
        }
    }
    dim3 g2(B, H);
    mla_reduce_batch_kernel<<<g2, 256, shm, stream>>>(
        scores.data_ptr<float>(), cache_ptrs.data_ptr<int64_t>(),
        o.data_ptr<float>(), H, (int)L, W, (int)rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

// Écriture du latent du pas pour les B créneaux en UN lancement (sage-duel
// verdict § 6.2, commit 1 : le chemin MLA du décodage tournait créneau par
// créneau — 12 index_copy_ + 12 len.add_ par couche). Un bloc par créneau :
// la ligne k_new[b] va à cache_b[len_b], puis len_b += 1. Les caches et les
// longueurs vivent chacun dans son tenseur (adresses stables sur la vie du
// serveur) : tables d'adresses [B] int64, comme mla_decode_batch. L'attention
// du même pas lit la copie ``lens`` prise AVANT ce lancement.
// Cache latent fp8 (sage-avis-exterieur-16-09 § 6, commit 3 du chantier MLA) :
// une ligne = W codes E4M3 + 16 octets dont l'échelle fp32 s = amax/448
// (foulée W+16 octets, alignée 16). Quantification : code = E4M3(v / s) en RN
// satfinite, division correctement arrondie — la référence torch divise par
// un TENSEUR (tensor/scalaire multiplie par l'inverse : 1 ulp d'écart).
constexpr int MLA_FP8_PAD = 16;
__device__ __forceinline__ int mla_fp8_foulee(int W) { return W + MLA_FP8_PAD; }

__global__ void mla_ecrit_latent_kernel(const __nv_bfloat16 *__restrict__ k_new,   // [B, W]
                                        const int64_t *__restrict__ cache_ptrs,    // [B]
                                        const int64_t *__restrict__ len_ptrs,      // [B]
                                        int W, int fp8) {
    const int b = blockIdx.x;
    long *lp = reinterpret_cast<long *>(len_ptrs[b]);
    const long len = *lp;
    if (!fp8) {
        const uint4 *src = reinterpret_cast<const uint4 *>(k_new + (size_t)b * W);
        uint4 *dst = reinterpret_cast<uint4 *>(reinterpret_cast<__nv_bfloat16 *>(cache_ptrs[b]) + len * W);
        for (int i = threadIdx.x; i < W / 8; i += blockDim.x) dst[i] = src[i];
    } else {
        __shared__ float red[32];
        const __nv_bfloat16 *src = k_new + (size_t)b * W;
        float amax = 0.f;
        for (int i = threadIdx.x; i < W; i += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
        __syncthreads();
        amax = 0.f;
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k) amax = fmaxf(amax, red[k]);
        const float sc = amax > 0.f ? __fdiv_rn(amax, 448.f) : 1.f;
        unsigned char *dst = reinterpret_cast<unsigned char *>(cache_ptrs[b]) + len * (long)mla_fp8_foulee(W);
        for (int i = threadIdx.x; i < W; i += blockDim.x)
            dst[i] = (unsigned char)__nv_cvt_float_to_fp8(__fdiv_rn(__bfloat162float(src[i]), sc),
                                                          __NV_SATFINITE, __NV_E4M3);
        if (threadIdx.x == 0) *reinterpret_cast<float *>(dst + W) = sc;
    }
    __syncthreads();
    if (threadIdx.x == 0) *lp = len + 1;
}

void mla_ecrit_latent(torch::Tensor k_new, torch::Tensor cache_ptrs, torch::Tensor len_ptrs, bool fp8) {
    CHECK_CUDA(k_new); ACVRAM_DEVICE_GUARD(k_new); CHECK_CONTIG(k_new);
    CHECK_CONTIG(cache_ptrs); CHECK_CONTIG(len_ptrs);
    TORCH_CHECK(k_new.dim() == 2 && k_new.scalar_type() == torch::kBFloat16 && k_new.size(1) % (fp8 ? 16 : 8) == 0,
                "ecrit_latent : k_new [B, W] bf16, W multiple de 8 (16 en fp8)");
    const int B = k_new.size(0), W = k_new.size(1);
    TORCH_CHECK(cache_ptrs.scalar_type() == torch::kInt64 && cache_ptrs.is_cuda() && cache_ptrs.numel() == B
                && len_ptrs.scalar_type() == torch::kInt64 && len_ptrs.is_cuda() && len_ptrs.numel() == B,
                "ecrit_latent : cache_ptrs et len_ptrs [B] int64 sur la carte");
    if (B == 0) return;
    mla_ecrit_latent_kernel<<<B, fp8 ? 128 : 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(k_new.data_ptr()),
        cache_ptrs.data_ptr<int64_t>(), len_ptrs.data_ptr<int64_t>(), W, fp8 ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// MLA à UNE PASSE (sage-duel-verdict-16-09 § 3) : les deux noyaux ci-dessus
// lisaient le cache latent [L, W] une fois PAR TÊTE dans mla_scores puis une
// seconde fois par tête dans mla_reduce — 40 lectures du cache par pas et par
// couche, ≈ 52 Go/pas à b=12 ctx 2 048 (les 32,8 ms mesurés). En forme
// absorbée les H têtes partagent la même tête KV : un CTA par (séquence,
// tranche de L) lit chaque tuile de TL lignes UNE fois en mémoire partagée,
// calcule les scores des H têtes (fp32, même arithmétique que mla_scores :
// Σ q·k en fp32 puis × scale), softmax en ligne par tête (m, l, rescalage de
// l'accumulateur), accumule o_lat[H, R] sur la même tuile, et écrit (o, m, l)
// partiels dans ws [B, S, H, R+2] ; mla_1p_combine recombine les S tranches
// (flash-decoding). Tout en fp32 : pas de mma bf16 sur q (l'erreur 2^-8 sur
// des scores de l'ordre de 10 aurait coûté le cos ≥ 0,9999 scellé) — le
// calcul (≈ 25 GFMA/pas) reste sous la lecture (1,3 Go/pas).
//
// Mémoire partagée : q fp32 [H][W] + tuile bf16 [TL][W+8] + S, P [HMAX][TL] +
// m, l, alpha [HMAX] + échelles et table fp8 — 89 968 o (88 Ko) à H=20,
// W=576, TL=32 (formule du lanceur) : UN CTA par SM (C14 : la grille doit
// donc venir du nombre de tranches).
// C14-c (verdict-c14-bis) : à b ≤ 2 la grille venait des seules tranches —
// 32 blocs pour 170 SM à L=512 (warps actifs 20 %). Les têtes sont
// indépendantes dans ce noyau (même tuile, q et o par tête) : la grille a
// une troisième dimension G = ⌈H/HMAX⌉ GROUPES DE TÊTES, le bloc (b, s, g)
// traite les têtes [g·HMAX, g·HMAX+HMAX) de la tranche s et écrit ses
// partiels aux mêmes places de ws [B, S, H, R+2] — le combine ne voit pas la
// répartition. Régime fin (b ≤ 2, H ≤ 20) : <5, 8, 1, FP8, MINB=2> :
// 256 blocs à L=512, 22 Ko de shared, chaîne d'un bloc = q 5 têtes (11,5 Ko)
// + tuile 8 lignes (9,2 Ko) + 8 scores × 5 têtes. Régime b ≥ 3 : inchangé
// (<20,32,4>, G=1).
// Scores : chaque warp prend RW lignes à la fois, les lanes se partagent W
// (paires bf16), q lu une fois pour RW lignes, réduction par __shfl_xor.
// o_lat : un fil = deux colonnes de R, toutes les têtes en registres.
// Lignes valides : r ≤ len (le jeton courant vient d'être écrit) — comme
// mla_scores ; les tranches entièrement au-delà écrivent m = −inf, l = 0.
// ============================================================================
constexpr int MLA1P_FILS = 256;
// C14 : borne de S (tranches par séquence) — les poids de recombinaison de
// mla_1p_combine tiennent en shared (4 Ko) ; mla_1p_tranches n'y arrive
// jamais (S ≤ SM/B ≤ 170 en pratique), la borne protège le combine.
constexpr int MLA1P_S_MAX = 1024;
// C14 : lectures globales groupées par PF par fil (registres d'abord, stores
// shared ensuite) : la copie q → shared (2 880 float4 à H=20, 12 tours) et
// la copie de la tuile (2 304 uint4 à TL=32, 9 tours) n'enchaînaient qu'une
// latence mémoire par tour ; PF=4 en émet quatre à la fois. Pure copie :
// aucune arithmétique, sortie identique au bit.
constexpr int MLA1P_PF = 4;
// C14-c : têtes par bloc et lignes par tuile du régime fin (b ≤ 2, H ≤ 20) ;
// la règle de grille (mla_1p_tranches_fin) suppose 2 blocs résidents par SM
// pour cette instanciation — __launch_bounds__(256, 2) (MINB=2) le garantit côté registres (≤ 128), la shared (22 Ko) en laisse 4.
// Le cinquième paramètre MINB est le second argument de __launch_bounds__ :
// 2 pour le régime fin, 0 (= non spécifié) pour les autres — un « 1 »
// explicite change l'allocation de ptxas (<20,32,4> passait de 170 à 232
// registres), 0 laisse les bornes d'avant.
constexpr int MLA1P_FIN_HMAX = 5, MLA1P_FIN_TL = 8, MLA1P_FIN_BLOCS_SM = 2;

template <int HMAX, int TL, int RW, bool FP8, int MINB>
__global__ void __launch_bounds__(MLA1P_FILS, MINB) mla_1p_kernel(
    const float *__restrict__ q,              // [B, H, W] fp32
    const int64_t *__restrict__ cache_ptrs,   // [B] (ou nullptr : cache0)
    const __nv_bfloat16 *__restrict__ cache0, // [L, W] si B = 1 sans table
    const long *__restrict__ lens,            // [B]
    float *__restrict__ ws,                   // [B, S, H, R+2]
    int H, int L, int W, int R, int S, int rows_par_cta, float scale) {
    // FP8 : lignes de W codes E4M3 + échelle fp32 (foulée W+16 o) ; la tuile
    // garde les codes, une table de 256 flottants en shared les décode ; le
    // score d'une ligne est multiplié par son échelle, et p_r l'absorbe pour
    // o_lat — la moitié des octets lus, même arithmétique fp32 ensuite.
    static_assert(HMAX <= 32, "HMAX <= 32 : une lane par tete pour l'ecriture des scores");
    const int b = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    // C14-c : groupe de têtes de ce bloc — h0 = g·HMAX, Hb têtes (≤ HMAX) ;
    // H reste le nombre total de têtes (foulées de q et de ws). Hors régime
    // fin (HMAX ≥ 20 : G=1) h0 et Hb sont fixés À LA COMPILATION à 0 et H,
    // pour que <20,32,4> et <32,16,2> gardent le SASS d'avant (avec h0/Hb
    // dynamiques ptxas passait de 170 à 232 registres sur <20,32,4>).
    constexpr bool GROUPES = HMAX <= MLA1P_FIN_HMAX;
    const int h0 = GROUPES ? (int)blockIdx.z * HMAX : 0;
    const int Hb = GROUPES ? min(HMAX, H - h0) : H;
    // foulée d'une ligne de tuile : W+8 éléments bf16 (1 168 o) ou, en FP8,
    // W+16 OCTETS — un multiple de 16 o dans les deux cas, condition des
    // lectures uint4 en shared (W+8 octets = 584 : adresse mal alignée,
    // CUDA_ERROR_MISALIGNED_ADDRESS, t-qa 89bb888)
    const int WP = FP8 ? W + 16 : W + 8;
    extern __shared__ __align__(16) unsigned char mla_smem[];
    float *q_s = reinterpret_cast<float *>(mla_smem);                        // [Hb][W]
    __nv_bfloat16 *tile = reinterpret_cast<__nv_bfloat16 *>(q_s + (size_t)Hb * W);   // [TL][WP] bf16 ou codes
    unsigned char *tile8 = reinterpret_cast<unsigned char *>(tile);          // FP8 : [TL][WP] codes
    const size_t tuile_octets = FP8 ? (size_t)TL * WP : (size_t)TL * WP * sizeof(__nv_bfloat16);
    float *S_s = reinterpret_cast<float *>(reinterpret_cast<unsigned char *>(tile) + tuile_octets);   // [HMAX][TL]
    float *P_s = S_s + HMAX * TL;                                            // [HMAX][TL]
    float *m_s = P_s + HMAX * TL, *l_s = m_s + HMAX, *a_s = l_s + HMAX;
    float *sc_s = a_s + HMAX;                                                // FP8 : [TL] échelles de ligne
    float *lut = sc_s + TL;                                                  // FP8 : [256] E4M3 -> float
    const __nv_bfloat16 *cache = cache_ptrs
        ? reinterpret_cast<const __nv_bfloat16 *>(cache_ptrs[b]) : cache0;
    const unsigned char *cache8 = reinterpret_cast<const unsigned char *>(cache);
    const long foulee8 = FP8 ? (long)mla_fp8_foulee(W) : 0;
    if (FP8) for (int i = tid; i < 256; i += MLA1P_FILS) lut[i] = e4m3_to_float((unsigned char)i);
    const int valide = min((int)lens[b] + 1, L);
    const int r0 = s * rows_par_cta, r1 = min(r0 + rows_par_cta, valide);
    float *out = ws + ((size_t)(b * S + s) * H + h0) * (R + 2);   // têtes h0..h0+Hb-1, contiguës
    if (r0 >= r1) {                                              // tranche vide
        for (int i = tid; i < Hb * (R + 2); i += MLA1P_FILS) {
            const int h = i / (R + 2), c = i - h * (R + 2);
            out[i] = c == R ? -INFINITY : 0.f;
        }
        return;
    }
    // q des Hb têtes du bloc en mémoire partagée (float4), MLA1P_PF lectures en vol par fil
    {
        const float4 *src = reinterpret_cast<const float4 *>(q + ((size_t)b * H + h0) * W);
        float4 *dst = reinterpret_cast<float4 *>(q_s);
        const int n4 = Hb * W / 4;
        for (int i0 = tid; i0 < n4; i0 += MLA1P_PF * MLA1P_FILS) {
            float4 v[MLA1P_PF];
            #pragma unroll
            for (int k = 0; k < MLA1P_PF; ++k) { const int i = i0 + k * MLA1P_FILS; if (i < n4) v[k] = src[i]; }
            #pragma unroll
            for (int k = 0; k < MLA1P_PF; ++k) { const int i = i0 + k * MLA1P_FILS; if (i < n4) dst[i] = v[k]; }
        }
    }
    if (tid < HMAX) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; a_s[tid] = 1.f; }
    float o[HMAX][2];
    #pragma unroll
    for (int h = 0; h < HMAX; ++h) { o[h][0] = 0.f; o[h][1] = 0.f; }
    const int c0 = 2 * tid;                                      // colonnes de o_lat de ce fil
    const int chunks_par_ligne = FP8 ? W / 16 : W / 8;           // uint4 = 8 bf16 ou 16 codes

    for (int t0 = r0; t0 < r1; t0 += TL) {
        const int n = min(TL, r1 - t0);
        __syncthreads();                                         // la tuile précédente est consommée
        // --- tuile [n][W] : lecture vectorisée 16 o, MLA1P_PF en vol par fil ---
        {
            const int nc = n * chunks_par_ligne;
            for (int i0 = tid; i0 < nc; i0 += MLA1P_PF * MLA1P_FILS) {
                uint4 v[MLA1P_PF];
                #pragma unroll
                for (int k = 0; k < MLA1P_PF; ++k) {
                    const int i = i0 + k * MLA1P_FILS;
                    if (i < nc) {
                        const int r = i / chunks_par_ligne, c = i - r * chunks_par_ligne;
                        v[k] = FP8 ? *reinterpret_cast<const uint4 *>(cache8 + (t0 + r) * foulee8 + c * 16)
                                   : *reinterpret_cast<const uint4 *>(cache + (size_t)(t0 + r) * W + c * 8);
                    }
                }
                #pragma unroll
                for (int k = 0; k < MLA1P_PF; ++k) {
                    const int i = i0 + k * MLA1P_FILS;
                    if (i < nc) {
                        const int r = i / chunks_par_ligne, c = i - r * chunks_par_ligne;
                        if (FP8) *reinterpret_cast<uint4 *>(tile8 + (size_t)r * WP + c * 16) = v[k];
                        else     *reinterpret_cast<uint4 *>(tile + (size_t)r * WP + c * 8) = v[k];
                    }
                }
            }
        }
        if (FP8 && tid < n) sc_s[tid] = *reinterpret_cast<const float *>(cache8 + (t0 + tid) * foulee8 + W);
        __syncthreads();
        // --- scores : le warp w prend les lignes w*RW .. w*RW+RW-1 (+ 8*RW…) ---
        for (int rb = warp * RW; rb < n; rb += 8 * RW) {
            float part[HMAX][RW];
            #pragma unroll
            for (int h = 0; h < HMAX; ++h)
                #pragma unroll
                for (int i = 0; i < RW; ++i) part[h][i] = 0.f;
            for (int j = lane; j < W / 2; j += 32) {
                float2 k2[RW];
                #pragma unroll
                for (int i = 0; i < RW; ++i) {
                    const int r = rb + i;
                    if (r >= n) { k2[i] = make_float2(0.f, 0.f); continue; }
                    if (FP8) {
                        const unsigned char *pc = tile8 + (size_t)r * WP + 2 * j;
                        k2[i] = make_float2(lut[pc[0]], lut[pc[1]]);
                    } else {
                        k2[i] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
                                    tile + (size_t)r * WP + 2 * j));
                    }
                }
                #pragma unroll
                for (int h = 0; h < HMAX; ++h) {
                    if (h < Hb) {
                        const float2 q2 = *reinterpret_cast<const float2 *>(q_s + (size_t)h * W + 2 * j);
                        #pragma unroll
                        for (int i = 0; i < RW; ++i) part[h][i] += q2.x * k2[i].x + q2.y * k2[i].y;
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < HMAX; ++h) {
                #pragma unroll
                for (int i = 0; i < RW; ++i) {
                    float v = part[h][i];
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
                    if (lane == h && h < Hb && rb + i < n)
                        S_s[h * TL + rb + i] = FP8 ? v * sc_s[rb + i] * scale : v * scale;
                }
            }
        }
        __syncthreads();
        // --- softmax en ligne, un fil par tête ---
        if (tid < Hb) {
            const float *Sh = S_s + tid * TL;
            float mt = -INFINITY;
            for (int r = 0; r < n; ++r) mt = fmaxf(mt, Sh[r]);
            const float m_old = m_s[tid], m_new = fmaxf(m_old, mt);
            const float a = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
            float lt = 0.f;
            float *Ph = P_s + tid * TL;
            for (int r = 0; r < n; ++r) {
                const float p = __expf(Sh[r] - m_new);
                Ph[r] = FP8 ? p * sc_s[r] : p;                   // l'échelle de ligne passe dans p pour o_lat
                lt += p;
            }
            l_s[tid] = l_s[tid] * a + lt;
            m_s[tid] = m_new;
            a_s[tid] = a;
        }
        __syncthreads();
        // --- o_lat : rescalage puis accumulation sur la tuile ---
        if (c0 < R) {
            #pragma unroll
            for (int h = 0; h < HMAX; ++h) {
                if (h < Hb) { const float a = a_s[h]; o[h][0] *= a; o[h][1] *= a; }
            }
            for (int r = 0; r < n; ++r) {
                float2 v2;
                if (FP8) { const unsigned char *pc = tile8 + (size_t)r * WP + c0; v2 = make_float2(lut[pc[0]], lut[pc[1]]); }
                else v2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(tile + (size_t)r * WP + c0));
                #pragma unroll
                for (int h = 0; h < HMAX; ++h) {
                    if (h < Hb) {
                        const float p = P_s[h * TL + r];
                        o[h][0] += p * v2.x; o[h][1] += p * v2.y;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (c0 < R) {
        #pragma unroll
        for (int h = 0; h < HMAX; ++h)
            if (h < Hb) { float *oh = out + (size_t)h * (R + 2) + c0; oh[0] = o[h][0]; oh[1] = o[h][1]; }
    }
    if (tid < Hb) { out[(size_t)tid * (R + 2) + R] = m_s[tid]; out[(size_t)tid * (R + 2) + R + 1] = l_s[tid]; }
}

// Recombinaison des S tranches : o = Σ_s o_s·e^{m_s−M} / Σ_s l_s·e^{m_s−M}.
//
// C14-c (verdict-c14-bis : 20 blocs de 128 fils, décrochage short-scoreboard,
// 8,4 µs/couche à S=32). L'ancien combine faisait, PAR FIL et PAR COLONNE,
// une boucle sérielle sur S avec un `if (m != −inf)` par tranche : 2·S
// lectures dépendantes du test par colonne (m relu S fois par colonne), et
// 20 blocs seulement à b=1. Ici :
//  - grille (B·H, ⌈R/CH⌉), 256 fils = SL voies de tranches × CH colonnes
//    (SL ∈ {1, 2, 4, 8}, CH = 256/SL, choisi par le lanceur pour que chaque
//    fil lise ≈ S/SL ≤ 8 partiels) : 320 blocs à b=1, S=64, R=512 ;
//  - m et l sont lus UNE fois par bloc (un fil par tranche), M et Σ l·w par
//    réduction de bloc, les poids w_s en shared ;
//  - la voie sl lit o_s[c] pour s ≡ sl (mod SL) : un warp = 32 colonnes
//    contiguës d'une même tranche (128 o coalescés), aucun test dans la
//    boucle (tranche vide : w = 0 et o_s = 0, écrits par le noyau), lectures
//    indépendantes émises en vol ; puis réduction des SL voies en shared.
// Chargements émis par (b, h) — avant : R·2·S (o et m par colonne) +
// 128 fils × 3·S (M, Lsum relus par chaque fil) ; après : R·S (o, une fois)
// + (R/CH)·2·S (m, l par bloc de colonnes). R=512 : S=64 → 90 176 → 34 816
// (−61 %) ; S=32 → 45 056 → 16 896. Octets utiles inchangés : S·H·(R+2)·4 =
// 2,63 Mo à S=64, 1,32 Mo à S=32 (b=1, H=20).
// Ordre des sommes : par voie puis réduction — pas identique au bit à
// l'ancien combine (test à sec : ≤ 8 ulp d'amplitude, indépendant de SL).
constexpr int MLA1P_CMB_FILS = 256;
template <int SL>
__global__ void __launch_bounds__(MLA1P_CMB_FILS) mla_1p_combine_kernel(
        const float *__restrict__ ws,   // [B, S, H, R+2]
        float *__restrict__ o,          // [B, H, R]
        int H, int R, int S) {
    constexpr int CH = MLA1P_CMB_FILS / SL;
    __shared__ float w_s[MLA1P_S_MAX];
    __shared__ float red[MLA1P_CMB_FILS / 32];
    __shared__ float part[SL][CH];
    const int bh = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int col = blockIdx.y * CH + (tid % CH), sl = tid / CH;
    const int b = bh / H, h = bh - b * H;
    const float *base = ws + ((size_t)b * S * H + h) * (R + 2);   // tranche 0 de (b, h)
    const size_t pas = (size_t)H * (R + 2);                       // d'une tranche à la suivante
    // --- M = max_s m_s (réduction de bloc) ---
    float mloc = -INFINITY;
    for (int s = tid; s < S; s += MLA1P_CMB_FILS) mloc = fmaxf(mloc, base[s * pas + R]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, off));
    if (lane == 0) red[warp] = mloc;
    __syncthreads();
    float M = red[0];
    #pragma unroll
    for (int w = 1; w < MLA1P_CMB_FILS / 32; ++w) M = fmaxf(M, red[w]);
    __syncthreads();                                    // red réutilisé ci-dessous
    // --- w_s = e^{m_s−M} (0 si tranche vide), Lsum = Σ l_s·w_s ---
    float lloc = 0.f;
    for (int s = tid; s < S; s += MLA1P_CMB_FILS) {
        const float m = base[s * pas + R];
        const float w = m != -INFINITY ? __expf(m - M) : 0.f;
        w_s[s] = w;
        lloc += base[s * pas + R + 1] * w;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) lloc += __shfl_xor_sync(0xffffffffu, lloc, off);
    if (lane == 0) red[warp] = lloc;
    __syncthreads();
    float Lsum = 0.f;
    #pragma unroll
    for (int w = 0; w < MLA1P_CMB_FILS / 32; ++w) Lsum += red[w];
    const float inv = Lsum > 0.f ? 1.f / Lsum : 0.f;
    // --- Σ_s o_s[col]·w_s sur la voie sl, lectures coalescées et en vol ---
    float acc = 0.f;
    if (col < R) {
        const float *oc = base + col;
        int s = sl;
        for (; s + 7 * SL < S; s += 8 * SL) {
            float v[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = oc[(size_t)(s + k * SL) * pas];
            #pragma unroll
            for (int k = 0; k < 8; ++k) acc += v[k] * w_s[s + k * SL];
        }
        for (; s < S; s += SL) acc += oc[(size_t)s * pas] * w_s[s];
    }
    if (SL == 1) {
        if (col < R) o[(size_t)bh * R + col] = acc * inv;
        return;
    }
    part[sl][tid % CH] = acc;
    __syncthreads();
    if (tid < CH && col < R) {
        float t = 0.f;
        #pragma unroll
        for (int k = 0; k < SL; ++k) t += part[k][tid];
        o[(size_t)bh * R + col] = t * inv;
    }
}

// C14-b (chantier-c14b-19-09, sage-fiches-c5b-c13c-c14b-20-09 § 3) : combine
// + v_b en UN lancement — y[b,h,v] = Σ_r v_b[h,v,r]·o_lat[b,h,r], écrit en
// bf16 [B, H, DV]. Remplace, par couche, l'einsum 'hvr,bhr->bhv' fp32 que
// cuBLAS sert à M=12 par gemmSN_TN (SIMT, 11,5 µs de latence pour 63 MFLOP)
// et la conversion bf16 de sa sortie. Un bloc par (b, h) : les R colonnes
// d'o_lat sont combinées par morceaux de CH (même arithmétique que
// mla_1p_combine_kernel, colonne par colonne : acc·inv) dans o_sh, puis
// chaque warp fait DV/8 lignes de v_b : la voie lit 8 bf16 contigus (uint4)
// tous les 256 r, produit fp32 (les bf16 de v_b convertis exactement — ce
// que lisait _v_b32), somme par arbre de shuffles ; bf16 par arrondi au plus
// proche = le .to(bf16) de torch sur le fp32 de l'einsum. Ordre des sommes ≠
// cuBLAS : ± quelques ulp fp32 avant l'arrondi bf16, jamais au bit (juge :
// ppl-decode-kv au lot de 12 ± 0,002 et test carte ≤ 4 ulp fp32 contre
// l'einsum ; REGLES § 4 bis). Octets : v_b bf16 128 Kio par (b, h), 20 × 12
// = 240 blocs lisent 2,6 Mo uniques par couche (L2 pour les 12 b).
constexpr int MLA1P_R_MAX = 2 * MLA1P_FILS;      // R ≤ 512 (borne du lanceur) : o_sh
template <int SL>
__global__ void __launch_bounds__(MLA1P_CMB_FILS) mla_1p_combine_vb_kernel(
        const float *__restrict__ ws,            // [B, S, H, R+2]
        const __nv_bfloat16 *__restrict__ v_b,   // [H, DV, R]
        __nv_bfloat16 *__restrict__ y,           // [B, H, DV]
        int H, int R, int S, int DV) {
    constexpr int CH = MLA1P_CMB_FILS / SL;
    __shared__ float w_s[MLA1P_S_MAX];
    __shared__ float red[MLA1P_CMB_FILS / 32];
    __shared__ float part[SL][CH];
    __shared__ __align__(16) float o_sh[MLA1P_R_MAX];
    const int bh = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int sl = tid / CH, cc = tid % CH;
    const int b = bh / H, h = bh - b * H;
    const float *base = ws + ((size_t)b * S * H + h) * (R + 2);
    const size_t pas = (size_t)H * (R + 2);
    // --- M, w_s, Lsum : à l'identique de mla_1p_combine_kernel ---
    float mloc = -INFINITY;
    for (int s = tid; s < S; s += MLA1P_CMB_FILS) mloc = fmaxf(mloc, base[s * pas + R]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, off));
    if (lane == 0) red[warp] = mloc;
    __syncthreads();
    float M = red[0];
    #pragma unroll
    for (int w = 1; w < MLA1P_CMB_FILS / 32; ++w) M = fmaxf(M, red[w]);
    __syncthreads();
    float lloc = 0.f;
    for (int s = tid; s < S; s += MLA1P_CMB_FILS) {
        const float m = base[s * pas + R];
        const float w = m != -INFINITY ? __expf(m - M) : 0.f;
        w_s[s] = w;
        lloc += base[s * pas + R + 1] * w;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) lloc += __shfl_xor_sync(0xffffffffu, lloc, off);
    if (lane == 0) red[warp] = lloc;
    __syncthreads();
    float Lsum = 0.f;
    #pragma unroll
    for (int w = 0; w < MLA1P_CMB_FILS / 32; ++w) Lsum += red[w];
    const float inv = Lsum > 0.f ? 1.f / Lsum : 0.f;
    // --- o_sh[col] = (Σ_s o_s[col]·w_s)·inv, par morceaux de CH colonnes ---
    for (int c0 = 0; c0 < R; c0 += CH) {
        const int col = c0 + cc;
        float acc = 0.f;
        if (col < R) {
            const float *oc = base + col;
            int s = sl;
            for (; s + 7 * SL < S; s += 8 * SL) {
                float v[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) v[k] = oc[(size_t)(s + k * SL) * pas];
                #pragma unroll
                for (int k = 0; k < 8; ++k) acc += v[k] * w_s[s + k * SL];
            }
            for (; s < S; s += SL) acc += oc[(size_t)s * pas] * w_s[s];
        }
        if (SL == 1) {
            if (col < R) o_sh[col] = acc * inv;
        } else {
            part[sl][cc] = acc;
            __syncthreads();
            if (tid < CH && col < R) {
                float t = 0.f;
                #pragma unroll
                for (int k = 0; k < SL; ++k) t += part[k][tid];
                o_sh[col] = t * inv;
            }
            __syncthreads();                            // part réutilisé au morceau suivant
        }
    }
    __syncthreads();
    // --- y[b,h,v] = Σ_r v_b[h,v,r]·o_sh[r] : une ligne de v_b par warp ---
    const __nv_bfloat16 *vb = v_b + (size_t)h * DV * R;
    for (int v = warp; v < DV; v += MLA1P_CMB_FILS / 32) {
        const __nv_bfloat16 *row = vb + (size_t)v * R;
        float p = 0.f;
        for (int r = lane * 8; r < R; r += 256) {
            const uint4 u = *reinterpret_cast<const uint4 *>(row + r);
            const __nv_bfloat162 *p2 = reinterpret_cast<const __nv_bfloat162 *>(&u);
            const float4 o0 = *reinterpret_cast<const float4 *>(o_sh + r);
            const float4 o1 = *reinterpret_cast<const float4 *>(o_sh + r + 4);
            float2 f;
            f = __bfloat1622float2(p2[0]); p += f.x * o0.x; p += f.y * o0.y;
            f = __bfloat1622float2(p2[1]); p += f.x * o0.z; p += f.y * o0.w;
            f = __bfloat1622float2(p2[2]); p += f.x * o1.x; p += f.y * o1.y;
            f = __bfloat1622float2(p2[3]); p += f.x * o1.z; p += f.y * o1.w;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) p += __shfl_xor_sync(0xffffffffu, p, off);
        if (lane == 0) y[((size_t)b * H + h) * DV + v] = __float2bfloat16(p);
    }
}

// Nombre de SM de la carte courante (lu une fois par carte) : la grille de
// mla_1p se dimensionne dessus, pas sur une constante.
static int mla_1p_sm_count() {
    static int cache[16] = {0};
    int dev = 0; cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 170;
    if (cache[dev] == 0) {
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) n = 170;
        cache[dev] = n;
    }
    return cache[dev];
}

// Chantier C14 (sage-glm-decode-budget-c14-c15-19-09) : l'ancienne règle
// « ≈ 64 lignes par CTA, S ≤ 32 » supposait 2 CTA par SM — faux : 88 Ko de
// mémoire partagée à H=20, TL=32 (formule du lanceur) n'en laissent qu'UN
// par SM (≤ 100 ou 128 Ko par SM selon la génération, ≤ 99 Ko par bloc). À
// b=1 sur le godet 512 mesuré par nsys (256 jetons d'invite + 70 pas,
// graphs.py:397) elle lançait 8 CTA sur 170 SM, chacun enchaînant 2 tuiles :
// 44 µs par couche, la même durée à b=12 (96 CTA, une vague) — le noyau est
// borné par la chaîne de latence d'un CTA, pas par les octets.
// Règle : autant de tranches que possible tant que le LOT tient en une vague
// (B·S ≤ SM, 1 CTA/SM), au plus une tranche par tuile, puis S ramené au
// nombre de tranches réellement non vides (rows arrondi à TL). Le même
// arrondi est refait par le lanceur pour rows_par_cta.
//   b=1  : L=512 → S=tuiles (16 à TL=32, 32 à TL=16) ; L=1 024 → 32 / 64 ;
//          L=2 048 → 64 / 128 ; L=4 096 → 128 (TL=32).
//   b=12 : L=512 → 8 (inchangé) ; L=2 048 → 13 (5 tuiles par CTA, une vague,
//          contre 384 CTA en 2,3 vagues avant).
// Plafond de S : rien d'autre ne le borne — ws vaut B·S·H·(R+2) floats
// (≤ SM·H·(R+2)·4 o = 7 Mo à H=20, R=512), mla_1p_combine boucle sur S sans
// borne (poids en shared : MLA1P_S_MAX), blockIdx.y ≤ 65 535.
static int mla_1p_tranches(int L, int TL, int B) {
    const int tuiles = (L + TL - 1) / TL;
    const int sm = mla_1p_sm_count();
    int S = max(1, min(tuiles, sm / max(1, B)));
    S = min(S, MLA1P_S_MAX);
    const int rows = ((((L + S - 1) / S) + TL - 1) / TL) * TL;   // même arrondi que le lanceur
    S = max(1, (L + rows - 1) / rows);                            // tranches non vides seulement
    return S;
}

// C14-c (verdict-c14-bis-occupation-19-09) : régime FIN, b ≤ 2 et H ≤ 20.
// La règle ci-dessus donnait 32 blocs pour 170 SM à L=512 (une tranche de
// 16 lignes par bloc, 1 CTA/SM par la shared) ; ici la grille est
// B × S × G avec G = ⌈H/5⌉ groupes de têtes (4 à H=20) et des tranches de
// TL=8 lignes, S = min(⌈L/8⌉, ⌊2·SM / (B·G)⌋) puis le même point fixe sur
// rows (tranches non vides). Le plafond 2·SM/(B·G) : DEUX blocs résidents
// par SM (22 Ko de shared, __launch_bounds__(256, 2) → ≤ 128 registres),
// donc B·S·G ≤ 2·SM tient en une vague ; il borne aussi le combine (S ≤ 85
// à b=1) et ws (B·S·G·… ≤ 2·SM·H/G·(R+2)·4 o = 7 Mo).
// Blocs = B·S·G (SM = 170, H = 20) :
//   b=1 : L=512 → S=64, rows=8  → 256 blocs (32 avant) ; L=1 024 → S=64,
//         rows=16 → 256 ; L=2 048 → S=64, rows=32 → 256 ; L=4 096 → S=74,
//         rows=56 → 296 ; L=128 → S=16 → 64 (contexte court : un bloc
//         coûte sa latence, pas sa grille).
//   b=2 : L=512 → S=32, rows=16 → 256 ; L=2 048 → S=37, rows=56 → 296.
// b ≥ 3 : mla_1p_tranches et <20,32,4> inchangés (b=12 : S=8, 96 blocs).
static int mla_1p_groupes_fin(int H) { return (H + MLA1P_FIN_HMAX - 1) / MLA1P_FIN_HMAX; }
static int mla_1p_tranches_fin(int L, int B, int G) {
    const int TL = MLA1P_FIN_TL;
    const int tuiles = (L + TL - 1) / TL;
    const int cap = MLA1P_FIN_BLOCS_SM * mla_1p_sm_count() / max(1, B * G);
    int S = max(1, min(tuiles, cap));
    S = min(S, MLA1P_S_MAX);
    const int rows = ((((L + S - 1) / S) + TL - 1) / TL) * TL;   // même arrondi que le lanceur
    S = max(1, (L + rows - 1) / rows);                            // tranches non vides seulement
    return S;
}

torch::Tensor mla_decode_1p(torch::Tensor q_eff, c10::optional<torch::Tensor> cache_ptrs,
                            c10::optional<torch::Tensor> cache, torch::Tensor lens,
                            int64_t L, int64_t rank, double scale, bool fp8,
                            c10::optional<torch::Tensor> v_b) {
    CHECK_CUDA(q_eff); ACVRAM_DEVICE_GUARD(q_eff); CHECK_CONTIG(q_eff); CHECK_CONTIG(lens);
    TORCH_CHECK(q_eff.dim() == 3 && q_eff.scalar_type() == torch::kFloat, "MLA 1p : q_eff [B, H, W] fp32");
    const int B = q_eff.size(0), H = q_eff.size(1), W = q_eff.size(2), R = (int)rank;
    TORCH_CHECK(W % 8 == 0 && R % 2 == 0 && R <= W && R <= 2 * MLA1P_FILS,
                "MLA 1p : W multiple de 8, R pair <= W et <= 512");
    TORCH_CHECK(!fp8 || W % 16 == 0, "MLA 1p fp8 : W multiple de 16 (foulee W+16 alignee)");
    TORCH_CHECK(H >= 1 && H <= 32, "MLA 1p : 1 <= H <= 32 tetes");
    TORCH_CHECK(lens.scalar_type() == torch::kLong && lens.numel() == B, "MLA 1p : lens [B] int64");
    const int64_t *ptrs = nullptr;
    const __nv_bfloat16 *cache0 = nullptr;
    if (cache_ptrs.has_value() && cache_ptrs->defined()) {
        CHECK_CONTIG(*cache_ptrs);
        TORCH_CHECK(cache_ptrs->scalar_type() == torch::kInt64 && cache_ptrs->is_cuda()
                    && cache_ptrs->numel() == B, "MLA 1p : cache_ptrs [B] int64 sur la carte");
        ptrs = cache_ptrs->data_ptr<int64_t>();
    } else {
        TORCH_CHECK(B == 1 && cache.has_value() && cache->defined(), "MLA 1p : cache [L, W] bf16 si B = 1 sans table");
        CHECK_CONTIG(*cache);
        if (fp8) {
            TORCH_CHECK(cache->scalar_type() == torch::kUInt8 && cache->size(1) == W + MLA_FP8_PAD && cache->size(0) >= L,
                        "MLA 1p fp8 : cache [>= L, W+16] uint8");
        } else {
            TORCH_CHECK(cache->scalar_type() == torch::kBFloat16 && cache->size(1) == W && cache->size(0) >= L,
                        "MLA 1p : cache [>= L, W] bf16");
        }
        cache0 = reinterpret_cast<const __nv_bfloat16 *>(cache->data_ptr());
    }
    // C14-c : régime FIN à b ≤ 2 et H ≤ 20 — <5, 8, 1> (5 têtes par bloc,
    // tuile de 8 lignes, RW=1 : 8 warps × 1 ligne), grille B × S × G ; il
    // remplace la demi-tuile <20,16,2> de C14 (32 blocs à L=512, verdict
    // C14-bis). b ≥ 3 : <20,32,4> et mla_1p_tranches inchangés ; H > 20 :
    // <32,16,2> inchangé (G=1).
    const bool fin = H <= 20 && B <= 2;
    const int HMAX = fin ? MLA1P_FIN_HMAX : (H <= 20 ? 20 : 32);
    const int TL = fin ? MLA1P_FIN_TL : (H <= 20 ? 32 : 16);
    const int G = fin ? mla_1p_groupes_fin(H) : 1;
    const int S = fin ? mla_1p_tranches_fin((int)L, B, G) : mla_1p_tranches((int)L, TL, B);
    const int rows = ((((int)L + S - 1) / S) + TL - 1) / TL * TL;
    auto ws = torch::empty({(long)B * S * H * (R + 2)}, q_eff.options());
    // C14-b : avec v_b [H, DV, R] bf16, le combine rend y = v_b·o_lat en bf16
    // [B, H, DV] (mla_1p_combine_vb_kernel) au lieu d'o_lat fp32 [B, H, R].
    const bool fusion = v_b.has_value() && v_b->defined();
    int DV = 0;
    if (fusion) {
        CHECK_CONTIG(*v_b);
        TORCH_CHECK(v_b->dim() == 3 && v_b->size(0) == H && v_b->size(2) == R && v_b->scalar_type() == torch::kBFloat16,
                    "MLA 1p fusion : v_b [H, DV, R] bf16 contigu");
        TORCH_CHECK(R % 8 == 0 && R <= MLA1P_R_MAX, "MLA 1p fusion : R multiple de 8 et <= 512");
        DV = (int)v_b->size(1);
    }
    auto o = fusion ? torch::empty({B, H, (long)DV}, q_eff.options().dtype(torch::kBFloat16))
                    : torch::empty({B, H, (long)rank}, q_eff.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    // shared d'un bloc : q de ses Hb ≤ HMAX têtes, tuile de TL lignes de (W+8)
    // bf16 (en fp8 TL lignes de (W+16) octets, plus petite), S/P, m/l/a,
    // échelles et table fp8
    const int Hb = min(H, HMAX);
    const size_t shm = (size_t)Hb * W * sizeof(float) + (size_t)TL * (W + 8) * sizeof(__nv_bfloat16)
                       + (size_t)2 * HMAX * TL * sizeof(float) + 3 * HMAX * sizeof(float)
                       + (size_t)(TL + 256) * sizeof(float);
    dim3 grid(B, S, G);
    #define MLA1P_LANCE_K(HM, T, RWW, F8, MB) do { \
        static size_t autorise = 0; \
        if (shm > autorise) { cudaFuncSetAttribute(mla_1p_kernel<HM, T, RWW, F8, MB>, \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); autorise = shm; } \
        mla_1p_kernel<HM, T, RWW, F8, MB><<<grid, MLA1P_FILS, shm, stream>>>( \
            q_eff.data_ptr<float>(), ptrs, cache0, lens.data_ptr<long>(), ws.data_ptr<float>(), \
            H, (int)L, W, R, S, rows, (float)scale); } while (0)
    #define MLA1P_LANCE(HM, T, RWW, F8) MLA1P_LANCE_K(HM, T, RWW, F8, 0)
    #define MLA1P_LANCE_FIN(HM, T, RWW, F8) MLA1P_LANCE_K(HM, T, RWW, F8, MLA1P_FIN_BLOCS_SM)
    TORCH_CHECK(shm <= 99 * 1024, "MLA 1p : memoire partagee > 99 Ko (H, W trop grands)");
    if (fin)             { if (fp8) MLA1P_LANCE_FIN(5, 8, 1, true); else MLA1P_LANCE_FIN(5, 8, 1, false); }
    else if (HMAX == 20) { if (fp8) MLA1P_LANCE(20, 32, 4, true); else MLA1P_LANCE(20, 32, 4, false); }
    else                 { if (fp8) MLA1P_LANCE(32, 16, 2, true); else MLA1P_LANCE(32, 16, 2, false); }
    #undef MLA1P_LANCE_FIN
    #undef MLA1P_LANCE
    #undef MLA1P_LANCE_K
    TORCH_CHECK(S <= MLA1P_S_MAX, "MLA 1p : S > MLA1P_S_MAX");
    // Combine C14-c : SL voies de tranches par bloc, ≈ S/SL ≤ 8 partiels par
    // fil ; grille (B·H, ⌈R/CH⌉) — b=1, S=64, R=512 : SL=8, CH=32, 320 blocs ;
    // b=12, S=8 : SL=1, CH=256, 480 blocs (240 de 128 fils avant).
    const int SL = S >= 64 ? 8 : S >= 32 ? 4 : S >= 16 ? 2 : 1;
    const int CH = MLA1P_CMB_FILS / SL;
    if (fusion) {
        // C14-b : un bloc par (b, h), les R colonnes puis v_b — b=12, H=20 : 240 blocs
        const float *w = ws.data_ptr<float>();
        const __nv_bfloat16 *vb = reinterpret_cast<const __nv_bfloat16 *>(v_b->data_ptr());
        __nv_bfloat16 *yy = reinterpret_cast<__nv_bfloat16 *>(o.data_ptr());
        switch (SL) {
            case 8:  mla_1p_combine_vb_kernel<8><<<B * H, MLA1P_CMB_FILS, 0, stream>>>(w, vb, yy, H, R, S, DV); break;
            case 4:  mla_1p_combine_vb_kernel<4><<<B * H, MLA1P_CMB_FILS, 0, stream>>>(w, vb, yy, H, R, S, DV); break;
            case 2:  mla_1p_combine_vb_kernel<2><<<B * H, MLA1P_CMB_FILS, 0, stream>>>(w, vb, yy, H, R, S, DV); break;
            default: mla_1p_combine_vb_kernel<1><<<B * H, MLA1P_CMB_FILS, 0, stream>>>(w, vb, yy, H, R, S, DV); break;
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return o;
    }
    dim3 g2(B * H, (R + CH - 1) / CH);
    switch (SL) {
        case 8:  mla_1p_combine_kernel<8><<<g2, MLA1P_CMB_FILS, 0, stream>>>(ws.data_ptr<float>(), o.data_ptr<float>(), H, R, S); break;
        case 4:  mla_1p_combine_kernel<4><<<g2, MLA1P_CMB_FILS, 0, stream>>>(ws.data_ptr<float>(), o.data_ptr<float>(), H, R, S); break;
        case 2:  mla_1p_combine_kernel<2><<<g2, MLA1P_CMB_FILS, 0, stream>>>(ws.data_ptr<float>(), o.data_ptr<float>(), H, R, S); break;
        default: mla_1p_combine_kernel<1><<<g2, MLA1P_CMB_FILS, 0, stream>>>(ws.data_ptr<float>(), o.data_ptr<float>(), H, R, S); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

// Préparation MLA du pas pour les B créneaux en UN lancement (sage-duel
// verdict § 6.2, marche « RoPE + cat » : 2 541 lancements/pas mesurés contre
// ≤ 2 500 scellés). Remplace, par couche : RoPE (≈ 10 élémentaires), trois
// cat, l'einsum k_b (bmm + copies), la norme kv_a et la conversion fp32 de
// q_eff. Même arithmétique que le chemin torch, opération par opération :
//  - RoPE « norm » (paires 2i, 2i+1) avec cos/sin bf16 (tables32 = les
//    tables bf16 en fp32) : y0 = bf16(bf16(x0·c) − bf16(x1·s)), y1 =
//    bf16(bf16(x0·s) + bf16(x1·c)) — ce que font deux produits et une somme
//    bf16 de torch ;
//  - q_abs[b,h,r] = Σ_n k_b[h,r,n]·q_nope[b,h,n] accumulé en fp32, arrondi
//    bf16 (la sortie de l'einsum) puis écrit en fp32 (le .to(float32) de
//    q_eff) — seul poste où l'ordre des sommes diffère de cuBLAS (≤ 1 ulp) ;
//  - norme kv_a : la réduction de rmsnorm_bf16_kernel à 256 fils, à
//    l'identique (même découpe, même arbre de shuffles) — bit-identique.
// Grille 1D : nh × (rank/64) blocs pour q (un bloc = 64 lignes de k_b[h] pour
// tous les b ; les blocs de première tranche tournent aussi q_pe), puis B
// blocs pour k_new (norme + RoPE d'un créneau).
//
// C14-b (chantier-c14b-19-09 « Fait le 20/09 ») : cette grille faisait 172
// blocs de 256 fils pour 170 SM à b=12 (un bloc par SM, 8 warps), chaque fil
// lisant ses 32 bf16 de k_b un par un, à foulée 4, depuis la mémoire globale
// (L1 après le premier b) — 20,6 µs par couche pour 2,6 Mo : latence, pas
// débit. Le noyau ci-dessous garde la MÊME arithmétique par (b, r) (même fil
// (r, quart), même ordre de somme sur n, mêmes deux shuffles) et change deux
// choses : (1) la tuile k_b[h][r0..r0+64] est chargée UNE fois en shared par
// lectures uint4 coalescées (rangées rembourrées à nope+8 : 8 rangées d'un
// warp sur des banques distinctes), les produits lisent la shared ; (2) les B
// créneaux sont découpés en groupes de MLAP_BG (4) : grille nh·NR·⌈B/BG⌉ + B
// = 492 blocs à b=12 (≈ 3 par SM, 24 warps), 172 inchangés à b=1. Le noyau
// d'avant reste sous `temoin=True` (mla_prep_batch_temoin_kernel) : le test
// carte compare les deux au bit.
constexpr int MLAP_FILS = 256, MLAP_LIGNES = 64, MLAP_BG = 4;
__global__ void __launch_bounds__(MLAP_FILS) mla_prep_batch_kernel(
    const __nv_bfloat16 *__restrict__ q,       // [B, nh, nope+rope]
    const __nv_bfloat16 *__restrict__ kvp,     // [B, rank+rope]
    const long *__restrict__ lens,             // [B] positions
    const float *__restrict__ cos32,           // [max_pos, rope] (nullptr : pas de RoPE)
    const float *__restrict__ sin32,
    const __nv_bfloat16 *__restrict__ k_b,     // [nh, rank, nope]
    const __nv_bfloat16 *__restrict__ w_norm,  // [rank]
    float *__restrict__ q_eff,                 // [B, nh, rank+rope] fp32
    __nv_bfloat16 *__restrict__ k_new,         // [B, rank+rope]
    int B, int nh, int nope, int rope, int rank, float eps, int GB) {
    extern __shared__ __align__(16) unsigned char mlap_smem[];
    const int W = rank + rope, QW = nope + rope, NR = rank / MLAP_LIGNES;
    const int tid = threadIdx.x;
    auto rot = [&](float x0, float x1, float c, float s, float &y0, float &y1) {
        const float a = __bfloat162float(__float2bfloat16(x0 * c));
        const float bb = __bfloat162float(__float2bfloat16(x1 * s));
        const float cc = __bfloat162float(__float2bfloat16(x0 * s));
        const float d = __bfloat162float(__float2bfloat16(x1 * c));
        y0 = __bfloat162float(__float2bfloat16(a - bb));
        y1 = __bfloat162float(__float2bfloat16(cc + d));
    };
    if ((int)blockIdx.x < nh * NR * GB) {
        const int hb = blockIdx.x / GB, g = blockIdx.x - hb * GB;
        const int h = hb / NR, r0 = (hb - h * NR) * MLAP_LIGNES;
        const int b0 = g * MLAP_BG, b1 = min(B, b0 + MLAP_BG);
        const int PADR = nope + 8;                                         // rangée de la tuile (bf16)
        __nv_bfloat16 *ks = reinterpret_cast<__nv_bfloat16 *>(mlap_smem);   // [64][PADR]
        float *qn = reinterpret_cast<float *>(mlap_smem + (size_t)MLAP_LIGNES * PADR * sizeof(__nv_bfloat16));  // [BG][nope]
        // tuile k_b[h][r0 .. r0+64] : 64 × nope/8 uint4, coalescés
        const int c8 = nope / 8;
        for (int i = tid; i < MLAP_LIGNES * c8; i += MLAP_FILS) {
            const int r = i / c8, c = i - r * c8;
            *reinterpret_cast<uint4 *>(ks + r * PADR + c * 8) =
                *reinterpret_cast<const uint4 *>(k_b + ((size_t)h * rank + r0 + r) * nope + c * 8);
        }
        for (int i = tid; i < (b1 - b0) * nope; i += MLAP_FILS) {
            const int b = i / nope, n = i - b * nope;
            qn[i] = __bfloat162float(q[((size_t)(b0 + b) * nh + h) * QW + n]);
        }
        __syncthreads();
        const int r = tid >> 2, quart = tid & 3;                          // 64 lignes × 4 quarts de nope
        const __nv_bfloat16 *kr = ks + r * PADR;
        for (int b = b0; b < b1; ++b) {
            float acc = 0.f;
            const float *qb = qn + (b - b0) * nope;
            for (int n = quart; n < nope; n += 4)
                acc += __bfloat162float(kr[n]) * qb[n];
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            if (quart == 0)
                q_eff[((size_t)b * nh + h) * W + r0 + r] = __bfloat162float(__float2bfloat16(acc));
        }
        if (r0 == 0 && cos32 != nullptr) {                                 // q_pe tourné, les b du groupe
            const int paires = rope / 2;
            for (int i = tid; i < (b1 - b0) * paires; i += MLAP_FILS) {
                const int b = b0 + i / paires, j = i - (b - b0) * paires;
                const long pos = lens[b];
                const __nv_bfloat16 *src = q + ((size_t)b * nh + h) * QW + nope + 2 * j;
                float y0, y1;
                rot(__bfloat162float(src[0]), __bfloat162float(src[1]),
                    cos32[pos * rope + j], sin32[pos * rope + j], y0, y1);
                float *dst = q_eff + ((size_t)b * nh + h) * W + rank + 2 * j;
                dst[0] = y0; dst[1] = y1;
            }
        } else if (r0 == 0) {
            for (int i = tid; i < (b1 - b0) * rope; i += MLAP_FILS) {
                const int b = b0 + i / rope, j = i - (b - b0) * rope;
                q_eff[((size_t)b * nh + h) * W + rank + j] =
                    __bfloat162float(q[((size_t)b * nh + h) * QW + nope + j]);
            }
        }
        return;
    }
    // --- k_new[b] : norme kv_a (rmsnorm_bf16 à 256 fils, à l'identique) + RoPE de k_pe ---
    const int b = blockIdx.x - nh * NR * GB;
    __shared__ float red[32];
    const __nv_bfloat16 *xr = kvp + (size_t)b * W;
    float ss = 0.f;
    for (int i = tid; i < rank; i += MLAP_FILS) { const float v = __bfloat162float(xr[i]); ss += v * v; }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.f;
    for (int k = 0; k < MLAP_FILS / 32; ++k) ss += red[k];
    const float rs = rsqrtf(ss / (float)rank + eps);
    __nv_bfloat16 *out = k_new + (size_t)b * W;
    for (int i = tid; i < rank; i += MLAP_FILS) {
        const float n = __bfloat162float(__float2bfloat16(__bfloat162float(xr[i]) * rs));
        out[i] = __float2bfloat16(n * __bfloat162float(w_norm[i]));
    }
    if (cos32 != nullptr) {
        const long pos = lens[b];
        for (int j = tid; j < rope / 2; j += MLAP_FILS) {
            float y0, y1;
            rot(__bfloat162float(xr[rank + 2 * j]), __bfloat162float(xr[rank + 2 * j + 1]),
                cos32[pos * rope + j], sin32[pos * rope + j], y0, y1);
            out[rank + 2 * j] = __float2bfloat16(y0); out[rank + 2 * j + 1] = __float2bfloat16(y1);
        }
    } else {
        for (int j = tid; j < rope; j += MLAP_FILS) out[rank + j] = xr[rank + j];
    }
}

// Témoin C14-b : la grille d'avant (nh·NR + B blocs, k_b lu depuis la
// mémoire globale), inchangée — lancée sous `temoin=True`, comparée au bit.
__global__ void __launch_bounds__(MLAP_FILS) mla_prep_batch_temoin_kernel(
    const __nv_bfloat16 *__restrict__ q,       // [B, nh, nope+rope]
    const __nv_bfloat16 *__restrict__ kvp,     // [B, rank+rope]
    const long *__restrict__ lens,             // [B] positions
    const float *__restrict__ cos32,           // [max_pos, rope] (nullptr : pas de RoPE)
    const float *__restrict__ sin32,
    const __nv_bfloat16 *__restrict__ k_b,     // [nh, rank, nope]
    const __nv_bfloat16 *__restrict__ w_norm,  // [rank]
    float *__restrict__ q_eff,                 // [B, nh, rank+rope] fp32
    __nv_bfloat16 *__restrict__ k_new,         // [B, rank+rope]
    int B, int nh, int nope, int rope, int rank, float eps) {
    extern __shared__ __align__(16) unsigned char mlap_smem[];
    const int W = rank + rope, QW = nope + rope, NR = rank / MLAP_LIGNES;
    const int tid = threadIdx.x;
    auto rot = [&](float x0, float x1, float c, float s, float &y0, float &y1) {
        const float a = __bfloat162float(__float2bfloat16(x0 * c));
        const float bb = __bfloat162float(__float2bfloat16(x1 * s));
        const float cc = __bfloat162float(__float2bfloat16(x0 * s));
        const float d = __bfloat162float(__float2bfloat16(x1 * c));
        y0 = __bfloat162float(__float2bfloat16(a - bb));
        y1 = __bfloat162float(__float2bfloat16(cc + d));
    };
    if ((int)blockIdx.x < nh * NR) {
        const int h = blockIdx.x / NR, r0 = (blockIdx.x - h * NR) * MLAP_LIGNES;
        float *qn = reinterpret_cast<float *>(mlap_smem);                 // [B][nope] fp32
        for (int i = tid; i < B * nope; i += MLAP_FILS) {
            const int b = i / nope, n = i - b * nope;
            qn[i] = __bfloat162float(q[((size_t)b * nh + h) * QW + n]);
        }
        __syncthreads();
        const int r = tid >> 2, quart = tid & 3;                          // 64 lignes × 4 quarts de nope
        const __nv_bfloat16 *kr = k_b + ((size_t)h * rank + r0 + r) * nope;
        for (int b = 0; b < B; ++b) {
            float acc = 0.f;
            for (int n = quart; n < nope; n += 4)
                acc += __bfloat162float(kr[n]) * qn[b * nope + n];
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            if (quart == 0)
                q_eff[((size_t)b * nh + h) * W + r0 + r] = __bfloat162float(__float2bfloat16(acc));
        }
        if (r0 == 0 && cos32 != nullptr) {                                 // q_pe tourné, tous les b
            const int paires = rope / 2;
            for (int i = tid; i < B * paires; i += MLAP_FILS) {
                const int b = i / paires, j = i - b * paires;
                const long pos = lens[b];
                const __nv_bfloat16 *src = q + ((size_t)b * nh + h) * QW + nope + 2 * j;
                float y0, y1;
                rot(__bfloat162float(src[0]), __bfloat162float(src[1]),
                    cos32[pos * rope + j], sin32[pos * rope + j], y0, y1);
                float *dst = q_eff + ((size_t)b * nh + h) * W + rank + 2 * j;
                dst[0] = y0; dst[1] = y1;
            }
        } else if (r0 == 0) {
            for (int i = tid; i < B * rope; i += MLAP_FILS) {
                const int b = i / rope, j = i - b * rope;
                q_eff[((size_t)b * nh + h) * W + rank + j] =
                    __bfloat162float(q[((size_t)b * nh + h) * QW + nope + j]);
            }
        }
        return;
    }
    // --- k_new[b] : norme kv_a (rmsnorm_bf16 à 256 fils, à l'identique) + RoPE de k_pe ---
    const int b = blockIdx.x - nh * NR;
    __shared__ float red[32];
    const __nv_bfloat16 *xr = kvp + (size_t)b * W;
    float ss = 0.f;
    for (int i = tid; i < rank; i += MLAP_FILS) { const float v = __bfloat162float(xr[i]); ss += v * v; }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.f;
    for (int k = 0; k < MLAP_FILS / 32; ++k) ss += red[k];
    const float rs = rsqrtf(ss / (float)rank + eps);
    __nv_bfloat16 *out = k_new + (size_t)b * W;
    for (int i = tid; i < rank; i += MLAP_FILS) {
        const float n = __bfloat162float(__float2bfloat16(__bfloat162float(xr[i]) * rs));
        out[i] = __float2bfloat16(n * __bfloat162float(w_norm[i]));
    }
    if (cos32 != nullptr) {
        const long pos = lens[b];
        for (int j = tid; j < rope / 2; j += MLAP_FILS) {
            float y0, y1;
            rot(__bfloat162float(xr[rank + 2 * j]), __bfloat162float(xr[rank + 2 * j + 1]),
                cos32[pos * rope + j], sin32[pos * rope + j], y0, y1);
            out[rank + 2 * j] = __float2bfloat16(y0); out[rank + 2 * j + 1] = __float2bfloat16(y1);
        }
    } else {
        for (int j = tid; j < rope; j += MLAP_FILS) out[rank + j] = xr[rank + j];
    }
}

std::vector<torch::Tensor> mla_prep_batch(torch::Tensor q, torch::Tensor kvp, torch::Tensor lens,
                                          c10::optional<torch::Tensor> cos32, c10::optional<torch::Tensor> sin32,
                                          torch::Tensor k_b, torch::Tensor w_norm,
                                          int64_t nope, int64_t rope, int64_t rank, double eps, bool temoin) {
    CHECK_CUDA(q); ACVRAM_DEVICE_GUARD(q);
    for (auto &z : {q, kvp, lens, k_b, w_norm}) CHECK_CONTIG(z);
    TORCH_CHECK(q.dim() == 3 && q.scalar_type() == torch::kBFloat16 && q.size(2) == nope + rope,
                "prep MLA : q [B, nh, nope+rope] bf16");
    const int B = q.size(0), nh = q.size(1), W = (int)(rank + rope);
    TORCH_CHECK(kvp.dim() == 2 && kvp.size(0) == B && kvp.size(1) == W && kvp.scalar_type() == torch::kBFloat16,
                "prep MLA : kvp [B, rank+rope] bf16");
    TORCH_CHECK(rank % MLAP_LIGNES == 0 && rope % 2 == 0 && nope % 4 == 0 && B * nope * 4 <= 48 * 1024,
                "prep MLA : rank multiple de 64, rope pair, nope multiple de 4, B*nope*4 o <= 48 Ko");
    TORCH_CHECK(k_b.dim() == 3 && k_b.size(0) == nh && k_b.size(1) == rank && k_b.size(2) == nope
                && k_b.scalar_type() == torch::kBFloat16, "prep MLA : k_b [nh, rank, nope] bf16");
    TORCH_CHECK(w_norm.numel() == rank && w_norm.scalar_type() == torch::kBFloat16, "prep MLA : w_norm [rank] bf16");
    TORCH_CHECK(lens.scalar_type() == torch::kLong && lens.numel() == B, "prep MLA : lens [B] int64");
    const float *c32 = nullptr, *s32 = nullptr;
    if (cos32.has_value() && cos32->defined()) {
        CHECK_CONTIG(*cos32); CHECK_CONTIG(*sin32);
        TORCH_CHECK(cos32->scalar_type() == torch::kFloat && cos32->dim() == 2 && cos32->size(1) == rope
                    && sin32->sizes() == cos32->sizes(), "prep MLA : cos32/sin32 [max_pos, rope] fp32");
        c32 = cos32->data_ptr<float>(); s32 = sin32->data_ptr<float>();
    }
    auto q_eff = torch::empty({B, nh, W}, q.options().dtype(torch::kFloat));
    auto k_new = torch::empty({B, W}, q.options());
    const int NR = (int)rank / MLAP_LIGNES;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (temoin) {
        const size_t shm = (size_t)B * nope * sizeof(float);
        mla_prep_batch_temoin_kernel<<<(unsigned)(nh * NR + B), MLAP_FILS, shm, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16 *>(kvp.data_ptr()), lens.data_ptr<long>(), c32, s32,
            reinterpret_cast<const __nv_bfloat16 *>(k_b.data_ptr()),
            reinterpret_cast<const __nv_bfloat16 *>(w_norm.data_ptr()),
            q_eff.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(k_new.data_ptr()),
            B, nh, (int)nope, (int)rope, (int)rank, (float)eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {q_eff, k_new};
    }
    // C14-b : tuile k_b [64][nope+8] bf16 + q_nope [BG][nope] fp32 en shared
    // (nope=128 : 17 408 + 2 048 o), groupes de MLAP_BG créneaux
    TORCH_CHECK(nope % 8 == 0, "prep MLA : nope multiple de 8 (tuile k_b en uint4)");
    const int GB = (B + MLAP_BG - 1) / MLAP_BG;
    const size_t shm = (size_t)MLAP_LIGNES * (nope + 8) * sizeof(__nv_bfloat16) + (size_t)MLAP_BG * nope * sizeof(float);
    TORCH_CHECK(shm <= 48 * 1024, "prep MLA : shared > 48 Ko (nope trop grand)");
    mla_prep_batch_kernel<<<(unsigned)(nh * NR * GB + B), MLAP_FILS, shm, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(kvp.data_ptr()), lens.data_ptr<long>(), c32, s32,
        reinterpret_cast<const __nv_bfloat16 *>(k_b.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(w_norm.data_ptr()),
        q_eff.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(k_new.data_ptr()),
        B, nh, (int)nope, (int)rope, (int)rank, (float)eps, GB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_eff, k_new};
}

// RMSNorm fusionnée (bf16 -> bf16, variance en fp32) : un bloc par ligne.
// Même arithmétique que la version torch : x normalisé arrondi en bf16, puis
// produit bf16 par le poids — bit-identique.
// ``res`` (optionnel) est le résidu à ajouter avant de normaliser : la somme
// part dans ``xn`` — le résidu de la couche suivante — et sa normalisation
// dans ``y``. Un lancement au lieu de deux, sur des tenseurs de quelques
// kilooctets où la latence pèse plus que le calcul.
__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16 *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ w,
                                    __nv_bfloat16 *__restrict__ y,
                                    const __nv_bfloat16 *__restrict__ res,
                                    __nv_bfloat16 *__restrict__ xn,
                                    float mult, int H, float eps) {
    __shared__ float red[32];
    const __nv_bfloat16 *xr = x + (size_t)blockIdx.x * H;
    __nv_bfloat16 *yr = y + (size_t)blockIdx.x * H;
    if (res != nullptr) {
        const __nv_bfloat16 *rr = res + (size_t)blockIdx.x * H;
        __nv_bfloat16 *nr = xn + (size_t)blockIdx.x * H;
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            nr[i] = __float2bfloat16(__bfloat162float(rr[i])
                                     + mult * __bfloat162float(xr[i]));
        __syncthreads();
        xr = nr;
    }
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

std::vector<torch::Tensor> rmsnorm_bf16(torch::Tensor x, torch::Tensor w, double eps,
                                        c10::optional<torch::Tensor> res,
                                        double mult) {
    CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(x);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16,
                "rmsnorm_bf16 : bf16 attendu");
    auto xc = x.contiguous();
    const int H = xc.size(-1);
    const long R = xc.numel() / H;
    auto y = torch::empty_like(xc);
    // Un bloc par ligne : à H = 2048 il ne reste que 256 fils pour 2048
    // éléments. Élargir le bloc raccourcit la réduction, seul coût réel ici.
    const int th = H >= 2048 ? 1024 : (H >= 1024 ? 512 : 256);
    torch::Tensor xn;
    const __nv_bfloat16 *pres = nullptr;
    __nv_bfloat16 *pxn = nullptr;
    if (res.has_value()) {
        auto rc = res->contiguous();
        TORCH_CHECK(rc.numel() == xc.numel(), "rmsnorm_bf16 : residu de meme taille");
        xn = torch::empty_like(xc);
        pres = reinterpret_cast<const __nv_bfloat16 *>(rc.data_ptr());
        pxn = reinterpret_cast<__nv_bfloat16 *>(xn.data_ptr());
        rmsnorm_bf16_kernel<<<(unsigned)R, th, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()),
            reinterpret_cast<const __nv_bfloat16 *>(w.contiguous().data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), pres, pxn,
            (float)mult, H, (float)eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {y, xn};
    }
    rmsnorm_bf16_kernel<<<(unsigned)R, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(w.contiguous().data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), nullptr, nullptr,
        1.f, H, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y};
}

// C15-prefill (chantier-c15-prefill-20-09) : la MÊME RMSNorm, un WARP par ligne
// au lieu d'un bloc. Au préfill (R = 2 047 lignes de H = 2 048) le noyau à bloc
// lance 2 047 blocs de 1 024 fils qui lisent chacun 4 Kio, avec deux
// __syncthreads et un tampon partagé : 21,9 µs par appel, 0,77 To/s (nsys P2
// 19/09). Ici huit lignes par bloc de 256 fils, sans mémoire partagée. La somme
// des carrés REJOUE l'ordre exact du noyau à bloc de TH fils : partiels par fil
// t = 32·wq + lane (éléments t, t + TH, …, la même expression `ss += v * v` —
// contractée en FFMA par les mêmes drapeaux), arbre xor par warp (les mêmes
// décalages 16, 8, 4, 2, 1), somme séquentielle des TH/32 warps depuis 0 ;
// rsqrtf(ss / H + eps) écrit à l'identique : mêmes flottants, mêmes bits que
// rmsnorm_bf16_kernel (juge : tests/test_prefill_compact.py sur carte, puis la
// PPL au bit). Le résidu suit la même formule que le noyau à bloc (écrit dans
// xn puis relu, __syncwarp entre les deux).
// Première version (a3a79e70, mesurée par Manon 07 h 22 : 26,4 µs par appel,
// PLUS LENTE que le bloc à 21,9) : une boucle `#pragma unroll 1` sur les 32
// warps rejoués, deux chargements par tour puis l'arbre — 32 latences de
// mémoire en série par ligne, et 2 048 warps pour 170 SM (12 par SM) ne
// couvrent rien. Ici la ligne entière est chargée d'abord dans des registres
// (EPT = éléments par fil rejoué, H / 32 flottants par lane : 64 à H = 2048,
// chargements indépendants émis d'un bloc), puis la somme est faite dans le
// même ordre qu'avant depuis les registres ; la phase de sortie relit les
// registres au lieu de la mémoire. Réservé à H ≤ 2 048 (au-delà, le bloc).
template <int TH, int EPT>
__global__ void rmsnorm_bf16_warp_kernel(const __nv_bfloat16 *__restrict__ x,
                                         const __nv_bfloat16 *__restrict__ w,
                                         __nv_bfloat16 *__restrict__ y,
                                         const __nv_bfloat16 *__restrict__ res,
                                         __nv_bfloat16 *__restrict__ xn,
                                         float mult, int H, float eps, long R) {
    constexpr int NW = TH / 32;                        // warps rejoués du bloc
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= R) return;
    const __nv_bfloat16 *xr = x + row * H;
    __nv_bfloat16 *yr = y + row * H;
    float v[NW * EPT];                                 // v[wq * EPT + r] = x[32·wq + lane + TH·r]
#pragma unroll
    for (int wq = 0; wq < NW; ++wq)
#pragma unroll
        for (int r = 0; r < EPT; ++r) {
            const int i = 32 * wq + lane + TH * r;
            v[wq * EPT + r] = (i < H) ? __bfloat162float(xr[i]) : 0.f;
        }
    if (res != nullptr) {
        // la somme bf16(res + mult·x) comme le bloc (FFMA), écrite dans xn et
        // gardée en registres — la valeur relue par le bloc est celle-ci
        const __nv_bfloat16 *rr = res + row * H;
        __nv_bfloat16 *nr = xn + row * H;
#pragma unroll
        for (int wq = 0; wq < NW; ++wq)
#pragma unroll
            for (int r = 0; r < EPT; ++r) {
                const int i = 32 * wq + lane + TH * r;
                if (i < H) {
                    const __nv_bfloat16 s = __float2bfloat16(__bfloat162float(rr[i]) + mult * v[wq * EPT + r]);
                    nr[i] = s;
                    v[wq * EPT + r] = __bfloat162float(s);
                }
            }
    }
    float ss = 0.f;
#pragma unroll
    for (int wq = 0; wq < NW; ++wq) {
        float p = 0.f;
#pragma unroll
        for (int r = 0; r < EPT; ++r) {
            const int i = 32 * wq + lane + TH * r;
            if (i < H) { const float t = v[wq * EPT + r]; p += t * t; }
        }
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
        ss += __shfl_sync(0xffffffffu, p, 0);          // red[wq] : la valeur du fil 0 du warp
    }
    const float rs = rsqrtf(ss / (float)H + eps);
#pragma unroll
    for (int wq = 0; wq < NW; ++wq)
#pragma unroll
        for (int r = 0; r < EPT; ++r) {
            const int i = 32 * wq + lane + TH * r;
            if (i < H) {
                const float n = __bfloat162float(__float2bfloat16(v[wq * EPT + r] * rs));
                yr[i] = __float2bfloat16(n * __bfloat162float(w[i]));
            }
        }
}

std::vector<torch::Tensor> rmsnorm_bf16_warp(torch::Tensor x, torch::Tensor w, double eps,
                                             c10::optional<torch::Tensor> res,
                                             double mult) {
    CHECK_CUDA(x); ACVRAM_DEVICE_GUARD(x);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16,
                "rmsnorm_bf16_warp : bf16 attendu");
    auto xc = x.contiguous();
    const int H = xc.size(-1);
    const long R = xc.numel() / H;
    auto y = torch::empty_like(xc);
    torch::Tensor xn;
    const __nv_bfloat16 *pres = nullptr;
    __nv_bfloat16 *pxn = nullptr;
    torch::Tensor rc;
    if (res.has_value()) {
        rc = res->contiguous();
        TORCH_CHECK(rc.numel() == xc.numel(), "rmsnorm_bf16_warp : residu de meme taille");
        xn = torch::empty_like(xc);
        pres = reinterpret_cast<const __nv_bfloat16 *>(rc.data_ptr());
        pxn = reinterpret_cast<__nv_bfloat16 *>(xn.data_ptr());
    }
    const unsigned blocs = (unsigned)((R + 7) / 8);        // 8 warps (lignes) par bloc de 256 fils
    auto px = reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr());
    auto pw = reinterpret_cast<const __nv_bfloat16 *>(w.contiguous().data_ptr());
    auto py = reinterpret_cast<__nv_bfloat16 *>(y.data_ptr());
    auto flux = at::cuda::getCurrentCUDAStream();
    const float m = (float)mult, e = (float)eps;
    // la découpe TH du noyau à bloc (1024 / 512 / 256 selon H) : c'est elle qui
    // fixe l'ordre de la somme, elle est rejouée telle quelle ; EPT = éléments
    // par fil rejoué = ceil(H / TH) ≤ 4 (H ≤ 2 048 : 64 registres par lane)
    const int TH = H >= 2048 ? 1024 : (H >= 1024 ? 512 : 256);
    const int EPT = (H + TH - 1) / TH;
    TORCH_CHECK(H <= 2048 && EPT <= 4, "rmsnorm_bf16_warp : H <= 2048 attendu (au-dela : rmsnorm_bf16)");
#define ACVRAM_RMSW(TH_, EPT_) rmsnorm_bf16_warp_kernel<TH_, EPT_><<<blocs, 256, 0, flux>>>(px, pw, py, pres, pxn, m, H, e, R)
    if (TH == 1024) ACVRAM_RMSW(1024, 2);
    else if (TH == 512) { if (EPT == 2) ACVRAM_RMSW(512, 2); else if (EPT == 3) ACVRAM_RMSW(512, 3); else ACVRAM_RMSW(512, 4); }
    else { if (EPT == 1) ACVRAM_RMSW(256, 1); else if (EPT == 2) ACVRAM_RMSW(256, 2); else if (EPT == 3) ACVRAM_RMSW(256, 3); else ACVRAM_RMSW(256, 4); }
#undef ACVRAM_RMSW
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (res.has_value()) return {y, xn};
    return {y};
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


// --------------------------------------------------------------------------
// Q3N — quantiles 3 bits, echelle FP8 e4m3 par bloc (spec du 8/09/2026).
// Huit poids par mot de 24 bits, faible vers fort ; un bloc de B poids porte
// une echelle FP8 relative a l'echelle globale. Version 1 : correcte et
// lisible — un bloc CUDA par groupe de ROWS lignes, fils en enjambee sur les
// mots de 24 bits. La reference numerique est dequantize_q3n (Python) ; le
// microbanc decidera des optimisations (chargements 12 octets, k_splits).
// --------------------------------------------------------------------------

// La table de niveaux arrive en argument (elle vit dans le manifeste depuis
// la table Lloyd-Max par modele) et se lit depuis la memoire PARTAGEE : la
// version __constant__ se serialisait des que les fils d'un warp lisaient
// des entrees differentes — le piege deja documente sur la table NVFP4 plus
// haut — et un symbole global serait un etat partage entre deux modeles
// charges avec deux tables. Huit entrees physiques toujours : le masque & 7u
// garantit l'index, la huitieme repete la septieme quand il n'y a que sept
// niveaux logiques.
template <int ROWS, typename XT, typename YT>
__global__ void q3n_gemv_kernel(
    const unsigned char *__restrict__ qw,     // [M, K*3/8]
    const unsigned char *__restrict__ bscale, // fp8 e4m3 [M, K/B]
    const float gscale,
    const float *__restrict__ table,          // [8] niveaux
    const XT *__restrict__ x,                 // [N, K]
    YT *__restrict__ y,                       // [N, M]
    int M, int K, int N, int B) {
    extern __shared__ float smem[];
    float *tab = smem;            // 8 niveaux
    float *red = smem + 8;        // zone de reduction
    if (threadIdx.x < 8) tab[threadIdx.x] = table[threadIdx.x];
    __syncthreads();
    const int nwarps = (blockDim.x + WARP - 1) / WARP;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    const int mots = K / 8;
    const long stride = (long)K * 3 / 8;

    for (int n = 0; n < N; ++n) {
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
        for (int i = threadIdx.x; i < mots; i += blockDim.x) {
            const int k0 = i * 8;
            float xs[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                xs[j] = to_float_q(x[(long)n * K + k0 + j]);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                const unsigned char *base = qw + (long)row * stride + (long)i * 3;
                const unsigned mot = base[0] | ((unsigned)base[1] << 8)
                                   | ((unsigned)base[2] << 16);
                const float es = e4m3_to_float(
                    bscale[(long)row * (K / B) + k0 / B]) * gscale;
                float somme = 0.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    somme += tab[(mot >> (3 * j)) & 7u] * xs[j];
                acc[r] += somme * es;
            }
        }
        block_reduce_rows<ROWS>(acc, red, nwarps);
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r)
                if (row0 + r < M)
                    y[(long)n * M + row0 + r] = from_float<YT>(acc[r]);
        }
        __syncthreads();
    }
}

torch::Tensor q3n_gemv_cuda(torch::Tensor qweight, torch::Tensor block_scale,
                            double global_scale, torch::Tensor table,
                            torch::Tensor x, int64_t K, int64_t B) {
    CHECK_CUDA(qweight); CHECK_CUDA(x); CHECK_CUDA(table);
    CHECK_CONTIG(table);
    TORCH_CHECK(table.numel() == 8 && table.scalar_type() == torch::kFloat,
                "q3n : table de 8 flottants attendue");
    ACVRAM_DEVICE_GUARD(qweight);
    CHECK_CONTIG(qweight); CHECK_CONTIG(block_scale);
    TORCH_CHECK(K % 8 == 0 && B % 8 == 0 && K % B == 0,
                "q3n : K et B multiples de 8, K divisible par B");
    auto xc = x.dim() == 1 ? x.reshape({1, -1}) : x.reshape({-1, K});
    const int M = qweight.size(0);
    const bool bf = xc.scalar_type() == torch::kBFloat16;
    xc = (bf ? xc : xc.to(torch::kFloat)).contiguous();
    const int N = xc.size(0);
    auto out = torch::empty({N, M}, xc.options());
    // Scalaire hote, jamais un tenseur : .item() sur un tenseur CUDA
    // synchronise le flux a chaque GEMV — 62 % du temps de decodage au profil
    // NVFP4, et une capture de graphe CUDA impossible. Meme signature que
    // nvfp4_gemv, pour la meme raison.
    const float g = (float)global_scale;
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int ROWS = 4;
    const int threads = 128;
    const int nwarps = (threads + 31) / 32;
    dim3 grid((M + ROWS - 1) / ROWS);
    const size_t shm = (8 + ROWS * nwarps) * sizeof(float);
    if (bf) {
        q3n_gemv_kernel<ROWS, __nv_bfloat16, __nv_bfloat16>
            <<<grid, threads, shm, stream>>>(
            qweight.data_ptr<unsigned char>(),
            block_scale.data_ptr<unsigned char>(), g,
            table.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()),
            M, (int)K, N, (int)B);
    } else {
        q3n_gemv_kernel<ROWS, float, float><<<grid, threads, shm, stream>>>(
            qweight.data_ptr<unsigned char>(),
            block_scale.data_ptr<unsigned char>(), g, table.data_ptr<float>(),
            xc.data_ptr<float>(), out.data_ptr<float>(), M, (int)K, N, (int)B);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto forme = x.sizes().vec();
    forme.back() = M;
    return out.reshape(forme).to(x.dtype());
}

// ============================================================================
// SwiGLU fusionne : act(gate) * up en un lancement, sur la sortie empilee.
//
// Le chemin dense bf16 lancait deux noyaux elementaires par couche -- le SiLU
// puis le produit -- soit ~96 lancements par pas la ou llama.cpp n'en a aucun
// (leur mmvf.cu les fusionne dans la GEMV). Mesure du 9 septembre 2026 sous
// ncu sur Qwen2.5-Coder-14B : 251 noyaux elementaires par pas chez nous contre
// 2 chez eux, pour 0,54 ms.
//
// L'ARITHMETIQUE EST CELLE DE TORCH, ET C'EST UNE CONTRAINTE, PAS UN DETAIL.
// torch calcule le SiLU en float et *arrondit en bf16*, puis relit ce bf16
// pour le produit, qu'il arrondit a nouveau. Calculer d'un trait en float
// donnerait un resultat plus exact -- et different, donc d'autres jetons. Une
// optimisation qui change la sortie est un bogue : on reproduit l'arrondi
// intermediaire tel quel.
__global__ void swiglu_bf16_kernel(const __nv_bfloat16 *__restrict__ gu,
                                   __nv_bfloat16 *__restrict__ y,
                                   int I, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long ligne = i / I, col = i % I;
    const float g = __bfloat162float(gu[ligne * 2 * I + col]);
    const float u = __bfloat162float(gu[ligne * 2 * I + I + col]);
    const float s = g / (1.f + __expf(-g));           // SiLU, comme torch
    const float sb = __bfloat162float(__float2bfloat16(s));   // arrondi rendu
    y[i] = __float2bfloat16(sb * u);
}

// Variante a DEUX entrees separees : meme calcul, mais sans exiger que gate et
// up soient contigus dans un seul tenseur.
//
// swiglu_bf16 n'etait atteignable que par la branche fusionnee, qui exige que
// gate/up soient empiles. Le nvfp4 ne fusionne pas -- ses echelles d'activation
// different entre projections -- et payait donc silu PUIS produit, deux noyaux
// par couche la ou le bf16 fusionne n'en paie qu'un. Mesure sous ncu :
// 48 `silu_kernel` par pas cote nvfp4, zero cote bf16.
__global__ void swiglu2_bf16_kernel(const __nv_bfloat16 *__restrict__ g,
                                    const __nv_bfloat16 *__restrict__ u,
                                    __nv_bfloat16 *__restrict__ y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = __bfloat162float(g[i]);
    const float uv = __bfloat162float(u[i]);
    const float s = gv / (1.f + __expf(-gv));                  // SiLU, comme torch
    const float sb = __bfloat162float(__float2bfloat16(s));    // arrondi rendu
    y[i] = __float2bfloat16(sb * uv);
}

torch::Tensor swiglu2_bf16(torch::Tensor g, torch::Tensor u) {
    CHECK_CUDA(g); ACVRAM_DEVICE_GUARD(g);
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16 &&
                u.scalar_type() == torch::kBFloat16,
                "swiglu2_bf16 : bf16 attendu");
    TORCH_CHECK(g.sizes() == u.sizes(), "swiglu2_bf16 : memes formes attendues");
    auto gc = g.contiguous(), uc = u.contiguous();
    auto y = torch::empty_like(gc);
    const long n = y.numel();
    const int th = 256;
    swiglu2_bf16_kernel<<<(unsigned)((n + th - 1) / th), th, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(uc.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


torch::Tensor swiglu_bf16(torch::Tensor gu) {
    CHECK_CUDA(gu); ACVRAM_DEVICE_GUARD(gu);
    TORCH_CHECK(gu.scalar_type() == torch::kBFloat16, "swiglu_bf16 : bf16 attendu");
    auto c = gu.contiguous();
    const int deux_i = c.size(-1);
    TORCH_CHECK(deux_i % 2 == 0, "swiglu_bf16 : derniere dimension paire attendue");
    const int I = deux_i / 2;
    auto tailles = c.sizes().vec();
    tailles.back() = I;
    auto y = torch::empty(tailles, c.options());
    const long n = y.numel();
    const int th = 256;
    swiglu_bf16_kernel<<<(unsigned)((n + th - 1) / th), th, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(c.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), I, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Controle positif du harnais : echelle de travail connue d'avance.
    m.def("paged_attn_tampon_octets",
          [] { return acvram_pa_tampon_octets; },
          "octets retenus par les tampons partiels — doit se stabiliser au "
          "pire cas vu, et non croitre a chaque nouvelle forme");
    m.def("paged_attn_participants", &paged_attn_participants,
          "nombre de blocs ayant reellement tourne a l'etape 0 (observe, pas "
          "reconstruit)", py::arg("remettre_a_zero") = true);
    m.def("banc_fma", &banc_fma,
          "K FMA chainees, grille imposee — chiffre le plancher de l'instrument",
          py::arg("gx"), py::arg("gy"), py::arg("gz"),
          py::arg("threads"), py::arg("K"));
    m.def("swiglu2_bf16", &swiglu2_bf16,
          "SwiGLU sur gate et up separes : SiLU(gate) * up",
          py::arg("g"), py::arg("u"));
    m.def("swiglu_bf16", &swiglu_bf16,
          "SwiGLU fusionne sur une sortie gate/up empilee : SiLU(gate) * up",
          py::arg("gu"));
    m.def("nvfp4_dequant", &nvfp4_dequant, "NVFP4 -> matrice dense",
          py::arg("qweight"), py::arg("block_scale"), py::arg("global_scale"),
          py::arg("K"), py::arg("dtype"), py::arg("gscale_rows") = py::none(),
          py::arg("rows_per_group") = 1);
    m.def("q3n_gemv", &q3n_gemv_cuda,
          "Q3N : quantiles 3 bits, dequantification + produit fusionnes",
          py::arg("qweight"), py::arg("block_scale"), py::arg("global_scale"),
          py::arg("table"), py::arg("x"), py::arg("K"), py::arg("B"));
    m.def("nvfp4_gemv", &nvfp4_gemv, "NVFP4 : dequantification + produit fusionnes",
          py::arg("qweight"), py::arg("block_scale"), py::arg("global_scale"),
          py::arg("x"), py::arg("K"), py::arg("global_scale_rows") = c10::nullopt);
    m.def("int4_dequant", &int4_dequant, "INT4 affine par groupes -> matrice dense");
    m.def("int4_gemv", &int4_gemv, "INT4 : dequantification + produit fusionnes");
    m.def("int8_dequant", &int8_dequant, "INT8 affine par groupes -> matrice dense");
    m.def("int8_gemv_norme", &int8_gemv_norme, py::arg("qweight"), py::arg("scales"), py::arg("zeros"),
          py::arg("x"), py::arg("group"), py::arg("res"), py::arg("w"), py::arg("eps"), py::arg("mult") = 1.0,
          "INT8 : y = W . rmsnorm(res + mult*x) et xout = res + mult*x en un lancement (poste F, 3b)");
    m.def("int8_gemv", &int8_gemv, py::arg("qweight"), py::arg("scales"), py::arg("zeros"), py::arg("x"),
          py::arg("group"), py::arg("sortie_fp32") = false,
          "INT8 : dequantification + produit fusionnes ; sortie_fp32 : x bf16, accumulation et sortie fp32 (tete)");
    m.def("nvfp4_gemv_grouped", &nvfp4_gemv_grouped,
          "NVFP4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
    m.def("nvfp4_gemv_grouped_xreg", &nvfp4_gemv_grouped_xreg,
          py::arg("qw"), py::arg("bscale"), py::arg("gscales"), py::arg("expert_ids"), py::arg("token_ids"),
          py::arg("x"), py::arg("K"), "NVFP4 : GEMV groupée, x en registres par tranche (K <= 2048), identique au bit");
    m.def("nvfp4_gemv_grouped_gateup_xreg", &nvfp4_gemv_grouped_gateup_xreg,
          py::arg("qg"), py::arg("bg"), py::arg("gsg"), py::arg("qu"), py::arg("bu"), py::arg("gsu"),
          py::arg("expert_ids"), py::arg("token_ids"), py::arg("x"), py::arg("K"), py::arg("act") = 0,
          "NVFP4 : gate/up fusionnés, x en registres par tranche (K <= 2048), identique au bit");
    m.def("nvfp4_gemv_grouped_v2", &nvfp4_gemv_grouped_v2,
          py::arg("qw"), py::arg("bscale"), py::arg("gscales"), py::arg("eid_s"), py::arg("tok_s"),
          py::arg("ordre"), py::arg("x"), py::arg("K"),
          "NVFP4 : GEMV groupée v2, paires triées par expert, poids lus une fois pour TPB jetons");
    m.def("nvfp4_gemv_grouped_gateup_v2", &nvfp4_gemv_grouped_gateup_v2,
          py::arg("qg"), py::arg("bg"), py::arg("gsg"), py::arg("qu"), py::arg("bu"), py::arg("gsu"),
          py::arg("eid_s"), py::arg("tok_s"), py::arg("ordre"), py::arg("x"), py::arg("K"), py::arg("act") = 0,
          "NVFP4 : gate/up fusionnés v2, paires triées par expert");
    m.def("nvfp4_gemv_marlin", &nvfp4_gemv_marlin,
          py::arg("w"), py::arg("s"), py::arg("g"), py::arg("expert_ids"), py::arg("token_ids"),
          py::arg("x"), py::arg("K"), py::arg("N"), py::arg("xscale") = c10::nullopt,
          "NVFP4 : GEMV groupée lisant la DISPOSITION MARLIN (forme (b), P1 disposition unique)");
    m.def("nvfp4_gemv_marlin_splitk", &nvfp4_gemv_marlin_splitk,
          "S du split-K que prendrait nvfp4_gemv_marlin(K, N, G) : 1 sauf ACVRAM_GEMV_SPLITK");
    m.def("nvfp4_gemv_marlin_gateup", &nvfp4_gemv_marlin_gateup,
          py::arg("wg"), py::arg("sg"), py::arg("gg"), py::arg("wu"), py::arg("su"), py::arg("gu"),
          py::arg("expert_ids"), py::arg("token_ids"), py::arg("x"), py::arg("K"), py::arg("N"), py::arg("act") = 0,
          py::arg("xscale") = c10::nullopt,
          "NVFP4 : gate et up fusionnés sur la disposition Marlin, sortie act(gate)*up");
    m.def("nvfp4_gemv_marlin_w13", &nvfp4_gemv_marlin_w13,
          py::arg("w13"), py::arg("s13"), py::arg("gg"), py::arg("gu"),
          py::arg("expert_ids"), py::arg("token_ids"), py::arg("x"), py::arg("K"), py::arg("N"), py::arg("act") = 0,
          py::arg("xscale") = c10::nullopt,
          "Pièce 82 : GEMV gate·up lisant la pile w13 [E, K/16, 4N] (largeur stockée 2N), au bit du chemin séparé");
    m.def("moe_slots", &moe_slots, py::arg("expert_ids"), py::arg("E"), py::arg("tpb") = 4,
          "Pièce 61 : créneaux (expert, ≤ TPB paires) à partir des paires du godet — (slot_e [G], slot_pair [G·TPB])");
    m.def("nvfp4_gemv_marlin_slots", &nvfp4_gemv_marlin_slots,
          py::arg("w"), py::arg("s"), py::arg("g"), py::arg("slot_e"), py::arg("slot_pair"), py::arg("token_ids"),
          py::arg("x"), py::arg("K"), py::arg("N"), py::arg("tpb") = 4, py::arg("xscale") = c10::nullopt,
          "Pièce 61 : GEMV Marlin par créneau d expert (tuile lue une fois par créneau), au bit avec nvfp4_gemv_marlin");
    m.def("nvfp4_gemv_marlin_gateup_slots", &nvfp4_gemv_marlin_gateup_slots,
          py::arg("wg"), py::arg("sg"), py::arg("gg"), py::arg("wu"), py::arg("su"), py::arg("gu"),
          py::arg("slot_e"), py::arg("slot_pair"), py::arg("token_ids"), py::arg("x"), py::arg("K"), py::arg("N"),
          py::arg("act") = 0, py::arg("tpb") = 4, py::arg("xscale") = c10::nullopt,
          "Pièce 61 : gate et up fusionnés par créneau d expert, au bit avec nvfp4_gemv_marlin_gateup");
    m.def("nvfp4_gemv_grouped_gateup", &nvfp4_gemv_grouped_gateup,
          py::arg("qg"), py::arg("bg"), py::arg("gsg"), py::arg("qu"), py::arg("bu"),
          py::arg("gsu"), py::arg("expert_ids"), py::arg("token_ids"), py::arg("x"),
          py::arg("K"), py::arg("act") = 0,
          "MoE NVFP4 : gate et up fusionnes, sortie act(gate)*up (act 0=SiLU, 1=GELU-tanh)");
    m.def("nvfp4_gemv_grouped_table", &nvfp4_gemv_grouped_table,
          py::arg("table_qw"), py::arg("table_bs"), py::arg("gscales"),
          py::arg("expert_ids"), py::arg("token_ids"), py::arg("x"),
          py::arg("M"), py::arg("K"),
          "Pendant table de nvfp4_gemv_grouped (bead pds) : chaque expert lu "
          "par adresse, resident ou epingle -- couche au placement heterogene");
    m.def("nvfp4_gemv_grouped_gateup_table", &nvfp4_gemv_grouped_gateup_table,
          py::arg("table_qg"), py::arg("table_bg"), py::arg("gsg"),
          py::arg("table_qu"), py::arg("table_bu"), py::arg("gsu"),
          py::arg("expert_ids"), py::arg("token_ids"), py::arg("x"),
          py::arg("M"), py::arg("K"), py::arg("act") = 0,
          "Pendant table de nvfp4_gemv_grouped_gateup (bead pds), meme motif "
          "que nvfp4_gemm_grouped_mma");
    m.def("nvfp4_gemm_grouped", &nvfp4_gemm_grouped,
          "MoE NVFP4 : GEMM groupee sur tuiles de jetons, poids lus en 4 bits");
    m.def("moe_act", &moe_act, py::arg("g"), py::arg("u"), py::arg("m"), py::arg("Kd"), py::arg("act"),
          py::arg("awq") = py::none(), py::arg("e_sorted") = py::none(), py::arg("gs_gate") = py::none(), py::arg("gs_up") = py::none(),
          "MoE prefill : act(g)*u en bf16 depuis les vues [:, :m], rembourre a Kd (act 0=SiLU, 1=GELU-tanh) ; awq [E,Kd] + e_sorted [G] : x / s[e] ; "
          "gs_gate/gs_up [E] fp32 (pièce 71 bis) : g·gs_gate[e], u·gs_up[e] avant l activation");
    m.def("nvfp4_moe_fused", &nvfp4_moe_fused,
          py::arg("tq_g"), py::arg("tb_g"), py::arg("gs_g"), py::arg("tq_u"), py::arg("tb_u"), py::arg("gs_u"),
          py::arg("tq_d"), py::arg("tb_d"), py::arg("gs_d"), py::arg("xq"), py::arg("xsf"),
          py::arg("tile_e"), py::arg("tile_t0"), py::arg("tile_n"), py::arg("ws"), py::arg("compteurs"),
          py::arg("K"), py::arg("I"), py::arg("M_out"), py::arg("act"), py::arg("tn"), py::arg("ordre"), py::arg("tw"),
          py::arg("k_top"), py::arg("t"), py::arg("atomique"), py::arg("etages"), py::arg("grow") = py::none(),
          "MoE decodage fusionne (port b12x) : gate+up+act+quant en shared, down par tranches, split-K seriel -> d bf16 [G, M_out]");
    m.def("narrow_gemm", &narrow_gemm,
          py::arg("qw"), py::arg("sc8"), py::arg("sc16"), py::arg("zr"), py::arg("x"), py::arg("K"),
          py::arg("group"), py::arg("gscale"), py::arg("rows"),
          "GEMM etroit (M<=16) bf16 tensor cores, poids int8 (groupes) ou nvfp4 (blocs 16) dequantifies en registres");
    m.def("moe_route_pack", &moe_route_pack, py::arg("topi"), py::arg("topw"), py::arg("x"), py::arg("E"), py::arg("bt"),
          py::arg("t_max"), py::arg("Hpad"), py::arg("awq") = py::none(), py::arg("awq2") = py::none(),
          "MoE decodage : routage + rassemblement en un lancement -> (xs, ordre, inv, tw, cnt, tile_e, tile_t0, tile_n, e_sorted, xs2) ; awq [E,Hpad] bf16 : x / s[e] ; awq2 (up distinct de gate) : xs2 = x / s_up[e], sinon xs2 est xs");
    m.def("moe_reduce_trie", &moe_reduce_trie,
          "MoE prefill : reduction ponderee par jeton depuis l'ordre trie par expert, sortie bf16");
    m.def("nvfp4_gemm_grouped_mma", &nvfp4_gemm_grouped_mma,
          py::arg("table_qw"), py::arg("table_bscale"), py::arg("gscales"), py::arg("xq"),
          py::arg("xsf"), py::arg("tile_e"), py::arg("tile_t0"), py::arg("tile_n"),
          py::arg("M"), py::arg("K"), py::arg("bt"), py::arg("etages") = 0, py::arg("ks") = 64,
          py::arg("grow") = py::none(), py::arg("marlin") = -1,
          "MoE NVFP4 : GEMM groupee W4A4 sur la MMA FP4 native de sm_120 (mxf4nvf4) ; "
          "etages 0 = chargements directs, 2-4 = pipeline cp.async en shared ; ks 64 ou 128 par etage ; "
          "marlin >= 0 (C17) : tables = disposition Marlin (w_marlin, s_marlin), valeur = decal d'exposant des echelles");
    m.def("nvfp4_gemm_grouped_mma_disponible", &nvfp4_gemm_grouped_mma_disponible,
          "vrai si la carte courante est sm_120 et le noyau MMA FP4 compile");
    m.def("nvfp4_quant_act", &nvfp4_quant_act,
          "activations bf16 [G, K] -> (E2M1 [G, K/2], echelles UE4M3 [G, K/16], echelle globale fp32 [G] "
          "par ligne = amax_r/(6*448), a passer en grow a la GEMM) ; awq [E, K] bf16 + e_sorted [G] : x/s[e] fusionne",
          py::arg("x"), py::arg("awq") = py::none(), py::arg("e_sorted") = py::none(),
          py::arg("compteurs") = py::none(), py::arg("hadamard") = 0);
    m.def("int4_gemv_grouped", &int4_gemv_grouped,
          "INT4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
    m.def("kv_write_int8", &kv_write_int8,
          "Cache KV : quantification INT8 et dispersion en un lancement");
    m.def("kv_write_int8_canal", &kv_write_int8_canal,
          "C5-b : cache KV int8, cles par canal sur chaque bloc (roles, ecriture, fermeture : "
          "trois lancements), V par jeton ; sc E4M3 [NB,HKV,D], tampon bf16 [R,16,HKV,D], "
          "tampon_de [NB], libres [R], sommet [1]");
    m.def("paged_attention_canal", &paged_attention_canal,
          "C5-b : attention de decodage fusionnee, cles int8 par canal (q x sc par bloc, "
          "bloc courant lu en bf16 dans la reserve)");
    m.def("rope_inplace", &rope_inplace_pos, "RoPE en place sur q et k",
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          py::arg("positions") = c10::optional<torch::Tensor>(),
          py::arg("wq") = c10::optional<torch::Tensor>(),
          py::arg("wk") = c10::optional<torch::Tensor>(),
          py::arg("eps") = 1e-6);
    m.def("moe_reduce", &moe_reduce,
          "MoE : ponderation et somme des top_k sorties d'un jeton (d fp32 ou bf16)");
    m.def("moe_aligner_petit", &moe_aligner_petit, py::arg("eid"), py::arg("E"), py::arg("bloc"),
          py::arg("sorted_ids"), py::arg("expert_ids"), py::arg("num_post"),
          "Pièce 63 : aligneur du chemin tensor en un lancement (fantômes −1 → expert 0), tampons fixes");
    m.def("moe_route", &moe_route, "routage MoE : scores, biais, top-k, poids");
    m.def("rmsnorm_bf16", &rmsnorm_bf16,
          "RMSNorm bf16 fusionnee (variance fp32), residu optionnel",
          py::arg("x"), py::arg("w"), py::arg("eps"),
          py::arg("residu") = c10::optional<torch::Tensor>(),
          py::arg("mult") = 1.0);
    m.def("rmsnorm_bf16_warp", &rmsnorm_bf16_warp,
          "C15-prefill : rmsnorm_bf16 un warp par ligne, meme ordre de somme (au bit)",
          py::arg("x"), py::arg("w"), py::arg("eps"),
          py::arg("residu") = c10::optional<torch::Tensor>(),
          py::arg("mult") = 1.0);
    m.def("kda_decode", &kda_decode, "KDA : pas de decodage fusionne (une sequence)");
    m.def("mla_decode", &mla_decode, "MLA absorbee : scores + softmax + lecture latente");
    m.def("mla_prep_batch", &mla_prep_batch, py::arg("q"), py::arg("kvp"), py::arg("lens"), py::arg("cos32"),
          py::arg("sin32"), py::arg("k_b"), py::arg("w_norm"), py::arg("nope"), py::arg("rope"), py::arg("rank"),
          py::arg("eps"), py::arg("temoin") = false,
          "MLA decodage : q [B,nh,nope+rope] + kvp [B,rank+rope] -> (q_eff [B,nh,W] fp32, k_new [B,W] bf16) : "
          "einsum k_b, RoPE, norme kv_a et cat en un lancement (cos32/sin32 = tables fp32 indexees par position, ou None) ; "
          "temoin=True : la grille d'avant C14-b (nh.NR + B blocs), meme sortie au bit");
    m.def("mla_ecrit_latent", &mla_ecrit_latent, py::arg("k_new"), py::arg("cache_ptrs"), py::arg("len_ptrs"),
          py::arg("fp8") = false,
          "MLA decodage : k_new [B, W] -> cache_b[len_b], puis len_b += 1, un lancement pour les B creneaux (tables d'adresses [B])");
    m.def("mla_decode_1p", &mla_decode_1p, py::arg("q_eff"), py::arg("cache_ptrs") = py::none(),
          py::arg("cache") = py::none(), py::arg("lens"), py::arg("L"), py::arg("rank"), py::arg("scale"),
          py::arg("fp8") = false, py::arg("v_b") = py::none(),
          "MLA absorbee a UNE PASSE (sage-duel-verdict-16-09) : le cache latent lu une fois pour les H tetes, "
          "softmax en ligne, tranches de L recombinees -> o_lat [B, H, rank] fp32 ; cache_ptrs [B] int64 ou cache [L, W] (B = 1) ; "
          "avec v_b [H, DV, rank] bf16 (C14-b) : le combine rend y = v_b . o_lat en bf16 [B, H, DV]");
    m.def("mla_decode_batch", &mla_decode_batch,
          "MLA : decodage de B creneaux en un lancement, caches par table d'adresses [B] int64");
    m.def("paged_attention", &paged_attention,
          "attention de decodage fusionnee sur cache KV int8 pagine");
}
