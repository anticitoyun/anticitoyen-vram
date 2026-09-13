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
template <int ROWS, int NV, typename XT, typename YT>
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

// CONTROLE POSITIF DU HARNAIS DE MESURE (poste1, 10/09/2026).
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

template <int D, typename QT, typename OT>
__global__ void paged_attn_partial_kernel(
    const QT *__restrict__ q,             // [B*QL, HQ, D]
    const signed char *__restrict__ kc,   // [NB, 16, HKV, D]
    const __half *__restrict__ ks,        // [NB, 16, HKV]
    const signed char *__restrict__ vc,
    const __half *__restrict__ vs,
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

    const long end = min(slen, (long)(c + 1) * chunk);
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
        default: I8G(12, XT, YT, PX, PY, SP); break; } } while (0)
    // Tranches de 12 activations : un lot de 12 séquences (le régime du
    // serveur) passe en UNE lecture des poids ; au-delà, une passe par tranche.
    const int Ntot = N;
    for (int base = 0; base < Ntot; base += 12) {
        const int N = min(12, Ntot - base);     // masque volontaire pour I8G_N
        if (bf) {
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
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 1;
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

// Activations bf16 [G, K] -> E2M1 [G, K/2] + échelles UE4M3 [G, K/16].
// Un fil par bloc de 16 : amax/6 arrondi en E4M3, puis chaque valeur divisée
// par l'échelle décodée et arrondie en E2M1 (satfinite : au-delà de 6 -> 6).
__global__ void nvfp4_quant_act_kernel(const __nv_bfloat16 *__restrict__ x,
                                       unsigned char *__restrict__ xq,
                                       unsigned char *__restrict__ xsf,
                                       long nblocs, int nblk) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocs) return;
    const long row = i / nblk;
    const int blk = (int)(i - row * nblk);
    const __nv_bfloat16 *src = x + row * (long)nblk * 16 + blk * 16;
    float v[16];
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) { v[j] = __bfloat162float(src[j]); amax = fmaxf(amax, fabsf(v[j])); }
    unsigned char sbits = 0;
    float sdec = 0.f;
    if (amax > 0.f) {
        // __fdiv_rn : l'extension est compilée avec --use_fast_math, dont la
        // division est un MUFU.RCP approché qui bascule les égalités E2M1
        // (mesuré : 5 codes sur 5 376 différaient de la référence Python)
        const float s = fminf(__fdiv_rn(amax, 6.f), 448.f);
        sbits = (unsigned char)__nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
        sdec = e4m3_to_float(sbits);
    }
    unsigned long long packed = 0ull;
    if (sdec > 0.f) {
        #pragma unroll
        for (int j = 0; j < 16; j += 2) {
            const float2 p = make_float2(__fdiv_rn(v[j], sdec), __fdiv_rn(v[j + 1], sdec));
            const __nv_fp4x2_storage_t q = __nv_cvt_float2_to_fp4x2(p, __NV_E2M1, cudaRoundNearest);
            packed |= (unsigned long long)(q & 0xFF) << (4 * j);
        }
    }
    *reinterpret_cast<unsigned long long *>(xq + row * (long)nblk * 8 + blk * 8) = packed;
    xsf[row * (long)nblk + blk] = sbits;
}

// Un bloc = 64 lignes de sortie x BT jetons d'une tuile ; 4 warps de 16
// lignes. Fragments A (jetons) et B (poids) chargés depuis la mémoire
// globale directement dans le layout de la MMA (aucune mémoire partagée :
// plus rien à convertir). Double tampon de registres sur K.
template <int BT>
__global__ void __launch_bounds__(128) nvfp4_gemm_grouped_mma_kernel(
    const float *__restrict__ gscales,
    const unsigned char *__restrict__ xq, const unsigned char *__restrict__ xsf,
    const int *__restrict__ tile_e, const int *__restrict__ tile_t0,
    const int *__restrict__ tile_n, const int64_t *__restrict__ table_qw,
    const int64_t *__restrict__ table_bscale,
    __nv_bfloat16 *__restrict__ y, int M, int K) {
#ifdef ACVRAM_MMA_FP4
    constexpr int MF = BT / 16;
    const int tile = blockIdx.y;
    const int e = tile_e[tile], t0 = tile_t0[tile], nt = tile_n[tile];
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
                    if (r < M)     dst[0] = __float2bfloat16(acc[mf][nf][2 * h] * gscale);
                    if (r + 1 < M) dst[1] = __float2bfloat16(acc[mf][nf][2 * h + 1] * gscale);
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

template <int BT, int S>
__global__ void __launch_bounds__(GM2_FILS) nvfp4_gemm_grouped_mma2_kernel(
    const float *__restrict__ gscales,
    const unsigned char *__restrict__ xq, const unsigned char *__restrict__ xsf,
    const int *__restrict__ tile_e, const int *__restrict__ tile_t0,
    const int *__restrict__ tile_n, const int64_t *__restrict__ table_qw,
    const int64_t *__restrict__ table_bscale,
    __nv_bfloat16 *__restrict__ y, int M, int K) {
#ifdef ACVRAM_MMA_FP4
    constexpr int MF = BT / 16;
    // shared dynamique : à bt=128 et 4 étages, 52 Ko dépassent les 48 Ko
    // statiques ; le découpage est fait à la main, un tableau par étage
    extern __shared__ __align__(16) unsigned char gm2_smem[];
    typedef unsigned char (*TA)[BT * GM2_LD];
    typedef unsigned char (*TB)[GM2_BM * GM2_LD];
    typedef unsigned char (*TSA)[BT * 4];
    typedef unsigned char (*TSB)[GM2_BM * 4];
    TA sA = reinterpret_cast<TA>(gm2_smem);
    TB sB = reinterpret_cast<TB>(gm2_smem + S * BT * GM2_LD);
    TSA sSA = reinterpret_cast<TSA>(gm2_smem + S * (BT + GM2_BM) * GM2_LD);
    TSB sSB = reinterpret_cast<TSB>(gm2_smem + S * (BT + GM2_BM) * GM2_LD + S * BT * 4);

    const int tile = blockIdx.y;
    const int e = tile_e[tile], t0 = tile_t0[tile], nt = tile_n[tile];
    const unsigned char *qw_e = reinterpret_cast<const unsigned char *>(table_qw[e]);
    const unsigned char *bs_e = reinterpret_cast<const unsigned char *>(table_bscale[e]);
    const int row0 = blockIdx.x * GM2_BM;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const long half_k = (long)K >> 1;
    const int nblk = K >> 4;
    const int KT = K / GM_KB;

    auto emettre = [&](int st, int k0) {
        const long kb = k0 >> 1, ks = k0 >> 4;
        for (int c = tid; c < 2 * BT; c += GM2_FILS) {
            const int r = c >> 1, h = c & 1;
            const long src = (long)(t0 + min(r, nt - 1)) * half_k + kb + 16 * h;
            cp_async16(&sA[st][r * GM2_LD + 16 * h], xq + src);
        }
        for (int c = tid; c < 2 * GM2_BM; c += GM2_FILS) {
            const int r = c >> 1, h = c & 1;
            const long src = (long)min(row0 + r, M - 1) * half_k + kb + 16 * h;
            cp_async16(&sB[st][r * GM2_LD + 16 * h], qw_e + src);
        }
        if (tid < BT) cp_async4(&sSA[st][tid * 4], xsf + (long)(t0 + min(tid, nt - 1)) * nblk + ks);
        else if (tid < BT + GM2_BM) {
            const int r = tid - BT;
            cp_async4(&sSB[st][r * 4], bs_e + (long)min(row0 + r, M - 1) * nblk + ks);
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
        if (s < KT) emettre(s, s * GM_KB);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        {
            const int kn = kt + S - 1;
            if (kn < KT) emettre(kn % S, kn * GM_KB);
            cp_async_commit();
        }
        const int st = kt % S;
        unsigned a[MF][4], sfa[MF], b[2][2], sfb[2];
        #pragma unroll
        for (int mf = 0; mf < MF; ++mf) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = mf * 16 + g + 8 * h;
                const unsigned char *p = &sA[st][j * GM2_LD + 4 * tq];
                const unsigned lo = *reinterpret_cast<const unsigned *>(p);
                const unsigned hi = *reinterpret_cast<const unsigned *>(p + 16);
                a[mf][h] = (j < nt) ? lo : 0u;
                a[mf][h + 2] = (j < nt) ? hi : 0u;
            }
            const int js = mf * 16 + (lane >> 2) + 8 * (lane & 1);
            const unsigned sv = *reinterpret_cast<const unsigned *>(&sSA[st][js * 4]);
            sfa[mf] = (js < nt) ? sv : 0u;
        }
        #pragma unroll
        for (int nf = 0; nf < 2; ++nf) {
            const int r = warp * 16 + nf * 8 + g;
            const bool ok = row0 + r < M;
            const unsigned char *p = &sB[st][r * GM2_LD + 4 * tq];
            const unsigned lo = *reinterpret_cast<const unsigned *>(p);
            const unsigned hi = *reinterpret_cast<const unsigned *>(p + 16);
            b[nf][0] = ok ? lo : 0u;
            b[nf][1] = ok ? hi : 0u;
            const unsigned sv = *reinterpret_cast<const unsigned *>(&sSB[st][r * 4]);
            sfb[nf] = ok ? sv : 0u;
        }
        #pragma unroll
        for (int mf = 0; mf < MF; ++mf)
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf)
                mma_mxf4nvf4(acc[mf][nf], a[mf], b[nf], sfa[mf], sfb[nf]);
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
                    if (r < M)     dst[0] = __float2bfloat16(acc[mf][nf][2 * h] * gscale);
                    if (r + 1 < M) dst[1] = __float2bfloat16(acc[mf][nf][2 * h + 1] * gscale);
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

std::tuple<torch::Tensor, torch::Tensor> nvfp4_quant_act(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIG(x); ACVRAM_DEVICE_GUARD(x);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "quant_act : activations bf16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 64 == 0, "quant_act : [G, K] avec K multiple de 64");
    const long G = x.size(0), K = x.size(1);
    const int nblk = (int)(K >> 4);
    auto opt = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto xq = torch::empty({G, K / 2}, opt);
    auto xsf = torch::empty({G, nblk}, opt);
    const long nblocs = G * nblk;
    if (nblocs > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int th = 256;
        nvfp4_quant_act_kernel<<<(unsigned)((nblocs + th - 1) / th), th, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
            xq.data_ptr<unsigned char>(), xsf.data_ptr<unsigned char>(), nblocs, nblk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {xq, xsf};
}

// table_qw / table_bscale [E] int64 : adresse octet (device) des piles
// [M, K/2] et [M, K/16] de chaque expert — contrat du bead pds (poste1) :
// résident = data_ptr() de sa tranche, froid = pointeur device zéro-copie
// d'un tampon épinglé ; jamais 0 ; mise à jour hors pas seulement. Le noyau
// ne connaît aucune autre adresse. gscales [E] reste résident, indexé par e.
torch::Tensor nvfp4_gemm_grouped_mma(torch::Tensor table_qw, torch::Tensor table_bscale,
                                     torch::Tensor gscales, torch::Tensor xq,
                                     torch::Tensor xsf, torch::Tensor tile_e,
                                     torch::Tensor tile_t0, torch::Tensor tile_n,
                                     int64_t M, int64_t K, int64_t bt, int64_t etages) {
    CHECK_CUDA(xq); ACVRAM_DEVICE_GUARD(xq);
    TORCH_CHECK(etages == 0 || etages == 2 || etages == 3 || etages == 4,
                "GEMM groupee MMA : etages dans {0 (direct), 2, 3, 4}");
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
    auto y = torch::zeros({G, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(xq.device()));
    if (T == 0) return y;
    auto stream = at::cuda::getCurrentCUDAStream();
    #define GM_ARGS gscales.data_ptr<float>(), \
        xq.data_ptr<unsigned char>(), xsf.data_ptr<unsigned char>(), \
        tile_e.data_ptr<int>(), tile_t0.data_ptr<int>(), tile_n.data_ptr<int>(), \
        table_qw.data_ptr<int64_t>(), table_bscale.data_ptr<int64_t>(), \
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), (int)M, (int)K
    if (etages == 0) {
        dim3 grid((M + GM_BM - 1) / GM_BM, T);
        #define GM_L(BT) nvfp4_gemm_grouped_mma_kernel<BT><<<grid, 128, 0, stream>>>(GM_ARGS)
        if (bt == 16) GM_L(16); else if (bt == 32) GM_L(32); else GM_L(64);
        #undef GM_L
    } else {
        dim3 grid((M + GM2_BM - 1) / GM2_BM, T);
        const size_t shm = (size_t)etages * ((bt + GM2_BM) * GM2_LD + (bt + GM2_BM) * 4);
        #define GM_L2(BT, S) do { \
            if (shm > 48 * 1024) cudaFuncSetAttribute(nvfp4_gemm_grouped_mma2_kernel<BT, S>, \
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); \
            nvfp4_gemm_grouped_mma2_kernel<BT, S><<<grid, GM2_FILS, shm, stream>>>(GM_ARGS); } while (0)
        #define GM_LS(S) do { if (bt == 16) GM_L2(16, S); else if (bt == 32) GM_L2(32, S); \
                              else if (bt == 64) GM_L2(64, S); else GM_L2(128, S); } while (0)
        if (etages == 2) GM_LS(2); else if (etages == 3) GM_LS(3); else GM_LS(4);
        #undef GM_LS
        #undef GM_L2
    }
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
    int H, int D, int bs) {
    __shared__ float red[8];
    const int t = blockIdx.x, h = blockIdx.y;
    const long slot = slots[t];
    if (slot < 0) return;
    const long pos = (slot / bs) * bs + (slot % bs);   // index à plat [bloc, offset]
    #pragma unroll 1
    for (int quel = 0; quel < 2; ++quel) {
        const __nv_bfloat16 *src = (quel ? v : k) + ((long)t * H + h) * D;
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
    auto kk = k.contiguous(), vv = v.contiguous();
    const int T = kk.size(0), H = kk.size(1), D = kk.size(2);
    dim3 grid(T, H);
    const int th = std::min(256, (D + 31) / 32 * 32);
    kv_write_int8_kernel<<<grid, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(kk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(vv.data_ptr()),
        slots.contiguous().data_ptr<long>(),
        reinterpret_cast<signed char *>(kc.data_ptr()),
        reinterpret_cast<signed char *>(vc.data_ptr()),
        reinterpret_cast<__half *>(ks.data_ptr()),
        reinterpret_cast<__half *>(vs.data_ptr()), H, D, (int)bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Réduction pondérée du MoE : les top_k lignes d'un jeton, multipliées par
// leur poids de routage et sommées. Le chemin PyTorch demandait une
// multiplication, une réduction et une conversion — trois lancements par
// couche pour quelques kilooctets.
__global__ void moe_reduce_kernel(const float *__restrict__ d,
                                  const float *__restrict__ topw,
                                  __nv_bfloat16 *__restrict__ y,
                                  int M, int k) {
    const int t = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;
    float s = 0.f;
    for (int e = 0; e < k; ++e)
        s += topw[t * k + e] * d[((long)t * k + e) * M + col];
    y[(long)t * M + col] = __float2bfloat16(s);
}

torch::Tensor moe_reduce(torch::Tensor d, torch::Tensor topw, int64_t k) {
    CHECK_CUDA(d); ACVRAM_DEVICE_GUARD(d);
    CHECK_CONTIG(d); CHECK_CONTIG(topw);
    TORCH_CHECK(d.scalar_type() == torch::kFloat, "moe_reduce : sorties fp32");
    const int M = d.size(1), T = d.size(0) / (int)k;
    auto y = torch::empty({T, M}, d.options().dtype(torch::kBFloat16));
    dim3 grid((M + 255) / 256, T);
    moe_reduce_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        d.data_ptr<float>(), topw.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), M, (int)k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
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
                               long n, int Mp, int m, int Kd, int act) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long r = i / Kd;
    const int c = (int)(i - r * Kd);
    float v = 0.f;
    if (c < m) {
        const float x = __bfloat162float(g[r * Mp + c]);
        const float y = __bfloat162float(u[r * Mp + c]);
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
    }
    out[i] = __float2bfloat16(v);
}

torch::Tensor moe_act(torch::Tensor g, torch::Tensor u, int64_t m, int64_t Kd, int64_t act) {
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
        n, (int)g.stride(0), (int)m, (int)Kd, (int)act);
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
    static const int rpw = std::getenv("ACVRAM_GROUPED_RPW") ? atoi(std::getenv("ACVRAM_GROUPED_RPW")) : 1;
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
        paged_attn_partial_kernel<DD, QT, OT><<<g1, threads, 0, stream>>>( \
            PQ, kc.data_ptr<signed char>(), \
            reinterpret_cast<const __half *>(ks.data_ptr()), \
            vc.data_ptr<signed char>(), \
            reinterpret_cast<const __half *>(vs.data_ptr()), \
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
    m.def("int8_gemv", &int8_gemv, "INT8 : dequantification + produit fusionnes");
    m.def("nvfp4_gemv_grouped", &nvfp4_gemv_grouped,
          "NVFP4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
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
    m.def("moe_act", &moe_act,
          "MoE prefill : act(g)*u en bf16 depuis les vues [:, :m], rembourre a Kd (act 0=SiLU, 1=GELU-tanh)");
    m.def("moe_reduce_trie", &moe_reduce_trie,
          "MoE prefill : reduction ponderee par jeton depuis l'ordre trie par expert, sortie bf16");
    m.def("nvfp4_gemm_grouped_mma", &nvfp4_gemm_grouped_mma,
          py::arg("table_qw"), py::arg("table_bscale"), py::arg("gscales"), py::arg("xq"),
          py::arg("xsf"), py::arg("tile_e"), py::arg("tile_t0"), py::arg("tile_n"),
          py::arg("M"), py::arg("K"), py::arg("bt"), py::arg("etages") = 0,
          "MoE NVFP4 : GEMM groupee W4A4 sur la MMA FP4 native de sm_120 (mxf4nvf4) ; "
          "etages 0 = chargements directs, 2-4 = pipeline cp.async en shared");
    m.def("nvfp4_gemm_grouped_mma_disponible", &nvfp4_gemm_grouped_mma_disponible,
          "vrai si la carte courante est sm_120 et le noyau MMA FP4 compile");
    m.def("nvfp4_quant_act", &nvfp4_quant_act,
          "activations bf16 [G, K] -> (E2M1 [G, K/2], echelles UE4M3 [G, K/16]) par bloc de 16");
    m.def("int4_gemv_grouped", &int4_gemv_grouped,
          "INT4 : GEMV pour tous les experts actifs d'une couche, en un lancement");
    m.def("kv_write_int8", &kv_write_int8,
          "Cache KV : quantification INT8 et dispersion en un lancement");
    m.def("rope_inplace", &rope_inplace_pos, "RoPE en place sur q et k",
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          py::arg("positions") = c10::optional<torch::Tensor>(),
          py::arg("wq") = c10::optional<torch::Tensor>(),
          py::arg("wk") = c10::optional<torch::Tensor>(),
          py::arg("eps") = 1e-6);
    m.def("moe_reduce", &moe_reduce,
          "MoE : ponderation et somme des top_k sorties d'un jeton");
    m.def("moe_route", &moe_route, "routage MoE : scores, biais, top-k, poids");
    m.def("rmsnorm_bf16", &rmsnorm_bf16,
          "RMSNorm bf16 fusionnee (variance fp32), residu optionnel",
          py::arg("x"), py::arg("w"), py::arg("eps"),
          py::arg("residu") = c10::optional<torch::Tensor>(),
          py::arg("mult") = 1.0);
    m.def("kda_decode", &kda_decode, "KDA : pas de decodage fusionne (une sequence)");
    m.def("mla_decode", &mla_decode, "MLA absorbee : scores + softmax + lecture latente");
    m.def("mla_decode_batch", &mla_decode_batch,
          "MLA : decodage de B creneaux en un lancement, caches par table d'adresses [B] int64");
    m.def("paged_attention", &paged_attention,
          "attention de decodage fusionnee sur cache KV int8 pagine");
}
