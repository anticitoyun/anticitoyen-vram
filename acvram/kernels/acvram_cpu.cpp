// Noyaux processeur de déquantification et multiplication, pour l'étage hôte.
//
// Une couche dont les poids résident en mémoire vive peut être calculée de deux
// façons :
//
//   transfert  copier les poids vers le GPU par le PCIe et calculer là-bas
//   processeur calculer sur place
//
// Sur la machine cible, la seconde est le meilleur pari. Le PCIe 5.0 x16 offre
// environ 54 Go/s de bande passante utile vers l'appareil ; la DDR5-6000 en
// double canal offre environ 70 à 75 Go/s en lecture séquentielle. Les deux
// chemins sont limités par la mémoire et lisent les mêmes octets : calculer sur
// place est donc environ 1,4 fois plus rapide, et laisse en outre le GPU libre
// au lieu de le faire attendre une copie.
//
// Cela ne tient que si le processeur lit réellement les poids empaquetés sur
// 4 bits, au lieu d'en matérialiser d'abord une copie 16 bits, ce qui
// triplerait le trafic et rendrait aussitôt l'avantage. D'où ces noyaux : ils
// dépaquettent les quartets dans des registres vectoriels et n'écrivent jamais
// un poids déquantifié en mémoire.
//
// La numérique doit correspondre exactement à acvram/quant/nvfp4.py et
// acvram/quant/int4.py ; tests/test_improvements.py le vérifie.

// Construit en bibliothèque partagée ordinaire à interface C et chargé par
// ctypes, et non en extension torch. Cela supprime la dépendance de compilation
// aux en-têtes Python et à ninja, qu'une machine de déploiement n'a aucune
// raison d'embarquer :
//
//     g++ -O3 -fPIC -shared -fopenmp -o libacvram_cpu.so acvram_cpu.cpp

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define ACVRAM_X86 1
#endif

namespace {

// Magnitudes E2M1, indexées par le champ de magnitude sur 3 bits.
const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

// FP8 E4M3 (variante OCP « fn » : pas d'infinis, un seul encodage NaN) -> float.
inline float e4m3_to_float(uint8_t bits) {
    const uint32_t sign = (bits >> 7) & 0x1u;
    const uint32_t exp = (bits >> 3) & 0xFu;
    const uint32_t man = bits & 0x7u;
    float mag;
    if (exp == 0) {
        // sous-normal : 2^-6 × (mantisse / 8)
        mag = static_cast<float>(man) * 0.001953125f;   // 2^-9
    } else {
        const int e = static_cast<int>(exp) - 7;
        mag = (1.0f + static_cast<float>(man) * 0.125f) * std::ldexp(1.0f, e);
    }
    return sign ? -mag : mag;
}

inline float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 0x1u;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    float mag;
    if (exp == 0) {
        mag = static_cast<float>(man) * 5.9604645e-8f;      // 2^-24
    } else if (exp == 31) {
        mag = man ? std::nanf("") : std::numeric_limits<float>::infinity();
    } else {
        mag = (1.0f + static_cast<float>(man) / 1024.0f) *
              std::ldexp(1.0f, static_cast<int>(exp) - 15);
    }
    return sign ? -mag : mag;
}

// -------------------------------------------------------------------------
// chemins scalaires de référence — toujours compilés, toujours corrects
// -------------------------------------------------------------------------

float int4_dot_scalar(const uint8_t *q, const uint16_t *scales,
                      const uint8_t *zeros, const float *x, int K, int group) {
    float acc = 0.f;
    const int half_k = K >> 1;
    for (int idx = 0; idx < half_k; ++idx) {
        const int k0 = idx << 1;
        const int g = k0 / group;
        const float s = half_to_float(scales[g]);
        const uint8_t zpacked = zeros[g >> 1];
        const float z = static_cast<float>((g & 1) ? ((zpacked >> 4) & 0xF)
                                                   : (zpacked & 0xF));
        const uint8_t b = q[idx];
        acc += ((static_cast<float>(b & 0xF) - z) * x[k0]
                + (static_cast<float>((b >> 4) & 0xF) - z) * x[k0 + 1]) * s;
    }
    return acc;
}

float nvfp4_dot_scalar(const uint8_t *q, const uint8_t *bscale, float gscale,
                       const float *x, int K) {
    float acc = 0.f;
    const int half_k = K >> 1;
    for (int idx = 0; idx < half_k; ++idx) {
        const int k0 = idx << 1;
        const float s = e4m3_to_float(bscale[k0 >> 4]) * gscale;
        const uint8_t b = q[idx];
        const uint8_t c0 = b & 0xF;
        const uint8_t c1 = (b >> 4) & 0xF;
        float v0 = kE2M1[c0 & 7];
        float v1 = kE2M1[c1 & 7];
        if (c0 & 8) v0 = -v0;
        if (c1 & 8) v1 = -v1;
        acc += (v0 * x[k0] + v1 * x[k0 + 1]) * s;
    }
    return acc;
}

// -------------------------------------------------------------------------
// chemins AVX2 — compilés inconditionnellement via un attribut de cible, si
// bien que le fichier se construit sur une machine incapable de les exécuter,
// et sélectionnés à l'exécution.
// -------------------------------------------------------------------------

#ifdef ACVRAM_X86

__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// 8 octets empaquetés -> 16 quartets dans l'ordre naturel, en deux vecteurs de 8 int32.
__attribute__((target("avx2,fma")))
inline void unpack16(const uint8_t *p, __m256i &n0, __m256i &n1) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(b, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
    // l'entrelacement donne [bas0, haut0, bas1, haut1, ...], soit l'ordre des quartets
    const __m128i inter = _mm_unpacklo_epi8(lo, hi);
    n0 = _mm256_cvtepu8_epi32(inter);
    n1 = _mm256_cvtepu8_epi32(_mm_srli_si128(inter, 8));
}

__attribute__((target("avx2,fma")))
float int4_dot_avx2(const uint8_t *q, const uint16_t *scales,
                    const uint8_t *zeros, const float *x, int K, int group) {
    __m256 acc = _mm256_setzero_ps();
    const int half_k = K >> 1;
    int idx = 0;
    for (; idx + 8 <= half_k; idx += 8) {
        const int k0 = idx << 1;                 // 16 poids par itération
        const int g = k0 / group;
        const float s = half_to_float(scales[g]);
        const uint8_t zpacked = zeros[g >> 1];
        const float z = static_cast<float>((g & 1) ? ((zpacked >> 4) & 0xF)
                                                   : (zpacked & 0xF));
        const __m256 vs = _mm256_set1_ps(s);
        const __m256 vz = _mm256_set1_ps(z);

        __m256i n0, n1;
        unpack16(q + idx, n0, n1);
        __m256 f0 = _mm256_sub_ps(_mm256_cvtepi32_ps(n0), vz);
        __m256 f1 = _mm256_sub_ps(_mm256_cvtepi32_ps(n1), vz);
        f0 = _mm256_mul_ps(f0, vs);
        f1 = _mm256_mul_ps(f1, vs);
        acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + k0), acc);
        acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + k0 + 8), acc);
    }
    float total = hsum256(acc);
    if (idx < half_k) {
        total += int4_dot_scalar(q + idx, scales, zeros, x, (half_k - idx) << 1,
                                 group);
    }
    return total;
}

__attribute__((target("avx2,fma")))
float nvfp4_dot_avx2(const uint8_t *q, const uint8_t *bscale, float gscale,
                     const float *x, int K) {
    const __m256 lut = _mm256_setr_ps(0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f);
    const __m256i seven = _mm256_set1_epi32(7);
    const __m256i eight = _mm256_set1_epi32(8);
    __m256 acc = _mm256_setzero_ps();
    const int half_k = K >> 1;
    int idx = 0;
    for (; idx + 8 <= half_k; idx += 8) {
        const int k0 = idx << 1;                 // exactement un bloc de 16
        const float s = e4m3_to_float(bscale[k0 >> 4]) * gscale;
        const __m256 vs = _mm256_set1_ps(s);

        __m256i n0, n1;
        unpack16(q + idx, n0, n1);
        // magnitude sur les 3 bits bas, signe du bit 3 basculé vers le bit 31
        __m256 v0 = _mm256_permutevar8x32_ps(lut, _mm256_and_si256(n0, seven));
        __m256 v1 = _mm256_permutevar8x32_ps(lut, _mm256_and_si256(n1, seven));
        const __m256i s0 = _mm256_slli_epi32(_mm256_and_si256(n0, eight), 28);
        const __m256i s1 = _mm256_slli_epi32(_mm256_and_si256(n1, eight), 28);
        v0 = _mm256_xor_ps(v0, _mm256_castsi256_ps(s0));
        v1 = _mm256_xor_ps(v1, _mm256_castsi256_ps(s1));
        v0 = _mm256_mul_ps(v0, vs);
        v1 = _mm256_mul_ps(v1, vs);
        acc = _mm256_fmadd_ps(v0, _mm256_loadu_ps(x + k0), acc);
        acc = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + k0 + 8), acc);
    }
    float total = hsum256(acc);
    if (idx < half_k) {
        total += nvfp4_dot_scalar(q + idx, bscale + ((idx << 1) >> 4), gscale,
                                  x + (idx << 1), (half_k - idx) << 1);
    }
    return total;
}

bool have_avx2() {
    static const bool yes = __builtin_cpu_supports("avx2")
                            && __builtin_cpu_supports("fma");
    return yes;
}

#else
bool have_avx2() { return false; }
#endif

}  // namespace

// -------------------------------------------------------------------------
// C ABI
// -------------------------------------------------------------------------

extern "C" {

int acvram_cpu_has_avx2() { return have_avx2() ? 1 : 0; }

// y[n, m] = somme_k dequant(W[m, k]) * x[n, k]
void acvram_int4_gemv(const uint8_t *q, const uint16_t *scales,
                      const uint8_t *zeros, const float *x, float *y,
                      int64_t M, int64_t K, int64_t N, int64_t group) {
    const int64_t row_bytes = K / 2;
    const int64_t ng = K / group;
    const int64_t zbytes = (ng + 1) / 2;
    const bool avx2 = have_avx2();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
        const uint8_t *qr = q + m * row_bytes;
        const uint16_t *sr = scales + m * ng;
        const uint8_t *zr = zeros + m * zbytes;
        for (int64_t n = 0; n < N; ++n) {
            const float *xn = x + n * K;
#ifdef ACVRAM_X86
            y[n * M + m] = avx2
                ? int4_dot_avx2(qr, sr, zr, xn, (int)K, (int)group)
                : int4_dot_scalar(qr, sr, zr, xn, (int)K, (int)group);
#else
            (void)avx2;
            y[n * M + m] = int4_dot_scalar(qr, sr, zr, xn, (int)K, (int)group);
#endif
        }
    }
}

void acvram_nvfp4_gemv(const uint8_t *q, const uint8_t *bscale, float gscale,
                       const float *x, float *y,
                       int64_t M, int64_t K, int64_t N) {
    const int64_t row_bytes = K / 2;
    const int64_t nblocks = K / 16;
    const bool avx2 = have_avx2();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
        const uint8_t *qr = q + m * row_bytes;
        const uint8_t *sr = bscale + m * nblocks;
        for (int64_t n = 0; n < N; ++n) {
            const float *xn = x + n * K;
#ifdef ACVRAM_X86
            y[n * M + m] = avx2 ? nvfp4_dot_avx2(qr, sr, gscale, xn, (int)K)
                                : nvfp4_dot_scalar(qr, sr, gscale, xn, (int)K);
#else
            (void)avx2;
            y[n * M + m] = nvfp4_dot_scalar(qr, sr, gscale, xn, (int)K);
#endif
        }
    }
}

}  // extern "C"
