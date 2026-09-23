// Test : mma.sync m16n8k64 kind::mxf4nvf4 block_scale 4X ue4m3 sur sm_120a (RTX 5090).
// Compilation : nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o test_mxf4 outils/test_mxf4_sm120.cu
//   (-arch=sm_120a seul produit du .target sm_120 et ptxas refuse : il faut le gencode explicite).
// 1) un MMA 16x8x64 contre une reference CPU (E2M1 x UE4M3 par bloc de 16, acc f32) : 128/128 bit-identiques le 13/09/2026 ;
// 2) micro-banc : 4000 iters x 4 chaines/warp sur 170 SM x 8 blocs x 4 warps, TFLOPS contre mma.sync bf16 m16n8k16
//    (13/09/2026 : 2059,6 TFLOPS contre 260,1, x7,9 ; SASS : OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X).
// Layouts des fragments decodes depuis cute/atom/mma_traits_sm120.hpp (CUTLASS, BSD-3) :
//   A (16x64 E2M1, 4 x b32) : a0 ligne g colonnes 8t+i, a1 ligne g+8, a2/a3 idem +32 en k (g=lane>>2, t=lane&3, nibble bas d abord) ;
//   B (64x8, 2 x b32) : colonne g, k = 8t+i (+32 pour b1) ;
//   SF-A (b32) : ligne (lane>>2)+8*(lane&1), 4 octets = 4 blocs de 16 en k ; SF-B : colonne lane>>2 ; byte-id et thread-id a 0.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

__device__ __forceinline__ void mma_mxf4(float d[4], const uint32_t a[4], const uint32_t b[2],
                                         const float c[4], uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),
      "r"(sfa),"h"(z),"h"(z),"r"(sfb),"h"(z),"h"(z));
}

__device__ __forceinline__ void mma_bf16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
    : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

// A : 16x64 nibbles E2M1 (ligne-major, A[m][k]), B : 64x8 (B[k][n]), SFA : 16x4, SFB : 8x4 (UE4M3)
__global__ void k_un(const uint8_t* A, const uint8_t* B, const uint8_t* SFA, const uint8_t* SFB, float* D) {
  int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  uint32_t a[4], b[2];
  for (int r = 0; r < 4; r++) {
    int m = g + 8 * (r & 1), kb = 8 * t + 32 * (r >> 1);
    uint32_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint32_t)(A[m * 64 + kb + i] & 0xF) << (4 * i);
    a[r] = v;
  }
  for (int r = 0; r < 2; r++) {
    int n = g, kb = 8 * t + 32 * r;
    uint32_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint32_t)(B[(kb + i) * 8 + n] & 0xF) << (4 * i);
    b[r] = v;
  }
  int msf = (lane >> 2) + 8 * (lane & 1);
  uint32_t sfa = 0, sfb = 0;
  for (int j = 0; j < 4; j++) { sfa |= (uint32_t)SFA[msf * 4 + j] << (8 * j); sfb |= (uint32_t)SFB[g * 4 + j] << (8 * j); }
  float c[4] = {0, 0, 0, 0}, d[4];
  mma_mxf4(d, a, b, c, sfa, sfb);
  D[g * 8 + 2 * t] = d[0]; D[g * 8 + 2 * t + 1] = d[1];
  D[(g + 8) * 8 + 2 * t] = d[2]; D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

template <int KIND> __global__ void k_banc(int iters, float* out) {
  uint32_t a[4] = {0x21212121u ^ threadIdx.x, 0x12121212u, 0x21212121u, 0x12121212u};
  uint32_t b[2] = {0x21212121u, 0x12121212u ^ threadIdx.x};
  uint32_t sf = 0x38383838u;  // 1.0
  float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0}, c3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; i++) {
    if (KIND == 0) { mma_mxf4(c0, a, b, c0, sf, sf); mma_mxf4(c1, a, b, c1, sf, sf);
                     mma_mxf4(c2, a, b, c2, sf, sf); mma_mxf4(c3, a, b, c3, sf, sf); }
    else           { mma_bf16(c0, a, b, c0); mma_bf16(c1, a, b, c1); mma_bf16(c2, a, b, c2); mma_bf16(c3, a, b, c3); }
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (s == 123456.f) out[0] = s;
}

static float e2m1(uint8_t v) {
  static const float t[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  return (v & 8) ? -t[v & 7] : t[v & 7];
}
static float ue4m3(uint8_t v) {
  int e = (v >> 3) & 0xF, m = v & 7;
  if (e == 0) return ldexpf((float)m / 8.f, -6);
  return ldexpf(1.f + m / 8.f, e - 7);
}

int main() {
  // donnees pseudo-aleatoires deterministes
  uint8_t A[16 * 64], B[64 * 8], SFA[16 * 4], SFB[8 * 4];
  uint32_t s = 12345;
  auto rnd = [&]() { s = s * 1103515245u + 12345u; return (s >> 16) & 0x7FFF; };
  for (auto& x : A) x = rnd() & 0xF;
  for (auto& x : B) x = rnd() & 0xF;
  static const uint8_t sfs[6] = {0x38, 0x40, 0x30, 0x3A, 0x44, 0x2C};  // 1, 2, .5, 1.25, 2.5, .25
  for (auto& x : SFA) x = sfs[rnd() % 6];
  for (auto& x : SFB) x = sfs[rnd() % 6];
  double ref[16 * 8];
  for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) {
    double acc = 0;
    for (int k = 0; k < 64; k++)
      acc += (double)e2m1(A[m * 64 + k]) * ue4m3(SFA[m * 4 + k / 16]) * e2m1(B[k * 8 + n]) * ue4m3(SFB[n * 4 + k / 16]);
    ref[m * 8 + n] = acc;
  }
  uint8_t *dA, *dB, *dSFA, *dSFB; float* dD;
  CK(cudaMalloc(&dA, sizeof A)); CK(cudaMalloc(&dB, sizeof B)); CK(cudaMalloc(&dSFA, sizeof SFA));
  CK(cudaMalloc(&dSFB, sizeof SFB)); CK(cudaMalloc(&dD, 16 * 8 * 4));
  CK(cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFA, SFA, sizeof SFA, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dSFB, SFB, sizeof SFB, cudaMemcpyHostToDevice));
  k_un<<<1, 32>>>(dA, dB, dSFA, dSFB, dD);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float D[128]; CK(cudaMemcpy(D, dD, sizeof D, cudaMemcpyDeviceToHost));
  double maxabs = 0, maxref = 0; int exact = 0;
  for (int i = 0; i < 128; i++) {
    double d = fabs((double)D[i] - ref[i]); if (d > maxabs) maxabs = d;
    if (fabs(ref[i]) > maxref) maxref = fabs(ref[i]);
    if ((float)ref[i] == D[i]) exact++;
  }
  printf("EXACTITUDE elements_bit_identiques=%d/128 ecart_max_abs=%.3e max_ref=%.3f D[0]=%.4f ref[0]=%.4f D[77]=%.4f ref[77]=%.4f\n",
         exact, maxabs, maxref, D[0], ref[0], D[77], ref[77]);

  // micro-banc
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = sms * 8, threads = 128, iters = 4000;  // 4 warps/bloc, 4 chaines/warp
  double warps = (double)blocks * threads / 32;
  float* dout; CK(cudaMalloc(&dout, 4));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int kind = 0; kind < 2; kind++) {
    if (kind == 0) k_banc<0><<<blocks, threads>>>(100, dout); else k_banc<1><<<blocks, threads>>>(100, dout);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int rep = 0; rep < 5; rep++) {
      cudaEventRecord(e0);
      if (kind == 0) k_banc<0><<<blocks, threads>>>(iters, dout); else k_banc<1><<<blocks, threads>>>(iters, dout);
      cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
      float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    double flop_per_mma = kind == 0 ? 2.0 * 16 * 8 * 64 : 2.0 * 16 * 8 * 16;
    double tflops = warps * iters * 4 * flop_per_mma / (best * 1e-3) / 1e12;
    printf("BANC %s : %d SM, %d blocs x %d fils, %d iters x 4 chaines : %.3f ms, %.1f TFLOPS\n",
           kind == 0 ? "mxf4nvf4 m16n8k64" : "bf16 m16n8k16   ", sms, blocks, threads, iters, best, tflops);
  }
  return 0;
}
