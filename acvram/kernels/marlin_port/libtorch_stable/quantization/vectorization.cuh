// Port acvram (23/09/2026, pièce 62) de vLLM v0.29.0 (commit 98dff2a), fichier d origine
// vllm/csrc/libtorch_stable/quantization/vectorization.cuh — licence Apache-2.0 (voir ../../LICENSE-vllm et NOTICE). Modifications : voir NOTICE.
#pragma once
/**
 * __device__ datatypes vectorized by 4
 */

// Include both AMD and NVIDIA fp8 types to avoid circular import
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>

namespace vllm {

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <typename quant_type_t, size_t vec_size>
struct __align__(vec_size * sizeof(quant_type_t)) q8_n_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>);
  quant_type_t val[vec_size];
};

template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;
template <typename quant_type_t>
using q8x4_t = q8_n_t<quant_type_t, 4>;

}  // namespace vllm
