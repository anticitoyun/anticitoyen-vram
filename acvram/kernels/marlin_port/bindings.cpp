// Port acvram (18/09) des noyaux Marlin de vLLM v0.29.0 (Apache-2.0, voir
// LICENSE-vllm) : enregistrement des deux ops, schémas copiés de
// csrc/libtorch_stable/{torch_bindings.cpp, moe/torch_bindings.cpp}. Les
// implémentations CUDA sont dans gptq_marlin_repack.cu et ops.cu
// (espace de noms `acvram_marlin`).
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(acvram_marlin, m) {
  m.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, int b_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_full_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float,"
      "int thread_k, int thread_n, int blocks_per_sm) -> Tensor");
}
