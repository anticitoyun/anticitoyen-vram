# M2 — pas b=12 Coder, régime naturel/mma2 (41 couches profilées × 48/41), W de M1 quand mesuré

| noyau | appels/pas | t ms/pas | W (M1) | J/pas | Go froid | Go chaud | chaud/froid | inst/oct | nJ/oct froid |
|---|---|---|---|---|---|---|---|---|---|
| `nvfp4_gemm_grouped_mma2_kernel<16, 4>` | 144 | 3.537 | n.m. | — | 3.490 | 3.472 | 0.99 | 0.15 | — |
| `_etroit_kernel` | 97 | 1.036 | 166.0 | 0.172 | 0.951 | 0.940 | 0.99 | 0.24 | 0.18 |
| `at::elementwise_kernel` | 146 | 0.512 | n.m. | — | 0.012 | 0.001 | 0.10 | 0.44 | — |
| `at::vectorized_elementwise_kernel` | 244 | 0.473 | n.m. | — | 0.023 | 0.000 | 0.00 | 0.15 | — |
| `moe_route_pack_kernel<__nv_bfloat16, int>` | 48 | 0.460 | n.m. | — | 0.005 | 0.001 | 0.25 | 8.79 | — |
| `at::reduce_kernel` | 97 | 0.454 | n.m. | — | 0.140 | 0.002 | 0.02 | 0.10 | — |
| `_route_fusee_kernel` | 48 | 0.410 | n.m. | — | 0.001 | 0.000 | 0.00 | 2.18 | — |
| `rmsnorm_bf16_kernel` | 97 | 0.378 | 121.0 | 0.046 | 0.014 | 0.001 | 0.08 | 0.84 | 3.26 |
| `nvfp4_quant_act_kernel` | 96 | 0.356 | n.m. | — | 0.037 | 0.001 | 0.03 | 0.62 | — |
| `_partiel_kernel` | 48 | 0.280 | n.m. | — | 0.085 | 0.080 | 0.93 | 0.48 | — |
| `rope_inplace_kernel` | 49 | 0.211 | n.m. | — | 0.008 | 0.000 | 0.00 | 2.09 | — |
| `0_wmma_tensorop_bf16_s161616gemm_bf16_16x16_` | 48 | 0.191 | 397.0 | 0.076 | 0.029 | 0.026 | 0.88 | 0.09 | 2.59 |
| `_reduce_kernel` | 48 | 0.177 | n.m. | — | 0.103 | 0.000 | 0.00 | 0.11 | — |
| `kv_write_int8_kernel` | 49 | 0.166 | n.m. | — | 0.001 | 0.000 | 0.00 | 1.38 | — |
| `moe_reduce_trie_kernel` | 48 | 0.126 | n.m. | — | 0.026 | 0.000 | 0.00 | 0.22 | — |
| `cublasLt::splitKreduce_kernel<32, 16>` | 48 | 0.112 | n.m. | — | 0.002 | 0.001 | 0.50 | 0.14 | — |
| `moe_act_kernel` | 48 | 0.110 | n.m. | — | 0.019 | 0.000 | 0.00 | 0.63 | — |
| `at::indexSelectSmallIndex` | 1 | 0.007 | n.m. | — | 0.000 | 0.000 | 0.00 | 5.12 | — |
| TOTAL | | 9.00 | | | 4.95 | 4.53 | 0.915 | | |
