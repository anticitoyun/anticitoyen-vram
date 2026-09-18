# Inventaire — cellules mesurées sous `certifie-b12-15-09.py` (et ses prédécesseurs `certifie-*`) : appel `nvidia-smi` en sous-processus tous les 200 pas DANS la fenêtre (poste7-profil-verdict-18-09)

Biais d'instrument : 0.136 ms par pas (27,1 ms par appel, mesuré verdict-profil-prefill-b1-18-09), le même quel que soit b ; effet relatif = 0,135 / pas_ms ; t/s corrigé = t/s × pas / (pas − 0,135) ; J/jeton : la fenêtre porte ~0,135 ms à puissance de repos par pas (biais < 1 % à b ≥ 4, ≤ 3 % à b=1). Étiquette, pas effacement : les chiffres publiés restent, l'écart est à confirmer par l'ABAB ancien/nouvel instrument (poste1). 197 passes, du 15/09 au 18/09, triées par date.

| passe (scratchpad/) | bras | modèle | b | pas ms | t/s | J/jeton | biais smi | t/s corrigé |
|---|---|---|---|---|---|---|---|---|
| ties-moe-15-09/certifie-moyen-20s-A1.json | A1 | (Coder par défaut) |  | 17.365 | 630.31 | 0.6195 | 0.8 % | 635.3 |
| ties-moe-15-09/certifie-moyen-20s-B1.json | B1 | (Coder par défaut) |  | 16.248 | 673.65 | 0.5371 | 0.8 % | 679.3 |
| ties-moe-15-09/certifie-moyen-20s-A2.json | A2 | (Coder par défaut) |  | 17.369 | 630.18 | 0.6213 | 0.8 % | 635.1 |
| ties-moe-15-09/certifie-moyen-20s-B2.json | B2 | (Coder par défaut) |  | 16.25 | 673.6 | 0.5403 | 0.8 % | 679.3 |
| ties-moe-15-09/certifie-b1-moyen-20s-A1.json | b1-A1 | (Coder par défaut) |  | 4.483 | 223.04 | 1.5099 | 3.0 % | 230.0 |
| ties-moe-15-09/certifie-b1-moyen-20s-B1.json | b1-B1 | (Coder par défaut) |  | 7.005 | 142.75 | 1.7241 | 1.9 % | 145.6 |
| ties-moe-15-09/certifie-b1-moyen-20s-A2.json | b1-A2 | (Coder par défaut) |  | 4.483 | 223.04 | 1.513 | 3.0 % | 230.0 |
| ties-moe-15-09/certifie-b1-moyen-20s-B2.json | b1-B2 | (Coder par défaut) |  | 7.004 | 142.77 | 1.727 | 1.9 % | 145.6 |
| courbe-lot-15-09/certifie-b2-moyen-20s-A1.json | b2-A1 | (Coder par défaut) |  | 6.165 | 324.39 | 1.1599 | 2.2 % | 331.7 |
| courbe-lot-15-09/certifie-b2-moyen-20s-B1.json | b2-B1 | (Coder par défaut) |  | 8.404 | 237.99 | 1.1221 | 1.6 % | 241.9 |
| courbe-lot-15-09/certifie-b2-moyen-20s-A2.json | b2-A2 | (Coder par défaut) |  | 6.166 | 324.37 | 1.163 | 2.2 % | 331.7 |
| courbe-lot-15-09/certifie-b2-moyen-20s-B2.json | b2-B2 | (Coder par défaut) |  | 8.402 | 238.05 | 1.1218 | 1.6 % | 242.0 |
| courbe-lot-15-09/certifie-b3-moyen-20s-A1.json | b3-A1 | (Coder par défaut) |  | 7.285 | 411.8 | 0.9602 | 1.9 % | 419.6 |
| courbe-lot-15-09/certifie-b3-moyen-20s-B1.json | b3-B1 | (Coder par défaut) |  | 9.244 | 324.52 | 0.8962 | 1.5 % | 329.3 |
| courbe-lot-15-09/certifie-b3-moyen-20s-A2.json | b3-A2 | (Coder par défaut) |  | 7.284 | 411.85 | 0.9529 | 1.9 % | 419.7 |
| courbe-lot-15-09/certifie-b3-moyen-20s-B2.json | b3-B2 | (Coder par défaut) |  | 9.244 | 324.55 | 0.8981 | 1.5 % | 329.4 |
| courbe-lot-15-09/certifie-b4-moyen-20s-A1.json | b4-A1 | (Coder par défaut) |  | 8.125 | 492.32 | 0.7947 | 1.7 % | 500.7 |
| courbe-lot-15-09/certifie-b4-moyen-20s-B1.json | b4-B1 | (Coder par défaut) |  | 10.085 | 396.62 | 0.813 | 1.3 % | 402.0 |
| courbe-lot-15-09/certifie-b4-moyen-20s-A2.json | b4-A2 | (Coder par défaut) |  | 8.125 | 492.28 | 0.7956 | 1.7 % | 500.6 |
| courbe-lot-15-09/certifie-b4-moyen-20s-B2.json | b4-B2 | (Coder par défaut) |  | 10.083 | 396.7 | 0.8127 | 1.3 % | 402.1 |
| courbe-lot-15-09/certifie-b6-moyen-20s-A1.json | b6-A1 | (Coder par défaut) |  | 10.923 | 549.29 | 0.7277 | 1.2 % | 556.2 |
| courbe-lot-15-09/certifie-b6-moyen-20s-B1.json | b6-B1 | (Coder par défaut) |  | 12.889 | 465.51 | 0.7475 | 1.1 % | 470.5 |
| courbe-lot-15-09/certifie-b6-moyen-20s-A2.json | b6-A2 | (Coder par défaut) |  | 10.923 | 549.29 | 0.7267 | 1.2 % | 556.2 |
| courbe-lot-15-09/certifie-b6-moyen-20s-B2.json | b6-B2 | (Coder par défaut) |  | 12.891 | 465.43 | 0.7461 | 1.1 % | 470.4 |
| 1aj-decodage-15-09/certifie-b1-20s-A1.json | 1aj-b1-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.485 | 222.97 | 1.4916 | 3.0 % | 229.9 |
| 1aj-decodage-15-09/certifie-b1-20s-B1.json | 1aj-b1-B1 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 4.297 | 232.71 | 1.3904 | 3.2 % | 240.3 |
| 1aj-decodage-15-09/certifie-b1-20s-A2.json | 1aj-b1-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.483 | 223.07 | 1.4992 | 3.0 % | 230.0 |
| 1aj-decodage-15-09/certifie-b1-20s-B2.json | 1aj-b1-B2 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 4.297 | 232.71 | 1.4042 | 3.2 % | 240.3 |
| 1aj-decodage-15-09/certifie-b12-20s-A1.json | 1aj-b12-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 16.249 | 673.62 | 0.5592 | 0.8 % | 679.3 |
| 1aj-decodage-15-09/certifie-b12-20s-B1.json | 1aj-b12-B1 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 17.923 | 610.7 | 0.6186 | 0.8 % | 615.4 |
| 1aj-decodage-15-09/certifie-b12-20s-A2.json | 1aj-b12-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 16.249 | 673.62 | 0.5603 | 0.8 % | 679.3 |
| 1aj-decodage-15-09/certifie-b12-20s-B2.json | 1aj-b12-B2 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 17.924 | 610.66 | 0.6206 | 0.8 % | 615.3 |
| gemvmax-bissect-15-09/m1-b12-A1.json | m1-A1 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 17.928 | 610.55 | 0.6152 | 0.8 % | 615.2 |
| gemvmax-bissect-15-09/m1-b12-B1.json | m1-B1 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 19.609 | 558.21 | 0.6839 | 0.7 % | 562.1 |
| gemvmax-bissect-15-09/m1-b12-A2.json | m1-A2 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 17.929 | 610.49 | 0.6196 | 0.8 % | 615.1 |
| gemvmax-bissect-15-09/m1-b12-B2.json | m1-B2 | Qwen3-Coder-30B-A3B-srcbf16-1ajvarD |  | 19.607 | 558.25 | 0.6872 | 0.7 % | 562.1 |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-A1.json | bis-c652947-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.71 | 1.4547 | 3.2 % | 240.3 |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-B1.json | bis-0c90017-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.484 | 223.03 | 1.5065 | 3.0 % | 230.0 |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-A2.json | bis-c652947-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.73 | 1.4609 | 3.2 % | 240.3 |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-B2.json | bis-0c90017-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.484 | 223.01 | 1.5101 | 3.0 % | 230.0 |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-A1.json | bis-97f2313-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.72 | 1.4574 | 3.2 % | 240.3 |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-B1.json | bis-7f3f422-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.484 | 223.02 | 1.5012 | 3.0 % | 230.0 |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-A2.json | bis-97f2313-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.71 | 1.4545 | 3.2 % | 240.3 |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-B2.json | bis-7f3f422-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.483 | 223.07 | 1.5097 | 3.0 % | 230.0 |
| v064-b1-15-09/certifie-b1-A1.json | v064-b1-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.296 | 232.76 | 1.4601 | 3.2 % | 240.3 |
| v064-b1-15-09/certifie-b1-A2.json | v064-b1-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.72 | 1.4607 | 3.2 % | 240.3 |
| v064-b1-15-09/certifie-b1-A3.json | v064-b1-A3 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.75 | 1.4629 | 3.2 % | 240.3 |
| v064-b1-15-09/certifie-b1-A4.json | v064-b1-A4 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.296 | 232.76 | 1.4656 | 3.2 % | 240.3 |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-A1.json | b1-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.298 | 232.68 | 1.4439 | 3.2 % | 240.3 |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-B1.json | b1-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 5.042 | 198.32 | 1.4506 | 2.7 % | 203.8 |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-A2.json | b1-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.72 | 1.4548 | 3.2 % | 240.3 |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-B2.json | b1-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 5.043 | 198.28 | 1.4566 | 2.7 % | 203.8 |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-A1.json | b2-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 6.165 | 324.41 | 1.1682 | 2.2 % | 331.7 |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-B1.json | b2-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 6.728 | 297.28 | 1.0504 | 2.0 % | 303.4 |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-A2.json | b2-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 6.165 | 324.43 | 1.1692 | 2.2 % | 331.7 |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-B2.json | b2-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 6.726 | 297.34 | 1.052 | 2.0 % | 303.5 |
| courbe-lot-rp-15-09/certifie-b3-moyen-20s-A1.json | b3-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 7.285 | 411.79 | 0.9615 | 1.9 % | 419.6 |
| courbe-lot-rp-15-09/certifie-b3-moyen-20s-B1.json | b3-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 7.842 | 382.55 | 0.9006 | 1.7 % | 389.3 |
| courbe-lot-rp-15-09/certifie-b3-moyen-20s-A2.json | b3-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 7.285 | 411.81 | 0.9639 | 1.9 % | 419.6 |
| courbe-lot-rp-15-09/certifie-b3-moyen-20s-B2.json | b3-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 7.845 | 382.41 | 0.9038 | 1.7 % | 389.1 |
| courbe-lot-rp-15-09/certifie-b4-moyen-20s-A1.json | b4-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 7.844 | 509.93 | 0.7823 | 1.7 % | 518.9 |
| courbe-lot-rp-15-09/certifie-b4-moyen-20s-B1.json | b4-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 8.124 | 492.37 | 0.74 | 1.7 % | 500.7 |
| courbe-lot-rp-15-09/certifie-b4-moyen-20s-A2.json | b4-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 7.847 | 509.72 | 0.7839 | 1.7 % | 518.7 |
| courbe-lot-rp-15-09/certifie-b4-moyen-20s-B2.json | b4-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 8.123 | 492.45 | 0.7402 | 1.7 % | 500.8 |
| courbe-lot-rp-15-09/certifie-b6-moyen-20s-A1.json | b6-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 10.644 | 563.72 | 0.6977 | 1.3 % | 571.0 |
| courbe-lot-rp-15-09/certifie-b6-moyen-20s-B1.json | b6-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 9.524 | 630.01 | 0.6216 | 1.4 % | 639.1 |
| courbe-lot-rp-15-09/certifie-b6-moyen-20s-A2.json | b6-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 10.643 | 563.74 | 0.6976 | 1.3 % | 571.0 |
| courbe-lot-rp-15-09/certifie-b6-moyen-20s-B2.json | b6-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 9.524 | 630.01 | 0.6215 | 1.4 % | 639.1 |
| duel-glm-15-09/acvram-decode-b1-p1.json | glm-acvram-b1-p1 | GLM-4.7-Flash-srcbf16-nvfp4 |  | 38.655 | 25.87 | 6.3578 | 0.4 % | 26.0 |
| narrow-15-09/b1-A1.json | narrow-b1-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.296 | 232.75 | 1.4386 | 3.2 % | 240.3 |
| narrow-15-09/b1-B1.json | narrow-b1-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.484 | 223.03 | 1.4636 | 3.0 % | 230.0 |
| narrow-15-09/b1-A2.json | narrow-b1-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.75 | 1.4556 | 3.2 % | 240.3 |
| narrow-15-09/b1-B2.json | narrow-b1-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.73 | 1.4578 | 3.2 % | 240.3 |
| narrow-15-09/b12-A1.json | narrow-b12-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.009 | 781.31 | 0.51 | 1.0 % | 788.9 |
| narrow-15-09/b12-B1.json | narrow-b12-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 11.765 | 930.32 | 0.4282 | 1.2 % | 941.2 |
| narrow-15-09/b12-A2.json | narrow-b12-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.011 | 781.22 | 0.5113 | 1.0 % | 788.8 |
| narrow-15-09/b12-B2.json | narrow-b12-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 11.767 | 930.22 | 0.4284 | 1.2 % | 941.1 |
| narrow-15-09/b1-B3.json | narrow-b1-B3 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.72 | 1.4558 | 3.2 % | 240.3 |
| narrow-15-09/b1-A3.json | narrow-b1-A3 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.297 | 232.7 | 1.4589 | 3.2 % | 240.3 |
| narrow-15-09/b1-B4.json | narrow-b1-B4 | Qwen3-Coder-30B-A3B-nvfp4 |  | 4.296 | 232.76 | 1.4611 | 3.2 % | 240.3 |
| awq-temoin-15-09/b12-A1.json | awq-temoin-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.014 | 781.07 | 0.5095 | 1.0 % | 788.7 |
| awq-temoin-15-09/b12-A2.json | awq-temoin-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.01 | 781.29 | 0.51 | 1.0 % | 788.9 |
| awq-temoin-15-09/b12-B1.json | awq-temoin-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.571 | 751.2 | 0.5172 | 0.9 % | 758.3 |
| awq-temoin-15-09/b12-B2.json | awq-temoin-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.57 | 751.26 | 0.5175 | 0.9 % | 758.3 |
| awq-temoin-15-09/b12-B3.json | awq-temoin-B3 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.57 | 751.27 | 0.5176 | 0.9 % | 758.3 |
| awq-temoin-15-09/b12-duree-A1.json | awq-temoin-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.926 | 785.98 | 0.5077 | 1.0 % | 793.7 |
| awq-temoin-15-09/b12-duree-A2.json | awq-temoin-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.958 | 784.18 | 0.5087 | 1.0 % | 791.9 |
| awq-temoin-15-09/b12-duree-B1.json | awq-temoin-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.056 | 778.7 | 0.5123 | 1.0 % | 786.3 |
| awq-temoin-15-09/b12-duree-B2.json | awq-temoin-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.073 | 777.77 | 0.5125 | 1.0 % | 785.3 |
| officiel-066-15-09/rondes-A1.json | off066-rondes-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.956 | 784.28 | 0.5079 | 1.0 % | 792.0 |
| officiel-066-15-09/rondes-B1.json | off066-rondes-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 11.718 | 934.1 | 0.4262 | 1.2 % | 945.0 |
| officiel-066-15-09/rondes-A2.json | off066-rondes-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.982 | 782.86 | 0.5101 | 1.0 % | 790.5 |
| officiel-066-15-09/rondes-B2.json | off066-rondes-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 11.716 | 934.26 | 0.4267 | 1.2 % | 945.2 |
| awq-temoin-15-09/b12-unite-A1.json | awq-temoin-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.948 | 784.77 | 0.508 | 1.0 % | 792.5 |
| awq-temoin-15-09/b12-unite-A2.json | awq-temoin-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.976 | 783.16 | 0.5104 | 1.0 % | 790.8 |
| awq-temoin-15-09/b12-unite-B1.json | awq-temoin-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.21 | 770.26 | 0.5158 | 1.0 % | 777.7 |
| awq-temoin-15-09/b12-unite-B2.json | awq-temoin-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 14.044 | 779.38 | 0.5121 | 1.0 % | 787.0 |
| b5-16-09/b5-A1.json | b5-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 9.585 | 521.63 | 0.7645 | 1.4 % | 529.1 |
| b5-16-09/b5-B1.json | b5-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 9.579 | 521.96 | 0.7331 | 1.4 % | 529.4 |
| b5-16-09/b5-A2.json | b5-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 9.627 | 519.39 | 0.768 | 1.4 % | 526.8 |
| b5-16-09/b5-B2.json | b5-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 9.575 | 522.18 | 0.7362 | 1.4 % | 529.7 |
| retampon-cellule-narrow-16-09/rondes-A1.json | retampon-off-A1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.584 | 805.8 | 0.4935 | 1.0 % | 813.9 |
| retampon-cellule-narrow-16-09/rondes-B1.json | retampon-off-B1 | Qwen3-Coder-30B-A3B-nvfp4 |  | 11.383 | 961.57 | 0.4114 | 1.2 % | 973.2 |
| retampon-cellule-narrow-16-09/rondes-A2.json | retampon-off-A2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.609 | 804.28 | 0.495 | 1.0 % | 812.4 |
| retampon-cellule-narrow-16-09/rondes-B2.json | retampon-off-B2 | Qwen3-Coder-30B-A3B-nvfp4 |  | 11.381 | 961.72 | 0.4117 | 1.2 % | 973.3 |
| retampon-cellule-narrow-16-09/rondes-M9a.json | retampon-mint-M9a | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.815 | 792.28 | 0.5026 | 1.0 % | 800.1 |
| retampon-cellule-narrow-16-09/rondes-M5a.json | retampon-mint-M5a | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.621 | 803.57 | 0.4956 | 1.0 % | 811.6 |
| retampon-cellule-narrow-16-09/rondes-M9b.json | retampon-mint-M9b | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.785 | 794.04 | 0.5022 | 1.0 % | 801.9 |
| retampon-cellule-narrow-16-09/rondes-M5b.json | retampon-mint-M5b | Qwen3-Coder-30B-A3B-nvfp4 |  | 13.622 | 803.55 | 0.4967 | 1.0 % | 811.6 |
| duel-glm-16-09/acvram-decode-b1-p1.json | glm-acvram-b1-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 12.739 | 78.5 | 2.8395 | 1.1 % | 79.3 |
| duel-glm-16-09/acvram-decode-b1-p2.json | glm-acvram-b1-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 12.709 | 78.69 | 2.8463 | 1.1 % | 79.5 |
| duel-glm-16-09/acvram-decode-b4-p1.json | glm-acvram-b4-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 32.008 | 124.97 | 2.1173 | 0.4 % | 125.5 |
| duel-glm-16-09/acvram-decode-b4-p2.json | glm-acvram-b4-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 32.127 | 124.51 | 2.131 | 0.4 % | 125.0 |
| duel-glm-16-09/acvram-decode-b12-p1.json | glm-acvram-b12-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 56.187 | 194.81 | 1.5751 | 0.2 % | 195.3 |
| duel-glm-16-09/acvram-decode-b12-p2.json | glm-acvram-b12-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 56.103 | 195.1 | 1.5856 | 0.2 % | 195.6 |
| eager-b12-16-09/rondes-eager-p1.json | glm-eager-b12-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 | 12 | 66.168 | 165.42 | 1.2301 | 0.2 % | 165.8 |
| eager-b12-16-09/rondes-eager-p2.json | glm-eager-b12-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 | 12 | 66.02 | 165.79 | 1.2405 | 0.2 % | 166.1 |
| post-correctif-16-09/rondes-b12-p1.json | glm-post-b12-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 21.272 | 514.56 | 0.773 | 0.6 % | 517.9 |
| post-correctif-16-09/rondes-b12-p2.json | glm-post-b12-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 21.317 | 513.47 | 0.7745 | 0.6 % | 516.8 |
| duel-b-16-09/rondes-b1-p1.json | duelb-b1-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 9.335 | 107.13 | 2.2641 | 1.5 % | 108.7 |
| duel-b-16-09/rondes-b1-p2.json | duelb-b1-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 9.308 | 107.44 | 2.2627 | 1.5 % | 109.0 |
| duel-b-16-09/rondes-b4-p1.json | duelb-b4-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 11.98 | 333.88 | 1.0276 | 1.1 % | 337.7 |
| duel-b-16-09/rondes-b4-p2.json | duelb-b4-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 |  | 11.982 | 333.84 | 1.0294 | 1.1 % | 337.7 |
| narrow-mla-16-09/rondes-A-p1.json | nm-A-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 | 12 | 21.056 | 519.84 | 0.7619 | 0.6 % | 523.2 |
| narrow-mla-16-09/rondes-A-p2.json | nm-A-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 | 12 | 21.153 | 517.45 | 0.7646 | 0.6 % | 520.8 |
| bloc-sage11-17-09/rondes-b12-p1.json | sage11-b12-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48 | 12 | 21.003 | 521.16 | 0.7587 | 0.6 % | 524.5 |
| bloc-sage11-17-09/rondes-b12-p2.json | sage11-b12-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48 | 12 | 21.024 | 520.64 | 0.7594 | 0.6 % | 524.0 |
| ppl-hadamard-17-09/rondes-b12-p1.json | hadamard-b12-p1 | GLM-4.7-Flash-srcbf16-nvfp4-hadamard51 | 12 | 22.569 | 485.0 | 0.783 | 0.6 % | 487.9 |
| ppl-hadamard-17-09/rondes-b12-p2.json | hadamard-b12-p2 | GLM-4.7-Flash-srcbf16-nvfp4-hadamard51 | 12 | 22.602 | 484.28 | 0.7846 | 0.6 % | 487.2 |
| coder-acvram-w4a16-17-09/rondes-b1-p1.json | coder-w4a16-b1-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 1 | 4.283 | 233.45 | 1.3643 | 3.2 % | 241.1 |
| coder-acvram-w4a16-17-09/rondes-b1-p2.json | coder-w4a16-b1-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 1 | 4.267 | 234.35 | 1.3652 | 3.2 % | 242.0 |
| coder-acvram-w4a16-17-09/rondes-b12-p1.json | coder-w4a16-b12-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 14.981 | 730.65 | 0.5461 | 0.9 % | 737.3 |
| coder-acvram-w4a16-17-09/rondes-b12-p2.json | coder-w4a16-b12-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 14.992 | 730.08 | 0.5468 | 0.9 % | 736.7 |
| profil-coder-pas-17-09/glm-rondes-b12-p1.json | glm-calibA-bf16-b12-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA | 12 | 20.268 | 540.05 | 0.7298 | 0.7 % | 543.7 |
| profil-coder-pas-17-09/glm-rondes-b12-p2.json | glm-calibA-bf16-b12-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA | 12 | 20.307 | 539.0 | 0.7334 | 0.7 % | 542.6 |
| profil-coder-pas-17-09/glm-rondes-b12-calibA-w8a8.json | glm-calibA-w8a8-b12 | GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA | 12 | 20.291 | 539.43 | 0.7308 | 0.7 % | 543.1 |
| profil-coder-pas-17-09/glm-rondes-b12-k48-bf16.json | glm-k48-bf16-b12 | GLM-4.7-Flash-srcbf16-nvfp4-k48 | 12 | 21.003 | 521.15 | 0.7577 | 0.6 % | 524.5 |
| poste-d-17-09/rondes-b12-plan3072.json | coder-b12-plan3072 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 16.143 | 743.38 | 0.5384 | 0.8 % | 749.7 |
| poste-d-17-09/pur-b12-plan3840.json | coder-pur-b12-plan3840 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 15.428 | 777.82 | 0.5135 | 0.9 % | 784.7 |
| poste-d-17-09/glm-rondes-b12-plan3072-p1.json | glm-calibA-bf16-b12-plan3072-p1 | GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA | 12 | 21.299 | 563.4 | 0.7016 | 0.6 % | 567.0 |
| poste-d-17-09/glm-rondes-b12-plan3072-p2.json | glm-calibA-bf16-b12-plan3072-p2 | GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA | 12 | 20.952 | 572.73 | 0.6981 | 0.6 % | 576.5 |
| palier2-qwen38-17-09/acvram-rondes-b1.json | qwen38-b1 | Qwen3.8-27B-srcexl3_6_00bpw-nvfp4 | 1 | 14.926 | 67.0 | 5.9676 | 0.9 % | 67.6 |
| palier2-qwen38-17-09/acvram-rondes-b12-p2.json | qwen38-b12-p2 | Qwen3.8-27B-srcexl3_6_00bpw-nvfp4 | 12 | 123.857 | 96.89 | 4.1062 | 0.1 % | 97.0 |
| palier2-qwen38-17-09/acvram-rondes-b12-p1.json | qwen38-b12-p1 | Qwen3.8-27B-srcexl3_6_00bpw-nvfp4 | 12 | 124.237 | 96.59 | 4.1265 | 0.1 % | 96.7 |
| coder-ec-17-09/rondes-b1.json | coder-ec-b1 | Qwen3-Coder-30B-A3B-nvfp4 | 1 | 3.879 | 257.77 | 1.214 | 3.5 % | 267.1 |
| coder-ec-17-09/rondes-b12-sansC.json | coder-ec-b12-sansC | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 13.519 | 887.67 | 0.4493 | 1.0 % | 896.7 |
| coder-c-fix-17-09/repro-b12.json | coder-cfix-b12-repro | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 12.281 | 977.15 | 0.4017 | 1.1 % | 988.1 |
| coder-c-fix-17-09/rondes-b12-p1.json | coder-cfix-b12-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 12.029 | 997.62 | 0.4001 | 1.1 % | 1009.0 |
| coder-c-fix-17-09/rondes-b12-p2.json | coder-cfix-b12-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 12.025 | 997.9 | 0.3989 | 1.1 % | 1009.3 |
| f-final-17-09/rondes-b1-p1.json | coder-F-b1-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 1 | 3.465 | 288.59 | 1.1759 | 3.9 % | 300.3 |
| f-final-17-09/rondes-b1-p2.json | coder-F-b1-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 1 | 3.476 | 287.72 | 1.1781 | 3.9 % | 299.4 |
| palier2-qwen38-17-09/acvram2-rondes-b1.json | qwen38v2-b1 | Qwen3.8-27B-nvfp4 | 1 | 14.881 | 67.2 | 5.9402 | 0.9 % | 67.8 |
| palier2-qwen38-17-09/acvram2-rondes-b12-p1.json | qwen38v2-b12-p1 | Qwen3.8-27B-nvfp4 | 12 | 123.874 | 96.87 | 4.1151 | 0.1 % | 97.0 |
| palier2-qwen38-17-09/acvram2-rondes-b12-p2.json | qwen38v2-b12-p2 | Qwen3.8-27B-nvfp4 | 12 | 124.075 | 96.72 | 4.1245 | 0.1 % | 96.8 |
| f-final-17-09/rondes-b1-F12-p1.json | coder-F12-b1-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 1 | 3.489 | 286.64 | 1.1858 | 3.9 % | 298.2 |
| f-final-17-09/rondes-b1-F12-p2.json | coder-F12-b1-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 1 | 3.478 | 287.49 | 1.1851 | 3.9 % | 299.1 |
| palier2-17-09/kimi-linear-35b/acvram-rondes-b1.json | kimi-linear-35b-b1 | Kimi-Linear-35B-kda-nvfp4 | 1 | 4.579 | 218.4 | 1.3968 | 3.0 % | 225.1 |
| palier2-17-09/kimi-linear-35b/acvram-rondes-b12-p1.json | kimi-linear-35b-b12-p1 | Kimi-Linear-35B-kda-nvfp4 | 12 | 23.076 | 520.01 | 0.689 | 0.6 % | 523.1 |
| palier2-17-09/kimi-linear-35b/acvram-rondes-b12-p2.json | kimi-linear-35b-b12-p2 | Kimi-Linear-35B-kda-nvfp4 | 12 | 23.091 | 519.68 | 0.6906 | 0.6 % | 522.7 |
| palier2-17-09/gemma-4-26b-a4b/acvram-rondes-b1.json | gemma-4-26b-a4b-b1 | gemma-4-26B-A4B-heretic-APEX-I-Quality | 1 | 4.383 | 228.18 | 1.3959 | 3.1 % | 235.5 |
| palier2-17-09/nemotron-3.5-30b-a3b/acvram-rondes-b1.json | nemotron-3.5-30b-a3b-b1 | Nemotron-3.5-Lightning-30B-A3B-srcexl3 | 1 | 3.759 | 266.05 | 1.1713 | 3.6 % | 276.0 |
| palier2-17-09/nemotron-3.5-30b-a3b/acvram-rondes-b12-p1.json | nemotron-3.5-30b-a3b-b12-p1 | Nemotron-3.5-Lightning-30B-A3B-srcexl3 | 12 | 26.264 | 456.9 | 0.7599 | 0.5 % | 459.3 |
| palier2-17-09/nemotron-3.5-30b-a3b/acvram-rondes-b12-p2.json | nemotron-3.5-30b-a3b-b12-p2 | Nemotron-3.5-Lightning-30B-A3B-srcexl3 | 12 | 26.217 | 457.72 | 0.7629 | 0.5 % | 460.1 |
| palier2-17-09/qwen38-calibA/rondes-b1.json | calibA-b1 | Qwen3.8-27B-nvfp4-calibA | 1 | 15.671 | 63.81 | 6.2486 | 0.9 % | 64.4 |
| palier2-17-09/qwen38-calibA/rondes-b1-cachevide.json | calibA-b1-cachevide | Qwen3.8-27B-nvfp4-calibA | 1 | 15.662 | 63.85 | 6.226 | 0.9 % | 64.4 |
| fla-17-09/fla-b12.json | calibA-fla-b12 | Qwen3.8-27B-nvfp4-calibA | 12 | 93.753 | 128.0 | 3.1242 | 0.1 % | 128.2 |
| fla-17-09/fla-b1.json | calibA-fla-b1 | Qwen3.8-27B-nvfp4-calibA | 1 | 14.309 | 69.89 | 5.7185 | 0.9 % | 70.6 |
| fla-17-09/torch-b12.json | calibA-torch-b12 | Qwen3.8-27B-nvfp4-calibA | 12 | 127.76 | 93.93 | 4.1756 | 0.1 % | 94.0 |
| fla-17-09/torch-b1.json | calibA-torch-b1 | Qwen3.8-27B-nvfp4-calibA | 1 | 15.708 | 63.66 | 6.2686 | 0.9 % | 64.2 |
| nemotron-srcbf16-17-09/rondes-b1.json | nemo-srcbf16-b1 | Nemotron-3.5-Lightning-30B-A3B-nvfp4 | 1 | 3.708 | 269.72 | 1.1698 | 3.7 % | 280.0 |
| nemotron-srcbf16-17-09/rondes-b12.json | nemo-srcbf16-b12 | Nemotron-3.5-Lightning-30B-A3B-nvfp4 | 12 | 17.397 | 689.76 | 0.5719 | 0.8 % | 695.2 |
| fla-17-09/5-narrow-calibA-b12.json | narrow-calibA-b12 | Qwen3.8-27B-nvfp4-calibA | 12 | 233.994 | 51.28 | 6.1246 | 0.1 % | 51.3 |
| fla-17-09/5-narrow-v2-b12.json | narrow-v2-b12 | Qwen3.8-27B-nvfp4 | 12 | 139.673 | 85.92 | 4.3922 | 0.1 % | 86.0 |
| gemm-dense-17-09/triton-b12-p1.json | dense-triton-b12-p1 | Qwen3.8-27B-nvfp4-calibA | 12 | 34.448 | 348.36 | 1.1456 | 0.4 % | 349.7 |
| gemm-dense-17-09/triton-b12-p2.json | dense-triton-b12-p2 | Qwen3.8-27B-nvfp4-calibA | 12 | 34.307 | 349.78 | 1.1432 | 0.4 % | 351.2 |
| gemm-dense-17-09/triton-b1.json | dense-triton-b1 | Qwen3.8-27B-nvfp4-calibA | 1 | 14.303 | 69.92 | 5.6961 | 0.9 % | 70.6 |
| nemotron-officiel-17-09/rondes-b1.json | nemo-officiel-b1 | Nemotron-3.5-Lightning-30B-A3B-nvfp4-p | 1 | 4.786 | 208.96 | 1.6244 | 2.8 % | 215.0 |
| nemotron-officiel-17-09/rondes-b12.json | nemo-officiel-b12 | Nemotron-3.5-Lightning-30B-A3B-nvfp4-p | 12 | 15.021 | 798.88 | 0.4957 | 0.9 % | 806.2 |
| gemm-dense-17-09/tete-b12-p1.json | tete-b12-p1 | Qwen3.8-27B-nvfp4-calibA | 12 | 30.586 | 392.34 | 1.0185 | 0.4 % | 394.1 |
| gemm-dense-17-09/tete-b12-p2.json | tete-b12-p2 | Qwen3.8-27B-nvfp4-calibA | 12 | 30.668 | 391.29 | 1.0216 | 0.4 % | 393.0 |
| gemm-dense-17-09/tete-b1.json | tete-b1 | Qwen3.8-27B-nvfp4-calibA | 1 | 14.286 | 70.0 | 5.7186 | 0.9 % | 70.7 |
| gemm-dense-17-09/fusion-b12-p1.json | fusion-b12-p1 | Qwen3.8-27B-nvfp4-calibA | 12 | 29.265 | 410.04 | 0.9667 | 0.5 % | 411.9 |
| gemm-dense-17-09/fusion-b12-p2.json | fusion-b12-p2 | Qwen3.8-27B-nvfp4-calibA | 12 | 29.023 | 413.47 | 0.9663 | 0.5 % | 415.4 |
| gemm-dense-17-09/fusion-b1.json | fusion-b1 | Qwen3.8-27B-nvfp4-calibA | 1 | 14.262 | 70.12 | 5.7018 | 1.0 % | 70.8 |
| nemotron-calibA-17-09/rondes-b1.json | nemo-calibA-b1 | Nemotron-3.5-Lightning-30B-A3B-nvfp4-c | 1 | 4.725 | 211.64 | 1.6167 | 2.9 % | 217.9 |
| nemotron-calibA-17-09/rondes-b12.json | nemo-calibA-b12 | Nemotron-3.5-Lightning-30B-A3B-nvfp4-c | 12 | 12.451 | 963.76 | 0.4051 | 1.1 % | 974.4 |
| gemv-experts-rpw-18-09/abab-cert-rpw1-p1.json | rpw1-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 11.579 | 1036.36 | 0.3846 | 1.2 % | 1048.6 |
| gemv-experts-rpw-18-09/abab-cert-rpw4-p2.json | rpw4-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.316 | 1163.2 | 0.343 | 1.3 % | 1178.7 |
| gemv-experts-rpw-18-09/abab-cert-rpw1-p3.json | rpw1-p3 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 11.622 | 1032.51 | 0.3854 | 1.2 % | 1044.7 |
| gemv-experts-rpw-18-09/abab-cert-rpw4-p4.json | rpw4-p4 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.346 | 1159.87 | 0.3441 | 1.3 % | 1175.3 |
| xreg-down-18-09/abab-cert-xreg0-p1.json | xreg0-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.518 | 1140.91 | 0.3508 | 1.3 % | 1155.8 |
| xreg-down-18-09/abab-cert-xregdown-p2.json | xregdown-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.034 | 1195.98 | 0.3341 | 1.4 % | 1212.4 |
| xreg-down-18-09/abab-cert-xreg0-p3.json | xreg0-p3 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.582 | 1133.99 | 0.352 | 1.3 % | 1148.7 |
| xreg-down-18-09/abab-cert-xregdown-p4.json | xregdown-p4 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.082 | 1190.27 | 0.3363 | 1.3 % | 1206.5 |
| coder-defaut-18-09/cert-p1.json | defaut-b12-p1 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.004 | 1199.57 | 0.3332 | 1.4 % | 1216.0 |
| coder-defaut-18-09/cert-p2.json | defaut-b12-p2 | Qwen3-Coder-30B-A3B-nvfp4 | 12 | 10.035 | 1195.85 | 0.334 | 1.4 % | 1212.2 |
| qwen38-defaut-18-09/cert.json | qwen38-defaut-b12 | Qwen3.8-27B-nvfp4-calibA | 12 | 29.372 | 408.55 | 0.9776 | 0.5 % | 410.4 |

## Cellules où le biais dépasse 2 % (48 passes, toutes à pas < 6,8 ms : b=1 et b=2 des modèles rapides)
| passe | bras | b | pas ms | t/s | biais |
|---|---|---|---|---|---|
| ties-moe-15-09/certifie-b1-moyen-20s-A1.json | b1-A1 |  | 4.483 | 223.04 | 3.0 % |
| ties-moe-15-09/certifie-b1-moyen-20s-A2.json | b1-A2 |  | 4.483 | 223.04 | 3.0 % |
| courbe-lot-15-09/certifie-b2-moyen-20s-A1.json | b2-A1 |  | 6.165 | 324.39 | 2.2 % |
| courbe-lot-15-09/certifie-b2-moyen-20s-A2.json | b2-A2 |  | 6.166 | 324.37 | 2.2 % |
| 1aj-decodage-15-09/certifie-b1-20s-A1.json | 1aj-b1-A1 |  | 4.485 | 222.97 | 3.0 % |
| 1aj-decodage-15-09/certifie-b1-20s-B1.json | 1aj-b1-B1 |  | 4.297 | 232.71 | 3.2 % |
| 1aj-decodage-15-09/certifie-b1-20s-A2.json | 1aj-b1-A2 |  | 4.483 | 223.07 | 3.0 % |
| 1aj-decodage-15-09/certifie-b1-20s-B2.json | 1aj-b1-B2 |  | 4.297 | 232.71 | 3.2 % |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-A1.json | bis-c652947-A1 |  | 4.297 | 232.71 | 3.2 % |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-B1.json | bis-0c90017-B1 |  | 4.484 | 223.03 | 3.0 % |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-A2.json | bis-c652947-A2 |  | 4.297 | 232.73 | 3.2 % |
| gemvmax-bissect-15-09/bis-b1-c652947-0c90017-B2.json | bis-0c90017-B2 |  | 4.484 | 223.01 | 3.0 % |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-A1.json | bis-97f2313-A1 |  | 4.297 | 232.72 | 3.2 % |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-B1.json | bis-7f3f422-B1 |  | 4.484 | 223.02 | 3.0 % |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-A2.json | bis-97f2313-A2 |  | 4.297 | 232.71 | 3.2 % |
| gemvmax-bissect-15-09/bis-b1-97f2313-7f3f422-B2.json | bis-7f3f422-B2 |  | 4.483 | 223.07 | 3.0 % |
| v064-b1-15-09/certifie-b1-A1.json | v064-b1-A1 |  | 4.296 | 232.76 | 3.2 % |
| v064-b1-15-09/certifie-b1-A2.json | v064-b1-A2 |  | 4.297 | 232.72 | 3.2 % |
| v064-b1-15-09/certifie-b1-A3.json | v064-b1-A3 |  | 4.297 | 232.75 | 3.2 % |
| v064-b1-15-09/certifie-b1-A4.json | v064-b1-A4 |  | 4.296 | 232.76 | 3.2 % |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-A1.json | b1-A1 |  | 4.298 | 232.68 | 3.2 % |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-B1.json | b1-B1 |  | 5.042 | 198.32 | 2.7 % |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-A2.json | b1-A2 |  | 4.297 | 232.72 | 3.2 % |
| courbe-lot-rp-15-09/certifie-b1-moyen-20s-B2.json | b1-B2 |  | 5.043 | 198.28 | 2.7 % |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-A1.json | b2-A1 |  | 6.165 | 324.41 | 2.2 % |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-B1.json | b2-B1 |  | 6.728 | 297.28 | 2.0 % |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-A2.json | b2-A2 |  | 6.165 | 324.43 | 2.2 % |
| courbe-lot-rp-15-09/certifie-b2-moyen-20s-B2.json | b2-B2 |  | 6.726 | 297.34 | 2.0 % |
| narrow-15-09/b1-A1.json | narrow-b1-A1 |  | 4.296 | 232.75 | 3.2 % |
| narrow-15-09/b1-B1.json | narrow-b1-B1 |  | 4.484 | 223.03 | 3.0 % |
| narrow-15-09/b1-A2.json | narrow-b1-A2 |  | 4.297 | 232.75 | 3.2 % |
| narrow-15-09/b1-B2.json | narrow-b1-B2 |  | 4.297 | 232.73 | 3.2 % |
| narrow-15-09/b1-B3.json | narrow-b1-B3 |  | 4.297 | 232.72 | 3.2 % |
| narrow-15-09/b1-A3.json | narrow-b1-A3 |  | 4.297 | 232.7 | 3.2 % |
| narrow-15-09/b1-B4.json | narrow-b1-B4 |  | 4.296 | 232.76 | 3.2 % |
| coder-acvram-w4a16-17-09/rondes-b1-p1.json | coder-w4a16-b1-p1 | 1 | 4.283 | 233.45 | 3.2 % |
| coder-acvram-w4a16-17-09/rondes-b1-p2.json | coder-w4a16-b1-p2 | 1 | 4.267 | 234.35 | 3.2 % |
| coder-ec-17-09/rondes-b1.json | coder-ec-b1 | 1 | 3.879 | 257.77 | 3.5 % |
| f-final-17-09/rondes-b1-p1.json | coder-F-b1-p1 | 1 | 3.465 | 288.59 | 3.9 % |
| f-final-17-09/rondes-b1-p2.json | coder-F-b1-p2 | 1 | 3.476 | 287.72 | 3.9 % |
| f-final-17-09/rondes-b1-F12-p1.json | coder-F12-b1-p1 | 1 | 3.489 | 286.64 | 3.9 % |
| f-final-17-09/rondes-b1-F12-p2.json | coder-F12-b1-p2 | 1 | 3.478 | 287.49 | 3.9 % |
| palier2-17-09/kimi-linear-35b/acvram-rondes-b1.json | kimi-linear-35b-b1 | 1 | 4.579 | 218.4 | 3.0 % |
| palier2-17-09/gemma-4-26b-a4b/acvram-rondes-b1.json | gemma-4-26b-a4b-b1 | 1 | 4.383 | 228.18 | 3.1 % |
| palier2-17-09/nemotron-3.5-30b-a3b/acvram-rondes-b1.json | nemotron-3.5-30b-a3b-b1 | 1 | 3.759 | 266.05 | 3.6 % |
| nemotron-srcbf16-17-09/rondes-b1.json | nemo-srcbf16-b1 | 1 | 3.708 | 269.72 | 3.7 % |
| nemotron-officiel-17-09/rondes-b1.json | nemo-officiel-b1 | 1 | 4.786 | 208.96 | 2.8 % |
| nemotron-calibA-17-09/rondes-b1.json | nemo-calibA-b1 | 1 | 4.725 | 211.64 | 2.9 % |

## Verdicts qui citent certifie (34) : verdict-1aj-decodage-15-09, verdict-b1-mma-repos-serve-15-09, verdict-bloc-sage11-17-09, verdict-budget-exil-prefill-17-09, verdict-budget-kv-chargeur-17-09, verdict-coder-acvram-w4a16-17-09, verdict-coder-b12-defaut-18-09, verdict-coder-c-mixte-17-09, verdict-coder-ec-remesure-17-09, verdict-courbe-lot-mma-15-09, verdict-courbe-lot-rp-15-09, verdict-duel-glm-prise-a-16-09, verdict-duel-glm-prise-b-16-09, verdict-f-cloture-17-09, verdict-f-final-17-09, verdict-fla-17-09, verdict-gemm-dense-fusion-17-09, verdict-gemm-dense-palier1-situ-17-09, verdict-gemm-dense-tete-17-09, verdict-gemv-experts-rpw-situ-18-09, verdict-gemv-experts-xreg-down-18-09, verdict-glm-b12-eager-16-09, verdict-glm-b12-plan-17-09, verdict-glm-slots12-erratum-16-09, verdict-mla-seuil-creneaux-16-09, verdict-mla1-16-09, verdict-mla2-16-09, verdict-narrow-mla-tete-16-09, verdict-nemotron-calibA-17-09, verdict-nemotron-officiel-17-09, verdict-nemotron-srcbf16-17-09, verdict-palier2-kimi-linear-17-09, verdict-palier2-llama-70b-final-17-09, verdict-palier2-qwen38-27b-17-09, verdict-post-correctif-b12-16-09, verdict-poste-d-17-09, verdict-ppl-hadamard-17-09, verdict-profil-coder-pas-17-09, verdict-profil-prefill-b1-18-09, verdict-qwen38-calibA-17-09, verdict-qwen38-v2-mesure-17-09, verdict-reprise-gui10-qwen38-18-09, verdict-retampon-cellule-narrow-16-09, verdict-ties-moe-decode-15-09, verdict-v064-b1-15-09

## Bras concurrents b=1 (question poste7, 18/09 13h) : harnais différent, sans smi dans la fenêtre
llama.cpp 341,4 (`banc-llamacpp-16-09.py`, llama-server + SSE + energie.py), EXL3 / vLLM (`banc-4moteurs.py`, serveurs HTTP + energie.py) : aucun `nvidia-smi` dans la fenêtre (banc-4moteurs.py:445-450 a retiré l'ancienne classe Watt ; `_temperature()` lu avant/après seulement, :669/:778). Seul acvram b=1 rondes (287,1, certifie) porte le biais 0,135 ms/pas ; les concurrents portent le coût HTTP/SSE. À instrument égal l'écart b=1 Coder acvram/llama.cpp ≈ −13 % au lieu de −16 % : ligne datée à ajouter au comparatif après l'ABAB (protocole-certifie-smi-abab-18-09). Bras B `CERT_SANS_SMI=1` = exemption datée jusqu'au commit NVML d'poste1.
