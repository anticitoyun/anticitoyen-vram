# Protocole — Qwen3.8-27B NVFP4 calibré GDN (Manon d391d45, `models_acvram/Qwen3.8-27B-nvfp4-calibA`, 496/866 tenseurs calibrés, SNR 20,5 → 22,8 dB) : PPL privée × bf16 + rondes b=1

instrument : même montage que `verdict-qwen38-v2-mesure-17-09` — `ppl-acvram-17-09.py` 3 tranches `corpus-prive/tranches-glm`, × bf16 géo 6,5344 ; `certifie-b12-15-09.py` b=1 ×1 ; chaîne `scratchpad/palier2-17-09/qwen38-calibA/chaine.sh`, arbre laure 6d18ad2 (= main ≥ ee69a7b, ppl-decode-kv 5,5426 à la fusion, témoin 5,544).
régime : classé (`ACVRAM_MOE_MMA=0 ACVRAM_MOE_DECODE_MMA=0 ACVRAM_NARROW_GEMM=1`), défauts 0.6.9, [gdn] torch de référence.
scellé (Sage) : géo ≤ 1,020 → classée ; 1,020-1,025 → notée ; > 1,025 → chantier calibration GDN clos. Repères : srcexl3 1,0282, srcbf16 non calibré 1,0306, Q4_K_M 4,92 bpw 1,0118, vLLM 1,0271.
falsification : une PPL ≥ 1,0306 dit que la calibration n'a rien changé (les 496 tenseurs calibrés ne portent pas la perte) ; b=1 hors 67,2 ± 5 % dit que la calibration a changé le chemin de calcul.
