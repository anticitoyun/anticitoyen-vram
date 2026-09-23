# Protocole — Nemotron-3.5-Lightning-30B-A3B acvram srcbf16 (Manon 8c9fc54, `models_acvram/Nemotron-3.5-Lightning-30B-A3B-nvfp4`, SNR 20,5 dB) : PPL × bf16 + rondes, sous FLA (Mamba2 fla, e28b7b1)

instrument : `ppl-acvram-17-09.py` 3 tranches `tranches-glm` × bf16 géo 13,416 (`palier2-17-09/nemotron-3.5-30b-a3b/bf16-ppl-tr*`), `certifie-b12` b=1 ×1 / b=12 ×1 ; régime classé ; chaîne `scratchpad/nemotron-srcbf16-17-09/chaine.sh` (en file après les profils GDN).
en face : srcexl3 double quantifié 1,0795 (b=1 266,1, b=12 457, prefill refusé) ; vLLM NVFP4 officiel 0,987 (b=1 401, b=12 1 799).
scellé : PPL ∈ [1,02 ; 1,05] (la double quantification EXL3 → NVFP4 vaut la moitié de l'écart : 1,0795 → ≈ 1,04 ; classé si ≤ 1,02 : improbable sans calibration sur un MoE) ; b=12 ≥ 457 et b=1 ≥ 266 (Mamba2 fla ne doit pas ralentir) ; falsification : PPL ≥ 1,07 ⇒ la source EXL3 n'était pas la cause ; b=12 < 400 ⇒ la voie fla Mamba2 coûte.
