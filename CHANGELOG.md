# Journal des changements

* **24/09/2026 — pièce 156** : les linéaires NVFP4 des modèles DENSES sont servis par défaut en disposition Marlin unique
  (GEMV v2, TPB par forme) : +57 à +90 % de débit à b = 8, b = 1 inchangé (0,979 à 0,996), sortie qualifiée (KL sous 2 ×
  témoin, PPL identique), TTFT +2 à +4 ms (B/A 1,001 à 1,012 sur gemma4 31B et Qwen3.8-27B, invites de 512, 2 048 et
  4 096 jetons : revue/poste6-piece147-verdict-24-09.md, dépaquetage CUDA de la pièce 147) ; MoE inchangés ; repli
  `ACVRAM_PROJ_MARLIN=0`.
