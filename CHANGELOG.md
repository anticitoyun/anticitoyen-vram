# Journal des changements

* **24/09/2026 — pièce 156** : les linéaires NVFP4 des modèles DENSES sont servis par défaut en disposition Marlin unique
  (GEMV v2, TPB par forme) : +57 à +90 % de débit à b = 8, b = 1 inchangé (0,979 à 0,996), sortie qualifiée (KL sous 2 ×
  témoin, PPL identique), +27 à 33 ms de TTFT (en réduction, pièce 147) ; MoE inchangés ; repli `ACVRAM_PROJ_MARLIN=0`.
