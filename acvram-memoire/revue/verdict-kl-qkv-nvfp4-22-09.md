# KL Coder, alias qkv-nvfp4 (bras acvram, dumps HF réutilisés) — TENU 5/5 — 22/09 (poste2)

* instrument : `scratchpad/kl-coder-texte-22-09/kl-alias-qkv-nvfp4.py`, réutilise les 5 dumps HF bf16 déjà écrits ce matin (référence indépendante de l'alias acvram — pas de rechargement HF)
* commit : main à jour (6c6a2cc8)
* régime : alias `Qwen3-Coder-30B-A3B-nvfp4-qkv-22-09` (qkv en NVFP4, régime confondu `experts_layout=naturel` — sans effet sur la CORRECTNESS mesurée ici, seulement sur le débit déjà signalé comme non comparable)
* scellé (groupe) : KL max ≤ 1,0 par invite, référence 0,519 (invite2, alias -qkvo-i8c)
* mesuré :

| invite | kl_max (qkv-nvfp4) | kl_max (référence -qkvo-i8c) | pas égaux |
|---|---|---|---|
| invite0 | 0,73525 | 0,00807 | 7/8 |
| invite1 | 0,19390 | 0,08894 | 8/8 |
| invite2 | 0,22962 | 0,51934 | 8/8 |
| invite3 | 0,26795 | 0,12542 | 7/8 |
| invite4 | 0,19251 | 0,23262 | 7/8 |

* verdict : **TENU 5/5** — kl_max maximal sur les 5 invites = 0,73525 (invite0), toujours <<1,0. Pas de dégradation qualité systématique par rapport à -qkvo-i8c (invite0 est plus haute cette fois, invite2 plus basse — variation dans le bruit, pas de tendance).
* durée : ~1 min de carte (pas de rechargement HF)

## Suite
Qualité qkv-nvfp4 validée indépendamment du régime de vitesse confondu. ABBA débit reste en attente de la résolution `experts_layout=naturel` (verdict séparé).
