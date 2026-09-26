# Pièce 142 — inventaire à sec : alias du menu que `ACVRAM_PROJ_MARLIN=1` ferait passer par la disposition Marlin — 24/09 07 h (poste1)

**Menu** : `~/TSV/acvram-chemins.tsv` (161 lignes, alias → dossier → contexte), lu par `acvram-serveur:9,17-20`.
**Dispatch** : `kernels/__init__.py:1042` `preparer_disposition_marlin` (appelée par `engine/loader.py:934` sous
`ACVRAM_PROJ_MARLIN=1`) prend TOUT poids nvfp4 2-D hors `MoEBlock` (`:1058`) avec N ≥ 2 048, K et N multiples de 64,
K ≥ 1 024 (`:1098`) ; au calcul, `:581` (vues) et `:600` (poids en disposition seule). Critère appliqué ici aux
manifestes réels du menu, clés `.experts.` et `mtp.` exclues (script : sortie `scratchpad/poste1-p142-24-09/inventaire-menu.txt`).

## Résultat
* **57 alias DENSES** portent des poids éligibles (0,1 à 17,6 Go chacun), en ~10 familles d'architecture/forme :
  qwen3_next/qwen3_5 27B (16 alias, **Qwen3.8-27B = déjà mesuré**) ; gemma4 31B (5) et 12B (2) ; qwen2/qwen3 32B (2) ;
  mistral/llama 24B (6, dont Skyfall 31B) ; granite 30B, muse-glimmer 30B ; 14B (phi-4, Qwen2.5-Coder-14B, Qwen3-14B) ;
  Nemo 12B (3) ; petits ≤ 9B (≈ 15, dont hybrides falcon_h1, nemotron_h).
* **32 alias MoE sont AUSSI touchés** (leurs linéaires hors experts : attention, GDN, etc., 0,01 à 1,0 Go) :
  nemotron_h 30B (7), qwen3_next 35B-A3B (8), gemma4 26B-A4B (2), qwen3_moe 30B-A3B (6), deepseek_v2/GLM (4), autres.
  La décision « PROJ_MARLIN=1 par défaut pour les denses » ne peut donc pas être une variable globale en l'état : elle
  s'appliquerait aussi à ces 32 alias (ou il faut une garde « modèle dense » dans la passe).
* 3 alias sans poids éligible (Coder nvfp4 servis, qkvo-i8c, Agents bf16), 1 dossier absent (gemma-4-12B-it-bf16).

## Plan proposé (au chef — 57 ABBA = ~28 h de carte)
Un REPRÉSENTANT par famille (même `model_type`, mêmes formes, même code) : ~8 prises denses (gemma4 31B, qwen2 32B,
mistral 24B, granite 30B, phi-4, Nemo 12B, un ≤ 9B hybride, Qwen3-14B), chacune : scellé → ABBA b=1/b=8 ≥ 5 lots,
-lgc 2700, J/jeton, KL contre témoins (critère 134), PPL sur les fenêtres de la 102 ; + 2 représentants MoE si la
variable reste globale. Prédiction commune à sceller par famille : b=8 gain ∝ part des GEMM denses dans le pas.
