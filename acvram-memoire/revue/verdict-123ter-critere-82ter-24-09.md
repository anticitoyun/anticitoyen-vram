# Verdict — 123-bis et 123-ter : correctif du pas de ligne, critère 82 ter (choix (i) du chef) — 24/09 00 h 5x (poste1)

* **instrument** : `scratchpad/poste1-123-24-09/prise-123bis.sh` (tests + bras cassant a) ; `prise-123ter.sh` : `verif-123ter.py` (crochet sur chaque MoEBlock : sortie servie contre témoin GEMV recalculé SUR LA MÊME ENTRÉE, `_MOE_TENSOR=False`) + `kl-123.py` témoin
* **commit** : 123-bis 622a5b0b ; 123-ter 2008050d (poste1-mtp ; correctif 4960574f)
* **régime** : S1b `Qwen3-Coder-30B-A3B-assemble-S1b-proj-tete-i8-23-09`, horloge libre (justesse) ; diagnostic sur `…-nvfp4-qkvo-i8c` (sans AWQ)
* **scellé** : `scratchpad/poste1-123-24-09/scelle-123ter.md` (écrit avant, commit 2008050d) — (1) T = 1 au bit, (2) T ≥ 8 : max|Δ| ≤ 2⁻⁷·max|témoin| par appel, (3)(4) KL b=1 / b=12 ≤ témoin + 0,025 par invite, (5) aligneur@T8+ et gemv@T1 atteints
* **mesuré** :
  * 123-bis : extension recompilée (moe_aligner_petit_xs présent), **28 passed** (awq_123 + glue_fusee + defaut + decodage) ; bras cassant (a) (fantômes → expert 1, cache séparé) : **ROUGE** 4/4 (« xs ≠ torch sur 12 220 … éléments »).
  * 123-ter S1b : (1) T = 1 au bit **1 680/1 680** ; (2) pire rapport **0,0105** au préfill b=1 (T = 30/32) et **0,0151** au décodage b=12 (T = 12, 0/1 680 au bit) — seuil 0,0078 ; (3) ΔKL b=1 max **+0,0217** ; (4) ΔKL b=12 max **+0,0039** ; (5) aligneur@T8+, gemv@T1 présents.
  * Diagnostic i8c (même crochet, aucune table AWQ, chemin tensor servi par défaut depuis la 82 ter) : T = 1 au bit 1 680/1 680 ; T ≥ 8 : **0,0075** (b=1, préfill) et **0,0122** (b=12, décodage).
* **verdict** :
  * 123-bis : **TENUE** — le correctif du pas de ligne est juste, et le test (a) casse bien si la faute revient.
  * 123-ter : **FAUX au critère (2)** tel qu'écrit, (1)(3)(4)(5) tenus. Réfuté reste réfuté.
  * Lecture (non jugée, pour le chef) : le chemin tensor DÉJÀ servi sur i8c dépasse le même critère à b=12 (0,0122). Le 2⁻⁷ de la 82 ter était un seuil de BANC (piles synthétiques), pas d'activations réelles de 48 couches : l'instrument ne sépare pas « tensor contre GEMV » de « tensor+AWQ contre GEMV ». L'AWQ ajoute un facteur × 1,24 (b=12) à × 1,39 (b=1), attendu de l'arrondi bf16 de xs (le GEMV divise en fp32 dans le noyau, dossier 123 § 3).
* **durée** : 123-bis prévue ≤ 15 min, 00:38:13 → 00:43:52 ; 123-ter prévue ≤ 15 min, → 00:47:34 ; diagnostic i8c 00:48:47 → 00:49:57 ; compute-apps début = fin (llama-server 4627 seul) à chaque prise

Décision au chef (deux options) : (a) critère relatif à écrire AVANT une nouvelle prise, du type « rapport S1b ≤ 1,5 × rapport i8c, même prise » — un seuil posé APRÈS avoir vu 1,24-1,39 se négocie : il ne vaudrait que sur des invites neuves ; ou (b) la 123 reste hors défaut, et le GEMV sert S1b.
