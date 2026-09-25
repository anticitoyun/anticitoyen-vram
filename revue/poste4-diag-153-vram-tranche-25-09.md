# Diagnostic 153 — VRAM eager + tranchage PPL (25/09, à sec, branche poste4-153b post-purge)

instrument : lecture de code seule (`acvram/engine/loader.py`, `acvram/evaluate.py`), aucune prise carte
commit : 0c54811f (origin/main post-purge)
régime : —
scellé : scratchpad/poste4-piece153-attn-gdn-int8-mlp-nvfp4-24-09/, revue/poste4-piece153-scelle-25-09.md
mesuré : —
verdict : deux causes distinctes, aucune n'est un bogue du planificateur
durée : 12 min

## 1. Repli eager (580 Mio libres après chargement)

Pas un défaut de `_replanifier` (KV déjà dimensionné pour 1 séquence depuis le correctif du 22/09,
confirmé par `ppl.json` : « budget : 1 séquence × 4112 jetons »). Cause réelle : le format mixte
promeut 308 tenseurs attention/GDN de nvfp4 (4 bit) à int8 (8 bit) — poids logiques 6,8 Gio int8 +
9,8 Gio nvfp4 + 3,3 Gio bf16 ≈ 19,9 Gio, CONTRE ~13 Gio pour un nvfp4 pur de même taille (scellé 102).
La promotion coûte réellement ses octets (2× les tenseurs promus), pas seulement en théorie SNR/PPL —
c'est cette marge perdue qui manque aux 1024 Mio exigés par la capture de graphes. Rien à corriger côté
loader ; c'est le coût attendu de `--attn-qkvo-int8-canal`/`--gdn-int8-canal`, à chiffrer dans le scellé
suivant (Gio supplémentaires par tenseur promu, pas juste le SNR).

## 2. Tranchage PPL : NON contourné (hypothèse du carnet réfutée à la lecture)

`_pertes_par_tranches` (evaluate.py:167-182) tranche bien AVANT `_tete`/`_logits_finaux` — `h` est
rendu une fois par `model(batch, return_hidden=True)` (états cachés normalisés, pas les logits), chaque
tranche de `_PPL_TRANCHE` (256 par défaut) applique la tête séparément (commentaire ligne ~145,
confirmé par le motif déjà couvert par `tests/test_ppl_tranches.py`, écrit pour un OOM antérieur du
même genre sur un i8c). Un tranche de 256 = 256×151936×4o ≈ 155 Mio, pas 2,49 Gio. L'allocation qui
échoue (2,49 puis 4,74 Gio dans `ppl-153.err`) n'est pas expliquée par le tranchage — probablement le
même déficit de marge que le point 1 (mixte int8/nvfp4/bf16 laisse ~0,4-0,6 Gio libres avant même
l'évaluation) plutôt qu'un bogue de `_pertes_par_tranches`.

## Suite proposée

Le format mixte lui-même est plus lourd, pas le planificateur. Options avant nouvelle mesure : (a)
réduire `--max-model-len`/window pour l'éval PPL sur ce format précis, (b) `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` (suggéré par le message CUDA), (c) mesurer sur une carte à plus de marge, ou
(d) accepter que ce format mixte n'a pas assez de VRAM libre pour CUDA-graphes/PPL fenêtre pleine sur
cette carte et l'écrire comme contrainte du format, pas comme régression à corriger.
