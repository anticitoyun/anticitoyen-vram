# banc-sampler (poste1), isolé du pipeline — 665eeacc — 21/09

* instrument : `outils/gpu/mesure/banc-sampler.py` (fichier suivi)
* commit : 665eeacc, CUDA_VISIBLE_DEVICES=0
* régime : lot 12, vocab 151936
* scellé : H1/H2/H3 selon le script (comparaison lent vs vectorisé, argmax seul, cast fp32)
* mesuré : `lent_glouton_bf16` 55,1 µs vs `lot_glouton_bf16` (vectorisé) 21,6 µs, delta −33,5 µs (vectorisé plus rapide isolément)
* verdict : **H2** — le sampler vectorisé lui-même n'est pas la cause de la perte de 2,2 % mesurée en cellule complète (b=12 6f-v2, `verdict-sampler-b12-6f-v2-21-09.md`) ; isolé, il est plus rapide ou égal. La perte vient de l'interaction avec le reste du pipeline (message du script), à instrumenter au nsys.
* durée : 3 s

nvidia-smi propre avant/après.
