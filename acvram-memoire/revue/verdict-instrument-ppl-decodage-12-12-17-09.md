# Verdict — instrument PPL décodage à 12/12 séquences (poste7 § 12.3) : cause de la troncature = allocateur par défaut de l'Engine sur les modèles MLA

instrument : `scratchpad/ppl-decode-prefixe-17-09.py` (252aab7 + ce commit) — journal `scratchpad/bloc-sage11-17-09/ppl-decode-tr0-12sur12.log`
commit : arbre poste3 780a067 (main 3082bda fusionné)
régime : GLM `-k48`, W4A16, SLOTS=12, préfixe `encode_brut`, tranche 0 privée, 12 séquences × 2 047 cibles ; une seule prise de validation d'instrument (la PPL obtenue n'est pas publiée : le chemin décodage attend le correctif de poste4)
scellé : 12/12 séquences complètes, 24 564 cibles, aucun exil hôte
mesuré : **12/12, 24 564 cibles, paramètres tous sur cuda:0** ; avant correctif : 8/12 (1 358 / 1 486 / 1 630 / 1 806 / 2 046 jetons pour les tronquées), 22 655 cibles
verdict : **la troncature ne venait pas du plan (kv_max_tokens 25 344, blocs réels 20 616) mais de l'Engine : `runner.py:348` prend `min(num_blocks des caches paginés, default=1024)` et GLM n'a AUCUN cache paginé (`model.caches` vide : le MLA garde un latent contigu par séquence, `config.py:272-280`) → allocateur de 1 024 blocs = 16 384 jetons quel que soit le budget, `_grow` refuse au-delà et finit les séquences « length » en silence (`runner.py:1013`). 12 × 1 365 = 16 384 : la première troncature à 1 358 le dit.** L'instrument redimensionne l'allocateur au besoin réel (1 636 blocs) quand `model.caches` est vide — l'allocateur ne gouverne que la décision sur ces modèles, le latent est par séquence (`mla.py new_static`, `max_len = max_model_len`).

## Ce que ça touche hors instrument (poste4)
- Le serveur passe par le même `Engine` : sur un modèle MLA, toute charge dont la somme des jetons vivants dépasse 16 384 (12 conversations de 2 048, ou 4 de 4 096) voit des séquences finies « length » avant leur `max_tokens`, sans message — défaut de production à vérifier sur `/v1/chat/completions` et à corriger dans `Engine.__init__` (budget de blocs = plan, pas `default=1024`, pour les modèles sans cache paginé).
- Toutes les PPL « mode décodage » GLM à 12 × 2 048 publiées avant ce jour portaient sur 22 655 cibles au lieu de 24 564 (`ppl-narrow-b12` du 16/09 : 22 664/24 564, noté alors comme « défaut connu » sans cause).
