# poste7 — P1 : porte corrigée à 137 TFLOPS par l'arithmétique de poste4 ; trois bandes nommées, scellé dérivé du micro-banc et non d'un chiffre concurrent ; deux dispositions d'experts gardées (1,7 Go) ; rien sur la carte avant le oui de l'utilisateur (18/09)

Entrée : poste4 65d5edc `note-marlin-classe-a-sec-18-09` — port du template Marlin (vLLM 0.29.0, Apache-2.0, attribution) faisable en 3-5 j ; réécriture 6-10 j non ; notre `NVFP4Tensor` est déjà le format ingéré (E2M1 [N,K/2], E4M3 [N,K/16], global fp32) ; 15 700 j/s ⇒ experts ≤ 54,3 ms ⇒ **137 TFLOPS** (7,43 TFLOP ; B0 = 93) ; à 110 le scellé tombe à 14 240.

## 1. Porte : 137, mais trois bandes — la règle du 17/09 (jamais deux nombres orphelins)

Ma porte 110 venait d'un ratio sur B1' (58) ; poste4 a refait la soustraction sur notre pas, c'est la bonne méthode (REGLES § 4, `poste7-b1-verdict`). Scellé dérivé, écrit avant : **j/s = 2 048 / (0,076 + 7,43 / X)**, X = TFLOPS effectifs du micro-banc (E = 128, T = 16 384, K = 2 048, N = 768, poids lus dans la tuile), 0,076 s = le reste du prefill mesuré (207 − 131 ms). Contrôle de la formule : X = 93 → 9 950 (B0 : 9 913, tenu à 0,4 %).

| micro-banc X | issue | scellé en situ (poste3, 20 min) |
|---|---|---|
| **≥ 137** | P1 ouvert, cible parité | prefill Coder ≥ **15 700 j/s** ; PPL = B0 ± 0,002 |
| **110 ≤ X < 137** | P1 ouvert, gain sous parité, **une seule passe**, chantier clos ensuite | prefill ≥ **0,95 × formule(X)** (X = 120 → ≥ 12 700) ; PPL idem ; publié « −n % vs llama.cpp », pas « parité » |
| **< 110** | P1 fermé sans carte | — |

La bande du milieu existe parce qu'une cellule classée perdue de 37 % qui passe à −10 % vaut d'être prise ; elle n'autorise pas de seconde passe (REGLES : pas de 4e tour). GLM dans la même passe de carte, prédiction formule sur ses propres postes (poste4 l'écrit avant).

## 2. Deux dispositions d'experts : garder les deux, budgétées

Repack MMA pour le prefill + disposition GEMV pour le décodage = +1,7 Go sur Coder : **gardées toutes les deux** à la première itération (Coder 17 Go sur 31,36 : tient). Condition : le `Plan` compte les deux (sinon exil silencieux, `poste7-glm-awq-mla` P6) ; `regime_ligne()` porte `experts_layout=double`. Re-permutation au vol = chantier séparé, seulement si un modèle classé n'entre plus.

## 3. Ce qui attend l'utilisateur (chef, la ligne § 5 de `poste7-reprise-ordre` complétée)

« Semaine CUDA = port du template Marlin de vLLM (Apache-2.0, attribution dans le dépôt), 3-5 j poste4, +1,7 Go VRAM Coder ; porte micro-banc à sec avant toute carte ; cibles : prefill ≥ 15 700 (parité llama.cpp) si le banc rend ≥ 137 TFLOPS, gain publié sous parité entre 110 et 137, fermé sous 110. Oui / non ? » Le micro-banc lui-même est à sec : poste4 peut l'écrire **avant** le oui (c'est le port du noyau, pas la carte) — décision de chef selon la charge ; aucune passe de carte P1 sans le oui.

## Ordre

* poste4 : micro-banc P1 à sec dès maintenant (le port, sans intégration moteur), verdict X + prédiction GLM par la formule ; intégration + tests ± 2⁻⁷ + bras cassant après le oui.
* chef : ligne § 3 à l'utilisateur ; ETAT : porte 137 / bandes § 1.
* poste3 : inchangé (profil (a)+(b) en cours).

## 4. GO utilisateur (18/09) : le micro-banc en situ d'abord — c'est la porte écrite, elle décide de l'intégration et de son scellé

Ordre, un seul bloc de carte poste3 (~40 min, machine calme : load1 < nproc/2 attesté par `energie.py`), dans cet ordre :
1. Porte cuBLASLt a8 (2 min) : `torch._int_mm` 4 formes q/k/v/o Coder 2048, ≤ 16 ms → in situ `a8-cublas` plus tard ; > 16 → P0-a8 fermé.
2. **Micro-banc P1** (20 min) : compilation à sec AVANT le verrou (`--compiler-seulement`, CUDA_VISIBLE_DEVICES=""), puis `outils/carte.sh python outils/banc-marlin-p1-18-09.py` : X TFLOPS effectifs, juge 2⁻⁷ contre déquant fp32, témoin B0 (attendu 90-100), en-tête `marlin_port_so`/`marlin_source`, BANC INVALIDE si JIT dans le processus. Verdict par les trois bandes § 1 : ≥ 137 → intégration, scellé ≥ 15 700 ; 110-137 → intégration, scellé ≥ 0,95 × formule(X), une passe ; < 110 → P1 fermé, la semaine CUDA n'a pas lieu (l'utilisateur en est informé par une ligne, pas par un chantier qui continue).
3. Harnais égal (15 min) : contrôle llama-server b=1 341 ± 3 %, puis acvram serve b=1 (275-295) et b=12 (1 150-1 200 bridé, `ignore_eos` porté, n_jetons = 12 × pas).
Puis poste4 (3-5 j, à sec sauf passes nommées) si X ≥ 110 : intégration prefill (`preparer_pile` au chargement, double disposition comptée par le `Plan`, `experts_layout=double` dans `regime_ligne()`, régime `ACVRAM_PREFILL_GROUPED=marlin` témoin `groupe`), test d'équivalence ± 2⁻⁷ contre B0 + bras cassant (échelle décalée d'un rang) dans le même commit, PPL prefill = B0 ± 0,002 ; poste3 : une passe de carte in situ, scellé de la bande. GLM dans la même passe, prédiction formule sur ses postes écrite par poste4 avant.
