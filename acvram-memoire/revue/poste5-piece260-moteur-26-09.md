# Pièce 260 — étape moteur (préfill par lot, KL 2b, PPL) : RÉSULTAT (poste5 26/09 06 h 4x)

Prise `poste5-p260-moteur` sur aab55ffb5 (0.7.0 + 260), 06:39:17-06:41:50, carte 0 sans PID hors prise avant/après.
Bascule à chaud : 305 tenseurs marqués (233 + piles). Brut : `eng260.{json,txt}`, `kl-260.{json,txt}`, `ppl-{A,B}.json`.

| | prédit | mesuré | issue |
|---|---|---|---|
| préfill par lot 8 × 78 | 0,410 → 0,27-0,33 s (−20 à −35 %), seuil −15 % | **0,4105 → 0,3005 s (−26,8 %)** (A 0,410/0,411, B 0,299/0,302) | **tenu** |
| mur par lot (256 jetons) | — | 4,470 → 4,363 s (−2,4 %) | — |
| pic mémoire | ≤ + 200 Mio | 26 860 → 26 795 Mio (−65) | tenu |
| chemins | B : cublas > A | A dequant 3 720 ; B cublas 3 576 + partagé 144, i8c_fabrique 696 | preuve de prise |
| KL 2b (b=8, 32 pas) | NON tenue (≈ 70 %) | max **0,215** ≫ seuil 0,0242 (2 × admis 0,0121) ; moy 0,0085 ; p99 0,094 ; **argmax 94,1 %** contre admis 99,6 % ; rejeux 0 | **non tenue** (prédit), et bien plus loin que prévu |
| PPL des 256 pas forcés | — | A 3,9112, B 3,8973 (−0,35 %) | bruit |
| **PPL wiki-gptq 2048/2048 (filtre)** | B/A 1,000-1,010 ; filtre ≤ 1,020 | **A 7,0273, B 7,1013 → 1,0105** | filtre **tenu** ; prédiction FAUSSE de 0,05 pt |

Lecture : le gain de temps est là (−27 % de préfill, conforme), la qualité paie plus que prévu. **L'argmax perd 5,5 points**
(15 jetons sur 256 changent de premier choix sur 32 pas après le préfill) : ce n'est pas l'ordre des sommes, c'est la
quantification par jeton des activations de 233 projections (GDN comprises, dont l'état récurrent porte l'erreur au décodage).
La PPL wiki (+1,05 %) passe le filtre de la P2 mais pas de beaucoup.

**Acceptation (décision chef) : panel de tâches Q19 — SUSPENDU** tant que l'outil (261b, poste2) n'est pas étalonné. En
l'état : opt-in seulement (`ACVRAM_I8C_FP8_PREFILL=cublas`), rien au défaut. Pistes si le panel échoue : W8A8 hors des
projections GDN (attention et mlp 56-63 seuls : la part n = 624 du gain, ≈ 50 ms sur 110), ou échelle par groupe de
l'activation au lieu de par jeton.
