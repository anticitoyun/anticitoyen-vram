# g6r + plongements, carte — gemma-4-31B à 65 536 : NOMINAL, 0 exilé, 31 j/s au premier chargement (39 exilés et 3,5 j/s hier), mais le régime ALTERNE d'un chargement à l'autre (2 MLP exilés au second) : pas fusionnable en l'état

instrument : `scratchpad/poste6-g6r/carte-g6r.sh` puis `carte-g6r-2.sh` → `outils/gpu/mesure/prise-s1-morceaux-kv31b.sh A1` (une prise par bras, sous `carte.sh`), lecture à sec des lignes `[acvram]`, de `metrics.json` et des empreintes d'ids (texte généré jamais affiché)
commit : poste6-gemma-anneau ee01f0272 (prise 1) et e3c5d15b7 (prise 2 ; même code, e3c5d15b7 n'ajoute que le scellé et le script), HEAD asserté par la prise, provenance `…/poste6-gemma-anneau/acvram/__init__.py` ; origin/main fusionné avant
régime : RTX 5090 seule (15 Mio occupés au départ et à la fin), `acvram-gemma-4-31b-it-nvfp4-4sur6-vision-nvfp4`, kv int8, b=1, plafond 400 W ; hors verrou sur la carte 1 (3080 Ti) : llama-server permanent, et un `leann` apparu pendant la prise 1 (2,7 Gio) ; charge 2,4-3,1
scellé : `poste6-g6r-scelle-02-10.md`, `poste6-plongements-hote-scelle-02-10.md` (G1', G5', G6' ; prédiction G1'' ajoutée et poussée à 12:50, avant la prise 2)
mesuré : 7 prises (3 chargements à 65 536, 1 bras manqué par ma faute, 2 bras d'ids à 8 192), 18 min 38 de carte
verdict : **G1' TENU** (65 536/65 536 tenus, 0/60 exilé, NOMINAL, graphes actifs) ; **G5' TENU** (30,8 et 31,0 j/s, prédit ≥ 20) ; **ids table hôte = table carte** à 8 192 (sha égal) ; **G1'' : prédiction tenue, donc DÉFAUT confirmé** — le second chargement exile 2 MLP et passe DÉGRADÉ, le troisième reviendrait NOMINAL.
durée : 12:42:09 → 13:03:09, 18 min 38 de carte tenue sur 30 accordées

## Mesuré

| prise | chargement | réserve appliquée | MLP exilés | régime | chauffe | pic / formule | décodage | 32 ids (sha) |
|---|---|---|---|---|---|---|---|---|
| g1-65k (12:42) | 1er | formule 3,35 Gio (plafond 5 120) | **0/60** | **NOMINAL**, graphes 6 captures | 65 536 tenus, 196 s | **4,09** / 3,35 Gio → +11 Kio/jeton enregistrés | **30,78 j/s** (76 jetons) | 5aabb6d5… ; requête 2 identique |
| g1-65k-2a (12:50) | 1er (fichier de chauffe remis à 0 par le bras 4 096) | formule 3,35 | 0/60 | NOMINAL | 65 536 tenus, 197 s | 4,09 / 3,35 → +11 | 30,98 j/s | 5aabb6d5… (= g1-65k) |
| g1-65k-2b (12:56) | **2e, excès appliqué** | 2,95 + 0,74 (plafond 4 096) | **2/60** | **DÉGRADÉ**, graphes coupés | 65 536 tenus, 194 s | 1,93 / 2,95 → +0 | 35,22 j/s (62 jetons) | 47fea92e… (≠ NOMINAL) |
| ids-8k-hote | `ACVRAM_EMBED=hote` | — | 0 | NOMINAL | 8 192 tenus | — | 41,02 j/s (44 jetons) | b839b1e4… |
| ids-8k-carte | `ACVRAM_EMBED=carte` | — | 0 | NOMINAL | 8 192 tenus | — | 40,97 j/s | b839b1e4… (égal) |

Invite : 61 942 jetons (README + REPRISE + ETABLI au commit a77970f45, sha de l'invite vérifié par la prise), 47 s par
requête en NOMINAL (cache de préfixe coupé sous l'anneau : la seconde requête repaie le préfill), 43 s en DÉGRADÉ.
Hier (G1, 5693846ae) : 39/60 exilés, 3,52 j/s, 51 s.

## Lecture

* **Le correctif fait ce qu'il annonçait** : cible KV au plancher d'une séquence (5,45 Gio, 4 096 blocs, 1 créneau), table de
  plongements en RAM hôte avant tout MLP, anneau pris d'office ; le premier chargement sert 65 536 en NOMINAL.
* **Le défaut** : le pic de préfill dépend du régime. En NOMINAL les MLP résidents passent par le chemin Marlin
  (`dense=…marlin(doubles=0,seuls=280)`) : 4,09 Gio mesurés pour 3,35 réservés. La chauffe enregistre l'excès ; au
  chargement suivant la réserve grossit de 0,74 Gio, le plan n'a plus la place, 2 MLP partent (prédit à sec : 1, fourchette
  0-2) ; le chemin Marlin se replie alors (`marlin(repli:exil (6 poids en flux))`), le pic retombe à 1,93, l'excès à 0 —
  et le chargement d'après repart NOMINAL. Le régime et les ids servis changent donc à chaque redémarrage.
* Le chiffre « pic 1,93 Gio » d'hier (G1) était celui du chemin de repli sous exil : il ne valait pas pour le NOMINAL.
* **Non expliqué, dit** : le DÉGRADÉ à 2 MLP exilés décode à 35,2 j/s, plus vite que le NOMINAL (31,0) — sur 62 et 76
  jetons, donc non conclu ; le modèle de coût de l'exil (8,8 ms par MLP et par jeton) prédisait ~22 j/s.
* G6' (débit à ± 1 % table hôte / carte, b=1 et b=12) : non jugé ici — 41,02 contre 40,97 j/s sur 44 jetons n'est qu'un
  indice ; instrument de cellule (poste2).
* Mon erreur de script : bras d'ids à 4 096 pour une invite de 7 953 jetons (HTTP 400, 22 s de carte perdues), rejoué à 8 192.

## Reste

1. Rendre le NOMINAL stable à 65 536 : la réserve compte 1,22 Gio de tampons denses (`_DENSE_SLOTS` × la plus grosse
   couche) qui ne servent que si un poids dense est exilé ; ne pas les compter quand le plan n'exile rien laisserait la
   place de l'excès (0,74). Scellé à sec avant, puis DEUX chargements consécutifs sur carte (≈ 12 min).
2. Corriger la formule pour le chemin Marlin résident plutôt que de s'en remettre à l'excès mesuré.
3. Tant que (1) n'est pas tenu : ne pas fusionner — ou fusionner en sachant que 65 536 alterne NOMINAL / DÉGRADÉ à 2 MLP,
   ce qui reste au-dessus de main (39 exilés, 3,5 j/s). Décision de chef.
