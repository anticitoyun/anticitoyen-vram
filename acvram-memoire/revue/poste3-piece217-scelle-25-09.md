# Scellé — pièce 217 (poste3, 25/09, ordre chef) : bilan de l'après-midi, A=be837ca1 → B=origin/main

Instrument : `outils/carte.sh` + `scratchpad/poste3-p217-25-09/cellule-217.sh` (gabarit de
`scratchpad/poste2-p190-25-09/cellule-190.sh`, ABAB×5, `-lgc 2700`, `ACVRAM_ATTENTE=5400`).
Worktrees : `travail/poste3-p217-A` (be837ca1), `travail/poste3-p217-B` (origin/main = 4d9a4e40f,
216 inclus).

## Couverture (leçon de la 190 : périmètre AVANT, pas après)

`git log be837ca1..origin/main --oneline` = 188 commits, fusions (`fusion `) recensées une à une.
Pièces qui touchent `acvram/` entre A et B (les autres = tests, instruments, CHANGELOG, mesures à
sec sans code — vérifié `git show --stat <sha> | grep acvram/` pour chacune, aucun résultat pour
185c, 190(doc), 191, 192b, 193, 196, 197, 199, 200(×2), 202, 203, 205, 206, 208, 210, 210b, 211, 216) :

* **194 (poste1, second flux GDN, AU DÉFAUT, 13h27)** — β‖α sur un second flux pour les modèles à
  couches GDN (mixte-i8c, attn-gdn-i8c, et l'architecture nvfp4 partagée). Mesuré (verdict) :
  mixte b=8 **+2,20 %** débit / **−1,9 %** J, b=1 +0,46 % ; **Qwen3.8-27B-nvfp4 INERTE** (−0,02 % /
  0,00 %, cité au bit dans le CHANGELOG). N'affecte ni Coder (pas de GDN) ni gemma31 (dense, pas
  de GDN).
* **195b (poste6, étroit int8 par canal, AU DÉFAUT, 14h08)** — `ACVRAM_ETROIT_CANAL=1`, cible les
  linéaires int8 par canal (mixte-i8c : attention + GDN). Mesuré (ABBA propre, même commit,
  après 194) : mixte b=8 **+4,01 %** débit / **−3,76 %** J. N'affecte pas nvfp4 (pas d'int8),
  Coder (pas d'int8 servi par défaut sur cet alias) ni gemma31 (dense fp/nvfp4).
* **201 (poste5, vision/MTP + copie i8c transitoire, 16h02)** — deux effets distincts : (a) capacité
  KV annoncée BAISSE (compte vision + MTP dans la borne) : Qwen3.8-27B-nvfp4 32k 128 960 →
  **119 440 jetons (−7,4 %, MTP 299,7 Mio)** ; gemma-4-31B-it-nvfp4-vision 8k 10 848 →
  **8 528 (−21,4 %, vision+MTP 1 098 Mio)** ; mixte-i8c : OOM → 9/64 couches exilées (pas de
  pourcentage simple, à relever au chargement). Coder : dense, non concerné. (b) copie i8c
  transitoire : coût mesuré au banc chat mixte b=8 **0,00 %** débit (426,05 → 426,05 t/s),
  **+0,28 %** J — NEUTRE au débit par construction (n'affecte que les modèles i8c).
* **209 (poste6, Marlin par ligne, AU DÉFAUT, 17h07)** — spécifique au Coder-30B (piles d'experts
  sous-normales, couches 0/1/2/4). Mesuré : Coder b=8 **+5,77 %** débit / **−8,58 %** J, b=1
  **+0,52 %**. N'affecte ni mixte, ni nvfp4, ni gemma31 (aucun n'a la pile d'experts concernée).
* **212 (poste4, marge KV GDN, 17h12)** — capacité KV seulement (aucun effet débit) : marge
  1 536 → **3 072 Mio** sur les modèles à couche `.linear_attn.` (les trois Qwen3.8 : mixte,
  nvfp4, attn-gdn-i8c), ≈ **−12 192 jetons** de capacité annoncée pour ces modèles. Coder et
  gemma31 : denses, marge inchangée (delta négligeable déjà sous l'ancienne marge, cité au
  CHANGELOG −868/266 Mio).
* **187 (déjà dans A)** — cité par chef car il avait fait échouer le périmètre du scellé de la
  190 (composition +24 % prédite contre +22,08 % mesuré, écart attribué après coup). **187 est
  fusionné AVANT be837ca1** (bilan 190 : "B be837ca1 = main + 175b" incluait déjà 172/175b/176/
  179/182/187) — mon A l'a déjà : delta 187 entre A et B = **nul**, cité ici pour ne pas répéter
  l'erreur de périmètre de la 190 (nommer même ce qui ne compte pour rien).

## Prédictions par cellule (ABAB×5, seuil de recouvrement des IQR/médianes comme la 190)

| cellule | débit prédit | J/jeton prédit | pièces qui portent | rôle |
|---|---|---|---|---|
| mixte-i8c b=8 | **+4 à +9 %** (194×195b composés : 1,022×1,0401−1≈+6,3 %, bande ±3 pt pour l'incertitude de composition, leçon 190) | **−3 à −8 %** | 194, 195b (201 neutre confirmé) | cellule pleine |
| Qwen3.8-27B-nvfp4 b=8 | **0 ± 2 %** | **0 ± 2 %** | *aucune* (194 inerte confirmé, 195b/209 hors périmètre, 201/212 KV seul) | **falsificateur** |
| Coder-30B-A3B-nvfp4 b=8 | **+4 à +7 %** | **−6 à −11 %** | 209 seul | cellule pleine |
| Coder-30B-A3B-nvfp4 b=1 | **0 à +2 %** | **±3 %** | 209 seul (petit effet à b=1 dans son propre verdict) | cellule pleine |
| gemma-4-31B-it-nvfp4-vision b=8 | **0 ± 2 %** | **0 ± 2 %** | *aucune* (201b neutre, reste hors périmètre) | **falsificateur** |

**Falsificateur du scellé** : nvfp4 b=8 OU gemma31 b=8 hors de 0 ± 2 % → une pièce non identifiée
touche un chemin partagé (attention dense, tête, échantillonneur) : le scellé est FAUX par
périmètre, pas par mesure — rejouer `git log A..B` avant de composer une nouvelle attribution
(exactement la faute de la 190). Mixte ou Coder hors bande → composition erronée (194/195b ou 209
mal isolées, ou interaction non prédite entre elles) : à décomposer piece par pièce avant de
conclure.

## Capacité KV annoncée — colonne factuelle (relevée au chargement, pas une prédiction)

Les chiffres 194/201/212 ci-dessus sont ceux DÉJÀ publiés par leurs verdicts respectifs (sourcés
`revue/poste5-piece201-verdict-25-09.md`, `revue/poste4-piece212-warmgraphs-25-09.md`) ; la
colonne du tableau 217 relève la capacité RÉELLEMENT annoncée par le serveur B (log de démarrage,
`kv_max_tokens` ou équivalent) pour les quatre alias de la cellule, à côté du débit — pas une
remesure de 201/212, un simple relevé qui doit être cohérent avec ces deux verdicts.

## Durée prévue

5 cellules × 10 passes (ABAB×5) + 1 relevé de capacité par alias intégré au démarrage serveur.
Prévu ≤ 90 min de carte (comparable à la 190, quatre cellules ~19 min hors attente ; ici cinq
cellules et deux modèles plus lourds — Coder 30B, gemma31 vision). `ACVRAM_ATTENTE=5400` (ordre
chef) : la file peut attendre jusqu'à 90 min avant chaque prise si la carte est tenue ailleurs.

## Addendum du 25/09 22h0x (poste3, après mesure) — deux suppositions fausses nommées, prédictions non réécrites

Cellule Coder-30B-A3B-nvfp4 b=8 mesurée : **FALSIFICATEUR net**, −8,91 % débit / +20,65 % J (prédit
+4 à +7 % / −6 à −11 %), journaux serveur vérifiés (`experts_layout` naturel→marlin-w13 comme
attendu, 0 refus en B). Deux suppositions du scellé ci-dessus étaient FAUSSES, nommées ici sans
retoucher les prédictions d'origine :

1. **« 209 seul » pour Coder** — la prédiction s'appuyait sur le verdict 209, mesuré et scellé sur
   l'alias **Coder qkvo-i8c**, jamais sur **Qwen3-Coder-30B-A3B-nvfp4** (celui de ma cellule).
   L'ampleur des échelles sous-normales diffère fortement entre les deux : 209 a/b testait « 8
   piles réelles / 70,8 M éléments » ; le journal de ma cellule 3 montre 4 refus en A dont
   **67 477** échelles sous-normales sur un seul up_proj (deux fois), contre 128/207/262 pour les
   trois autres — un ordre de grandeur au-dessus de ce que 209 a mesuré. Je n'avais pas vérifié que
   le scellé de 209 couvrait CET alias avant de composer la prédiction.
2. **« 195b inerte sur Coder (pas d'int8 servi) »** — FAUX : le journal serveur de B porte
   `etroites=serie+canal(table)`, donc le noyau de la 195b est bien actif sur des linéaires int8 de
   cet alias Coder. Je l'avais classée inerte sans vérifier la ligne de régime réellement servie.

Piste ouverte, pas encore vérifiée par moi : le facteur par (expert, ligne) de la 209 pourrait avoir
un surcoût qui domine quand la fraction sous-normale d'un up_proj est énorme (quasi une couche
entière) plutôt que quelques piles isolées — ou une interaction 195b×209 non prédite. poste6 isole
(220, PAR_LIGNE=0/CANAL=0/les deux, processus séparés).
