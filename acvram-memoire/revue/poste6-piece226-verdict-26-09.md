# Verdict — pièce 226 : la « régression » de la 209 sur le Coder-30B-A3B-nvfp4 PUR (217 : −8,9 %, 220 : −8,3 %) est un ARTEFACT du banc 217 (invites = jetons tirés, sorties dégénérées qui dépendent du bras) ; à invites RÉELLES la 209 gagne **+12,8 % de débit et −18,4 % de J/jeton** à b=8 ; le mécanisme « préfill » de la 222 est FAUX, le code C17-préfill n'est pas justifié (poste6, 26/09)

* **instrument** : `scratchpad/poste6-p226-26-09/` — étape 1 `prise-226-preuve.sh` (banc 217 + `/metrics` : préfill et pas par bras) ; 1 bis
  `banc-colonne-226.py` (harnais 214) ; 2 `chaine-nsys.sh` (chariot 203/116, boucle moteur nue, 20 pas) ; 3 `prise-service-nsys.sh` (nsys sur
  `acvram serve` + banc 217, 10 s) + `classer-226.py`, `trous-pas-206.py` ; 4 `prise-routage.sh` (NULLE) ; 4 bis `client-textes.py` ; 5
  `prise-abba.sh` (ABBA banc chat 102 à invites réelles, celui du 209 c). Résultats `metrics-*.json`, `colonne.json`, `nsys-{A,B}/`,
  `service-{A,B}/`, `classes-*`, `trous-service-*`, `textes-{A,B}/textes.json`, `abba/cellule-*.jsonl`.
* **commit** : poste6-226 = origin/main d782caa78 + instruments ; prises f38f30225 (1), a52d84e8a (1 bis), 877ac7f58 (2-3), 218d14dc1 (4),
  b13e026c7 (4 bis, 5). Noyau servi inchangé ; branche poste6-226-dev 44235edc1 (C17-préfill, tests non joués) NON fusionnable.
* **régime** : carte 0, -lgc 2700 (horloges 2 632-2 692), llama-server 4219 (5,6 Go, tiers) début = fin ; A/B = `ACVRAM_MARLIN_PAR_LIGNE`
  1/0 (étapes 1-4) ; étape 5 : A = 0 (défaut depuis la 220 b), B = 1 (209) ; graphes on sauf étape 4 (coupés, nulle).
* **scellé** : `scratchpad/poste6-p226-26-09/scelle.md`, un bloc daté AVANT chaque étape, résultats consignés à la suite.
* **mesuré** : (1) préfill par 2 048 jetons : 209 226-229 ms, défaut 238-246 (**−12 ms**, pas +100) ; pas de décodage 5,23 contre 4,64-4,80 ms.
  (1 bis) épilogue par colonne +0,6 (w13) / +1,4 µs (down), au bit du scalaire ; chaîne tensor 113 µs contre decode_mma 126 (64 distincts).
  (2) boucle nue, jetons fixes : pas GPU 209 **4 862** contre défaut **4 946 µs** (209 plus rapide), noyaux 92,6 contre 92,9 µs/couche ; le
  défaut n'est PAS « naturel » : 44 couches Marlin-w13 + 4 decode_mma. (3) service : pas 5 249 contre 4 672 µs, +577 tout en noyaux, trous
  égaux ; GEMM Marlin **23,0 µs/lancement (209) contre 19,7** à toutes les couches, mêmes piles sur 44 ; B accélère le long du lot (22 → 15 µs),
  209 non ; 209 +25 W à horloges égales. (4 bis) invites du banc 217 = `invite(k, n)` pseudo-aléatoire, greedy : 8/16 sorties = un caractère
  répété, 9/16 divergent entre bras. **(5) ABBA invites réelles, b=8, 5 + 5 : défaut 1 630,9 t/s · 0,1364 J (1 624-1 690, bridage puissance)
  contre 209 1 839,7 · 0,1113 (1 833-1 840, aucun bridage) : +12,80 % / −18,4 %.**
* **verdict** : étape 1 FALSIFIÉE (le préfill n'y est pour rien : erratum 222 § 2-3), 1 bis FALSIFIÉE (épilogue ≤ 2 µs/couche), 2 : 209 plus
  rapide à jetons fixes, 3 : l'écart du service est dans le GEMM experts à piles égales → dépend des DONNÉES (routage), 4 bis : le texte
  dégénéré diffère entre bras, **5 TENUE (B ≥ A − 1 % ; falsificateur B < A − 3 % non atteint)**. La 217/220 mesuraient le débit d'un texte
  dégénéré propre à chaque bras ; à invites réelles la 209 gagne, comme sur i8c (209 c +5,77 %). Résidu nommé : le compte direct d'experts
  distincts par pas (trace de routage non écrite à l'arrêt du serveur).
* **durée** : preuves 4 × 1 min + colonne 1 min + nsys 2 × 30 s + service 2 × 70 s + routage 2 × 70 s + textes 2 × 20 s + ABBA 10 × 46 s ;
  files 953 s (1), 163-1 447 s ; deux instruments nuls consignés (routage ; « PASSE NULLE : A sans refus » = contrôle copié de la 209 qui
  cherche « sous-normales » dans la ligne de régime, où le pur ne l'écrit pas — les 5 passes A sont valides, régime `experts_layout=naturel`).

## Ce que ça change
1. **`ACVRAM_MARLIN_PAR_LIGNE=1` peut revenir au défaut** (condition de chef tenue) : sur le Coder pur, +12,8 % / −18,4 % à invites
   réelles, sortie exacte (209 a) ; le témoin 0 et le test cassant de la 220 b à inverser (tests/test_marlin_pile_par_ligne_209.py::test_220…).
2. **Le banc 217 (`banc-llamacpp-16-09.py decode`) ne mesure pas un MoE en génération libre** : invites tirées → sorties dégénérées dont le
   routage dépend des numériques du bras ; pour une cellule de débit MoE, invites réelles (banc chat 102) ou jetons fixes (chariot 116).
   Proposition REGLES § 4 : « une cellule de débit d'un MoE en génération libre se prend à invites réelles ; jetons tirés = préfill seul ».
3. **Ligne de régime** : `experts_layout` ne nomme que la couche 0 (« naturel » avec 44 couches Marlin) — à remplacer par un compte
   `marlin-w13(44/48)` (bead à ouvrir, 20 lignes).
4. La 222 § 2-3 (préfill W4A4 → Marlin W4A16, règle C17-préfill) tombe : erratum ; poste6-226-dev reste une réserve (noyau mma2 `ldn` /
   `gs_ldn`, w13 en vues, `facteur_pile`, tests au bit écrits, jamais compilés) — ne pas fusionner sans besoin.
