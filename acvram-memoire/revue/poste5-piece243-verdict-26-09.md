# Pièce 243 (179 b) — seuil GEMV → GEMM int8 sous B′ : VERDICT (poste5 26/09 04 h 3x)

Scellé : `poste5-piece243-scelle-26-09.md` (poussé avant mesure). Instruments : `scratchpad/poste5-p243-26-09/`
(prise-a.sh, prise-b.sh, prise-p4.sh). Prises A/B sur 2a130d1b4, P4 sur 88eb99af9 (seul ajout : /metrics `int8_chemins`).
Carte 0 : aucun PID hors prise avant/après ; llama-server permanent 4219 sur l'autre carte.

| | prédit | mesuré | issue |
|---|---|---|---|
| P1 GEMV qkv n=78 | 655 µs ± 5 % | 636,4 µs | tenu |
| P1 déquant + GEMM NON partagée n=78 | 120-220 µs, n* 16-40 | 733,7 µs ; GEMV meilleur jusqu'à 78 | **FAUX** (issue i : déquant bornée mémoire) |
| P1 déquant partagée sur 8, n=78 | 40-90 µs, croisement ≤ 17 | 171,7 µs ; croisement ≈ 17 (142,0 / 138,1) | fourchette FAUSSE, croisement tenu |
| P2 préfill/lot 8 × 78 | −54 à −66 %, seuil −30 % | 0,821 → 0,410 s (−50,0 %), mur/lot −8,6 %, pic +29 Mio | tenu |
| P3 KL b=8, 32 pas | max ≤ 2 × admis (0,00793) | 0,00603 ; argmax 254/256 (admis 255/256) ; rejeux 0 ; PPL 6,0483 → 6,0657 | tenu |
| P4 service b=8, ABBA ×5, -lgc 2700 | +6 à +14 %, seuil +3 % et > 2 × étendue A | 423,6 → 464,7 t/s (**+9,70 %**) ; étendue A 2,9, B 4,7 ; J/jeton net 0,7615 → 0,6906 (−9,3 %) | tenu |

Preuve de prise (P3 et P4) : chemins int8 comptés — P3 B : dequant 1 240, gemv 0 ; P4 /metrics B : dequant 9 904, gemv 2 622
(décodage) contre A : 2 992 / ~10 000. Horloges moyennes A 2 545-2 590, B 2 555-2 588 MHz ; fenêtres toutes valides ; load1 ≤ 3.
Réserves : les 10 fenêtres portent « bridage pendant la fenêtre : puissance » (plafond 400 W, 393-397 W) — les DEUX bras
également, B produit plus de jetons au même wattage ; `regime` du client banc = « indisponible » (défaut P2 connu des outils
lancés depuis un worktree) : le régime SERVEUR est prouvé par /metrics (regime_ligne et int8_chemins de chaque passe).

**Défaut du scellé** : seuil P2 ABSOLU (≤ 0,83 s) déduit d'un témoin supposé à 1,19 s ; témoin mesuré 0,821 s → ce seuil ne
pouvait pas rendre « faux ». Jugé sur le relatif (−30 %). Leçon : un seuil de gain se pose en relatif au témoin de la même prise.

Lecture : le gain vient de la déquant PARTAGÉE (une déquant pour 8 séquences) ; non partagée, la bascule régresserait (P1) →
le seuil global hors B′ reste 80 (issue iv). Hors bit (KL dans la tolérance admise, argmax 1 jeton sur 256 de moins que le témoin).

Décision à chef : passer `INT8_GEMV_MAX_PARTAGE` à 16 par défaut (hors bit, KL tenue) ou garder l'opt-in. Restes : issue (v)
Coder -qkvo-i8c non mesuré ; 245 (poste1, GDN varlen) doit garder la portée `depaquetage_partage` autour des projections.
Dump logits P3 : `dumps.txt` (sha256, hors dépôt).
