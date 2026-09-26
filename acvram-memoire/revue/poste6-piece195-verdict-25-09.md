# Verdict — pièce 195 : étroit int8 hors bit en opt-in (K entier par canal, géométrie NInfer) — +4,0 % de débit, −3,8 % J/jeton, KL NON tenue sur le mixte au critère scellé (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p195-25-09/` — `banc-canal.py` (géométrie, L2 froid), `kl-decode-195.py` (KL décodage
  b=8, 8 × 78 + 32 pas forcés, témoins T1 = séquence 0 seule à b=1, T2 = rejeu, même prise), `prise-abba.sh` (banc chat de la
  102, copie de la prise 173, A B B A A B B A A B, serveur neuf par passe, fenêtre 20 s), `chaine.sh` ; résultats
  `banc-canal-resultat.txt`, `kl-*.json`, `cellule-*-b8.jsonl` ; logits `kl-*-logits.pt` hors git (sha256 en fin de note).
* **commit** : 7b632d4ba (poste6-195 = origin/main 0c54811f2 + noyau, table, tests, instruments) ; banc de géométrie à 5a996bf27.
* **régime** : carte 0 ; KL : -lgc non posé (eco=2700(2692)) ; ABBA : -lgc 2700, cpu-safe 100, ACVRAM_ECO=off, b=8, 256 jetons,
  A `etroites=serie`, B `etroites=serie+canal(table)` (journaux serveur ; `/metrics` tronqué à 600 car. → drapeau « NULLE » de la
  prise = artefact, ignoré sur preuve des journaux) ; llama-server 4242 (5,6 Go, tiers) présent début = fin, comme à la 185 b.
* **scellé** : `scelle-etape1.md` (5a996bf27), `scelle-qualite.md` (670101ed8), `scelle-abba.md` (46955e525), tous avant leur prise.
* **mesuré** : banc 0,86 ms/pas (prédit 0,6-1,1, seuil 0,5) ; KL mixte max 0,0049 nat > seuil 0,00109 (2 × T1) — NON tenue,
  argmax 0,984 ≥ 0,964 tenu, PPL 9,728 → 9,732 ; KL attn-gdn-i8c max 0,0038 ≤ 0,0202 tenue, argmax 0,992 ; ABBA mixte b=8 :
  A 397,7 t/s (395,4-398,8), B 413,6 (412,4-414,5) = **+4,01 %** (prédit +2,5 à +4,5, seuil +1,5, gain 15,9 > 2 × étendue A 3,4) ;
  J/jeton net 0,804 → 0,773 = **−3,76 %** (prédit −2 à −4), W 396 = 396 ; tests étroits 121 verts sous verrou.
* **verdict** : levier RÉEL en service (+4,0 % / −3,8 % J) et opt-in propre (défaut au bit, 121 verts) ; **mais le critère de
  qualité scellé n'est pas tenu sur l'alias mixte** (KL max 4,5 × T1, 10/256 positions au-dessus du seuil ; moyenne 1,6 × T1,
  p99 0,0015 ; sur la seule séquence que T1 couvre, max AB 0,0004 < seuil) → reste opt-in ; décision du défaut à chef.
* **durée** : prévu ≤ 6 + 24 + 20 min ; tenu banc 19 s + KL 17 + 22 min + tests 1 min + ABBA 9 min (`tenue=` au journal) ;
  une 1re chaîne perdue (40 min de carte) sur deux fautes d'instrument, § 5.

## 1. Ce qui est livré (opt-in `ACVRAM_ETROIT_CANAL`, jamais au défaut, jamais posé par un lanceur)
`gemm_etroit.py` : `_etroit_canal_kernel` — poids int8 symétrique par canal (échelle [N, 1], zéro 128, G = K : tous les int8 du
mixte-i8c, attention + GDN de l'alias de poste4), **K entier par programme**, sans tranche, partiel ni atomique ; `(q − 128)`
exact en bf16 donc aucun terme Σx·z ; `tl.dot` accumulé fp32 ; échelle appliquée une fois. `gemm_canal`, `canal_eligible`
(vérifié une fois par tenseur), table `GEOMETRIE_CANAL` par forme (N, K) ; **une forme absente de la table reste sur le noyau
servi** (α/β int8 48 × 5120 de l'alias de poste4 : 2 programmes à K entier, prouvé par le compte des chemins). Crochet
`kernels/__init__.py:int8_matmul` (n ≥ 2, ≤ 16, bf16, pas la tête fp32), AVANT la vue g128, compteur `etroit_canal`. Variable
au régime (`regime.py`), ligne `etroites=serie+canal(table|BNxBKxWxS)`. Tests : référence ± 2⁻⁸ (3 formes), refus du non-par-canal
et du zéro ≠ 128, opt-in fermé au défaut et forme inconnue → servi. Pourquoi hors bit : une somme fp32 sur K entier n'a pas
l'ordre des 2-5 tranches du noyau servi (REGLES § 1, mode ± 1 ulp).

## 2. Étape 1 — géométrie (banc, M = 8, L2 froid, % du plancher 1,55 To/s)
| forme (N × K) | appels/pas | servi µs | canal µs (BN, BK, w, s) | NInfer | gain ms/pas |
|---|---|---|---|---|---|
| o‖out 5120 × 6144 | 64 | 25,23 (80 %) | **22,70** (32, 128, 4, 4) — 89 % ; servi T=4 : 22,91 | 20,79 | 0,162 |
| qkv attn 14336 × 5120 | 16 | 50,30 (94 %) | 49,80 (32, 256, 4, 3) | 45,95 | 0,008 |
| GDN qkv‖gate 16384 × 5120 | 48 | 65,57 (83 %) | **56,47** (64, 256, 8, 2) — 96 % | 52,36 | 0,437 |
| down 5120 × 17408 | 8 | 66,27 (87 %) | 58,92 (64, 512, 8, 3) — 98 % | 55,99 | 0,059 |
| gate‖up 34816 × 5120 | 8 | 141,41 (81 %) | **117,01** (64, 256, 4, 2) — 98 % | — | 0,195 |
Total **0,86 ms/pas** sur 19,90 (173). BN 16 (NInfer littéral) perd partout en Triton (tuile de dot 16 × 16) ; BN 64 × BK 512 × 4
étages déborde la mémoire partagée. « T = 4 sur o/out » (193) est absorbé : le canal fait aussi bien (22,70 contre 22,91).

## 3. Étape 3 — qualité (décodage b=8, KL(A‖B) par position, 256 positions ; T1 sur 32 positions)
| alias | chemins B (par pas) | rejeu | KL AB max / moy / p99 | T1 max / moy | seuil 2 × T1 | > seuil | argmax AB / T1 | PPL A → B (T1) |
|---|---|---|---|---|---|---|---|---|
| mixte-i8c | canal 144, tête triton 1 | 0 / 0 | **0,0049** / 0,00024 / 0,0015 | 0,00054 / 0,00015 | 0,00109 | 10 / 256 | 0,984 / 0,969 | 9,728 → 9,732 (seq 0 : 2,365 ; T1 2,358) |
| attn-gdn-i8c | canal 128, triton 96 (α/β + tête) | 0 / 0 | 0,0038 / 0,00042 / 0,0020 | 0,0101 / 0,0019 | 0,0202 | 0 / 256 | 0,992 / 0,969 | 9,348 → 9,332 (T1 1,676 ; A seq 0 1,653) |
Lecture : sur le mixte, le témoin T1 (b=1 CUDA par canal contre b=8 Triton g128) est 20 × plus petit que sur l'alias de poste4,
et il ne couvre qu'une séquence : le max AB sur les 8 séquences dépasse 2 × ce témoin (pire position : seq 3 pas 26, p(top) 0,187
→ 0,178, même argmax), le max AB sur la séquence 0 seule (0,0004) ne le dépasse pas. PPL par fenêtre de 16 pas : B dans l'écart
A/T1 sur toutes les fenêtres. Le critère scellé est ce qu'il est : **NON tenu sur le mixte**. Si chef veut un témoin à
échantillon égal, T1 sur les 8 séquences seules coûte ≈ 8 min de carte de plus (à sceller avant, pas ici).

## 4. Étape 4 — ABBA servi (mixte-i8c, b=8, banc chat 102, 5 A / 5 B)
A 395,4 · 396,6 · 398,8 · 398,7 · 398,8 → 397,7 t/s ; B 414,5 · 412,4 · 414,3 · 414,3 · 412,5 → 413,6 : **+4,01 %** ; J/jeton net
0,804 → 0,773 (**−3,76 %**), W 396,1 / 396,5, horloge 2700 posée. Prédit +3,5 % (bande +2,5 à +4,5) : TENU ; 0,86 ms sur 24,5
ms/pas servis = 3,5 %, mesuré 4,0 % (le banc HTTP rejoue aussi des lots plus pleins : B sert 5 lots par fenêtre contre 4).
Contre la cellule 190 (394,8 t/s, poste2 11 h 26) : A 397,7 la reproduit à +0,7 %.

## 5. Fautes et pièges du jour
* 1re chaîne : `nll` lisait la cible du dernier pas hors séquence (IndexError après 17 min de carte : tester le dépouillement à
  sec sur des tenseurs factices AVANT la prise) ; le banc chat de la 102 insère un `~` non développé dans sys.path → `energie`
  introuvable (`PYTHONPATH=$PWD:$PWD/outils/gpu/mesure`) ; la prise ABBA rendait rc=0 avec 10 passes vides — un `set -e` ne suffit
  pas quand chaque passe est en `|| echo`. 40 min de carte perdues. L'alias de poste4 vit hors `racine_modeles()` (chemin absolu).
* `/metrics` tronque `regime_ligne` à 600 caractères dans la prise : le contrôle « B porte +canal(table) » doit lire le journal
  serveur, pas /metrics tronqué (corrigé ici par la preuve des journaux, à corriger dans le script à la prochaine prise).

## Suite (décision chef)
(i) défaut : NON au critère scellé (mixte) ; (ii) témoin à échantillon égal (8 × b=1) si le critère doit être rejugé ; (iii) reste
hors bit et hors table : α/β int8 48 × 5120 (poste4) et la tête fp32 ; (iv) 194 b2 d'poste1 est orthogonale (gdn.py, second flux).
sha256 : kl-Qwen3.8-27B-unsloth-mixte-i8c-logits.pt d7c59562051848a0… · kl-Qwen3.8-27B-nvfp4-attn-gdn-i8c-logits.pt 0996f96c5775c0b2…

## 6. Relecture 198 (poste2) et ordre de chef : trois trous comblés (13 h 2x, commit ci-dessous)
* (a) **au bit du défaut** : `test_195_le_defaut_est_au_bit_du_chemin_servi_et_l_opt_in_prend` — sans variable et à `0`,
  `int8_matmul` sur un poids par canal 5120 × 6144 = `gemm_etroit(vue_g128, compact)` **au bit** (`torch.equal`), compteur
  `etroit_canal` = 0 ; à `1`, compteur = 1 et sortie dans ± 2⁻⁷ de la référence fp64 ; retour à `0` : au bit de nouveau.
  Passé sur carte (prise poste6-p195-tests-bit, `tests-bit.log` : 27 verts, 1 skip préexistant).
* (b) **zéro-point ± 1** et (c) **K tronqué sur la plus petite forme** : `test_195_le_zero_point_et_k_entier_sont_juges`
  (x de moyenne non nulle, juge ± 2⁻⁷ du max de la référence, formes 8 × 300 × 640 et 2 × 100 × 200) exige en plus que les
  deux bras faux (zéro 127, K/2) soient REFUSÉS par le même juge. Fautes injectées à sec une fois puis annulées (arbre
  restauré, `git status` propre) : zéro 127 → 2 échecs (les deux formes) ; `range(0, K // 2, BK)` → 4 échecs, dont
  2 × 100 × 200 ; noyau restauré → 7/7 verts.

## 7. Prise kl2 (ordre chef 13 h 4x : témoins à échantillon égal) — TENU
* scellé `scelle-kl2.md` (3bb46bad6, poussé avant) ; instrument `kl-decode-195-kl2.py` ; prise poste6-p195-kl2, tenue 21 s
  (poids en cache de pages), mixte-i8c, mêmes invites, même code ; A et B identiques à la prise précédente (rejeu 0, B déterministe).
* T1 = 8 séquences décodées seules à b=1 sous A (chemins : gemv 38 656 = 8 × 4 832), T2 = KL(A‖A₂) = 0.

| | max | moy | p99 | max par séquence |
|---|---|---|---|---|
| KL(A‖B) | 0,0049 | 0,00024 | 0,0015 | 0,0004 · 0,0012 · 0,0009 · 0,0049 · 0,0015 · 0,0021 · 0,0012 · 0,0014 |
| KL(A‖T1) | **0,0052** | 0,00038 | 0,0036 | 0,0005 · 0,0019 · 0,0009 · 0,0052 · 0,0035 · 0,0048 · 0,0016 · 0,0015 |
Seuil 2 × max(T1, T2) = **0,0104** ; KL AB max 0,0049 ≤ 0,0104 : **TENU** (prédit T1 max 0,001-0,006, pari « tenu à 55-65 % » :
tenu). Argmax AB 0,984 = argmax T1 0,984 (tenu) ; PPL A 9,728 · B 9,732 · T1 9,716 ; fenêtres de 16 pas : B dans l'écart A/T1.
Lecture : T1 suit AB séquence par séquence (0,0052 / 0,0049 sur la 3e, 0,0005 / 0,0004 sur la 0e) — les deux sont un changement
d'ordre des sommes fp32 dans les mêmes 145 appels par pas, que le service accepte déjà entre b=1 et b=8 ; le « NON tenu » du § 3
était un défaut d'échantillon du témoin (32 positions contre 256), pas un résultat. Logits hors git : kl2-…-logits.pt aeac38e3995442d9….
**Décision du défaut : à chef** (critère scellé tenu dans les deux sens, ABBA +4,01 % / −3,76 % J, tests 198 en place).
