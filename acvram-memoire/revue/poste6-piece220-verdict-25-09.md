# Verdict — pièce 220 : la régression du Coder-30B-A3B-nvfp4 PUR à b=8 (217) est la 209 — `ACVRAM_MARLIN_PAR_LIGNE=0` rend +8,3 % de débit et −14,5 % de J/jeton ; le canal 195b est hors de cause (poste6, 25/09)

* **instrument** : celui de la 217 (`scratchpad/banc-llamacpp-16-09.py decode`, slots 8, invite 256, 256 jetons, fenêtre 20 s, énergie nvml
  nette), `scratchpad/poste6-p220-25-09/prise-220.sh` : serveur neuf par passe (`acvram serve --max-batch 8 --max-model-len 4096`),
  quatre bras en processus séparés — a défaut · b `ACVRAM_MARLIN_PAR_LIGNE=0` · c `ACVRAM_ETROIT_CANAL=0` · d les deux ; ordre
  a b c d d c b a puis a b c d (3 passes par bras) ; `cellule-Qwen3-Coder-30B-A3B-nvfp4-b8.jsonl` (12 lignes RESULTAT), `serveur-<bras>-<n>.log`.
* **commit** : 6f74cf762 (poste6-220 = origin/main 9bc3674b0 + scellé + prise) ; HEAD asserté (rc 65 sinon).
* **régime** : -lgc 2700 posé par la prise (horloge 2 685-2 692 à chaque passe), cpu-safe 100, llama-server 4219 (5,6 Go, tiers) présent
  début = fin ; a et c : `experts_layout=marlin-w13` ; b et d : `experts_layout=naturel` (= be837ca1 : disposition Marlin refusée, 67 477
  sous-normales) ; c et d : `etroites=serie+canal(temoin)`. Les 12 cellules portent `invalidations: bridage pendant la fenêtre : puissance`
  (plafond de puissance atteint, comme les cellules b=8 de la 217 sur le même instrument) — le comparatif reste entre bras du même régime.
* **scellé** : `scratchpad/poste6-p220-25-09/scelle.md` (93c937650, avant toute cellule) : b ≥ a × 1,07, d ≈ b, c ≈ a ± 1,5 %.
* **mesuré** (médiane de 3, t/s · J/jeton net) : **a 1 539,8 · 0,1521** (1 464-1 575) ; **b 1 667,4 · 0,1300** (1 600-1 689) ; **c 1 533,2 · 0,1541**
  (1 493-1 538) ; **d 1 664,2 · 0,1323** (1 624-1 664). Contre a : b **+8,29 % / −14,5 %**, d +8,08 % / −13,0 %, c −0,43 % / +1,3 %.
  Prise 1 annulée après 2 passes nulles (`BANC_MOTEUR=acvram-<bras>` = protocole llama.cpp, 0 jeton décodé ; `nul/`), corrigée, rejouée.
* **verdict** : prédiction TENUE aux trois clauses — **la 209 est la coupable** (b et d rendent la même chose, à 0,2 % près) ; **la 195b
  n'y est pour rien** (c = a). Sur ce modèle la 209 ne bascule pas 4 couches mais 48 : le 1 fait passer tout le Coder pur de la
  naturelle (préfill « groupe », décodage `decode_mma`) à Marlin-w13 par colonne. Décision chef (prise sur a/b, avant c et d) :
  **défaut 0** — branche poste6-220b (code, régime, test cassant, CHANGELOG), tests sous verrou en file.
* **durée** : prévu ≤ 20 + 10 min ; tenu 8 passes 7 min 47 (22:21:51-22:29:38) + 4 passes 3 min 52 (22:43:15-22:47:07), ≈ 1 min par passe
  (chargement 35 s + fenêtre 20 s) ; files 163 s et 1 447 s ; prise nulle 2 passes.

## Lecture
* Le banc de la 217 compte autant de jetons de **préfill** que décodés (32 768 / 32 768 par fenêtre) : un bras qui perd au préfill perd
  dans cette cellule même à décodage égal. La 147 L2 avait mesuré le préfill Marlin plus lent que la naturelle (TTFT +27 / +35 % à 2 048 /
  4 096 jetons), et la 203 le décodage tensor Marlin plus rapide que `decode_mma` (50,5 contre 58,7 µs/couche) : sur 48 couches
  basculées, le préfill l'emporte ; sur 4 (i8c, 209 c) le décodage et le préfill « groupe » évité l'emportaient (+5,77 %). Mécanisme
  probable, non prouvé ici : pièce à venir (chef) — Marlin MoE contre naturelle par phase et par fraction de piles basculées.
* L'épilogue par colonne (`marlin_template.h`, g lu par élément écrit) coûte ≤ 2 µs/couche à sec : il n'explique pas 8 %.
* Dispersion intra-bras jusqu'à 7 % (a : 1 464-1 575) sous plafond de puissance : la médiane de 3 sépare a de b à 2 σ, pas a de c.

## Suite (à chef)
poste6-220b : `ACVRAM_MARLIN_PAR_LIGNE=0` au défaut, 1 = opt-in (gagne sur qkvo-i8c), test cassant, CHANGELOG — sha après les tests.
