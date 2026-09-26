# Verdict — 129 (2)-(3) : disposition Marlin mixte, preuve mémoire et KL b=1 (ABBA NON lancé) — 24/09 02 h 5x (poste1)

* **instrument** : `scratchpad/poste1-p129-24-09/preuve-memoire.py` (chargement réel de Qwen3.8-27B-nvfp4, 8 séquences × 8 192) ; `kl-chemins.py` (5 invites neuves, b=1, 8 pas gloutons du bras A, teacher forcing du bras B, KL(A ‖ B) par pas) ; prise `prise-abba.sh … kl-seul`
* **commit** : f5cad85c (poste1-mtp ; code 129 (2) : 6a5fa367 → 4912d0ae → f5cad85c)
* **régime** : RTX 5090, horloge libre (grandeurs numériques et mémoire) ; cpu-safe=off (100/100) ; compute-apps début = fin (llama-server sur la 3080 Ti)
* **scellé** : `scratchpad/poste1-p129-24-09/scelle-abba.md` (avant la mesure) — FAUX si refus au chargement (8 × 8 192), KL > 0,01, b=8 < +15 %, b=1 > −3 %
* **mesuré** :
  * Tests 129 (2) sur carte : **19 passed** ; bras cassants vues / moitié de K / gcol / rôle doublé : **4/4 ROUGES**
    (6a5fa367 → 4912d0ae). À sec : 344 + 13 passed.
  * Preuve mémoire, 8 × 8 192 demandés : **défaut** : chargé, capacité KV **55 136 jetons** (déjà sous 65 536),
    18,04 Gio alloués, 12,36 libres ; **PROJ_MARLIN=1** : **refus nommé au chargement**, capacité KV **32 768**
    jetons < 65 536. Disposition : 158 poids doublés, 129 seuls, **8,50 Gio** doublés (prévu 10,6 au manifeste, écart
    non expliqué : 158 au lieu des 176 attendus, 64 + 64 + 48, à lire).
  * KL(A ‖ B), b=1 (chargement de B à 1 × 1 024, accepté) : **kl_max 0,0137**, argmax égaux **39/40** ; ligne de
    régime B `dense=…+marlin(doubles=158,seuls=129,8.50Go)`.
* **verdict** : **FAUX au scellé, deux fois** :
  1. **Mémoire** : B ne tient pas 8 × 8 192 — mais le DÉFAUT non plus (55 136 < 65 536). La condition du chef n'est
     tenue par aucun bras ; la disposition mixte coûte **41 % de la capacité KV** (55 136 → 32 768 jetons, soit
     8 × 4 096). Le refus nommé au chargement a fonctionné, et l'on n'a pas vu d'OOM en service.
  2. **KL** : 0,0137 > 0,01. **Mon seuil était posé sans témoin** (REGLES § 3 : pas de scellé sous 2 × l'écart du
     témoin mesuré avant) : deux chemins W4A16 valides (Marlin fp32 contre GEMV/cuBLAS) diffèrent par l'ordre
     d'accumulation, dont le bruit n'a pas été mesuré. Réfuté reste réfuté ; il fallait le témoin avant le seuil.
  * ABBA de vitesse **non lancé** : l'étalon de poste2 occupe la carte de 03:00 à 06:00, et le scellé est déjà FAUX sur
    la mémoire.
* **durée** : 02:50:09 → 02:50:41 (prise courte)

## Au chef (deux décisions)
* **Capacité** : garder (A) à 8 × 4 096 (32 768 jetons), ou réduire les doubles (gate‖up seul : 6,5 Go → capacité
  ≈ 41 k jetons, perte b=1 8,1 % au banc) — ou (B), le GEMV Marlin rapide, qui rend 0 double.
* **KL** : mesurer d'abord le TÉMOIN (défaut contre `ACVRAM_DENSE_NVFP4=gemv`, deux chemins W4A16 existants) sur les
  mêmes 5 invites, puis sceller « KL(A ‖ B) ≤ 2 × KL témoin » avant toute nouvelle prise. Le verdict ci-dessus n'est
  pas rouvert.
