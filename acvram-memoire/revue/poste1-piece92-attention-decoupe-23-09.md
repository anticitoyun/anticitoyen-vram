# Pièce 92 — attention : le découpage n'est PAS le levier (réfuté au banc) ; réduction déroulée au bit par défaut (−1,9 %) ; 4 warps −5 % au banc mais contredit par le servi du 20/09 — 23/09 (poste1)

* **instrument** : `outils/gpu/mesure/banc-attn-decoupe.py` (nouveau, suivi) — 48 caches distincts (L2 froid, comme un pas servi), table au godet nblk du service, 48 appels `paged_attention(compact=True)` rejoués en un graphe ; variantes par substitution de `_tranches`, `PAGES_PAR_TUILE`, `WARPS_COMPACT`, `REDUC_DEROULEE` ; justesse : `torch.equal` contre le servi, écart en 2⁻⁸, distance à la référence fp64 rapportée à celle du servi ; test `tests/test_glue_compact.py::test_reduction_deroulee_au_bit` (5 formes) ; `capture-godets.py` (godets 1/2/8/16)
* **commit** : banc e0d0bad7/b76cdfad ; prises f7bccbf0 (banc 1), acfe6149 (banc 2), eaa45017 (capture) ; défaut eaa45017 ; ligne de régime (témoin nommé `reduc=serie`) dans le commit de ce verdict
* **régime** : -lgc 2700 posé et rendu par la prise ; horloge moyenne de l'échantillonneur 2 453 MHz (repos compris, prise de 6 s) ; carte seule (llama-server 4627 au début et à la fin)
* **scellés** : étape 1 `scelle-banc.md`, étape 2 `scelle-banc2.md` (tous deux écrits avant leur prise)
* **mesuré** (µs par couche, b = 12, moyenne des 8 tranches de ctx de la p91 ; ms/pas = × 48) :

| variante | moyenne du lot | ms/pas | au bit du servi | b=1 ctx 2 048 | b=4 ctx 768 |
|---|---|---|---|---|---|
| servi (C = 8, BN 64, 8 warps, boucle série) | 15,46 | 0,742 | — | 15,41 | 12,08 |
| C × 2 / C × 4 / une tuile par programme | 18,46 / 20,54 / 20,54 | 0,886 / 0,986 / 0,986 | non (C change l'ordre) | = | = |
| BN 32 | 17,27 | 0,829 | non | 16,65 | 13,94 |
| 4 warps | 14,67 | 0,704 | 10/11 cellules (ctx 704 : 0,41 × 2⁻⁸, distance fp64 × 1,00) | 13,15 | 10,37 |
| **réduction déroulée** | **15,17** | **0,728** | **11/11** | 14,07 | 11,23 |
| déroulée · 4 warps | 14,53 | 0,697 | 10/11 | 12,70 | 9,77 |
| déroulée · C × 2 | 17,95 | 0,861 | non | 14,06 | 11,27 |

  Représentativité (prédiction 1) : le servi du banc suit la p91 servie à ± 15 % (9,9 · 11,7 · 14,5 · 14,6 · 15,0 · 17,6 · 20,0 · 20,3 contre 9,5 · 10,6 · 13,6 · 14,2 · 14,9 · 15,6 · 20,3 · 21,3), marches aux ctx 512 et 1 024 comprises : **tenue**.
* **verdict** :
  1. **Prédiction 2 (découpage, 12-13 µs) RÉFUTÉE.** Plus de programmes, c'est plus lent (+19 à +33 %), pas plus rapide. La marche à ctx 512 ne vient pas d'un manque de parallélisme. Je l'avais nommée « chunk fixé par le godet » dans la 91 ; c'est vrai de la cause (le godet double la tranche) mais faux du remède.
  2. **Étape 2 (réduction série sur le chemin critique) RÉFUTÉE à < 5 %.** Déroulée, elle gagne 1,9 % au bit (15,46 → 15,17) ; à C × 2, elle reste plus lente que le servi. Le coût est donc dans le travail par programme (une tuile de 64 jetons ≈ 3-4 µs de latence de chaîne k → softmax → v), pas dans la réduction.
  3. **Retenu, par défaut** : la réduction déroulée (`ACVRAM_ATTN_REDUC_DEROULEE=1`, témoin `=0` nommé `glue=compact(8,reduc=serie)`). Elle est au bit sur 11/11 cellules et au test (5 formes, dont lot mêlé b=3). Capture 8/8 (Coder qkvo-i8c, gemma-4-26B-A4B). Gain : −0,014 ms/pas sur le lot b=12, −8,7 % à b=1 ctx 2 048, −7 % à b=4.
  4. **4 warps** : −5,1 % au banc (−0,038 ms/pas ; −15 % à b=1 ctx 2 048). Mais le défaut 8 vient d'un ABAB SERVI (poste2, 20/09 05 h 15 : B8 +11,5 % contre B4 +9,1 %, `regime.py`). Le banc et le servi se contredisent ; **ne se tranche qu'en service, par poste2**. Je ne le passe pas au défaut.
  5. **Ce qui reste de l'écart d'attention contre vLLM** (+0,145 ms/pas à la 91) : environ +0,12 après déroulé et 4 warps. Il est dans la **tuile** : vLLM fait 5,9 µs pour une tranche courte (ctx 320) contre nos 9,3-9,9. Le levier suivant est un noyau à tuiles de 16 jetons et à segments fixes, lus depuis `slen` (dessin de `kernel_unified_attention`). C'est une pièce de noyau (≥ 1 jour), pas un réglage.
* **durée** : prévue 3-5 h + 10 min de carte ; tenue ≈ 1 h 30, carte ≈ 4 min (banc 1 : 6 s de noyaux, banc 2 : 15 s, capture 1 min 48)
