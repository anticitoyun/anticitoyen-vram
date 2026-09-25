# Verdict — pièce 177 : le coût des préfills au banc chat servi, alias mixte b=8 (poste5, 25/09)

* **instrument** : `scratchpad/poste5-p177-25-09/prise.sh` (banc de la 102 servi, 3 passes N1, N2, SYNC, `/metrics`
  avant et après) + `iso177.py` (préfill isolé en processus, médiane de 5) → `prise.txt`, `metrics-*.json`,
  `cellule.jsonl`, `iso177.json`
* **commit** : 34bdaa82 · **régime** : `Qwen3.8-27B-unsloth-mixte-i8c` défaut (Marlin, F1-F6, B'), -lgc 2700,
  `repli_eager=0` sur les 3 passes, `prefill_int8=bf16(origine fp8 …)` · **scellé** : `revue/poste5-piece177-scelle-25-09.md`
* **durée** : 02:52:54-02:5x (`poste5-p177`)

## Ce que fait le banc (compteurs du moteur, par passe ; identiques aux trois passes)
* 34 requêtes (2 de chauffe + 4 lots de 8), 1 092 pas, dont **9 avec préfill** : ≈ 2 pas d'admission par lot, soit 7
  requêtes puis 1 (cohérent avec la 150 : 1,7).
* **Invite L = 92 jetons** (3 120 jetons de préfill / 34) : le gabarit de chat compte. Prédiction FAUSSE (30-45).
* Décodage : 21,38 s pour 1 083 pas = **19,74 ms/pas**, le 19,90 de la 173. Débit servi 325,6 / 326,6 t/s
  (SYNC 319,4, la synchronisation coûte 2 %).
* Fenêtre du banc : 4 lots en 25,1 s = 6,29 s par lot, dont 255 × 19,74 = 5,03 s de décodage : **1,26 s par lot hors
  décodage** (≈ 4,9 ms par pas de sortie).

## Coût réel du préfill (isolé, en processus, 8 × 92)
| forme | ms (carte) | lecture |
|---|---|---|
| 8 × 92 groupé (un pas) | **906,9** | le préfill d'un lot si les 8 arrivent au même pas |
| 7 × 92 puis 1 × 92 (deux pas, le cas servi) | **1 002,5** | +96 ms : le coût de l'admission en deux pas |
| 1 × 92 | 190,7 | coût marginal par séquence ≈ 102 ms (8 × L − 1 × L) / 7 |
| 8 × 92 sous LOT=1 (projections groupées, pas au bit) | 397,8 | −509 ms (−56 %) |
| compteur B' (réutilisations) | **0** | B' ne s'applique pas au mixte, comme prédit |

**Décomposition des 1,26 s par lot** : préfill ≈ 1,00 s (79 %), dont 0,10 s d'admission en deux pas ; reste ≈ 0,26 s
(aller-retour HTTP, émission, pas vides entre deux lots). Prédiction (0,6-1,0 s, 50-85 %) : tenue, en haut de la
fourchette. Pas de nsys ici : le temps hôte vaut le temps carte (906,8 contre 906,9 ms), ce qui signale une
synchronisation interne au passage. La part « lancement », celle que viserait un graphe de préfill, n'est donc PAS
mesurée par cet instrument.

**Arrêt des décodages** : au banc, les lots sont synchrones ; il ne coûte que dans l'admission en deux pas. Les 7
premières séquences attendent alors le préfill de la 8ᵉ (0,19 s), et le lot finit un pas plus tard : ≈ 0,10 s par lot
(8 %). Prédiction (≤ 10 %) tenue.

## Leviers chiffrés (par lot de 6,29 s ; débit servi 326 t/s)
| levier | gain par lot | débit servi estimé | au bit ? |
|---|---|---|---|
| **B' étendu à l'int8** : dépaquetage bf16 des poids int8 (`int8_matmul`, M > 80 → déquant + GEMM, coût FIXE par appel, `kernels/__init__.py`, docstring « la déquantification … un coût fixe ») partagé par la boucle par séquence | ≈ −0,40 à −0,48 s (80-95 % du gain de LOT, car LOT = UNE déquant + un GEMM ; B' = une déquant + 8 petits GEMM) | ≈ 348-352 t/s (+7 à +8 %) | **oui** (mêmes appels, même W) |
| LOT=1 (déjà opt-in) | −0,51 s | ≈ 354 t/s (+8,8 %) | non (165) |
| admettre les 8 au même pas (attendre un pas d'admission court) | −0,10 s | ≈ 331 t/s (+1,6 %) | oui, mais retarde le 1er jeton des 7 premiers |
| préfill par tranches intercalé | ≈ 0 à ce banc (invites de 92 jetons, lots synchrones) | — | — |
| graphe de préfill | non mesuré (demande nsys) | — | — |

Contre NInfer (463,3 au même banc) : un lot à 463 t/s dure 4,42 s ; notre décodage pur seul en prend 5,03. Même avec
un préfill gratuit, l'écart restant tient au décodage (173 : int8 étroit, α/β bf16, Marlin).

## Verdict
* Le poste « préfill » est bien le préfill : 79 % des 1,26 s par lot, 8 × 92 jetons en 0,9-1,0 s. Le reste (≈ 0,26 s)
  est hors moteur de calcul.
* **Levier proposé : B' pour l'int8** (au bit, même démarche que la 172). Gain estimé +7 à +8 % de débit servi à ce
  banc ; prédiction à sceller et à mesurer.
* Prédictions : L fausse (92 contre 30-45) ; coût du préfill, arrêt des décodages et B' = 0 sur le mixte tenus.
