# Verdict — pièce 172 : B' au défaut, poids déquantifié partagé par la boucle par séquence (poste5, 25/09) — TENU

* **instrument** : `scratchpad/poste5-p172-25-09/prise.sh` (tests + 2 bras cassants ; `diag172.py` sur deux modèles ;
  TTFT servi par `ttft-charge.py` de la 165) → `prise-1.txt`, `prise-ttft.txt`, `diag-*.json`, `ttft.jsonl`
* **commit** : f0795b95 · **régime** : défaut (B' = `ACVRAM_DEPAQ_PARTAGE=1`), -lgc 2700 ; TTFT : 20/20 passes
  `repli_eager=0`, A porte `DEPAQ_PARTAGE=0` à sa ligne de régime, 0 passe nulle ; Qwen3.5-35B : Marlin refusé (157)
* **scellé** : `revue/poste5-piece172-scelle-25-09.md` (avant) · **durée** : tests 00:55-00:58:28, diag → 00:59:2x,
  TTFT → 01:13:26

## Au bit
* Tests : 171 verts (dont `test_depaquetage_partage_172.py` : naturel et Marlin, C1 et C2, 1 fabrication + 7
  réutilisations, témoin « GEMM groupée ≠ » vert ; couche GDN au bit, LOT=1 ≠). Bras cassants : GEMM groupée dans la
  couche → ROUGE ; partage perdu → ROUGE (5/5).
* Modèles : logits fp32 **égaux au bit** entre A et B', C1 et C2, sur Qwen3.8 ET Qwen3.5-35B ; bras cassant B' + LOT=1
  ≠ A sur les deux. Réutilisations par passage : 1 680 (Qwen3.8 : 48 couches × 5 linéaires × 7), 1 050 (Qwen3.5-35B).

## Vitesse (prédictions écrites avant)
| modèle | forward C1 (A → B') | forward C2 | prédit | TTFT servi C1 (8 simultanées) | TTFT C2 | prédit |
|---|---|---|---|---|---|---|
| Qwen3.8 | 365,1 → 319,4 ms **−12,5 %** | 417,4 → 370,4 **−11,3 %** | −10 à −15 | 429,2 → 389,8 **−9,2 %** | 464,6 → 427,0 **−8,1 %** | −9 à −13 / −7 à −11 |
| Qwen3.5-35B | 191,3 → 184,1 **−3,8 %** | 192,6 → 185,6 **−3,6 %** | −3 à −8 | 213,8 → 207,1 **−3,1 %** | 216,8 → 209,5 **−3,4 %** | −2 à −6 |

Toutes les prédictions sont tenues. Forward : A B' B' A en processus, médiane de 5 par passe ; TTFT : 5 passes par bras,
σ ≤ 2,3 ms.

## Verdict
* **TENU** : au bit (tests, bras cassants, logits de deux modèles), gain de préfill −11 à −13 % sur Qwen3.8 et −3,6 à
  −3,8 % sur Qwen3.5-35B, TTFT servi −8 à −9 % et −3 %. Environ la moitié du gain de LOT=1, sans rien changer à la sortie.
* **Limite** : le coût mémoire annoncé (≈ +230 Mo de pic, le temps d'une couche) n'est PAS mesuré en différence. `pic_Mo`
  cumule A et B' dans un même processus (27,0-27,5 Go sur Qwen3.8). Aucun OOM ni refus de capacité dans les 20 passes
  servies.
* Reste à faire avant le push : la suite complète sous verrou.
