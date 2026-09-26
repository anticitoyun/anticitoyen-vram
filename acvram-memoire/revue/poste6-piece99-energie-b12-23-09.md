# Pièce 99 — porte énergie b=12 : l'écart de J/jeton est dans les PROJECTIONS d'attention, et l'énergie suit les octets — 23/09 (poste6)

* **instrument** : étape 1 à sec sur les 20 fenêtres RESULTAT de la 96 (poste2, `poste2-w-21-09/scratchpad/poste2-piece96-vllm-rejeu-23-09/banc-b12-*.log`) ; prise A `scratchpad/poste6-p99-23-09/prise-a.sh` (nsys autour de `acvram serve` b=12, recette p79, client `banc-llamacpp-16-09.py` 1 024 jetons, fenêtre 5 s) → `outils/gpu/mesure/familles-comparees.py` contre la trace vLLM de la p76 ; prise B `outils/gpu/mesure/energie-familles-p99.py` (fichier suivi) via `prise-b.sh` : W nets par famille, isolée, graphe L2 froide rejoué ≥ 2 s sous le compteur NVML (mJ), repos mesuré dans chaque processus, mêmes octets que les bancs 75 (MoE) et 43 (étroites), venv vLLM 0.29.0 puis le nôtre
* **commit** : 79846d74 (branche poste6 = main b4981f09, moteur au bit de la 96 c8da314c) ; `acvram.__file__` sous le worktree vérifié dans les deux prises
* **régime** : -lgc 2700 posé par les chaînes pour les deux venvs ; prise A horloge médiane sous charge 2 662 MHz (vLLM p76 : 2 639) ; prise B 2 677-2 685 sur tous les bras sauf la tête (voir mesuré) ; mclk 13 801 MHz partout ; compute-apps début = fin = llama-server 4627 (les deux prises) ; alias `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`, ACVRAM_MOE_TENSOR=1, w13 par défaut
* **scellé** : `scratchpad/poste6-p99-23-09/scelle.md`, écrit avant : P1 Δt_noyaux ∈ [+0,05 ; +0,25] ms/pas ; P2 MoE |ΔW| ≤ 3 % (5 % = trouvaille), int8 étroites W ∈ [0,85 ; 1,10] × fp4, ΔJ_proj +0,04 à +0,07 J/pas, ΔJ_tête −0,03 à −0,05 ; P3 Σ W_f·t_f reproduit 1,617 / 1,513 J/pas à ± 20 % et Σ ΔJ_f ≥ 0,06 J/pas, classement projections > attention > MoE ≈ 0 > tête ; hypothèse principale « l'énergie suit les octets HBM, 25-35 pJ/bit »
* **mesuré** :

  Étape 1 (96, n = 10 par moteur) : t/s 1 995,1 contre 2 027,0 (−1,58 %, sous 2 σ) ; **W 342,4 contre 329,0 (+13,4 W), repos 73,5 les deux, nets 268,8 contre 255,5 (+5,2 %)** ; horloge 2 665 contre 2 666 MHz ; bridage « puissance » 10/10 des deux côtés ; J/pas nets 1,617 contre 1,513 (**+0,104**). ln(1,0696) = 76 % puissance + 24 % débit. Tension non lisible (`-q -d VOLTAGE` vide), même point V/F à -lgc 2700 sans bridage différent.

  Prise A (ms/pas, noyaux seuls, aujourd'hui, w13 par défaut) : proj_dense **1,029 / 0,787 (+0,242)** · moe_gemm 2,942 / 2,729 (+0,213, 96/96 lancements) · rope+normes+autres 0,452 / 0,416 (+0,036) · attention 0,588 / 0,538 (+0,050) · moe_glue −0,036 · échantillonnage −0,049 · tête 0,201 / 0,378 (−0,177) · **NOYAUX 5,811 / 5,526 (+0,285)**.

  Prise B (isolé, 2 685 MHz, W nets, mJ nets par appel) :

  | famille | acvram | vLLM | W ratio | t ratio | Δ J/pas (× 48) |
  |---|---|---|---|---|---|
  | qkv [5120×2048] | int8 g128 9,71 µs · 312 W · 3,029 mJ (10,7 Mo, 1,10 To/s) | Marlin fp4 7,31 µs · 254 W · 1,856 mJ (5,9 Mo, 0,81 To/s) | 1,23 | 1,33 | **+0,056** |
  | o [2048×4096] | 11,03 µs · 237 W · 2,619 mJ (8,6 Mo) | 9,01 µs · 166 W · 1,496 mJ (4,7 Mo) | 1,43 | 1,22 | **+0,054** |
  | MoE (p75, mêmes octets) | A3 servie 72,45 µs · 328 W · 23,74 mJ ; A2 75,65 · 311 · 23,54 | V 68,73 µs · 332 W · 22,82 mJ | 0,99 / 0,94 | 1,05 / 1,10 | +0,044 / +0,034 |
  | tête [151 936×2048] | int8 215,6 µs · 338 W · 72,9 mJ · **horloge 1 777** | bf16 377,3 µs · 336 W · 126,9 mJ · **horloge 2 587** | — | — | (−0,054, paire hors bande : plafond 400 W atteint en isolé) |

  **Énergie par bit des projections : acvram 35,3 (qkv) / 38,1 (o) pJ/bit ; vLLM 39,3 / 39,6** — deux noyaux, deux formats, une seule constante ; la tête donne 28,6 / 25,5 pJ/bit (à horloge réduite). Contrôle P3 : Σ W_f(isolé)·t_f(service) = 1,70 J/pas (acvram) et 1,60 (vLLM) contre 1,617 / 1,513 mesurés en service : +5 / +6 %, tenu ; Σ ΔJ_f = +0,119 (projections) + 0,03 à 0,04 (MoE) − 0,054 (tête) ≈ **+0,10 à +0,11 J/pas** contre +0,104 mesuré.

* **verdict** : la porte énergie est **une seule famille : les projections d'attention int8 (0,90 Go/pas) contre nvfp4 (0,48)**, +0,119 J/pas = 115 % de l'écart, masquée pour un tiers par notre tête int8 (−0,054, vLLM la sert en bf16) ; le MoE ajoute +0,03 à 0,04 (même noyau, +0,21 ms/pas de contexte de service — pas de puissance : ΔW −1 %). Ce n'est ni l'horloge, ni la tension, ni l'attente : **l'énergie d'un noyau borné mémoire est ses octets, ~37 pJ/bit quel que soit le noyau**. Scellé : P1 **réfutée à la marge** (+0,285 > +0,25 : le temps porte ~¾ de l'écart, la puissance plus haute de l'int8 — 312 contre 254 W — le quart, dans la même famille, donc même levier) ; P2 projections **réfutée sur le W ratio** (1,23-1,43 hors [0,85 ; 1,10]) mais tenue sur la direction et dépassée sur l'ampleur (+0,110 contre +0,04 à +0,07) ; MoE A2 −6 % hors ± 5 % (moins de W pour plus de temps, énergie +3 %) ; tête ordre de grandeur tenu, paire invalide à mon propre garde ; P3 tenue (± 6 %, Σ ΔJ ≥ 0,06). **Ce qui change de lecture : la pièce 71 avait écarté le Marlin dense de vLLM comme levier de VITESSE (−16/−22 % sous le seuil de 25 %) ; comme levier d'ÉNERGIE il vaut la moitié des octets, donc −0,11 J/pas.**
* **durée** : prévue A ≤ 4 min + B ≤ 3 min ; tenue A essai 1 3 min 02 (échec : chargement > 180 s, disque froid, aucune mesure), A essai 2 58 s (chargement 18 s), B 2 min 14 s ; `nsys stats` 2 min hors carte ; total carte 6 min 14.

## Levier nommé, gain prédit

**Projections q/k/v/o en nvfp4 servies par un noyau dense au débit du Marlin de vLLM (8 µs par lancement)** :
−0,42 Go/pas ⇒ **−0,11 J/pas = −6,8 % de J/jeton** (ferme la porte à elle seule : 0,1349 → 0,1257 contre 0,1261 vLLM), et
−0,24 ms/pas (+4 % de débit). Conditions : (1) le noyau — notre nvfp4 étroit de la 42 (0,44 To/s) et la GEMM groupée à
E = 1 de la 71 (0,35 / 0,19 To/s) sont trop lents ; l'énergie suivant les octets, même un noyau à la vitesse de l'int8
(9,7 / 11 µs) rendrait −0,10 J/pas sans gain de débit — c'est le bras à mesurer d'abord, avant tout portage
(`gptq_marlin_gemm`, ~2 000 lignes, jalon de la 71) ; (2) qualité : KL 5/5 déjà tenue sur l'alias nvfp4-qkvo (0,735,
pièce 42) ; (3) la pile qkv en une GEMM (la 42 chargeait q/k/v en 3 GEMM, +0,9 ms, layers.py:1207).
Prédiction pour ce bras : alias nvfp4-qkvo servi, même client que la 96, ABBA contre l'alias i8c : **J/jeton −5 à −7 %
si le pas ne s'allonge pas de plus de 0,3 ms ; réfuté si ΔJ > −3 %** (alors l'énergie ne suit pas les octets en
service, contre la prise B).
Second levier : le MoE en service (+0,21 ms/pas à noyau égal, +0,03-0,04 J/pas, −2 %) — la 76 l'attribuait au contexte
de service (L2, lancements) ; la tête en bf16 chez vLLM est un handicap qu'ils ont, pas un levier pour nous.

## Restes et limites

* Attention : Δt seul (+0,050 ms en service aujourd'hui, la 91 disait +0,145 sur une autre prise), pas de bras isolé —
  ≤ 0,02 J/pas dans tous les cas.
* Familles non mesurées en isolé (normes, rope, glue, routeur, 1,6 ms/pas) comptées à 250 W nets pour les deux : l'écart
  de Δt y est +0,005 ms, l'hypothèse ne pèse pas sur le verdict.
* `energie.py` marque « fenêtre trop courte < 10 s » sur les bras de 2 s : assumé (compteur mJ exact, joules et
  intégrale trapèze à 2-8 % l'un de l'autre) ; les bras MoE ont duré 13-14 s (rejeux asynchrones), sans effet.
* Le bras tête sature le plafond 400 W en isolé (12 jetons, 1,5-1,6 To/s) : à refaire avec un cycle utile < 100 % si
  la tête devient une pièce ; ici seul l'ordre de grandeur compte.
