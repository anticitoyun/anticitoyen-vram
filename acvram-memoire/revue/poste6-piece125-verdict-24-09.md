# Verdict — pièce 125 : préfill ou décodage, lequel est le plus juste ? (poste6, 24/09)

instrument : `outils/gpu/mesure/kl-chemins-p125.py` (suivi) ; prises `scratchpad/poste6-p125-24-09/prise.sh {4b,coder}`, sorties `prise-4b.txt`, `prise-coder.txt`, `*/compare-<alias>.json`
commit : 91593bf6 (branche poste6 = main b6f290c6 + pièce), asserté rc 65 ; venv de mesure, PYTHONPATH de l'arbre
régime : b=1, à sec pour HF (bf16 eager processeur, 8 fils, sous le verrou) ; acvram défaut servi, `force` eager + témoin graphes ; cpu-safe=off (max_perf_pct 100 au début et à la fin des deux prises) ; compute-apps début = fin = llama-server 4627
scellé : `revue/poste6-piece125-scelle-24-09.md` (49fed384, écrit avant) — S0 instrument, S1 « moins juste » = KL_moy > 1,25 × l'autre sur ≥ 4/5 ET L2 d'invite > 1,25 × sur ≥ 4/5
mesuré : S0 tenu (bf16-acvram préfill KL_moy 3e-5 – 1,8e-3, L2 0,013-0,015, 5/5 ; graphes = eager 0 ulp 15/15). **Coder i8c (servi) : équivalents** — KL_moy préfill 0,054 contre forcé 0,038 (préfill pire sur 3/5, ratios 1,16 · 2,1 · 3,35 · 1,43 · 0,78), L2 d'invite 0,13-0,19 des deux, 0/5 hors 25 %. 4B nvfp4 : équivalents (0,333 / 0,340, 0/5). 4B bf16 : **décodage le moins juste** (KL 4/5, L2 5/5 ; 0,00067 contre 0,0135)
verdict : **FAUX** (prédiction « préfill le plus juste sur les trois » : tenue sur bf16 seul) ; **pas de défaut servi au sens du scellé** ; sur i8c le préfill est plutôt le MOINS juste (3/5, jusqu'à ×3,35), sous le seuil
durée : prévu ≤ 15 + ≤ 25 min ; tenu 4b 50 s de carte (+ HF 74 s, dumps réutilisés à la 2e prise), coder 15 min 20 s dont HF 13 min 54 s (`tenue=` au journal)

## Chiffres (KL(HF ‖ chemin) moyenne sur 32 positions de réponse, nats ; L2 relative du flux résiduel à la dernière couche, positions d'invite)

| alias | invite | KL préfill | KL forcé | KL préfill‖forcé | L2 inv. préfill | L2 inv. forcé | argmax p / f |
|---|---|---|---|---|---|---|---|
| 4B bf16-acvram | 0-4 | 1,8e-3 · 8,9e-4 · 3,8e-4 · 3e-5 · 3,1e-4 | 1,0e-2 · 6,0e-4 · 2,7e-2 · 1,1e-4 · 3,0e-2 | 6,8e-3 · 1,8e-3 · 2,1e-2 · 2e-5 · 3,0e-2 | 0,015 · 0,014 · 0,013 · 0,014 · 0,014 | 0,028 · 0,018 · 0,057 · 0,024 · 0,072 | 139/140 / 136/140 |
| 4B srcgguf-nvfp4 | 0-4 | 0,504 · 0,257 · 0,460 · 0,017 · 0,424 | 0,472 · 0,261 · 0,499 · 0,023 · 0,446 | 8,8e-3 · 2,4e-4 · 2,8e-3 · 1,6e-3 · 2,1e-3 | 0,64 · 0,63 · 0,62 · 0,61 · 0,62 | 0,63 · 0,62 · 0,61 · 0,61 · 0,61 | 128/140 / 128/140 |
| **Coder i8c (servi)** | 0-4 | **0,103 · 0,0076 · 0,077 · 0,057 · 0,026** | **0,089 · 0,0036 · 0,023 · 0,040 · 0,034** | 0,015 · 0,0016 · 0,016 · 0,005 · 0,009 | 0,149 · 0,127 · 0,132 · 0,187 · 0,136 | 0,137 · 0,127 · 0,132 · 0,173 · 0,137 | 133/141 / 133/141 |

KL max par invite, Coder : préfill 2,18 · 0,18 · 0,99 · 1,06 · 0,45 ; forcé 1,15 · 0,05 · 0,19 · 0,89 · 0,67 (seuil kl-gabarit 1,0 : préfill 2/5 au-dessus, forcé 1/5).

## Lecture
* **La question de poste5 a une réponse par alias.** Poids bf16 : le décodage (KV int8 par jeton, GEMV, attention paginée)
  coûte 20 × la KL du préfill (0,0135 contre 0,00067) et jusqu'à 0,89 nat sur un pas : le chemin de décodage est le moins
  juste quand rien d'autre ne fait de bruit. Poids quantifiés : l'écart préfill/forcé (KL 0,002-0,016) est 10 à 100 ×
  sous l'écart à HF (0,04-0,5) — les 32-51 % de codes de poste5 sont réels mais **sous le plancher de la quantification
  des poids** ; aucun des deux chemins n'a « raison » à ce niveau.
* **Sur l'alias servi, le signe s'inverse** : le préfill est le moins juste sur 3/5 invites, ×3,35 sur l'invite 2, KL max
  > 1,0 sur 2/5 (0,99 et 1,06 ; le forcé 1/5). Sous le seuil du scellé (4/5 sur les deux mesures), donc pas un défaut
  déclaré — mais un candidat nommé : le préfill du Coder i8c passe les projections en **W8A8** (`ACVRAM_PREFILL_INT8=cublas`,
  activation quantifiée int8 par jeton, défaut depuis poste7-p2-au-defaut-19-09), le décodage en W8A16 (GEMM étroit, activation
  bf16). Une quantification d'activation que le décodage ne paie pas explique un préfill moins juste que le décodage
  sur i8c seulement (le 4B nvfp4 n'a pas de projections int8 : équivalents). Hypothèse, non vérifiée ici.
* L2 par couche : sur le 4B bf16 le préfill et le forcé divergent de 12-39 % dès la couche 0 (positions d'invite) et
  convergent à 2-7 % en fin de pile ; sur le Coder 1 % à la couche 0, 4-9 % en fin. Le profil du 4B (grand à la couche 0)
  n'est pas celui de poste5 (V à la couche 1) : à regarder si la 125 bis rouvre le sujet, pas ici.
* Réserve tenue : `Qwen3-4B-srcgguf-nvfp4` est à 0,61-0,64 de L2 et 0,33 nat de HF hub sur les DEUX chemins — sa source
  GGUF n'est pas le bf16 du hub (pas le même point de départ, ou pas le même modèle) ; la comparaison appariée reste
  valable, la valeur absolue ne juge pas cet alias.
* Témoin graphes : 0 ulp sur 15/15 (alias, invite) — le forçage eager est bien le chemin servi.

## Suite proposée (à chef ; rien lancé)
**125 bis, 3 min de carte** : même instrument, Coder i8c, bras `ACVRAM_PREFILL_INT8=bf16` (témoin : déquant + cutlass, sortie
« inchangée » par construction du régime) contre `cublas` (défaut). Scellé à écrire avant : si KL_moy(préfill bf16) ≤
KL_moy(forcé) sur ≥ 4/5 ET ≤ 0,8 × préfill cublas sur ≥ 4/5 → le W8A8 du préfill est le coupable, et c'est un **défaut servi**
(poste7-p2-au-defaut a été jugé sur la PPL de tranches, pas sur la KL par position : les deux instruments ne voient pas
la même chose, pièce 37 / 107). Sinon le préfill du Coder est équivalent au décodage et la 125 se ferme.
Coût du candidat s'il est coupable : le W8A8 était le levier du préfill C15 (× 1,29-1,34 de débit) — un correctif
se juge en débit ET en KL, pas en KL seule.
