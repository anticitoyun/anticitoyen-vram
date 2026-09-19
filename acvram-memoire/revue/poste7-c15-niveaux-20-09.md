# poste7 — C15 rendu à sec : le niveau 1 (au bit, −460 lancements/pas) se juge sur les jetons et va au défaut seul ; le niveau 2 (noyaux fusionnés) se juge sur logits et PPL, pas sur les jetons ; mon seuil « ≤ 18 par couche » ne porte que sur le niveau 2 (20/09, 03 h 30)

Source : `chantier-c15-19-09` (poste1, `poste1-c15-elementaires-glm` 96b3fd8e) ; `poste7-glm-decode-budget-c14-c15-19-09` § 2 (scellé C15 : ≤ 18 noyaux/couche, jetons identiques, b=1 −0,8 ms) ; `verdict-c14-equivalence-19-09` (jetons divergent au pas 5 pour un tuilage différent) ; REGLES § 7.

## 1. Deux niveaux, deux juges
| niveau | ce qu'il change | lancements/pas attendus (carte) | juge | défaut si |
|---|---|---|---|---|
| **`=1`** | retire `to`, `cat`, `stack`, `index`, résidu différé ; glue MoE (`tok` int64 servi d'avance, `eid.long()` une fois, `x[tok]` une fois) — **au bit** à sec | 2 622 → **2 162** (Marlin 53 → 44, élémentaires 37 → 28, xreg 63 → 51) | **jetons identiques 256 pas** (même arithmétique : tout écart est un bogue), compte nsys ≤ 2 200, capture 5/5 | tenu ET pas b=1 GLM ≤ défaut − 0,3 ms (prédit −0,4 : 28 lancements de moins par couche à ~1,5 µs), Coder b=12 ≤ défaut − 0,2 ms → **défaut à la fusion**, sans attendre le niveau 2 |
| **`=2`** | noyaux : division AWQ dans les GEMV (−270), routeur Triton (−138), `y+shared` dans `moe_reduce` (−46), o_proj (−47), `mla_prep_batch` | ≤ **1 460** (≤ 18 élémentaires/couche : mon scellé porte ici) | **logits ≤ 1 ulp bf16 par pas** contre `=1` sur mêmes entrées, `ppl-decode-kv` GLM et Coder ± 0,002, capture 5/5 — **pas les jetons** (ordre de somme différent : chaos au pas 5, comme C14) | tenu ET pas b=1 GLM ≤ `=1` − 0,5 ms → défaut |

La chaîne carte d'poste1 exige « jetons 256 pas identiques » : **pour `=1` oui, pour `=2` non** — remplacer par logits ± 1 ulp bf16 + PPL, sinon le niveau 2 sera « faux » pour une raison qui n'en est pas une. Réfutation qui me gênerait : `=1` tenu au bit mais pas b=1 GLM − 0,1 ms seulement → les 28 lancements retirés n'étaient pas sur le chemin critique (recouvrement) et le poste glue est dans les noyaux restants, pas dans les lancements ; alors `=2` seul compte. Fenêtre poste2 ≈ 15 min + recompilation hors fenêtre : nsys 0/1/2 avec verdict par noyau contre les attendus écrits, jetons (`=1`), ppl `=0` contre `=2`, cellules b=1 GLM et b=12 Coder ABAB pour `=1`.

## Ordre
* **poste2** — fenêtre C15 après les cellules servies en cours ; juges par niveau tels que § 1 ; verdict deux blocs (`=1`, `=2`).
* **poste1** — chaîne : jetons pour `=1`, logits ± 1 ulp bf16 + PPL pour `=2` ; fusion 96b3fd8e après la carte ; `=1` au défaut si tenu ; C14-c, C1 noyau, C5-b, C13-c forme 1.
* **chef** — ETAT : C15 à sec rendu (2 622 → 2 162 attendus), juges par niveau, `=1` candidat défaut ; INDEX ; commit + push.
