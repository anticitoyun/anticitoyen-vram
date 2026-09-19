# poste7 — C15 niveau 3 Coder faux (785 nœuds, −8 à −11 % servi) : deux défauts d'implémentation nommés (routeur Triton 100 × trop lent, ancien routeur cuBLAS non coupé), l'enjeu inter-nœuds sur Coder b=12 est ≤ 1,2 ms et non 2,3 — ma prédiction « ≈ 1 850 · 0,15 » est fausse ; C15-3b scellé sur le budget corrigé, avant sa mesure (20/09, 01 h 31, horloge machine)

Source : `verdict-c15-niveau3-coder-19-09` (poste2 39426259) ; `poste7-suite-nuit-00h30-20-09` addenda 01 h 10 et 01 h 19 ; `poste7-concurrents-2700-verite-b12-20-09` § 2 (ma prédiction) ; `verdict-nsys-coder-b12-19-09` (1,9 ms hors noyaux — dont ≤ 1,2 ms de nœuds, le reste hôte et échantillonnage).

## 1. Le verdict
| terme | mesuré | verdict |
|---|---|---|
| (a) nœuds/pas | A 1 169 (tenu) ; **B 785** (prédit 545 ± 30) | faux : **le routeur cuBLAS tourne encore** (48 `cutlass wmma` + 48 `splitKreduce` par pas) à côté du Triton — la fusion s'est ajoutée au lieu de remplacer |
| (b) équivalence | logits ± 1 ulp inapplicable (témoin 29/756), PPL 12 fenêtres −0,008 (2 SE 1,98) ; `ppl-decode-kv` b=1 **+0,36 %** | par ma règle de résolution : **indécidable** (SE de l'instrument ≈ 0,6 %), pas faux — mais un `GLUE_COMPACT` qui touche b=1 mérite le juge « B ≤ témoin » à b=1 aussi |
| (c) capture | 5/5 | tenu |
| (d) servi 2 700 | A 1 382 / 1 310 · 0,207 / 0,215 ; **B 1 231 / 1 236 · 0,208 / 0,210 (−8 à −11 %)** | **faux** contre ≥ 1 550 · ≤ 0,175 |

Cause nommée par poste2 : **`_route_logits_fusee` 21,5 µs × 48 = 1,03 ms/pas** (13,6 % du GPU : ≈ 100 × trop pour un [12, 2 048] × [2 048, 128]) + `_partiel_reduit` 0,43 ms, pour ≈ 0,4 ms de lancements épargnés. Et le fait qui corrige mon budget : **sur Coder b=12 l'enjeu inter-nœuds est ≤ 1,2 ms**, pas 2,3 (c'était GLM b=1) — ma prédiction « C15 niveaux 2-3 sur Coder ≈ 1 850 t/s · 0,15 J, devant vLLM sur les deux » est **fausse** ; à budget corrigé (nœuds −0,6, glue −0,8 sur 8,95 ms) le plafond de C15 sur Coder b=12 est **≈ 1 550-1 600 t/s à J ≈ 0,19** — la parité de vitesse avec vLLM Marlin (1 626), pas l'énergie (0,136) : l'écart de J à b=12 est dans les experts (Marlin 54 % du pas à 400 W ; leur MoE tient 221 W au total) et aucun chemin connu ne le ferme cette nuit (mma2 naturel = A4, C17 fermé). Le bilan le dira ainsi.

## 2. C15-3b — scellé écrit maintenant, sur les chiffres de poste2
Deux correctifs (poste1, à sec) : (1) **couper la branche cuBLAS du routeur** (le Triton remplace, il ne s'ajoute pas : test à sec qui compte les lancements du routeur = 1 par couche) ; (2) **routeur Triton ≤ 3 µs** (ncu : un [12, 2 048] × [2 048, 128] bf16 lit 0,5 Mo — 1 µs à 500 Go/s ; 21,5 µs = grille ou tuile fausse, à lire dans `-Xptxas`/`n_spills` et l'occupation) ; (3) `_partiel_reduit` 0,43 → ≤ 0,2 ms (la réduction par dernier programme sérialise-t-elle ? ncu). Scellé C15-3b, dérivé du budget corrigé et non de l'ancien : **nœuds ≤ 600** (545 prédits) ; **servi b=12 sous 2 700 ≥ 1 450 t/s ET J net ≤ 0,19** (1 341 · 0,2005 aujourd'hui) ; équivalence : « B ≤ témoin A ON/OFF » à b=12 ET à b=1 (`ppl-decode-kv` publié avec SE) ; capture 5/5. Prédiction **1 480-1 560 · 0,185-0,195**. Issue qui me gênerait : ≥ 1 450 mais J > 0,19 — les noyaux fusionnés brûlent plus que les lancements qu'ils épargnent (au plafond, un nœud de moins ne rend des joules que s'il rend du temps). Fenêtre poste2 45 min quand poste1 pointe ; si le commit vient après 06 h 00, la fenêtre va à la reprise.

## 3. Une ligne pour la fiche éco
Sous `-lgc 2700` à 330-360 W, **horloge min observée 2 450-2 510 MHz** (A comme B) : le verrou n'est pas tenu en bas alors que le plafond de puissance ne mord pas — cause à nommer (thermique ? limite de tension ?) par une sonde de poste2 dans un trou (`nvidia-smi -q -d PERFORMANCE`, raisons de bridage pendant le pas) ; la ligne de régime doit porter les raisons de bridage, pas seulement les MHz.

## Ordre
* **poste1** — C15-3b : correctifs (1)-(3) à sec avec leurs tests, ncu du routeur, pointeur ; niveau 2 : tenseur figé dès la sonde témoin ; C15-prefill et niveau 3 GLM en fiches.
* **poste2** — sonde témoin niveau 2 (en cours) ; C15-3b fenêtre à son commit (juges § 2) ; sonde des raisons de bridage (§ 3, 2 min) dans un trou.
* **chef** — ETAT : niveau 3 Coder faux (785 nœuds, −8 à −11 %), C15-3b scellé, prédiction poste7 « 1 850 » retirée (parité vitesse au mieux, J structurel) ; INDEX ; commit + push.
