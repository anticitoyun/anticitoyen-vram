# Scellé : g9m retenu — `est_hybride = couches_recurrentes > 0` comme définition (poste6, 01/10, écrit AVANT la prise, ordre chef)

Branche `poste6-g9m` 5d9ac10b8 (origin/main 88c63bacc + retrait d'`ACVRAM_HYBRIDE_PAR_RECURRENCE` + tests cassants, 453 verts à sec).
Bras AVANT = arbre témoin détaché sur 88c63bacc (`travail/poste6-g9m-temoin`, variable absente ⇒ ancien comportement : gemma « hybride ») ;
bras APRÈS = `travail/poste6-g9m` 5d9ac10b8. Modèle : gemma-4-31B `acvram-gemma-4-31b-it-nvfp4-4sur6-vision-nvfp4` (kv int8), le même que
les prises g9m du matin (0,144 / 0,027) et que le S1 bis. Carte tenue à l'écriture par la campagne e50.2 de poste2 (PID 1063167, pause coopérative `~/.config/acvram/campagne-e50.2.pause`) : fenêtre au feu de chef, après poste5 (ddw).

## (a) Témoin reprise et équivalence — `scratchpad/poste6-g9m-retenu-a.sh`, lecture `scratchpad/g9m-retenu-comparer.py`
Instrument : `prise-s1-morceaux-kv31b.sh A1` (seul tenant, `--max-batch 1 --speculative none`, glouton 32 jetons, logprobs 10, REQUETES=2 :
la même requête rejouée = reprise), chaînes court (README 87d8bfe0a, 7 953 jetons, ctx 10 240) et long (README + REPRISE, 17 859, ctx 20 480),
4 prises (avant/après × court/long). Δ lu comme au S1 bis : Δmax sur ids égaux, Δ logprob à la position 0 si les ids divergent.
Reconstruit à sec sur les prises existantes (S1 bis coupé contre insta = sans coupe, même commit ; g9m du matin) : court coupé/sans coupe
Δ pos 0 = **0,737, ids divergents** ; long 0,031 ; g9m du matin = insta au bit (G6).

| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| R1 | cache servi APRÈS (`cached_prompt_tokens`, 2e requête) | court ≥ 7 900 (7 952 ce matin), long ≥ 17 800 | court < 7 000 ou long < 17 000 |
| R2 | cache servi AVANT | **0** (le défaut est bien sur main : contrôle qui peut rendre faux) | > 0 — le diagnostic g9m était faux |
| R3 | témoin reprise APRÈS (reprise/A1) | court 0,10-0,20, ids 32/32 (0,144 ce matin) ; long 0,015-0,05, ids divergents au pas 2 (0,027) | court < 0,05 ou > 0,30 |
| R4 | APRÈS A1 au bit avec le g9m du matin (158b86530) | court sha `d3449ae4b18ee762`, long `f34d6b5f6a7384b0` | autre sha : le moteur a changé ailleurs entre 158b86530 et 5d9ac10b8 — bissecter AVANT de lire R6 |
| R5 | AVANT A1 au bit avec le S1 bis coupé | court `3750296adf4a7743`, long `67baf588c4616849` | autre sha |
| R6 | équivalence APRÈS / AVANT SANS COUPE (référence tranchée par chef, § ci-dessous : arbre témoin, ancien prédicat, `ACVRAM_INSTA_PAS=32768` repousse la frontière au-delà de l'invite, seul tenant, 1re requête) ≤ 2 × R3 | **au bit** (Δ 0, ids 32/32) en court comme en long — G6 du matin : g9m = insta au bit ; tenu a fortiori | Δ > 0 : phénomène lic dans le même commit, tenu si ≤ 2 × R3 ; NON TENU au-delà |
| R6 bis | APRÈS / AVANT COUPÉ (ancienne sortie servie) — publié comme **changement de sortie attendu**, attribué à la référence lic (SDPA par blocs de clés : la passe coupée 7 936 + 17 n'est pas la passe dense) | court ≈ 0,74, ids divergents ; long ≈ 0,031 | court < 0,29 (dans le témoin : alors la coupe ne changeait rien et R4/R5 sont à relire) |
| R7 | régime | AVANT `prefill=bf16(coupé@256)` ; APRÈS `prefill=bf16` sans coupe ; exil MLP égal entre bras d'une chaîne (8 court, 45 long) | exil différent : comparaison contaminée |
| R8 | durée | ≤ 20 min (6 chargements, le premier 60-240 s) | > 25 |

Issue nommée à la première écriture : contre l'ancienne sortie servie, R6 court était prédit NON TENU (0,74 > 0,29).
**Décision chef (01/10, avant la prise, commit à part) : la référence est la passe dense non coupée.** La coupe à la frontière d'instantané
était un artefact du mauvais prédicat, pas un comportement à préserver ; le Δ 0,737 avec/sans coupe est le phénomène lic (le SDPA dépend de la
longueur totale des clés), pas une faute du correctif. R6 se juge donc contre un témoin SANS coupe, même commit (arbre témoin) et même passe ;
l'écart entre l'avant coupé et l'après (R6 bis) est publié comme changement de sortie attendu, chiffré, attribué à lic. Deux prises de plus
(avant sans coupe, court et long) : durée R8 portée à ≤ 20 min (6 chargements).
Autres issues : R2 > 0 ⇒ diagnostic faux, tout s'arrête ; R4 faux ⇒ autre changement de moteur, bissection avant tout verdict ; exil ≠ ⇒ rejeu.

## (b) Débit b=12 ABBA — `scratchpad/poste6-g9m-retenu-b12.sh`
Instrument : banc decode de poste2 (`poste2-w-21-09/scratchpad/banc-llamacpp-16-09.py`, BANC_URL vers un `acvram serve` sous carte.sh, invite 256
ids, BANC_SLOTS=12,12,12 — premier palier jeté, moyenne des deux autres —, BANC_JETONS=256, fenêtre ≥ 10 s, 7 passes courtes), serveur
`--max-batch 12 --max-model-len 4096`, spéculation par défaut (`ngram`), ordre A1 B1 B2 A2 (A = avant, B = après), même alias que (a). Effet 3
seul : AVANT, gemma « hybride » ⇒ `_plain_decode` dès que > 1 séquence (pas de lot spéculatif) ; APRÈS ⇒ décodage spéculatif par lot comme un dense.

| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| D1 | débit B/A (`jetons_s`, moyenne des paliers 2-3, A = moyenne A1/A2, B = B1/B2) | **+0 à +8 %** (scellé poste2 : ngram b=1 +1,5 / +3,3 %) ; TENU si ≥ 0 | **≤ −2 % : BAISSE** — l'issue qui me gêne : ngram accepte peu sur du texte libre et la vérification par séquence coûte plus que les jetons acceptés en lot ; alors proposer `--speculative none` par défaut pour gemma (effets 1-2 acquis indépendamment) ; > +10 % hors fourchette, nommé |
| D2 | dispersion A1/A2 et B1/B2 | ≤ 3 % | > 3 % : B/A illisible sous 3 %, rejeu |
| D3 | refus : HTTP ≠ 200 (`erreurs_finales_parseur` du banc), lignes « refus » au journal, `sequences_tronquees_budget` | 0 des deux côtés | B en a et A non : le lot spéculatif (k+1 jetons par pas) déborde le budget KV — à nommer |
| D4 | preuve de l'effet 3 : `/metrics` `proposed_tokens` | A = 0 (`_plain_decode`), B > 0 ; `acceptance_rate` B 0,2-0,6 | B = 0 : D1 ne mesure rien, la pièce n'est pas prise |
| D5 | durée | ≤ 24 min (4 bras × 5-6 min) | > 30 |

Carte : `nvidia-smi --query-compute-apps` relevé au début et à la fin de chaque script (`releve-*.txt`) ; les deux scripts ≤ 30 min chacun, sous
carte.sh par bras ; aucun pytest pendant.
