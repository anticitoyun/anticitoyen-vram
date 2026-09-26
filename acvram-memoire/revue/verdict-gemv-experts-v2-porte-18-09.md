# Verdict — porte GEMV groupée des experts v2 (poste4 c904806) : **FAUX** — v2 9,66 ms/pas contre v1 7,88 (+23 %), identique au bit 20/20 ; in situ non ouvert

instrument : `pytest tests/test_gemv_experts_v2.py` (9 cas, compilation réelle) puis `outils/carte.sh python outils/banc-gemv-experts-18-09.py` (E=128, top-8, b=12, K=2048, I=768, 48 couches, 20 routages, gate/up + down, rejeu graphe) ; `scratchpad/gemv-experts-v2-18-09/{banc.log,banc.json}`
commit : 8a75b48 (poste3 = main c904806), carte sous bridage habituel
régime : synthèse hors moteur, poids aléatoires quantifiés NVFP4, 65-75 experts distincts par routage (172-199 Mo relus par couche)
scellé (poste7) : v2 ≤ 5,3 ms/pas ET identique au bit → ouvre ; prédiction poste4 v1 6,5-7 ms (1,2 To/s), v2 4,6-5,3 ms (1,55-1,8 To/s)
mesuré : médianes v1 **0,1641 ms/couche → 7,88 ms/pas (1,13 To/s, 63 % de 1,79)** ; v2 **0,2013 ms/couche → 9,66 ms/pas (0,92 To/s, 51 %)** ; gate/up seul v1 0,080 / v2 0,117 ms ; identique au bit 20/20 routages ; v2 plus lent sur 20/20 routages (+18 à +26 %)
verdict : **FAUX** — v2 dépasse 5,3 ms de 4,4 ms et est plus lent que v1 ; la prédiction v1 (6,5-7) est aussi dépassée (7,88) ; porte fermée, pas d'in situ, `ACVRAM_MOE_GEMV=v1` reste le défaut

## Lecture
- Le mécanisme visé (poids relus une fois par ≤ 4 jetons du même expert au lieu d'une fois par paire) ne rend pas de bande : à 65-75 experts distincts pour 96 paires, le facteur de relecture de v1 n'est que 96/70 ≈ 1,37 et une bonne part se sert déjà en L2 ; v2 paie le tri, les blocs meneurs par segment et une occupation moindre (moins de blocs en vol) — le coût fixe dépasse le gain de relecture.
- La bande atteinte par v1, 1,13 To/s, dit que la GEMV experts n'est pas au plafond mémoire (63 %) : la marge existe mais elle n'est pas dans la relecture des poids ; les postes candidats sont l'occupation (nombre de blocs par expert), la lecture des activations et les échelles par bloc.
- Défaut de l'instrument (hors verdict) : `tests/test_gemv_experts_v2.py:35` — dans `_routage`, `eid = topi.reshape(...)` est exécuté avant la branche `fantomes` qui définit `topi` → `UnboundLocalError` sur les 2 cas `fantomes` (2 failed, 7 passed). Rejoué avec la branche corrigée (copie locale jetable, `else` + `eid[-2k:] = -1` après) : 9/9 passent, v2 identique au bit avec jetons fantômes. Correction à porter par poste4.
