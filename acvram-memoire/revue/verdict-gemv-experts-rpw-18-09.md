# Verdict — campagne RPW de la GEMV groupée des experts (Laurine a850937) : min(rpw=2, 4) = **7,01 ms/pas** > 6,7 → **FAUX de peu** ; bit-exact tenu (banc 20/20, ppl-decode-kv Coder 1 024 NLL identiques) ; gain réel −11 % sur le témoin

instrument : `outils/banc-gemv-experts-18-09.py` un processus par valeur (rpw=2, 4, témoin 1 en fin, 06:31-06:32, `scratchpad/gemv-experts-rpw-18-09/banc-rpw*.{log,json}`) ; bit-exact in situ : `ppl-decode-kv-17-09.py` Coder 256+1024, `ACVRAM_GROUPED_RPW=4` vs `=1` (`situ.sh`), 1 024 `nll_par_jeton` comparés
commit : 5bb3862 (laure = main a850937), carte sous bridage habituel
régime : banc synthèse E=128 top-8 b=12 K=2048 I=768 48 couches, 20 routages ; in situ régime classé, graphes on
scellé (Sage, sage-gemv-experts-rpw-18-09) : seuil unique 6,7 ms/pas sur min(rpw=2, rpw=4), bit-exact au témoin ; prédiction Laurine rpw=2 6,9-7,4, rpw=4 6,6-7,2 (« non tenu de peu » annoncé)
mesuré : rpw=1 **7,88** ms/pas (1,13 To/s, 63 %) · rpw=2 **7,11** (1,25 To/s, 70 %) · rpw=4 **7,01** (1,26 To/s, 70 %) ; v2 sous RPW : 9,67 / 8,38 / 8,06 (toujours > v1) ; identique au bit v2/v1 20/20 dans chaque processus ; in situ rpw=4 et rpw=1 : PPL **5,5426** toutes deux, 1 024 NLL par jeton identiques
verdict : **FAUX** — 7,01 > 6,7 de 0,31 ms (4,6 %) ; prédiction rpw=4 tenue (6,6-7,2), rpw=2 tenue (6,9-7,4) ; bit-exact tenu ; rpw=4 rend −0,87 ms/pas (−11 %) sur le témoin, sans changer la sortie

## Lecture
- Le levier (activation étagée une fois par bloc de GW_WARPS×RPW lignes) rend 70 % de la bande contre 63 % : le gain vient de moins de relectures d'activations, pas des poids ; il sature entre rpw=2 et 4 (7,11 → 7,01) — le reste du plafond (30 %) n'est pas là.
- Défaut ou non : ce n'est pas mon rôle de décider (règle 9 : équivalence tenue au bit, mécanisme compris, −11 % mesuré) ; si Sage retient rpw=4 en défaut, l'in situ b=12 Coder reste à mesurer (nu et J/jeton) — le banc ne dit rien de l'occupation en présence des autres noyaux du pas.
- v2 reste plus lente que v1 sous toutes les valeurs de RPW ; le chantier v2 n'a plus d'argument.
