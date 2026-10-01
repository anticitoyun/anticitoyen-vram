# kv31b levier 2, étape 1 — preuve carte S1 bis : les morceaux ne tiennent pas 2 × le témoin reprise ; la frontière d'instantané explique le 0,60 de gemma (poste6, 01/10 06:53-07:06, branche poste6-reserve-attention)

instrument : `prise-s1-morceaux-kv31b.sh` (REQUETES=2 sur A1 = témoin reprise, INVITE_FICHIERS, OPTIONS_SERVE, HEAD asserté, relevés) + `s1-morceaux-comparer.py` (REGLES § 4 : P3 = Δ(B, A1) ≤ 2 × Δ(reprise)) ; glouton 32 jetons, logprobs=10, SAMPLER_LENT=1, sans spéculation
commit : f634d5016 (moteur = 72521cb1b + témoin RELIRE_KV inactif)
régime : 5090 seule (4436 sur la 3080 Ti), carte vide au départ (15 Mio) ; gemma-4-31B 4sur6 kv=int8 : 8/60 MLP exilés à 10 240, 45/60 à 20 480 ; Devstral-24B : 0 exilé, graphes on ; ACVRAM_POSTE=poste6, feu chef après la 0.7.17
scellé : `poste6-s1-morceaux-scelle-carte-bis-01-10.md` (Q1-Q7, verdict attendu « critère non tenu » écrit avant)
mesuré : 11 prises, 28-177 s chacune (long A1 : 177 s = chargement + chauffe 20 480), 06:53:23-07:06:23 ; 4 chaînes (court 7 953 jetons, long 17 859, insta `ACVRAM_INSTA_PAS=16384`, Devstral 7 865)
verdict : **critère NON tenu, comme prédit (Q4)** — Devstral : B/A1 Δ 0,041 à la position 0 contre témoin reprise 0,004 (seuil 0,008) : 5 × au-dessus ; gemma : témoin reprise VIDE (0 jeton servi, g9m : Δ 0) et morceaux à 0,088-0,60. **Q5 TENU** : sans coupe (`INSTA_PAS` 16 384) gemma tombe à 0,088 avec les 32 ids identiques — le 0,60 du matin venait de la frontière d'instantané (7 936) × morceaux, pas des morceaux seuls. **Q6 TENU** : long (17 859, 4 morceaux + 1 475) Δ pos 0 0,092, ids identiques sur 20 pas, Δmax 0,22. Q1, Q2, Q3 (0,004), Q7 tenus
durée : 13 min 0 s de carte (prévu ≤ 10 : le long dépasse de 3 min, dit) ; carte rendue 07:06:23, poste5 prévenue

## Chiffres (B contre A1 ; A1 = A2 au bit dans les 4 chaînes ; reprise = même requête rejouée sur le serveur A1)
| chaîne | modèle, ctx | `prefill_morceaux` B | ids B/A1 | Δ lp pos 0 | top-10 communs | Δmax lp (préfixe commun) | témoin reprise (pos 0 / cache servi) | seuil 2 × | P3 |
|---|---|---|---|---|---|---|---|---|---|
| court | gemma 10 240, coupe 7 936 | 3 | diffèrent dès 0 | **0,601** | 5 | — | 0 / 0 jeton (g9m) | 0 | FAUX |
| insta | gemma 10 240, sans coupe | 3 | **identiques 32/32** | 0,088 | 8 | 0,29 sur 32 | 0 / 0 jeton | 0 | FAUX |
| long | gemma 20 480, 45 exilés | 3 | identiques 20 pas | 0,092 | 10 | 0,22 sur 20 | 0 / 0 jeton | 0 | FAUX |
| dense | Devstral 10 240 | 4 | diffèrent dès 0 (quasi-égalité −3,19/−3,15) | 0,041 | 10 | — | **0,004** / 9 712 jetons | 0,008 | FAUX |
Note : sur gemma les ids du seul tenant eux-mêmes changent avec la coupe (A1 court 3750…, A1 insta d3449…), et B court = A1 insta (d3449…) :
la coupe à 7 936 déplace le seul tenant d'un côté, les morceaux de l'autre — même mécanisme (longueur des appels d'attention), cf. lic.

## Lecture
* Le seul témoin reprise mesurable aujourd'hui est Devstral (gemma ne sert pas son cache : g9m, correctif sur poste6-g9m pour la 0.7.18).
  Sur Devstral les morceaux font 10 × l'écart de la reprise (0,041 contre 0,004) : la reprise relit l'int8 pour une ligne, les morceaux pour
  ~3 800 lignes et composent l'erreur sur 40 couches — ce n'est pas la même quantité d'erreur, et le critère de REGLES § 4 le dit.
* gemma : 0,088-0,092 sans coupe, court comme long, ids stables (32/32 et 20/32) ; le 0,60 est un artefact de la coupe d'instantané que g9m
  supprime (plus de `est_hybride` → plus de coupe). Après g9m, gemma aura un témoin reprise réel ET plus de coupe : la chaîne court/long est à
  rejouer telle quelle sur poste6-g9m (prédiction : témoin ≈ 0,005-0,02 avec 60 couches, B ≈ 0,09 → encore 5-10 ×, critère non tenu).
* Pour séparer int8 et chemin sur carte : bras Devstral KV bf16 sur poste1-lic 7450cd5c0 (poste1 : 640 blocs prédits, invite admise) —
  prédiction poste6 : B/A1 bf16 Δ pos 0 ≤ 0,01 (la relecture int8 pèse l'essentiel) ; > 0,02 = le noyau GPU (flash/paginé par longueur) domine.

## Reste
* `ACVRAM_PREFILL_MORCEAU` reste opt-in ; levier 2 étapes 2-4 non ouvertes.
* À l'ordre de chef : (1) Devstral KV bf16 (poste1-lic, 3 prises, 3 min) ; (2) gemma court/long sur poste6-g9m (témoin reprise réel) ;
  (3) si le critère reste hors de portée : seuil absolu scellé par le chef ou abandon de l'étape 1 au profit d'un KV borné sans morceaux.
* Sorties : `scratchpad/s1bis-{court,long,insta,devstral}/` (completion*.json, metrics, journaux, relevés).
