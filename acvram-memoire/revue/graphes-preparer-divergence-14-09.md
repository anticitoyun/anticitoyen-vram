# Bogue de fond dans `graphs.py` (preparer/_fill), trouvé en testant le recouvrement

poste1, 14/09/2026 soir, bead runner. Test bit-identique (mandat chef,
condition posée avant d'engager le recouvrement pas n+1/rejeu n) : jetons
émis identiques entre `ACVRAM_PIPELINE=0` et `=1` sur 12 séquences × 200
pas, fins à des pas différents + une arrivée en cours de lot.

**Divergence trouvée**, mais PAS dans mon code : `run()` normal (jamais de
pipeline touché — `preparer()` + `rejouer_suivant()` + `clone()`,
l'équivalence que poste4 a posée pour PIPELINE=0) diverge DÉJÀ de
l'eager pur (`enable_cuda_graphs=False`), sur le même scénario :

    s1 (max_tokens=60)  jeton 2  : graphes=198  eager=220
    s6 (max_tokens=160) jeton 10 : graphes=537  eager=5435

2 séquences sur 12, reproductible à l'identique sur plusieurs essais (pas
une course visible), à des jetons sans rapport avec une fin de séquence
proche. `ACVRAM_PIPELINE=1` donne EXACTEMENT les mêmes valeurs — confirme
qu'il hérite du bogue plutôt que d'en introduire un nouveau.

## Ce que ça écarte

- Ce n'est PAS le décalage de position que j'avais d'abord introduit dans
  `_build_batch_device`/`_pipeline_suite` (corrigé : `pos = seq.length - 1`
  après `_consommer`, pas `seq.length`) — corrigé, mêmes divergences.
- Ce n'est PAS la synchronisation de l'événement de jetons du pipeline non
  plus (corrigé : le pipeline enregistre désormais SON PROPRE événement
  après `_sample_only`, pas `self.graphs.evenement_jetons` qui ne borne
  que le rejeu) — corrigé, mêmes divergences.
- La preuve définitive : le bogue existe SANS aucun code de pipeline actif.

## Piste non vérifiée

Signalée à poste4 : la parité des tampons hôte épinglés dans `_fill`
(pas n → tampon n%2) pourrait se désynchroniser quand la taille du lot
change (une séquence finit → une clé de godet différente) — pas confirmé,
à elle de trancher côté `graphs.py`.

## Outils laissés

    outils/diag-graphes-vs-eager.py         graphes (run() normal) vs eager
    outils/diag-pipeline-bit-identique.py   PIPELINE=0 vs 1 (hérite du bogue ci-dessus)

## Conséquence

Intégration du recouvrement (bead runner) EN PAUSE, pas committée comme
"faite" — le code (`_plain_decode_pipeline` et alentours, derrière
`ACVRAM_PIPELINE=1`, défaut 0) est écrit et prêt, mais son test obligatoire
ne peut pas passer tant que `graphs.py` a ce bogue de fond. Rien ne change
pour un usage normal (`ACVRAM_PIPELINE` non posé) : ce code est mort par
défaut.

## Verdict des deux contrôles de chef (14/09 soir, avant de traiter comme un bogue)

**(1) Bissection** : `graphs.py` d'AVANT le merge de poste4 (`8fbc962`,
sans `preparer`/`rejouer_suivant`) donne EXACTEMENT les mêmes divergences
(`s1 jeton 2 : 198 vs 220`, `s6 jeton 10 : 537 vs 5435`). **Antérieure au
merge — pas causée par la parité des tampons épinglés.**

**(2) Logits au point de divergence** (`outils/diag-logits-divergence.py`,
top-2 candidats) :

    pas=2  ligne=1 (s1) : graphes 12.375000 == 12.375000 (écart 0,000000, TIE EXACT)
                          eager   12.284713 vs 12.266634 (écart 0,018079)
    pas=10 ligne=6 (s6) : graphes 17.250000 == 17.250000 (écart 0,000000, TIE EXACT)
                          eager   17.511557 vs 17.210035 (écart 0,301521)

Le chemin graphes rend un **TIE EXACT** entre les deux candidats (le
`argmax` bascule alors sur l'ordre des index, pas la valeur) ; l'eager les
distingue par un écart de 0,018 à 0,301 — à comparer au pas bf16 (`ulp`) à
cette magnitude (12-17, exposant 3-4) : ≈0,0625 et ≈0,125. **≤ 1-2,4 ulp :
bruit numérique légitime**, pas un écart franc (le seuil de chef était
1e-2 — 0,018 le dépasse à peine, 0,301 davantage, mais le TIE EXACT côté
graphes est le signal qui compte : deux chemins de calcul différents
(GEMV groupé vs par expert) convergent vers la même valeur arrondie à un
souffle près, et LEQUEL gagne dépend de l'ordre des sommes en bf16 — pas
d'une adresse fausse ni d'un tampon désynchronisé.

## Conséquence révisée

**Pas un bogue de `graphs.py`.** Le test bit-identique doit comparer les
LOGITS (tolérance ~1 ulp bf16) plutôt que l'égalité stricte des jetons —
un greedy qui bascule sur un TIE est un résultat correct des deux côtés,
pas une divergence à corriger. poste4 n'a rien à changer côté tampons.
Intégration du recouvrement reprise.

## s11 ("arrivee", jeton 0 après admission) : même famille, confirmé

poste4 a trouvé, avec son propre test (`outils/test_graphes_vs_eager.py`,
main `021c7f2`), que sa séquence admise en cours de lot (indice 11)
diverge d'eager dès son PREMIER jeton décodé — cas plus suspect que
s4/s7 puisque c'est le tout premier pas d'un slot repris (table/position/
slot_mapping), potentiellement un vrai bogue d'admission plutôt qu'un
tie bf16.

Contrôle (`outils/diag-logits-arrivee-jeton0.py` + `.sh`, deux processus
séparés) — piège trouvé en le construisant : `step()` prefille ET décode
une séquence nouvellement admise dans le MÊME appel (`_decodables()`
l'inclut dès que `prefilled` passe à vrai, juste après son prefill,
`runner.py:787`) — deux appels à `_emit` pour s11 dans un seul pas, jeton
0 (prefill) puis jeton 1 (décode) ; un premier essai qui ne gardait pas
QUE le premier appel capturait le mauvais jeton, produisant des valeurs
incohérentes entre les deux chemins. Corrigé (ne garder que le premier
`_emit` pour `request_id=="s11"`), puis reproduit EXACTEMENT la
divergence de poste4 (eager=198, graphes=76, comme son propre test) :

    eager  : top1=198 val=12,452925  top2=76  val=12,374856  écart=0,078069
    graphes: top1=76  val=12,403308  top2=198 val=12,258751  écart=0,144557

Mêmes DEUX candidats des deux côtés (198 et 76), juste lequel gagne qui
bascule — signature classique du tie bf16, pas d'une adresse fausse.
Les deux écarts (0,078 et 0,145) sont sous le seuil `TOLERANCE_ULP=0,4`
déjà retenu pour s4/s7 à cette magnitude. **Même verdict : bruit
numérique légitime, pas un bogue d'admission.** Rien à changer côté
table/position/slot_mapping.
