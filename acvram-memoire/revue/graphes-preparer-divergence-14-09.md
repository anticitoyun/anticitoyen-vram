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
