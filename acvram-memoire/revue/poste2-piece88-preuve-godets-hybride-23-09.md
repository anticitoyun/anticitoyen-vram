# Pièce 88 (reste a) — preuve sur carte, sortie graphes==eager, hybride GDN, godets {1,2,8,16} (poste2, 23/09)

instrument : `scratchpad/poste2-p88-23-09/preuve-godets-eager.py` — deux moteurs INDÉPENDANTS (un
  `enable_cuda_graphs=True`, un `False`, aucun état partagé), chacun laissé tourner par son propre
  `step()` normal ; logits captés au vol sans appel supplémentaire (monkeypatch sur `GraphRunner.run`
  côté graphe, sur `Model.forward` côté eager), comparés par KL (seuil `conftest.assert_logits_proches`,
  1e-2). `ACVRAM_PIPELINE=0` imposé (sous pipeline=1, le décodage à un pas de retard ne passe pas par
  `GraphRunner.run` de façon interceptable — preuve de justesse, pas de débit, le pipeline n'est pas en
  cause). warm_graphs au godet lui-même, REGLES § 3.
commit : 01476adc
régime : Agents-A1-4B-kimi-nvfp4 (attention linéaire GDN, hybride), max_model_len=512, éco libre
  (mesure à sec de justesse, aucune horloge posée), 8 pas de décodage purs par godet, N_PAS=8
scellé : `scratchpad/poste2-p88-23-09/scelle.md` — sortie graphes identique à l'eager (KL < seuil du
  projet) aux quatre godets ; deux méthodes précédentes explicitement écartées et documentées dans le
  script (comparaison sur état partagé : contamine l'état récurrent, invalide sur hybride ; comparaison
  par argmax de jetons : réintroduit l'instabilité que `test_graphs.py` évite exprès par KL)
mesuré : b=1 KL_max 2,78e-17 ; b=2 KL_max 5,04e-17 ; b=8 KL_max 8,10e-17 ; b=16 (isolé, après un premier
  essai groupé faussé par une accumulation VRAM entre 4 chargements séquentiels dans le même processus,
  `graphes_disponibles=false` à tort) KL_max 1,29e-16 — tous largement sous le seuil 1e-2, à la précision
  machine. `n_replis_eager=0` et `graphes=on` aux quatre godets. `warm_graphs` a bien capturé (2-3
  clés/godet, 11 rejeux) avant les 8 pas mesurés.
verdict : reste (a) de la pièce 88 TENU — sortie identique entre graphes et eager sur un hybride GDN
  réel, aux quatre godets prescrits par REGLES § 3. Aucun défaut de justesse trouvé (la restauration
  d'état de la pièce 88 fonctionne comme annoncé). Parties « injection d'échec activée sur une clé » et
  « ligne de régime graphes=on(refus=…) » : couvertes par les tests unitaires d'poste1 et par cb7c4147
  (poste3, confirmé fusionné), hors scope mesure — pas repris ici, sur accord explicite de chef.
  Reste nommé : le premier essai groupé (b=16 faux-négatif) montre qu'un harnais qui charge 4+ moteurs
  successivement dans UN processus peut épuiser la VRAM en silence (`Engine.__init__` retombe sur
  `e.graphs=None` sans erreur) — un artefact d'instrument, pas un défaut moteur ; à garder en tête pour
  tout futur script qui enchaîne plusieurs chargements sans `del`+`empty_cache` suffisant.
durée : trois essais de méthode (rejetés, aucune carte gaspillée au-delà de leur propre prise : 2,9 à
  7,7 s par godet) + essai retenu (4 godets, ~8 s de carte) + b=16 isolé (3,6 s) ; verrou journal `tenue=`
