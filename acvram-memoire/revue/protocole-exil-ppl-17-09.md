# Protocole — l'exil change-t-il la sortie ? (Sage, sage-exil-ppl-89-priorite-17-09 ; règle 9 : une voie qui change la sortie est un bogue moteur)

objet : `models_acvram/Qwen3-4B-srcgguf-nvfp4` (36 couches, 3,2 Go : 30 s par point), tranche 0 de `corpus-prive/tranches-glm`, `ppl-acvram-17-09.py`, régime classé, arbre laure 27f2e8d (main ≥ 7ca777d).
bras : R = résident (défaut) ; E1 = `ACVRAM_EXIL_COUCHES=18` (loader.py:1636, exil forcé sans dépendre de la taille) ; E2 = E1 + `ACVRAM_POOL_SYNC=1` (layers.py:363 : copies sur le flux de calcul) ; E3 = E1 + `ACVRAM_SANS_PRECHARGE=1` (layers.py:455) ; E4 = `ACVRAM_EXIL_COUCHES=1`.
scellé (Sage) : voie juste ⇒ E1..E4 = R ± 0,002 (géo de la tranche). Prédiction : E1 ≠ R et E2 = R (course flux/événements layers.py:388-399). Si E1 = R : le défaut est propre au 70B ou à la taille → seule une fenêtre 70B POOL_SYNC + PPL (~25 min) est alors autorisée.
falsification : E1 = R ET 70B toujours à +89 % ⇒ la cause n'est pas la voie d'exil générique ; E1 ≠ R ET E2 ≠ R ⇒ pas la course des flux (autre chose dans le chemin exilé : déquantification, préchargement).
menus : toute ligne exilée porte « exil : sortie fausse (+89 % PPL, cause ouverte) » tant que non clos.
ordre carte : ce test (≤ 10 min) après l'essai 70B déjà en cours (lancé 10:38, avant réception), puis TRT-LLM.
