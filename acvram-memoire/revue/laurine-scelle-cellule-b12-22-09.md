# Scellé — cellule b=12 officielle (Coder, graphe+épinglé) contre vLLM, 6 fenêtres intercalées (Laurine, 22/09, à sec)

**Arbre** : `main` avec les deux défauts déjà bascule (levier 1 `sampler=graphe`, levier 2 `rapatriement=epingle`) — aucune variable posée, la ligne de régime seule fait foi (`sampler=` et `rapatriement=` dedans, vérifiés avant chaque fenêtre).

**Harnais** : même modèle/paramètres que la cellule 1 540 du 21/09 et que vLLM 1 596 (`verdict-t1-poste-1030-20-09`) — Coder `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`, b=12, invite 256, `max-model-len 2304`, `-lgc 2700`, `ACVRAM_CPUS=0-15`. acvram via `banc-llamacpp-16-09.py decode` sur un `acvram.cli serve` relancé à chaque fenêtre (comme les chaînes ABBA) ; vLLM via `scratchpad/decode-vllm-17-09.py` (`BANC_SLOTS=12`, `BANC_CTX=2304`, `BANC_MODELE=` le checkpoint W4A16 Marlin déjà utilisé au T1) — **rejouable, vérifié présent** (`/opt/ia/vLLM/.venv/bin/python`, checkpoint sur disque) ; si l'un des deux manque à l'exécution, Manon le nomme et retombe sur la référence figée 1 596,1 t/s du T1 pour les fenêtres manquantes, sans reconstruire un chiffre.

**Ordre, 6 fenêtres ≥ 20 s** : acvram A1 / vLLM V1 / acvram A2 / vLLM V2 / acvram A3 / vLLM V3 (le témoin vLLM encadre chaque paire acvram, même discipline que l'ABBA des leviers 1/2 — bracketing, pas un bloc puis l'autre).

**Par fenêtre** : `nvidia-smi -i <carte> --query-gpu=clocks.sm --format=csv,noheader,nounits` × 3 échantillons pendant la fenêtre, médiane retenue ; garde de charge étrangère instantanée (`ps -eo pcpu,args` hors {serve, banc, LLM vLLM, Xorg/cinnamon, ps} > 50 % → fenêtre invalidée) ; `nvidia-smi --query-compute-apps` avant/après toute la chaîne, PID hors verrou = prise invalide.

**Prédiction** : médiane acvram 1 590-1 660 t/s (recoupe le levier 2, médiane observée 1 624,3) ; **réfuté si médiane acvram < 1 596** (vLLM T1) — objectif « devant vLLM » non tenu. Écart horloge médiane A(acvram)/V(vLLM) par paire encadrante ≤ 3 %, sinon la paire est invalidée (pas toute la chaîne) et une paire de repli est jouée si le temps le permet.

**Ce qui invalide une fenêtre** : échantillon de charge étrangère > 50 % ; watts moyens > 395 (dérive de régime) ; `eager_raisons`/repli non vide côté acvram ; `fenetre_valide=false` côté vLLM (durée < 20 s, JSON) ; écart horloge > 3 % dans la paire qui l'entoure ; ligne de régime sans `sampler=graphe`/`rapatriement=epingle`.

**En-tête TSV publié** (une ligne par fenêtre) : `bras nom jetons_s watts horloge_med charge_etr_max valide`.

durée : 0 min de carte (note à sec)
suite : Manon joue la chaîne (protocole déjà reçu de la Maîtresse) ; à réception, verdict 7 lignes de ma part comme pour les leviers 1 et 2.
