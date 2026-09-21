# chef — Reprise 21/09 08 h 59 : tout ce qui reste, dans l'ordre, jusqu'à l'objectif

Utilisateur (08 h 49 → 08 h 59) : groupe = **chef** (ex-poste7 : décide, scelle, fusionne, tient ETAT et le lien utilisateur), **poste1** (moteur), **poste2** (mesure, seule main sur la carte), **poste4** (neuve : ordonnanceur b=12 + MoE, branche `poste4`), **poste3** (neuve : conversion, poste, parc, .deb, VM ; hérite poste9 et la part mécanique de chef). chef et poste9 absorbés. Mot d'ordre : « tout ce qui n'a pas été fait, puis tout le reste dans l'ordre, sans erreur, jusqu'à ce que tout soit mesuré et l'objectif atteint ».

Objectif initial, état (poste 1030, `comparatif-cinq-moteurs-17-09`) : b=1 380,8 t/s (llama.cpp 323,6, vLLM 290,6) **tenu** ; prefill 22 707 (vLLM 21 054) **tenu** ; **b=12 1 378,6 contre vLLM 1 596,1 : non tenu** ; TensorRT-LLM : non mesuré ; J/jeton à horloge égale contre les trois : à refaire après 0.6.34. L'objectif est atteint quand ces quatre cellules sont tenues et publiées avec en-tête de poste.

## File unique (une prise de carte à la fois, poste2 ; à sec en parallèle)

| # | pièce | qui | porte / réfutation |
|---|---|---|---|
| 0 | trou 5 min : tests processeur (commande-unique ×5, poste3) | poste2 déclare | passed/failed dans les carnets |
| 1 | n = 60 kv int8/bf16 Qwen3-VL-2B | poste2, 1 min | géo − 2 SE > +1 % → kv=bf16(vision) en 0.6.35 |
| 2 | k48 : capture des graphes après le clamp ; i8c : libération entre essais ; commande unique | poste1, à sec | k48 tient 31744 ; i8c ≥ 12288 ; imprimé == exécuté |
| 3 | 4 alias 0 × 500 sur le SHA final → § 1b commande exacte → GLM b=1 charge < 1 | poste2 | −2 % sur 168,4 t/s = défaut |
| 4 | .deb 0.6.34 sur ce SHA, sha256, `sudo dpkg -i` (utilisateur) ; 19/19 rejoué installé | poste3, poste2 | tout 500 silencieux = défaut |
| 5 | colonnes ctx règle 2b, modeles-a-jour saute si .qui, verifier-contexte 0 FAUX | poste3 | colonne ≠ ligne servie = FAUX |
| 6 | VM parc nspawn → --decoder sous carte (5 min) → feu vert parc → installation | poste3, poste2 | dépendance manquante ou chemin machine = faux |
| 7 | P3 (3)(4) Qwen3-VL-30B contre l'AWQ déquantifié, duel vLLM AWQ | poste1 instruments, poste2 2 prises | « non établi pire > 3 % » à 2 SE |
| 8 | scellé b=12 (poste4) → profil sur trou ≤ 5 min → correctif → cellule b=12 | poste4, poste2 | ≥ 1 596 t/s à horloge égale, sortie au bit inchangée |
| 9 | qualité nvfp4 31B par bande (scellé poste1 05 h 28, amendé) | poste2 | T tenu sinon rien ; défaut attendu KL max |
| 10 | TensorRT-LLM : cellules b=1/b=12/prefill/J à horloge égale, même poste | poste3 installe, poste2 mesure | acvram ≥ sur les quatre ou écart nommé |
| 11 | J/jeton contre llama.cpp, vLLM, TRT-LLM (régime éco 2700 et 5090 pleine) | poste2 | publié avec en-tête ; objectif = tenu 4/4 |
| 12 | poste THP/EPP (dernier : change la signature) ; C9 119B et bf16 30B (60 Go) = questions à l'utilisateur | poste3 ; chef | — |

Règles inchangées : scellé avant mesure, verrou `carte.sh`, aucun pytest hors « trou : n min », pointeurs d'une ligne, verdict sept lignes, heures lues sur `date`. Un poste qui attend plus d'un tour le dit à la chef.
