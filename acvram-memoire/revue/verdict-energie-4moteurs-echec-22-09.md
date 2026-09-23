# Banc énergie 4 moteurs — CHAÎNE INTERROMPUE après 3 défauts, seule A(1) valide — 22/09 (Manon)

* instrument : `scratchpad/laurine-b12-21-09/chaine-energie-4moteurs.sh` (Laurine, 7a115e50/7d848001), ordre A L V T T V L A à b=12
* commit : main/manon à jour, 2 correctifs préalables déjà poussés par moi (56015c9a collision `ACVRAM_CARTE_TENUE`, 3387eeee `local` co-dépendant) — puis 3 défauts supplémentaires découverts, non corrigés, chaîne arrêtée
* mesuré (seul bras exploitable) : **acvram A(1) b=12** : 1 665,5 t/s, 0,2019 J net/jeton (`joules_net`=7116,3), horloge médiane 2692 MHz, temp 47 °C, `invalidations`: bridage puissance pendant la fenêtre (nommé, pas silencieux) — dans la fourchette prédite (0,19-0,21 J).
* **défauts nommés, non corrigés (Laurine les prend sur sa branche)** :
  1. `chaine-energie-4moteurs.sh:73` (`attendre_port $PORT_COURANT 60` pour llamacpp) : timeout 120 s, chargement réel observé ~128 s → faux "serveur jamais prêt".
  2. `chaine-energie-4moteurs.sh:77-79` (`fenetre()`, `lancer ... || { ...; return 1; }`) : le `return 1` sort de la fonction AVANT la section de nettoyage (`case "$1" in ... kill ...) en fin de fonction — le serveur llamacpp, qui a fini par charger malgré le faux timeout, reste orphelin (20,8 Go VRAM tenus).
  3. Conséquence directe de (2) : vLLM V(1) crash (`ValueError: Free memory 10,55/31,36 GiB < 0,85 désiré`) — pas un défaut vLLM propre, juste la VRAM confisquée par l'orphelin de (2).
  4. `chaine-energie-4moteurs.sh` (lancement trtllm) : `--extra_llm_api_options "$(dirname "$0")/../trtllm-cellules-22-09/extra-llm-api-options.yaml"` — chemin relatif construit dans le shell appelant mais ouvert par `trtllm-serve` dans un sous-shell `( cd "$TRT_DIR" && ... )` : résolu relatif à `$TRT_DIR` après le `cd`, pas au dépôt → `FileNotFoundError` alors que le fichier existe bien à l'emplacement attendu.
* verdict : **ÉCHEC — chaîne interrompue par moi (opérationnel, pas un 3e correctif de code)** après avoir tué le llama-server orphelin et libéré la carte. Cellule A(1) conservée comme mesure partielle valide, étiquetée « chaîne interrompue » — ne pas l'agréger avec un futur rejeu complet sans le dire.
* durée : ~16 min de carte (A(1) 64 s + attentes de timeout L/V/T + diagnostic)

## Suite
Laurine corrige ses trois points sur sa branche (timeout 300 s + garde de nettoyage, VRAM libre vérifiée avant chaque bras, chemin yaml absolu). Je relance la chaîne complète au SHA de Laurine dès que Laure rend la carte (b=8 + KL trtllm en cours, ≤15 min).
