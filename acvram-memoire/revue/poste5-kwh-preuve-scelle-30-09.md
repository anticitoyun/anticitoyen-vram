# kwh — preuve réelle : scellé (poste5, 30/09, écrit AVANT la prise)

Prise : `scratchpad/poste5-kwh-preuve-30-09/prise.sh` sous carte.sh mesure (ACVRAM_DUREE_MAX 900 s), ACVRAM_POSTE=poste5,
lanceurs du dépôt sans verrou service. Étapes : (1) acvram qwen3-coder-30b nvfp4 ; (2) acvram qwen25-coder-3b, qui
remplace (1) (arrêt + attente de mort + garde) ; (3) arrêt EXTERNE de (2) par SIGTERM sans attendre, puis vllm-serveur
qwen3-coder-30b FP4 32 768 (seule la garde protège) ; (4) llamacpp-serveur qwen3-4b Q4_K_M 32 768, qui arrête vLLM
lui-même ; (5) arrêt de llama.cpp, carte rendue. Mémoire de la carte 0 relevée toutes les 250 ms.

Prédictions :
* P1 — les 4 lancements rendent 0 ; aucun « mort au démarrage », « out of memory », « OOM », « refusé » dans 1-4.log.
* P2 — étape 3 : une ligne « vllm : carte rendue en N s », 1 ≤ N ≤ 20 (B vient de recevoir SIGTERM, il meurt et rend
  sa VRAM après) ; dans memoire.csv, la VRAM de B retombe AVANT que celle de vLLM monte.
* P3 — étape 4 : « llama.cpp : carte rendue en N s » ou rien (llamacpp-serveur a déjà attendu 6 s après son kill) ;
  aucun OOM.
* P4 — prise ≤ 15 min ; compute-apps fin = début (appoint 8081 carte 1 seul).
FAUX si : un OOM de démarrage (P1), une refus « carte non rendue », ou vLLM qui alloue avant la chute de B (P2).
NON PROUVÉ (pas VRAI) si l'étape 3 n'affiche aucune ligne de garde : la garde n'aurait pas été exercée.
