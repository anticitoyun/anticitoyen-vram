# Banc énergie 4 moteurs (protocole 2.5), Coder b=12/b=1 — llama.cpp seul TENU au protocole strict, acvram/vLLM/trtllm rejetés — 22/09 (poste2)

* instrument : `scratchpad/poste4-b12-21-09/chaine-energie-4moteurs.sh` (poste4, sha final `a5cbca1b` puis `da83a843` — mes 2 correctifs + les 3 de poste4 fusionnés), `resumer-energie-4moteurs.py` (protocole 2.5 : rejette sd>10 %, `invalidations`≠"aucune" [throttle/bridage inclus], écart horloge par PAIRE MIROIR de l'ordre imbriqué >3 % — b=1 sans paire, horloge non jugée)
* commit : main/poste2 `10c14b0e` au moment de la mesure
* régime : ordre imbriqué A L V T T V L A (b=12), puis A L V T (b=1, un passage), 4 serveurs relayés sur la même carte, un seul verrou
* incident préalable (avant cette prise) : ma **1ʳᵉ tentative de chaîne** (PID 1338030), « arrêtée » par un `kill` insuffisant (le wrapper `carte.sh` meurt mais le script bash enfant survit, détaché — vLLM 18,3 Go restait actif hors verrou), a **perturbé le créneau b=8+KL trtllm de poste3** vers 12:30 (`httpx.ConnectError` côté client, VRAM partagée). Nommée, tuée avant cette 5ᵉ tentative (`kill 1338030 1371657 1371810`).
* mesuré (`protocole25.tsv`, 12 fenêtres, jugées par le protocole strict, PAS ma lecture des JSON bruts) :

| moteur | fenêtres valides | j/jeton médian | t/s médian | rejets |
|---|---|---|---|---|
| **llamacpp** | **2/3** | **0,2157** | **1022,6** | b=1 rejeté (bridage puissance) |
| acvram | 0/3 | — | — | **bridage puissance actif sur les 2 fenêtres b=12** (939-1613 t/s bruts, plausibles mais invalidés par le protocole) ; b=1 rejeté (sd 12,5 %>10 %) |
| vLLM | 0/3 | — | — | **0 jeton décodé sur les 3 fenêtres** (bug nommé ci-dessous) |
| trtllm | 0/3 | — | — | **0 jeton décodé sur 2/3 fenêtres** ; 3ᵉ (T2) « serveur jamais prêt » (timeout au 2ᵉ chargement, signal distinct non diagnostiqué) |

* **défaut nommé, non corrigé (vLLM et trtllm)** : `chaine-energie-4moteurs.sh:fenetre()` passe `BANC_MOTEUR="$moteur"` (donc littéralement `vllm` ou `trtllm`) au client `banc-llamacpp-16-09.py` — ce nom pilote aussi le CHOIX D'ENDPOINT HTTP dans le client (`if MOTEUR == "acvram": /v1/completions [OpenAI] else: /completion [natif llama.cpp]`) : vLLM et trtllm-serve n'implémentent que l'endpoint OpenAI, donc chaque appel échoue silencieusement (0 jeton, pas de crash) — **exactement le même bug que j'ai diagnostiqué et corrigé ce matin dans mon script `nsys-trtllm-b12.sh`** (fix : `BANC_MOTEUR=acvram` toujours, quel que soit le moteur réel mesuré, convention déjà utilisée par `cellule.sh` de poste3).
* verdict : **TENU uniquement pour llamacpp** (2 fenêtres b=12 propres, 0,2157 J/jeton net médian, 1022,6 t/s médian). acvram REJETÉ par le protocole (throttle puissance, un défaut d'environnement du poste, pas de l'instrument — cohérent avec les bridages vus ailleurs aujourd'hui sur ce même GPU). vLLM et trtllm NON MESURÉS (bug d'endpoint nommé, correctif connu non appliqué sur ce script).
* durée : ~34 min de carte sur cette 5ᵉ prise (12:45-13:19), + ~10 min de diagnostic/nettoyage des chaînes orphelines

## Suite
Correctif `BANC_MOTEUR=acvram` (fenetre()) à appliquer par poste4 si un rejeu vLLM/trtllm est voulu ; T2 « jamais prêt » à investiguer séparément (timeout ou VRAM résiduelle du 1er chargement trtllm). acvram : le throttle puissance récurrent sur ce poste (déjà vu nsys trtllm, cellule A/V) mériterait sa propre pièce si l'énergie acvram doit être publiée proprement.
