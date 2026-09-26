# 262 — TTFT servi à 12 séquences sur 0.7.0 : 0,24 s sans instrument parasite ; le duel en mesurait 0,80 (poste1, 26/09)

* instrument : `outils/gpu/mesure/duel-moteurs.py` tel quel (prise 1) ; `scratchpad/poste1-p262-26-09/` — `serveur-trace-262.py`
  (horodatage par requête : http, submit, admission, fenêtre 179, pas, premier jeton), `client-ttft-262.py` (tour TTFT du duel,
  horodaté ; VEILLEUR=1 = le veilleur /metrics du duel), `analyse-262.py` ; prises `prise1-duel.sh`, `prise2-trace.sh`, `prise3-veilleur.sh`
* commit : poste1-262 854492261 (prises 1-2), efb4f67c7 (prise 3) ; base origin/fusion-070 180876914 = moteur 0.7.0 (main v0.7.0
  3d2993ccc n'en diffère que par deux listes de variables sans effet sur le calcul, `cli.py`, `regime.py`)
* régime : Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4 (acvram) / Q4_K_M GGUF (llama.cpp officiel, -np 1, KV q8_0), -lgc 2700
  pour les deux, ACVRAM_ECO=off, spéculation coupée, max-batch 16, cache de préfixe au défaut, cpu-safe 100
* scellé : `scratchpad/poste1-p262-26-09/scelle.md` (f6dc0d081, avant toute prise) + addendum 1 (efb4f67c7, avant la prise 3)
* mesuré : duel A L L A × 7 tours ; trace 7 tours à 12 + 5 à 1 ; veilleur A B B A × 5 tours, même serveur
* verdict : le « TTFT à 12 » d'acvram = **0,242 s** ; le duel le triple (0,80 s) par son propre veilleur : **/metrics dure 343 ms
  et bloque la boucle HTTP**. Contre llama.cpp -np 1 (0,652-0,662 s, duel) : acvram **2,7 × plus rapide**, pas 2,29 × plus lent
* durée : prévu 10 + 5 min ; tenu (`carte.sh` journal `tenue=`) prise 1 469 s, prise 2 20 s, prise 3 28 s

## Chiffres
| mesure | acvram | llama.cpp |
|---|---|---|
| duel (veilleur /metrics 20 Hz), mur du tour, médiane de 7 | 0,801 / 0,811 s (préfill serveur ≈ 0,56 s) | 0,662 / 0,652 s |
| même tour SANS veilleur (prise 3, A) | **0,242 / 0,242 s** | non mesuré (son /metrics ne répond pas en 343 ms : hors cause probable) |
| même tour AVEC veilleur (prise 3, B, même serveur) | 0,782 / 0,789 s ; /metrics 343 ms médian (max 357) | — |
| débit décodage agrégé à 12 (duel) | 1 102 / 1 092 t/s | 301 / 301 t/s |

## Découpe (prise 2, sans veilleur, sous trace : mur 266,6 ms)
| poste (dernière requête du tour, médiane) | ms | part du mur |
|---|---|---|
| entrée (client → http) | 1,5 | 0,6 % |
| gabarit + tokeniseur | 3,0 | 1,1 % |
| file (dont fenêtre 179 : 5,5 ms, et attente du 1er pas de préfill) | 41,8 | 15,7 % |
| préfill (fin d'admission → premier jeton) | 215,5 | 80,8 % |
| sortie + retour | 2,0 | 0,8 % |
Pas de préfill par tour : 2, 2, 1, 2, 2, 2, 1 — la rafale de 12 fils arrive en deux vagues, la fenêtre de 5 ms coupe 5 tours sur 7.
Témoin solo (1 requête ≈ 470 jetons) : 38,5 ms, dont préfill 36,3.

## Prédictions scellées
* P1 (mur 0,40-0,80 s ; ratio 0,6-1,3) : tenu par le duel (0,80, 1,23), FAUX pour le moteur (0,24, 0,37) — l'instrument était faux.
* P2 (préfill ≥ 60 %) : TENU, 81 %.  P3 (un seul pas dans ≥ 5 tours / 7) : FAUX, 2 / 7 — la file (16 %) est le 2e poste.
* P4 (gabarit ≤ 60 ms sur le mur) : TENU, 3 ms.  P5 (sortie + retour ≤ 5 ms) : TENU.  Témoin solo : TENU (38,5 ms).
* Addendum 1 (B ≥ 0,55 s, A ≤ 0,32 s) : TENU.

## Ce qui en suit
1. Tout TTFT d'acvram mesuré par `duel-moteurs.py` à CONC > 1 depuis l'entrée du veilleur dans le fichier suivi (9e6725b10,
   11/09 ; l. 139-146) porte le coût de /metrics — à requalifier ; le chiffre moteur est celui de la prise 3 A. Le « × 2,29 » du
   10/09 est antérieur à ce commit : je ne sais pas s'il portait le veilleur, je ne le lui impute pas.
2. **Défaut de service réel** : `/metrics` est un `async def` qui coûte 343 ms dans la boucle HTTP (`app.py:984-1029` : six
   `engine.regime()`, `_regime_ligne()`, énergie). Toute console ou supervision qui le lit gèle le service pendant ce temps. Pièce
   proposée : le rendre `def` (fil de travail) et/ou calculer `regime()` une fois — puis nommer le poste lent par cProfile, à sec.
3. Levier TTFT restant : la file (42 ms) — la rafale de 12 arrive en deux vagues ; une fenêtre d'admission plus longue en rafale
   (ex. 10-15 ms) ferait un seul pas de préfill. À mesurer contre son coût en solo (règle 179 b).
4. Le bras llama.cpp -np 1 sert en série ; une comparaison juste à 12 opposerait llama.cpp -np 12 — non mesuré ici.
