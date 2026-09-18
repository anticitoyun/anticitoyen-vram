# poste7 — Harnais égal : acvram Coder b=1 361,5 t/s (+5 % devant llama.cpp 344,0) et b=12 1 106 (+56 %) ; la revendication « llama.cpp devant à b=1 » se retire avec sa cause (deux harnais, deux contextes), et la règle entre dans REGLES § 4 ; un contrôle à long contexte avant d'écrire « devant » sans réserve (18/09)

Entrée : poste3 ea3981c `verdict-harnais-egal-coder-18-09-b` — machine calme (charge 1,9), contrôle llama.cpp b=1 **344,0** (341,4 + 0,8 %, tenu : la passe de 12 h 39 était la charge) ; acvram serve, même script SSE, `ignore_eos`, 256 + 1 024 jetons exacts : **b=1 361,5 t/s · 0,970 J** (prédiction 275-295 dépassée) ; **b=12 1 106,4 t/s · 0,372 J** (prédiction 1 150-1 200 fausse de −3,8 % : le harnais compte le prefill 12 × 256 dans la fenêtre ; llama.cpp 709,2).

## 1. Ce que le chiffre dit, et dans quel régime

Le −16 % à b=1 n'a jamais été mesuré au même régime : 287,1 venait de `certifie` (en processus, smi, contexte 256 → 2 044), 341,4 du harnais SSE à 256 + 1 024. Au même harnais et au même contexte, acvram est devant de 5 % en t/s. **Chiffre exact hors de son régime (REGLES § 4)** : la comparaison, pas l'un des deux chiffres, était fausse. Réfuté sur moi deux fois : « b=1 perdu pour des raisons GPU » (§ 2 de `poste7-profil-verdict`) reposait sur cette comparaison ; et 275-295 sous-estimait de 20 % ce que le contexte fixe rend.

## 2. Avant d'écrire « devant » : un contrôle qui peut rendre faux (poste3, 3 min de carte)

L'attention paginée coûte +0,21 ms de ctx 300 à 2 000 (verdict profil) ; llama.cpp a sa propre pente. Même harnais, **256 + 1 792 jetons** (fin ≈ 2 048, le contexte des rondes), b=1, les deux moteurs, 20 s chacun. Prédiction : acvram **≥ 0,95 × llama.cpp** (≈ 300 contre ≈ 315 : llama.cpp repasse devant de peu ou égalité) ; issue qui me gênerait : acvram < 0,90 × llama.cpp à 2 048 — alors « devant » ne vaut qu'à contexte court et la ligne le dit. J llama.cpp b=1 et b=12 de cette passe : à publier (le 1,108 du 16/09 vient d'un autre run).

## 3. Écriture (chef, après § 2), à chaque endroit où la revendication a été dite — comparatif, ETAT, `poste7-objectif-b1`, `poste7-reprise-ordre`, menus (poste8)

Ligne du comparatif : « **harnais égal 18/09** (SSE, `energie.py`, `ignore_eos`, 256 + 1 024) : acvram b=1 361,5 / 0,970 J, b=12 1 106,4 / 0,372 J ; llama.cpp b=1 344,0 / J, b=12 709,2 / J » ; « à 256 + 1 792 : … » (§ 2). Cellules moteur `certifie` (287,1 ; 1 198) conservées à part, étiquetées « en processus, contexte 256 → 2 044 ». Retrait daté : « la revendication “llama.cpp devant à b=1 (+19 %)” (17-18/09) comparait deux harnais et deux contextes ; retirée le 18/09, cause nommée ». Revendication Coder : « seul classé ; devant tous les classés à b=1 et b=12, t/s et J, au harnais égal (contexte 1 280 ; à 2 048 : § 2) ; prefill −37 % vs llama.cpp (P1 en cours) ».
REGLES § 4, nouvelle entrée : **deux cellules d'un comparatif se comparent au même harnais ET au même contexte ; le contexte fait partie du régime** (18/09, b=1 Coder −16 % → +5 %).

## Ordre

* poste3 : § 2 (3 min), J des deux moteurs ; puis PPL `-qkvo-i8c` quand poste2 livre.
* chef : § 3 après § 2 ; REGLES § 4 ; ETAT.
* poste8 : menus Coder après § 3.

## 4. Addendum (poste3 694a95a) : « devant » tient à 2 048 et croît (1,094) ; llama.cpp b=12 remesuré 971,1 (pas 709,2) ; et au J brut b=12, llama.cpp est devant

* Contexte : b=1 à 256 + 1 792 — llama.cpp 333,2 (1,190 J brut / 0,969 net), acvram **364,5** (1,010 / 0,735) : rapport 1,094 ≥ 0,95, tenu large. À 1 024 : 1,051. « Devant à b=1 » s'écrit sans réserve de contexte, avec les deux points.
* **llama.cpp b=12 : 971,1 t/s · 0,306 J brut / 0,226 net** aujourd'hui, même harnais, même binaire, machine calme attestée ; le 709,2 du 16/09 ne se reproduit pas et n'a pas de relevé de charge → **trois états : 709,2 = « non reproduit, charge non relevée »**, il ne porte plus la ligne. Une mesure existe en deux exemplaires (REGLES § 4) : **jumelle llama.cpp b=12, 20 s** (poste3) ; 971 ± 3 % → la ligne porte 971,1 et 709,2 reste lisible, étiqueté ; hors bande → les deux sont indécidables et la ligne dit « à refaire ».
* **J brut b=12 : acvram 0,372 contre llama.cpp 0,306 — llama.cpp plus économe de 18 % ; en t/s acvram devant de 14 % (1 106 / 971).** C'est un résultat, il se publie tel quel : à b=12 nous tirons ≈ 411 W (0,372 × 1 106) contre ≈ 297 W pour llama.cpp — le mécanisme est connu depuis le 14/09 (`instr-par-octet` : nos GEMV saturent le plafond 400 W, ×7 d'instructions par octet DRAM). Contrôle avant d'écrire : en-tête des deux bras = **mêmes cartes sommées** (`CUDA_VISIBLE_DEVICES`, garde multi-cartes d'`energie.py`) et J **net** acvram b=12 publié à côté du brut (llama.cpp 0,226 net). Si les cartes diffèrent, la comparaison de J est fausse avant d'être un résultat.
* Revendication Coder, version honnête (après la jumelle) : « seul classé ; **b=1 : devant llama.cpp en t/s (+5 % à 1 024, +9 % à 2 048) et en J (−16 %)** ; **b=12 : devant en t/s (+14 %), derrière en J (+18 %)** ; prefill −37 % (P1 en cours) ». Le J à b=12 devient un poste nommé — **après P1/P2**, pas un chantier de plus maintenant : les leviers connus (eco `-lgc 2100` : J −8,7 % pour t/s −20 %, `verdict-modes-energie`) ne suffisent pas (≈ 885 t/s / 0,34 J, encore derrière sur les deux) ; le levier réel est le noyau de décodage à moins d'instructions, même famille que P1 côté décodage — une ligne à l'utilisateur, pas une promesse.

## Ordre (complète)

* poste3 : jumelle llama.cpp b=12 (20 s) ; en-têtes cartes des deux bras ; J net acvram b=12. Puis P1 dès la réponse AWQ.
* chef : § 3 avec § 4 (971 après jumelle ; J b=12 derrière, dit tel quel) ; REGLES § 4 ; ETAT ; ligne utilisateur « J b=12 : llama.cpp devant de 18 %, cause connue (puissance 411 vs 297 W), poste après P1/P2 ».

## 5. Addendum 1f0afe4 : le contrôle des cartes a rendu faux (acvram sommait [0,1]) ; cellules à cartes égales ; jumelle llama.cpp b=12 hors bande → troisième passe

* **Cartes** : llama.cpp `cartes[0]` sous `carte.sh`, acvram `[0,1]` (client HTTP hors `carte.sh`, donc sans `CUDA_VISIBLE_DEVICES`). Remesuré à `CUDA_VISIBLE_DEVICES=0` : **b=1 366,4 t/s · 0,907 J brut / 0,689 net ; b=12 1 126,4 · 0,347 / 0,282** (24 576 jetons). Le 14/09 avait la même faute (vLLM sommait 5090 + 3080 Ti) ; la garde d'`energie.py` déclare, elle ne compare pas. Remède structurel (poste1, à sec, 20 min) : le harnais du comparatif **refuse d'écrire une ligne dont les deux bras n'ont pas le même `cartes`** dans l'en-tête, et un client HTTP de mesure pose lui-même `CUDA_VISIBLE_DEVICES` = carte servie (lue de `/metrics`), verrou ou pas. Test : deux en-têtes différents → refus.
* **llama.cpp b=12** : 971,1 puis 1 020,1 (+5,0 %, hors ± 3 %) → « à refaire », comme écrit : **troisième passe de 20 s** dans la fenêtre de poste3 ; la ligne publie les trois valeurs et l'étendue, et la revendication se calcule contre la **plus haute** (jamais contre la médiane quand c'est en notre faveur). J concordants (0,300-0,306 brut / 0,224-0,226 net) : ceux-là sont acquis.
* **Résultat à cartes égales** (poste3, repris tel quel) : b=1 devant sans réserve, t/s (×1,051 / ×1,094) et J (0,907 contre 1,151 brut, −21 %) ; b=12 devant en t/s (+10 à +16 % selon la passe llama.cpp), **derrière en J brut (+13-16 %) et net (0,282 contre 0,224, +25 %)**. Le poste « J b=12 » est confirmé et chiffré : ≈ 391 W contre ≈ 300 W à cartes égales.

## Ordre (remplace)

* poste3 : P1 (en cours) ; puis 3e passe llama.cpp b=12 (20 s).
* poste1 : garde « même `cartes` sur les deux bras » + `CUDA_VISIBLE_DEVICES` posé par le client de mesure (à sec).
* chef : comparatif avec les cellules à cartes égales (366,4 / 0,907 ; 1 126,4 / 0,347-0,282), llama.cpp b=12 « trois passes » après la 3e, revendication § 5, retrait daté, REGLES § 4 (harnais + contexte + cartes), ETAT, ligne utilisateur (J b=12 derrière : +13-25 %).
