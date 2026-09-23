# Levier 2 : d où viennent les +6,84 % quand le trou mesuré ne valait que 2,4 % (22/09, Océane, à sec)

* sources : `frontiere-epingle-22-09-v2` A0/A1/B0/B1 (frontiere-pas, 300 pas pleins, Manon), `verdict-levier2-3-abba-22-09` (service HTTP, `chaine-sampler-abba.sh` : `acvram.cli serve` + `banc-llamacpp-16-09.py` 12 flux, A B B A A B B A), `verdict-m2-b12-21-09` (service : mur 7,42 = rejeu 6,77 + **trou 0,61 ms**), `verdict-abba-graphe-22-09` (levier 1 : A 1 531 → B 1 550)
* alarme de ma note § 4 : gain > 2,6 % = « autre chose bouge ». Réponse : oui — et c est le service, pas la carte.

## 1. Ce que frontiere-pas v2 dit (µs, médianes A0/A1 → B0/B1)
| | A | B | Δ |
|---|---|---|---|
| pas_gpu | 6 948 / 6 941 | 6 846 / 6 846 | **−98 (−1,4 %)** |
| graphe | 6 776 / 6 771 | 6 819 / 6 819 | **+45 (+0,7 %)** |
| trou_gpu | 166 / 164 | 24 / 25 | −141 |
| attente_evt | 5,6 | 6 631 / 6 622 | la signature prédite |
| consommer | 6 746 | 15 | |
| suite_prep | 150 | 156 | (pendant le graphe désormais) |
| moyenne pas_gpu | 7 183 / 7 176 | 7 061 / 7 059 | −120 (−1,7 %) |
| moyenne trou_gpu | 456 / 453 | 289 / 286 | queue de pas lents inchangée (p90 193 → 33, la moyenne reste 10 × la médiane) |
Sur CE harnais (hôte minimal : une boucle `engine.step()` sans tokenizer ni HTTP), le levier vaut **−1,4 à −1,7 %** de pas, moins que mes 2,0-2,4 : le graphe s allonge de 45 µs (+0,7 %) quand la carte n a plus de repos — régime de puissance : horloge B 2 527 contre A 2 561 MHz dans l ABBA (−1,3 %), même signe. **Le trou de 165 µs était celui de frontiere-pas, pas celui du service.**

## 2. Le service n est pas frontiere-pas : son trou était 0,61 ms, pas 0,165
`verdict-m2-b12-21-09`, service HTTP, pipeline : **mur 7,42 = rejeu 6,77 + trou 0,61 ms** (p90 0,73). Dans le service, entre deux rejeux, l hôte fait ce que frontiere-pas ne fait pas : `_decode_delta` (tokenizer) × 12, `GenerationOutput` × 12, émission SSE, boucle asyncio, `_admit`, journal — ≈ 0,45 ms de plus que les 165 µs du harnais nu. En A (flux), TOUT cela est sérialisé après le `.tolist()` qui attend la fin du graphe : carte oisive 0,61 ms par pas = **8,2 % du mur**. En B (épinglé), l hôte revient de `_consommer` en 15 µs, fait tokenizer + SSE + admission **pendant** le graphe suivant, et il lui reste 6,6 ms de marge (attente_evt) : le trou du service tombe au même ≈ 25 µs que celui du harnais.
Arithmétique : mur A ≈ 6,77 + 0,61 = 7,38 ; mur B ≈ 6,82 (graphe +0,7 %) + 0,03 + queue ≈ 6,90 → **B/A ≈ 1,070**, mesuré **1,0684**. Les « 4,4 % restants » sont les 0,45 ms de travail hôte du service que mon instrument ne portait pas — et que ma note § 2 avait pourtant nommés (« `_emit` avec tokenizer aussi ») sans les chiffrer dans la prédiction. **Faute de prédiction, pas de mesure** : j ai prédit le service avec le trou du harnais.

## 3. A était-il dégradé ? Non, à la dispersion près
A = 1 533 / 1 514 / 1 576 / 1 540 (méd 1 536) ; hier même chaîne, A graphe = 1 556 / 1 542 / 1 570 / 1 544 (méd 1 550) : −0,9 %, dans la dispersion connue (± 30-50 t/s par fenêtre). Le « 1 550 » de la Maîtresse est ce B d hier. Pas de dégradation nommable ; l écart B/A porte sur les 4 paires, A1 B1 B2 A2 A3 B3 B4 A4, sans ordre A A A B B B.

## 4. Prédiction pour la 2e passe ABBA (écrite avant)
* **B/A = 1,055 à 1,080** (service, 4 paires) ; horloge B inférieure à A de 0,5 à 2 % (carte sans repos, plafond de puissance) ; **J/jeton B ≤ 0,95 × A** (moins de temps oisif à 380 W) — si Manon relève J.
* Réfuté si B/A < 1,040 (la 1re passe avait un A faible ou un B chanceux : on republie la médiane des deux passes, pas la meilleure) ; alarme si > 1,090 (au-delà du trou de service 0,61 ms : autre chose).
* Invariant : ids/logprobs au bit tenus 3/3 (`verdict-levier2-1`), PPL A/B sur wiki-gptq à jouer = témoin de qualité, prédit identique au bit (même arithmétique).

## 5. Ce que le levier vaut selon l instrument — à écrire dans la fiche
| instrument | trou hôte sérialisé en A | gain B |
|---|---|---|
| frontiere-pas (boucle nue) | 165 µs | −1,4 à −1,7 % de pas |
| certifie-b12 (in-process, sans tokenizer, NVML tous les 200 pas) | ≈ 170-250 µs | prédit +2 à +3 % |
| **service HTTP** (tokenizer, SSE, asyncio) | **0,61 ms** | **+6,84 % mesuré** |
Le gain du levier 2 est proportionnel au travail hôte par pas : plus le service en fait (tokenizer lent, `logprobs`, journal), plus il gagne — et il ne peut dépasser le trou de service (8,2 %). Après lui, le pas servi est borné par le graphe (6,82 ms → 1 760 t/s théoriques ; le reste est la queue de pas lents, moyenne − médiane ≈ 210 µs = 3 %, à nommer un jour : `_admit` ? ramasse-miettes ? — hors levier).
