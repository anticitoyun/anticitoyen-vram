# 269 — fenêtre d'admission en rafale : pourquoi la 262 fait deux pas de préfill, et à quelle fenêtre elle n'en ferait qu'un (à sec, scellé AVANT mesure — poste6, 26/09)

instrument (prévu) : `scratchpad/poste6-p269-26-09/prise-269.sh` = les instruments de la 262 d'poste1 recopiés tels quels
  (`serveur-trace-262.py`, `client-ttft-262.py`, `analyse-262.py`, `commun.sh` — origin/poste1-262 ad4da2f70), un serveur neuf par bras ;
  contrôle b=8 : banc chat de la 102 (`prise-abba.sh` de la 226), 3 + 3 tours
commit : à sec sur main e3c345bf7 (0.7.0 + 266) ; aucune mesure avant la 268 d'poste1 (/metrics hors boucle), ordre de chef
régime : celui de la 262 (Coder-30B nvfp4, -lgc 2700, ECO=off, spéculation coupée, max-batch 16, cpu-safe 100)
scellé : ci-dessous (P1-P6), avant toute prise
mesuré : rien encore
verdict : à venir
durée : prévu 3 bras × (7 tours à 12 + 5 tours solo) ≈ 3 × 4 min + contrôle b=8 ≈ 8 min → 2 prises ≤ 15 min

## 1. Ce que dit le code (lu, pas supposé)
* `acvram/server/app.py:113-131` `_attendre_les_arrivees` : n'attend QUE si le moteur est vide (`eng.running` faux) ET ≥ 2 requêtes
  déjà en file au réveil du fil ; puis boucle à 0,5 ms jusqu'à ce que la file n'ait pas grossi pendant `FENETRE` (défaut 5 ms,
  `regime.py:347`), plafond 4 × FENETRE. Le fil moteur dort 2 ms entre deux sondages quand il est `idle` (`_run`, app.py:154).
* `acvram/engine/runner.py:1100` `_admit` : admet toute la file jusqu'à `max_batch_size` (16) tant que les blocs KV suffisent — aucun
  budget de jetons de préfill, pas de préfill par morceaux (runner.py:153-158) : 12 invites de ≈ 450 jetons (5 400) tiennent dans UN pas.
* Client 262 (`client-12.json`) : les 12 envois tiennent dans 1,2-2,7 ms. Côté serveur, chaque requête paie gabarit + tokeniseur ≈ 3 ms
  DANS la boucle HTTP (262 : 3,02 ms/req), donc les arrivées dans `eng.waiting` s'étalent sur plusieurs ms, une par une.

## 2. Le mécanisme des deux pas (262 : premiers [2, 10] × 2, [1, 11] × 3, [12] × 2)
* Tours [1, 11] (3/7) : le fil se réveille avec UNE requête en file → la porte « ≥ 2 » ne s'ouvre pas → pas de préfill de 1 (42 ms) ;
  les 11 autres arrivent pendant ce pas et attendent sa fin (file 42 ms = la durée du pas, 262). La fenêtre, quelle que soit sa
  longueur, ne peut rien pour ces tours : elle n'est jamais ouverte.
* Tours [2, 10] (2/7) : réveil à 2 → fenêtre ouverte, close par 5 ms de tranquillité (`fenêtre 5,0-5,5 ms` mesurés) → les 10 autres
  arrivent juste après (elles font ensuite la file du pas entier, 42-62 ms) : un trou > 5 ms entre la 2e et la 3e arrivée serveur.
* Tours [12] (2/7) : fenêtre 10-12 ms = arrivées sur ≈ 5-7 ms, puis 5 ms de calme → un pas de 228-240 ms.
* Coût d'un pas de 12 : 228-240 ms ; coût de « 1-2 puis 10-11 » : 43-62 + 194-220 = 258-277 → le second schéma coûte 20-35 ms de
  plus au mur ET donne au p50 par requête le TTFT du second pas (263-274 ms contre 243-257).

## 3. Ce que la fenêtre peut et ne peut pas faire
* Une fenêtre plus longue (10 ms) couvre le trou des tours [2, 10] → 1 pas ; elle ne change RIEN aux tours [1, 11] (porte fermée).
  Au mieux la fenêtre seule passe de 2/7 à 4/7 tours à un pas — pas 7/7. C'est la prédiction P1, et son issue défavorable est nommée.
* Le levier des tours [1, 11] est la PORTE, pas la fenêtre : ouvrir un micro-guet à 1 requête (attendre ≤ 1 ms l'arrivée d'une 2e)
  coûterait ≤ 1 ms à une requête seule (38,5 ms : +2,6 %) — contraire à la règle 179 b (« une requête seule ne paie rien »), donc
  PAS dans cette pièce : bras D, à sec seulement, décision de chef si P1-P3 tiennent et que 4/7 ne suffit pas.
* Toute fenêtre s'applique aussi aux rafales déjà en un pas : +5 ms de tranquillité de plus par tour (244 → ≈ 249 ms). C'est le prix.
* La fenêtre ne s'ouvre jamais pendant le service (`eng.running`) : une rafale qui arrive sur un moteur occupé reste admise pas à pas.
  Hors périmètre ici, nommé.

## 4. Scellé (avant mesure) — bras A = 5 ms (défaut), B = 10, C = 15 ; ordre A B C C B A ; 7 tours à 12 + 5 solo par bras
* P1 (pas de préfill à 12) : A 2/7 tours à un pas (262 : 2/7) ; B ≥ 4/7 ; C ≥ 4/7 et C = B ± 1 tour. FAUX si B ≤ 2/7.
* P2 (TTFT par requête, 84 requêtes par bras) : p50 A 263-268 ms (262 : 265) ; B ≤ 258 ; p95 A ≈ 274, B ≤ 262. FAUX si p50(B) > 260.
* P3 (mur médian) : A 262-270 (262 : 266,6) ; B 250-262.
* P4 (règle 179 b, solo) : TTFT solo A = B = C à ± 1 ms (262 : 38,5) — la porte reste fermée à 1. FAUX si B ou C > 40 ms.
* P5 (contrôle, débit servi b=8, banc chat 102, 3 + 3) : B/A dans ± 1,5 %. FAUX au-delà, et alors pas de défaut.
* P6 (issue défavorable, NOMMÉE) : le TTFT MINIMUM par tour monte de 45-65 ms (la 1re vague) à ≈ 245 ms dès qu'un tour passe à un pas :
  1-2 requêtes par tour perdent ≈ 200 ms pour que 10-11 en gagnent ≈ 20. Le p50 et le p95 gagnent, le min perd ; si l'on juge à
  l'équité par requête, B n'est pas un gain — décision de chef, le chiffre sera publié dans les deux lectures.
* Issue qui me gênerait : P1 tenu (4/7) mais P2 FAUX (p50 ≤ 260 non atteint) = la tranquillité ajoutée (+5 ms sur tous les tours)
  mange le gain des tours fusionnés → défaut inchangé, et la porte (bras D) devient le seul levier.
* Décision annoncée : P1-P5 tenus → défaut `ADMISSION_FENETRE_MS=10` (regime.py, CHANGELOG, test cassant « défaut 10, 5 témoin ») ;
  sinon défaut 5 inchangé, note de verdict dans les deux cas.

## 5. Décision de chef AVANT la mesure (26/09, écrite ici pour que rien ne soit réécrit après)
* Lecture de P6 : ce qui compte est le PIRE cas — p50, p95 ET TTFT max par tour ; une requête qui passe de 45 à 245 ms reste sous le
  max actuel (263-274) : pas une perte d'équité au sens du service. Le min par tour est relevé à titre d'information, il ne juge pas.
  → P2 devient : p50 B ≤ 258, p95 B ≤ 262, max par tour B ≤ max A (médiane des 7 tours). FAUX si l'un des trois manque.
* Bras D (micro-guet à 1 requête) : ÉCARTÉ — il coûte au solo, contre la 179 b ; on ne le mesure pas.
* Cause (b) (gabarit + tokeniseur ≈ 3 ms/req dans la boucle HTTP) : signalée à poste1 pour la 268 (/metrics hors boucle), qui pourrait
  en sortir aussi la tokenisation. Si la 268 le fait, l'étalement des arrivées se resserre et la fenêtre de 5 ms peut suffire aux
  tours (b) : la 269 se mesure APRÈS la 268, sur son arbre, et le bras A est alors le témoin de ce que la 268 a déjà changé.
