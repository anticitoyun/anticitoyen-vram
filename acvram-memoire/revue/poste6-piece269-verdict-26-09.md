# 269 — fenêtre d'admission en rafale : 10 et 15 ms n'apportent RIEN après la 268 et coûtent leur durée ; défaut 5 ms inchangé (poste6, 26/09)

instrument : `scratchpad/poste6-p269-26-09/prise-269.sh` (instruments 262 d'poste1 recopiés : `serveur-trace-262.py`, `client-ttft-262.py`,
  `analyse-262.py`, `commun.sh`), serveur neuf par bras ; contrôle b=8 : `prise-b8-269.sh` (banc chat 102, `prise-abba` de la 226)
commit : poste6-269 5ba155ef7 = main e3c345bf7 (0.7.0 + 266) + poste1-268 eef137839 (/metrics et tokenisation hors boucle) ; prise b=8 : fb04bc254
régime : celui de la 262 (Coder-30B nvfp4, -lgc 2700, ECO=off, spéculation coupée, max-batch 16, cpu-safe 100) ; carte : llama-server 4219 seul
scellé : `revue/poste6-piece269-a-sec-26-09.md` § 4-5 (P1-P6, lecture de chef : p50, p95, max par tour ; min informatif), écrit AVANT
mesuré : A B C C B A (A = 5 ms défaut, B = 10, C = 15), 7 tours à 12 + 5 solo par bras, 2 passes par bras (14 tours, 10 solo) ; 07:34:43 → 07:39:20
verdict : **FAUX pour B et C — P1, P2, P3 tous FAUX** ; défaut `ADMISSION_FENETRE_MS=5` INCHANGÉ. Le levier de la fenêtre est épuisé après la 268.
durée : prévu 2 × 15 min ; tenu prise 1 (12 + solo) 4 min 37 s après 303 s d'attente ; prise 2 (b=8) 4 min 46 s après 355 s (une 1re prise
  b=8 tuée par moi : dossier `abba/` non créé, serveur jamais lancé, banc muet 900 s — même famille que la 266, corrigé fb04bc254)

## 1. Chiffres (TTFT en ms ; par requête = 12 par tour ; « 1 pas » = tours dont la rafale tient dans un seul pas de préfill)
| bras | 1 pas / 14 | mur médian | p50 par requête | max par tour (médiane) | max max | min par tour (médiane) | solo médian (min-max) |
|---|---|---|---|---|---|---|---|
| A 5 ms | **8** | **264,5** | **262,5** | **264,0** | 274 | 241 | 39,0 (37-41) |
| B 10 ms | 7 | 270,5 | 268,5 | 270,5 | 281 | 150 | 41,0 (38-46) |
| C 15 ms | 7 | 274,0 | 272,5 | 274,0 | 288 | 160 | 39,5 (38-44) |
Par passe : A 4/7 puis 4/7 ; B 5/7 puis 2/7 ; C 4/7 puis 3/7. Fenêtre médiane mesurée (trace) : A 8,1-8,8 ms, B 10,5-13,4, C 15,5-20,6.

## 2. Prédictions
* P1 (B ≥ 4/7 tours à un pas) : **FAUX** — B 7/14 (3,5/7), C 7/14 ; et A, le témoin après la 268, est déjà à 8/14. La 262 (avant la 268) : 2/7.
* P2 (p50 B ≤ 258, p95 ≤ 262, max par tour B ≤ max A) : **FAUX** ×3 — p50 268,5 (A 262,5), max médian 270,5 (A 264,0).
* P3 (mur B 250-262) : **FAUX** — 270,5.
* P4 (solo A = B = C ± 1 ms) : à la lettre FAUX pour B (41,0 contre 39,0) ; la trace dit fenêtre 0,00-0,01 ms dans les 30 tours solo (porte
  fermée à 1, comme prévu) et l'écart est dans le PRÉFILL (39,4 contre 37,3 ms), instance de serveur à instance : la 2e passe de B donne 39,0.
  Le mécanisme tient (la fenêtre ne coûte rien au solo) ; le seuil ± 1 ms était sous la variance entre serveurs (A : 37-41 selon l'instance).
* P5 (débit servi b=8, B/A ± 1,5 %) : § 3.
* P6 (issue défavorable nommée) : dans tout tour à deux pas le min par requête tombe à 47-66 ms (la 1re requête) et les 11 autres portent
  le 2e pas ; à un pas, min ≈ max. Lecture de chef (pire cas) : B et C n'améliorent ni p50, ni p95, ni max → rien à trancher.
* Issue qui me gênait, nommée d'avance (§ 4 du scellé) : « P1 tenu mais P2 faux = la tranquillité ajoutée mange le gain ». C'est pire :
  P1 n'est même pas tenu, et la tranquillité ajoutée est payée à chaque tour (+6 ms à 10, +10 à 15, sur le mur médian).

## 3. Ce que la mesure dit du mécanisme (lu dans les traces, pas supposé)
* Après la 268 (tokenisation hors boucle), les tours [2, 10] de la 262 ont DISPARU : les 12 arrivées tiennent en ≈ 5-7 ms (fenêtre A 8-9 ms
  = arrivées + 5 ms de calme). La cause (b) de la note à sec était bien la tokenisation dans la boucle — réglée par la 268, pas par la fenêtre.
* Tous les tours à deux pas restants sont [1, 11] (une fois [2, 10]) : le fil moteur se réveille avec UNE requête en file, la porte « ≥ 2 »
  ne s'ouvre pas, pas de 1 (41-49 ms) ; la requête (max_tokens=1) termine, le moteur est vide, les 11 autres ouvrent alors la fenêtre — la
  « fenêtre 10,5 / 15,4 ms » des tours à deux pas est celle de la 2e vague, payée en pure perte (les 11 sont déjà là). Une fenêtre plus longue
  ne peut pas agir sur la porte ; elle ajoute sa durée aux deux schémas. Le seul levier restant est la porte (bras D), écarté par la 179 b.
* Le hasard porte/pas-porte (4/7, 5/7, 2/7, 3/7 selon la passe) est celui du réveil du fil (sommeil 2 ms) contre la première arrivée : pas
  un effet de la fenêtre.
* **P5 (contrôle b=8, banc chat 102, A B B A A B, 20 s par passe, 34 816 jetons décodés par passe) : TENU** — A 1 682,7 / 1 683,0 / 1 684,4 t/s
  (médiane 1 683,0), B 1 684,9 / 1 684,6 / 1 683,4 (1 684,6) : **B/A +0,10 %** ; J/jeton net A 0,1264, B 0,1272 (+0,6 %). repli_eager=0 sur les
  6 passes, `ACVRAM_ADMISSION_FENETRE_MS=10` imprimé dans la ligne de régime de B (la variable a pris). Les 6 passes portent « bridage
  puissance » (plafond 400 W, horloge moyenne 2 675-2 676 MHz, identique A/B) : régime servi de la 226, comparaison à horloge égale. Attendu : la fenêtre ne s'ouvre que
  moteur vide, jamais en service établi — elle n'a aucun effet sur le débit servi.

## 4. Ce qui en suit
1. Défaut 5 ms inchangé ; aucune pièce sur la fenêtre n'a plus de sens : après la 268 elle n'a qu'un coût.
2. Le seul reste du TTFT en rafale est la porte « ≥ 2 au réveil » (3/7 tours en moyenne, +20-35 ms de mur, +200 ms pour 11 requêtes sur 12
   de ces tours). Bras D (micro-guet ≤ 1 ms à 1 requête) écarté par la 179 b. Une autre voie sans coût pour le solo : ouvrir la fenêtre à 1
   requête SEULEMENT si une 2e connexion HTTP est déjà acceptée (compteur de requêtes entrées mais non encore soumises, côté app.py) — à sec,
   si chef le demande ; le solo ne paie rien puisque le compteur est à 0.
3. La 262 avait attribué à la file 16 % du mur ; après la 268 la file d'un tour à un pas est de 10-11 ms (fenêtre 5 + arrivées), soit 4 %.
