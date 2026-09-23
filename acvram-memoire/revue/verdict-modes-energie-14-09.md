# Verdict — trois modes d'énergie (point 4) et P5 garde de décodage sous prefill

Laure, 14/09/2026, 10:38-10:46. Protocole et prédictions scellées avant :
[`protocole-modes-energie-14-09.md`](protocole-modes-energie-14-09.md) (laure
9fbf481). Décideuse : Sage
([`sage-modes-energie-14-09.md`](sage-modes-energie-14-09.md), P5 dans
[`sage-veille-trtllm-ggrun-14-09.md`](sage-veille-trtllm-ggrun-14-09.md)).

## En-tête de mesure (REGLES §3)

    instrument   energie.py, compteur NVML TotalEnergyConsumption (pas de médiane)
    cartes       [0] seule (CUDA_VISIBLE_DEVICES=0, champ `cartes` du JSON)
    fenêtre      23-38 s par cellule (rondes ≥ 20 s), repos 30 s avant chacune
    plafond      relevé DANS le processus : 400 / 600 / 400 W
    horloge      min/moy/max de clocks.sm échantillonné toutes les 2 s (JSON)
    mode         porté par le nom du fichier : moyen.json / max.json / eco.json
    modèle       Qwen3-Coder-30B-A3B-nvfp4, ctx 2048, invite 256, main du jour
    verrou       carte.sh tenu 10:38:41 → 10:45:22 (journal du verrou, 401 s)
    données      scratchpad/modes-energie-14-09/{moyen,max,eco,p5}.json

**Réserve sur trois cellules** : un processus GPU étranger (PID 264421, ni
dans le journal du verrou, ni identifié) est apparu sur la carte de ~10:39:30
à ~10:41:10 — la garde « les processus ont changé » d'`energie.py` l'a signalé
sur moyen b=1 (fin de fenêtre) et max b=1 (début) ; il était présent pendant
tout moyen b=12 (pas de changement, donc pas d'alarme — voir ci-dessous).
Les cellules eco (×2) et max b=12 sont propres. Le moyen b=12 reproduit hier
au dixième près (630,7 vs 630,6 t/s ; 0,6191 vs 0,619 J), donc l'intrus était
inactif ; mais un chiffre sous carte partagée reste un chiffre sous carte
partagée : à refaire si Sage veut un moyen certifié, ~3 min.

## Résultats

    mode   réglage    b   t/s      Δ t/s    J/jeton  Δ J      W moy   sm moy   repos chaud
    moyen  -pl 400    1   233,0    —        1,4174   —        330     2717     47,6 W
    moyen  -pl 400   12   630,7    —        0,6191   —        390,5   2336     49,2 W
    max    -pl 600    1   233,0    +0,0 %   1,4512   +2,4 %   338     2728     49,0 W
    max    -pl 600   12   724,0   +14,8 %   0,8121  +31,2 %   587,9   2678     50,6 W
    eco    -lgc 2100  1   178,7   −23,3 %   1,0623  −25,1 %   190     2028     41,9 W
    eco    -lgc 2100 12   514,6   −18,4 %   0,5654   −8,7 %   291     1926     41,8 W

### Contre les seuils scellés

* **max** : test d'acceptation de Sage (t/s ≥ 1,08 × moyen = 681) **TENU** à
  b=12 (724). Sa prédiction « J ±5 %, réfuté si > +15 % » est **RÉFUTÉE** :
  +31 %. Ma fourchette « J +5 à +15 % » est réfutée aussi. Mécanisme lu dans
  les horloges : à 400 W le plafond rabat la SM à 2 336 MHz (b=12), à 600 W
  elle tient 2 678 ; +15 % d'horloge donne +15 % de débit, mais la puissance
  passe de 390 à 588 W (+51 %) : le mode « max » achète 15 % de débit avec
  31 % d'énergie par jeton de plus. À b=1 le plafond ne mord pas (330 W),
  rien ne change — comme prédit.
* **eco** : test de Sage (J ≤ 0,90 × moyen = 0,557) **NON TENU** à b=12 :
  0,5654 (−8,7 %, il manque 1,5 point). Ma fourchette « J −12 à −20 % »
  réfutée. À b=1 le mode tient largement (−25 % J pour −23 % t/s). Note :
  la carte n'a pas tenu 2 100 MHz à b=12 (moy 1 926, min 1 912) — `-lgc`
  fixe un maximum, le limiteur a encore rabattu sous charge ; la garde
  « bridage : puissance » a fait feu à 291 W, donc ce n'est pas le plafond
  400 : autre raison de bridage (tension/thermique), non instrumentée ici.
* **Repos chaud (question de Sage : attente active ou P0 ?)** : ni l'un ni
  l'autre. Modèle chargé, graphes capturés, processus vivant, `sleep` pur :
  P1 pendant ~20 s (70-73 W, 2 572 MHz) puis **P8 à 16-17 W, 225 MHz** ; en
  eco (`-lgc`) la descente s'arrête à P5, 20 W, 2 092 MHz. Moyenne 30 s :
  42-51 W. Ma prédiction « P0 maintenu » est **réfutée** : la carte redescend
  seule, il n'y a pas de contexte qui la retient ; le « 68-76 W chaud » d'hier
  est la moyenne d'une fenêtre plus courte que la descente (~20 s), pas un
  plancher. Reste ouvert : sous `acvram serve` (boucle de service réelle), le
  chiffre peut différer — non mesuré, à demander si utile.

## P5 — garde de décodage sous prefill

Montage : b=1 établi (A, 300 pas seule), B = invite 8 192 admise sous
`ACVRAM_BUDGET_JETONS`. Config prouvée par le compte de pas.

    budget   pas B (attendu)   A seule   A sous prefill   chute    durée prefill B   pas moyen
    512      16 (16)           244,6     11,3 t/s         95,4 %   1,419 s           88,7 ms
    2048      4 (4)            244,9      4,7 t/s         98,1 %   0,858 s           214 ms
    8192      1 (1)            243,9      2,3 t/s         99,1 %   0,437 s           437 ms

Seuil de Sage « chute ≤ 30 % à 512, ≥ 70 % à 8 192 » : la seconde moitié
tient, la première est **réfutée** (95 %). Ma prédiction (≈ 90 / 98 / 99 %,
dépendante du budget mais jamais ≤ 30 %) est **tenue**. Lecture : le budget
découpe la latence de B (mais la triple : 1,42 s contre 0,44 s, les tranches
de 512 tournent à ~5 800 j/s contre 18 700 j/s d'un seul tenant) sans
protéger le débit de A, parce qu'un pas = une tranche de prefill + un jeton
de A, en série (`runner.py:748-757` puis décodage), et qu'une tranche de 512
coûte 16 fois le pas seul. La garde de décodage n'est pas le budget : il
faudrait recouvrir (prefill sur un flux, décodage sur l'autre) ou borner la
tranche à ~la durée d'un pas (≈ 30 jetons à ce débit — sans intérêt). W/J
de ces fenêtres (1-2 s) invalidés par la garde `< 10 s`, non publiés.

## Ce que je ne conclus pas

Rien sur `acvram serve` (aucune boucle de service ici) ; rien sur la 3080 Ti ;
rien sur vLLM/llama.cpp sous ces modes (non mesurés). Le moyen b=12 sous carte
partagée est à refaire avant de servir de référence certifiée.
