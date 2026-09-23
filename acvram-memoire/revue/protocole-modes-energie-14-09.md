# Protocole — trois modes d'énergie (point 4 de l'ordre de Sage)

Laure, 14/09/2026. Suite à
[`sage-modes-energie-14-09.md`](sage-modes-energie-14-09.md) (Sage,
décideuse technique) et à la transmission de Jérôme. Prédictions et seuils
sont ceux de Sage, écrits avant cette mesure — je ne les réécris pas,
j'exécute et je rapporte le verdict.

## Prédictions de Sage (rappel, ne pas modifier après coup)

    mode        réglage         prédit (Coder-30B b=12, 20s, réf=moyen)   réfuté si
    maximal     -pl 600         +15-20 % t/s, J/jeton ±5 %                t/s<+8% ou J>+15%
    moyen       400 (actuel)    630,6 t/s / 0,619 J (déjà mesuré)         —
    économique  -pl 300         J −15 à −20 %, t/s −15 à −25 %            J < −10 %

Test d'acceptation (Sage) : `eco` doit rendre J ≤ 0,90 × moyen ; `max` doit
rendre t/s ≥ 1,08 × moyen — sinon le mode ment.

## Ce que j'ajoute : b=1 en plus de b=12

Jérôme demande b=1 ET b=12 pour chaque mode (Sage n'avait chiffré que
b=12). Je n'ai pas de prédiction scellée de Sage pour b=1 — je rapporte
les chiffres sans les évaluer contre un seuil qu'elle n'a pas posé, et je
la laisse juger si le mode tient aussi à b=1.

## Plancher de repos — attente active ou P0 maintenu ?

Question de Sage : le repos chaud (68-76 W) vient-il d'une attente active
de la boucle de service, ou d'un contexte CUDA qui maintient la carte en
P0 (haute performance) même sans travail ? Mesure : `nvidia-smi -q -d
PERFORMANCE` interrogé pendant une fenêtre de repos (30 s, serveur chargé,
aucune requête), en lisant l'état de performance (`Performance State`)
plutôt que la seule puissance. Réfuté si le repos bloqué reste ≥ 60 W
(déjà l'ordre de grandeur mesuré — l'important ici est la CAUSE, P-state
ou boucle, pas de refaire la mesure de puissance elle-même).

## Montage

    modèle     Qwen3-Coder-30B-A3B-nvfp4 (acvram, régime des campagnes du
               jour)
    b          1 et 12
    fenêtre    >= 20 s par cellule (rondes répétées, comme la campagne
               principale du jour), repos 30 s avant chaque cellule
    réglage    `sudo -n nvidia-smi -i 0 -pl <600|300>` avant chaque mode,
               `-pl 400` remis à la sortie ET sur signal (trap), jamais
               sans le verrou `carte.sh` tenu de bout en bout (un seul
               verrou continu pour toute la campagne, pas un par cellule)
    nommage    dossier de sortie porte le nom du mode (règle 4 de Sage :
               « le mode est porté par le nom »)

## Reprise du 14/09 (matin) — corrections avant relance

**Ce qui change et pourquoi.** (1) « refus -pl 300 » d'hier soir n'était pas
sudo : la 5090 n'accepte que 400-600 W (`docs/MATERIEL.md` §plage,
1103825). **eco = `-lgc 2100`** (horloge bloquée, plafond 400 inchangé),
alternative déjà nommée par Sage. (2) La lecture P-state d'hier (`P8`) était
prise **après la sortie du processus** — à froid : elle ne répondait pas à
la question. Le P-state se relève maintenant **dans le processus, modèle
chargé, graphes capturés, pendant 30 s sans aucun pas** (sonde toutes les
5 s : état, W, MHz). (3) `moyen` est **remesuré dans la même campagne**
(dérive thermique : on ne compare pas à un chiffre d'hier). (4) En-tête
REGLES §3 dans chaque enregistrement JSON (instrument, cartes, fenêtre,
plafond, horloge min/moy/max, mode).

**Seuils (Sage, inchangés)** : `max` t/s ≥ 1,08 × moyen ; `eco` J ≤ 0,90 ×
moyen — le moyen de la MÊME campagne. Alarme publiée d'avance : si le moyen
remesuré à b=12 s'écarte de > 5 % de 630,6 t/s / 0,619 J (hier), les deux
régimes diffèrent et je le dis avant de comparer.

**Mes prédictions ajoutées (scellées, à réfuter)** :

    max b=12   : le plafond 400 W mordait à b=12 (moyenne 395-400 W hier,
                 401-435 W à b=2-4 avant régulation) → t/s +8 à +15 %,
                 J/jeton +5 à +15 % (plus de W pour un peu plus de t/s).
                 Réfuté si t/s < +3 % (le plafond ne mordait pas, comme au
                 3/09 à b=1 : +0,3 %) — alors « max » ne vaut rien.
    max b=1    : +0 à +2 % t/s (b=1 ≈ 330 W < 400, plafond inactif).
    eco b=12   : 2 100 MHz vs ~2 600 libre → t/s −15 à −22 %, J −12 à −20 %
                 (13/09 : −20 % / −19 %). Réfuté si J > −10 % (Sage).
    P-state    : P0 maintenu pendant tout le repos chaud, W 65-80 → la cause
                 du plancher est le contexte CUDA vivant, pas une boucle de
                 service (il n'y en a aucune dans ce script : sleep pur).
                 Réfuté si l'état lu est P8/P5 avec W ≥ 60 (autre cause :
                 mémoire/horloge mémoire) ou si W < 30 (le plancher 68-76
                 venait du serveur, à remesurer sous `acvram serve`).

## P5 — garde de décodage sous prefill (même prise de carte, après les modes)

Montage : b=1 établi (A, invite 256, 300 pas chronométrés seule), puis B
(invite 8 192, `max_tokens=1`) admise sous `ACVRAM_BUDGET_JETONS` = 512 /
2 048 / 8 192. Fenêtre = admission de B → `B.prefilled`. Chute = 1 −
t/s(A sous prefill)/t/s(A seule). Preuve que la config a pris : pas du
prefill de B = 16 / 4 / 1 exactement, sinon cellule invalide.

Sage : chute ≤ 30 % à 512 et ≥ 70 % à 8 192 → le budget est la garde ;
indépendant du budget → la garde est ailleurs.

Ma prédiction : chaque pas sous prefill coûte (tranche de prefill + 1 pas de
décodage) ; à 6 400 j/s de prefill, 512 jetons ≈ 80 ms contre 5,5 ms le pas
seul → **chute ≈ 90 % à 512, ≈ 98 % à 2 048, ≈ 99 % à 8 192** — dépendante
du budget mais jamais ≤ 30 % : le budget découpe la latence de B, il ne
protège pas le débit de A tant que la tranche coûte plus qu'un pas. Réfuté
si chute ≤ 30 % à 512 (le prefill se recouvre avec le décodage, ce que je
ne vois nulle part dans `runner.py:705-760`) ; à l'inverse, chute égale aux
trois budgets = config pas prise (voir preuve) avant toute autre lecture.
Durée prefill de B seule : ≈ 1,3 s à 8 192 (borne 6 400 j/s), plus à 512
(16 lancements + 16 pas de décodage intercalés).
