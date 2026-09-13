# Protocole — trois modes d'énergie (point 4 de l'ordre de poste7)

poste3, 14/09/2026. Suite à
[`poste7-modes-energie-14-09.md`](poste7-modes-energie-14-09.md) (poste7,
décideuse technique) et à la transmission de chef. Prédictions et seuils
sont ceux de poste7, écrits avant cette mesure — je ne les réécris pas,
j'exécute et je rapporte le verdict.

## Prédictions de poste7 (rappel, ne pas modifier après coup)

    mode        réglage         prédit (Coder-30B b=12, 20s, réf=moyen)   réfuté si
    maximal     -pl 600         +15-20 % t/s, J/jeton ±5 %                t/s<+8% ou J>+15%
    moyen       400 (actuel)    630,6 t/s / 0,619 J (déjà mesuré)         —
    économique  -pl 300         J −15 à −20 %, t/s −15 à −25 %            J < −10 %

Test d'acceptation (poste7) : `eco` doit rendre J ≤ 0,90 × moyen ; `max` doit
rendre t/s ≥ 1,08 × moyen — sinon le mode ment.

## Ce que j'ajoute : b=1 en plus de b=12

chef demande b=1 ET b=12 pour chaque mode (poste7 n'avait chiffré que
b=12). Je n'ai pas de prédiction scellée de poste7 pour b=1 — je rapporte
les chiffres sans les évaluer contre un seuil qu'elle n'a pas posé, et je
la laisse juger si le mode tient aussi à b=1.

## Plancher de repos — attente active ou P0 maintenu ?

Question de poste7 : le repos chaud (68-76 W) vient-il d'une attente active
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
    nommage    dossier de sortie porte le nom du mode (règle 4 de poste7 :
               « le mode est porté par le nom »)
