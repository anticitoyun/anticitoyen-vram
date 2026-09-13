# Verdict — horloge SM au décodage, Coder-30B b=12 — 13/09/2026

poste3. Campagne exécutée en service systemd détaché (`horloge-poste3.service`,
2 min 9 s de mur, 5,4 Gio pic) après quatre échecs du superviseur de session
(tue sur MemFree brut — cf. `outils/carte.sh:76-84`, diagnostic de chef).
Contre la prédiction scellée dans
[`protocole-horloge-decodage-13-09.md`](protocole-horloge-decodage-13-09.md).

## Les chiffres

    palier   tok/s    perte débit   J/jeton net   gain J/jeton   bridages
    défaut   568,63       —          0,6009           —          puissance
    2400     501,80    -11,75 %      0,5113        +14,91 %       puissance
    2100     452,06    -20,50 %      0,4851        +19,27 %       puissance
    1800     392,03    -31,06 %      0,4643        +22,73 %       puissance
    1500     337,69    -40,61 %      0,4646        +22,68 %       aucun

## Contre la prédiction

    palier   perte prédite   perte mesurée   gain prédit   gain mesuré
    2400        ≤ 5 %          11,75 %         ≥ 12 %        14,91 %   (gain OK, perte 2,3× trop optimiste)
    2100        ≤ 12 %         20,50 %         ≥ 20 %        19,27 %   (perte 1,7× trop optimiste, gain manqué de 0,7 pt)
    1800        ≤ 20 %         31,06 %         ≥ 25 %        22,73 %   (perte 1,6× trop optimiste, gain manqué de 2,3 pt)
    1500     coude : perte accélère, J/jeton plafonne/remonte
             MESURÉ : perte continue d'accélérer (+9,5 pt vs 1800, plus que
             les +10,6 pt de 2100→1800) ; J/jeton plafonne bien
             (0,4643 → 0,4646, quasi identique) — **cette partie tient**.

**Aucune des deux réfutations nommées ne s'est produite** : la perte à 2400
(11,75 %) reste sous le seuil de réfutation (15 %), et J/jeton s'améliore à
chaque palier jusqu'à 1800. **La prédiction n'est donc pas réfutée**, mais
**ses seuils numériques étaient systématiquement trop optimistes d'un facteur
1,6 à 2,3 sur la perte de débit** — Coder-30B au décodage b=12 est plus
sensible à la fréquence SM que les kernels déjà mesurés ailleurs dans ce
dépôt (MLA, paged_attn), ou bien un autre facteur domine ici (voir réserve
ci-dessous). Le sens de la thèse de poste7 (l'horloge, pas le plafond,
récupère de l'énergie) **se confirme** ; son ampleur relative au débit
(« quasi égal », 2605.11999/2501.08219) **ne se transporte pas** à ce
modèle/cette carte : le débit chute nettement plus que ce que H200/2842→180 MHz
laissait attendre.

## Réserve méthodologique, à dire

La fenêtre d'énergie de chaque palier enveloppe `while running or waiting:
step()` — donc le décodage à 12 séquences PUIS la traîne pendant que les
séquences finissent une à une (toutes à `max_tokens=200`, mais la longueur du
lot rétrécit en fin de fenêtre). **Ce n'est pas un régime stationnaire pur à
b=12** : les derniers pas de chaque fenêtre tournent à un lot plus petit, ce
qui abaisse mécaniquement le débit et l'occupation SM en fin de mesure — un
biais qui touche également tous les paliers (même protocole), donc probablement
sans effet sur le CLASSEMENT relatif, mais qui gonfle l'incertitude sur les
valeurs absolues. Le bridage « puissance » signalé à 4 paliers sur 5 (jamais
à 1500, jamais réellement contraignant en moyenne — `watts` max mesuré
352,9 W < 400 W) est cohérent avec un **pic transitoire** (démarrage/replay du
premier pas) plutôt qu'un bridage soutenu — mais la fenêtre reste,
techniquement, invalidée par le contrôle de `energie.py`. À refaire avec une
fenêtre qui exclut la traîne (mesurer un nombre fixe de PAS à lot constant,
pas un nombre fixe de JETONS) si une décision plus ferme est nécessaire.

## Décision — `ACVRAM_HORLOGE_DECODAGE`

Critère scellé (§4 du protocole) : le palier retenu maximise le gain de
J/jeton **sous une perte de débit ≤ 10 %**, et le geste devient un réglage
si ce gain dépasse 15 %.

**Aucun palier ne satisfait la contrainte de perte ≤ 10 %** — même le premier
palier testé (2400 MHz) perd déjà 11,75 %. **Par le critère scellé, il n'y a
pas de geste gagnant : je ne propose pas `ACVRAM_HORLOGE_DECODAGE`.**

Si chef souhaite assouplir la contrainte de perte (le protocole ne l'exclut
pas, c'est un choix de produit, pas un fait de mesure) : 2100 MHz est le
meilleur point s'il accepte une perte ≥ 20 % contre un gain de 19,3 % en
J/jeton — mais c'est alors un compromis débit/énergie explicite, pas un gain
« quasi gratuit » comme le décrivaient 2605.11999/2501.08219 pour leur régime.
Aucune donnée ici ne soutient un déploiement par défaut ; ce reste une note
de mesure.

## bd

Aucun bead créé/fermé pour ce résultat : ni confirmation franche ni
implémentation à faire, seulement une note de mesure et une décision de
produit renvoyée à chef.

## Décision — 13/09/2026

**Non retenu.** L'utilisateur, par chef : aucun réglage d'horloge —
`ACVRAM_HORLOGE_DECODAGE` n'est pas créé, ce verdict reste une note de
mesure. Le compromis à 2100 MHz (§ ci-dessus) n'est pas adopté par défaut.
