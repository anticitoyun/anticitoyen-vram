# Verdict — taux d'acceptation n-gram sur code, Coder-30B (item A4, audit Sage)

Laure, 13/09/2026. Génération réelle (systemd-run, verrou, 20 invites de
`outils/trace-taux-succes-experts.py`, température 0, max_tokens=512,
6 527 jetons produits) puis rejeu hors ligne
(`outils/mesure-taux-ngram-hors-ligne.py`), contre la prédiction scellée de
[`protocole-taux-ngram-code-13-09.md`](protocole-taux-ngram-code-13-09.md).

## Le chiffre qui tranche

**Taux réel, comportement par défaut du moteur (`NGramProposer()`, min=2,
max=4, adaptatif=True — exactement `cli.py:585`)** :

    tokens_par_pas = 1,6137   (6 467 positions rejouées, k=8)

Par `n` fixé séparément (isole l'apport de chaque longueur, demande de
Jérôme) :

    n=3   tokens/pas = 1,6262   taux_proposition = 24,9 %
    n=4   tokens/pas = 1,5662   taux_proposition = 18,1 %
    n=5   tokens/pas = 1,4980   taux_proposition = 14,1 %

Par famille (n=3) — l'écart est net et cohérent avec la nature du code :

    format        2,0042   (JSON schema, SQL, regex, bash — motifs très structurés)
    algorithme    1,8717   (quicksort, LRU, recherche binaire — boilerplate répété)
    bogue         1,3562
    refactor      1,2494
    fonction      1,1333
    explication   1,1111
    test          1,1055   (le plus proche du « prose » du 8/09, 1,105)

## Contre ma prédiction — RÉFUTÉE

**Prédite** : taux combiné 1,10-1,20 (« entre le prose et le répétitif du
8/09 »). **Mesuré** : 1,61 — bien au-dessus de ma borne haute, et deux
familles sur sept (format, algorithme) dépassent même le seuil `≥ 2` du
chantier. **Ma réfutation nommée s'est produite** (`> 1,30`) : je tenais le
code de nos 20 invites pour moins répétitif que le régime artificiel du
8/09 ; c'est l'inverse — du code réel, avec ses motifs syntaxiques rigides
(indentation, mots-clés, structures de données), répète PLUS qu'une prose,
au moins sur ce modèle et ce jeu d'invites.

## Contre le critère du chantier (`chantier-speculation.md` §5)

    taux < 1,3                          → chantier fermé                    NON APPLICABLE (taux = 1,61)
    taux ≥ 1,3                          → gain réel, mtp à mesurer aussi     APPLICABLE
    taux ≥ 2 (format, algorithme)       → échelle qui change, prioritaire    APPLICABLE PAR FAMILLE

**Le chantier n'est pas fermé.** Au contraire : le taux réel sur notre
modèle et notre domaine dépasse largement le seuil qui aurait justifié de
l'arrêter, et deux familles de requêtes (format, algorithme — probablement
une bonne part du trafic réel d'un serveur de code) atteignent le régime où
l'amortissement de la tête devient significatif.

## Gain de débit recalculé — f(taux réel, coût de vérification)

Formule inchangée (`protocole-taux-ngram-code-13-09.md` §4) :
`gain = taux / coût_vérification`. Avec `taux = 1,61` (au lieu de 1,10-1,20) :

    régime          coût_vérif (conjecture, jamais mesuré)   gain recalculé
    b=1             1,08 - 1,15 (rétro-calculé du 8/09)       1,40 à 1,49  → +40 à +49 %
    b=12            1,3 - 1,6 (conjecture, occupation pleine) 1,01 à 1,24  → +1 à +24 %

**Le signe s'inverse par rapport à ma prédiction initiale à b=12** : avec un
taux de 1,10-1,20 je prédisais un gain négatif (-8 à -31 %) ; avec le taux
réel de 1,61, même la conjecture pessimiste de `coût_vérification(b=12)`
(jusqu'à 1,6) donne un gain quasi nul à légèrement positif, pas franchement
négatif. **L'inquiétude de l'audit de Sage (« le défaut ngram pourrait
coûter 31 % à b=12 ») reposait sur un taux sous-estimé — la mienne autant
que la sienne.**

## Ce qui reste non mesuré — la vraie question maintenant

`coût_vérification(b=12)` **n'a jamais été mesuré**, ici ni ailleurs dans le
dossier. C'est lui, pas le taux, qui décide du signe à b=12 :

    si coût_vérif(b=12) < 1,61   gain positif, garder ngram par défaut même à b=12
    si coût_vérif(b=12) > 1,61   gain négatif, le défaut coûte réellement à b=12

Avec un taux confirmé à 1,61 (pas 1,10-1,20), le seuil de bascule est
beaucoup plus haut qu'estimé — `coût_vérification` devrait être
**anormalement élevé** (> 1,61, c'est-à-dire qu'un pas de vérification à
`k` jetons coûterait plus de 61 % de plus qu'un pas simple) pour que le
défaut actuel soit nuisible à b=12. C'est possible (occupation déjà pleine,
chantier §2) mais ce n'est plus la conclusion à laquelle mènent les chiffres
disponibles — **ça reste À MESURER, pas à trancher par le calcul seul.**

## Recommandation

**Ne pas désactiver `ngram` par défaut sur la seule base de ce résultat.**
Le taux réel est bon, meilleur que prévu par tout le monde (moi y compris).
La question qui reste ouverte — et qui est maintenant la SEULE qui compte
pour la décision de `serve` à b=12 — est le coût réel d'un pas de
vérification à occupation déjà pleine. Protocole C du chantier
(`chantier-speculation.md` §4, « ÉNERGIE ET DÉBIT ») reste à faire, sur
carte, à `BQ=12` réaliste, graphes déclarés — ce que ce document ne peut pas
trancher sans mesure de temps.

## bd / audit Sage

Item A4 de `sage-audit-connaissances-14-09.md` : **traité**, mais pas fermé
— la génération et le rejeu répondent à « le corpus n'a jamais été généré »,
mais la décision (garder/retirer le défaut) attend encore la mesure C du
chantier (coût de vérification à b=12 réel). Pas de bead créé : c'est une
suite du chantier existant, pas un item nouveau.
