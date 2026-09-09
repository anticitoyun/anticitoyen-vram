# Prédiction écrite AVANT la campagne de clôture du 9/09/2026

Ce fichier est commité **avant** que la campagne soit lancée. Son horodatage
git est la preuve qu'il n'a pas été ajusté après coup — *écrire un contrôle
d'avance ne le rend pas juste, seulement non ajustable*.

## Les quatre correctifs du jour

| correctif | gain mesuré isolément | chemin |
|---|---|---|
| fusion bf16, décodage | +2,60 % | décodage |
| fusion bf16, prefill (88 jetons) | +4,23 % | **TTFT** |
| double tampon nvfp4 | +6,90 % | décodage |
| cache d'échelle nvfp4 | +7,91 % | décodage |

## Prédiction

Composition **multiplicative**, jamais une somme. Et **deux lignes, jamais un
titre** : le prefill porte sur le TTFT, les trois autres sur le débit. Les
additionner fabriquerait un nombre qui ne correspond à aucune grandeur
mesurable.

    decodage   1,026 x 1,069 x 1,0791 - 1  =  +18,4 %
    prefill                                   +4,23 %   (TTFT, ligne separee)

## Règle de lecture, posée maintenant

* **mesuré ≥ prédit − résolution** → les chemins sont disjoints, les quatre
  gains sont confirmés, et la composition multiplicative est validée comme
  méthode d'estimation ;
* **mesuré < prédit − résolution** → les chemins se recouvrent. **On ne garde
  alors pas les quatre chiffres avec un total plus petit.** Il faut
  ré-attribuer : le double tampon et le cache d'échelle touchent tous deux le
  décodage, c'est la paire suspecte, et elle se départage par un test à une
  variable — chacun seul contre la même base. À ne lancer que si le déficit
  dépasse la résolution.

Résolution du banc : 2,8 · σ · √(2/n), soit ≈ 0,4 % avec σ ≈ 0,2 % et n = 4.

## Barrière de qualité — condition ferme

**Quatre correctifs ont atterri aujourd'hui et aucun n'a été confronté à
l'étalon.** Trois se disent numériquement neutres, le quatrième change les
bits par construction (float32 au lieu de fp16 sur les échelles).

Avant que quoi que ce soit soit dit *acquis* : **un point de perplexité sur la
construction finale contre l'étalon consigné** — cumul, `min_context` apparié,
corpus au sha enregistré. Une exécution.

C'est le contrôle qu'on saute quand tout va bien, et le seul qui attrape une
régression silencieuse dans une journée à quatre correctifs.

## Sur les empreintes, pour éviter un faux signalement

Une empreinte est un contrôle d'**équivalence**, pas de **justesse** : elle dit
« est-ce le même calcul », jamais « est-ce le bon ». **Le passage des échelles
en float32 fait délibérément changer les bits** — `fp32 → fp16 → bf16` est un
double arrondi et peut différer d'un ulp de `fp32 → bf16` direct.

**Un changement d'empreinte est donc ATTENDU sur un modèle reconverti.** Une
empreinte inchangée signifierait au contraire que le correctif ne s'applique
pas. Ce n'est ni une régression ni du bruit.

## Conditions de forme

n ≥ 3 · passages de **régime** seulement · `W_passages` et `jkj_passages` à
deux décimales · empreintes sur le même périmètre que le passage publié ·
TTFT en **deux lignes**, froid et régime.

## Portée

« Publiable » ne veut pas dire publié. Cette campagne produit un résultat
**défendable** ; toute publication reste soumise à l'autorisation explicite de
l'utilisateur.
