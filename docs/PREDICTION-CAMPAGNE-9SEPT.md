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

---

# RECTIFICATION, avant toute mesure

**La prédiction de +18,4 % ci-dessus est fausse, et elle reste écrite pour que
l'erreur soit visible.** Elle compose trois facteurs qui **ne s'appliquent pas
au même modèle**.

Vérifié par poste2 en chargeant les deux :

    bf16-pur   scalers actifs 0     tenseurs nvfp4 0    fusions 44 MLP + 48 attentions
    nvfp4      scalers actifs 336   tout en nvfp4       fusions 0 (96 refus)

* **sur le bf16**, la fusion opère — mais le double tampon vit dans
  `nvfp4_gemv`, sans un seul tenseur nvfp4 à toucher, et le cache d'échelle
  n'a **aucun scaler** à mettre en cache. **Seul le premier facteur
  s'applique** ;
* **sur le nvfp4**, le double tampon et le cache opèrent — mais la fusion est
  refusée sur les 96 groupes. **Seuls les deux derniers s'appliquent.**

**Aucun modèle ne porte les trois.** Le +18,4 % ne décrit aucune construction
mesurable.

## Prédiction corrigée

    modele bf16-pur    decodage  +2,6 %                     (fusion seule)
    modele nvfp4       decodage  1,069 x 1,0791 = +15,3 %   (tampon x cache)
    modele bf16-pur    prefill   +4,23 % sur le TTFT        (ligne separee)

**La règle de lecture reste entière** — elle s'applique à deux compositions au
lieu d'une. Et la paire suspecte qu'elle nomme, double tampon et cache
d'échelle, est **exactement** celle qui reste composée : le test à une
variable garde tout son sens si le nvfp4 déçoit.

## Ce que l'erreur enseigne

Composer multiplicativement était le bon geste ; **vérifier qu'un modèle porte
tous les facteurs ne l'était pas moins, et personne ne l'a fait.** Une
composition n'a de sens que sur une population où chaque terme s'applique. Ici
deux populations disjointes ont été traitées comme une seule parce que les
quatre correctifs venaient de la même journée.

*L'erreur va vers la conclusion bienvenue* : +18,4 % était le plus beau chiffre
disponible, et c'est celui qui n'a pas été vérifié.

## Condition d'exécution, ajoutée

Le chargement bf16 refuse les graphes à 32k de contexte — *poids en flux depuis
la RAM hôte*, 44 MLP fusionnés sur 48. **Le bf16 se mesure à 8192 ou 16384 de
contexte**, apparié des deux côtés, et la fenêtre retenue se déclare dans le
tableau.
