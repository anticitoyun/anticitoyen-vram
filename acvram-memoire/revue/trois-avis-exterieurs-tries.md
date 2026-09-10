# Trois avis extérieurs, triés par ce qui est vérifiable

Trois IA ont répondu à la question ancrée du 10/09. **Les identifiants arXiv ont
été vérifiés un par un contre l'API arXiv, avec témoin.** Le tri qui en sort est
plus instructif que les réponses elles-mêmes.

## D'abord : mon vérificateur était mort, et il accusait

Premier passage, `curl` sans `-L` : **les neuf identifiants sont sortis
« inexistant »**, y compris `2508.08192` qui était plausible. J'ai failli
rapporter que trois IA avaient inventé toutes leurs références.

Le témoin l'a arrêté : `1706.03762` (*Attention Is All You Need*) sortait aussi
« inexistant ». L'API arXiv répond **301** et `curl` sans `-L` ne suit pas la
redirection. **Une absence lue comme un résultat, et l'accusation la plus grave
possible fondée sur un instrument muet.** Refait avec `-L` et deux témoins qui
répondent.

## Le tri, et il est net

```
avis   identifiants cites   verifies exacts   pointant ailleurs   sans identifiant
 1              0                  —                 —              toutes
 2              4                  0                 4                 —
 3              5                  4                 0                 —
```

**Avis 2 : quatre identifiants sur quatre pointent vers des articles réels et
sans aucun rapport.** C'est la forme la plus dangereuse, parce que les liens
**s'ouvrent** :

```
2407.07722  annonce FlashAttention-3  ->  Characterization of SiPM Performance in a Small Satellite
2211.09835  annonce SmoothQuant       ->  Closed-form analytic expressions for shadow estimation
2401.07351  annonce EAGLE             ->  On min-base palindromic representations of powers of 2
2401.14431  annonce MTP               ->  Primordial naked singularities
```

**Avis 3 : quatre sur cinq exacts, titre et date compris.** Un seul écart, sur
le nom :

```
2606.23521  Concordia: JIT-Compiled Persistent-Kernel Checkpointing        22/06/2026  EXACT
2605.18475  GAMMA: Global Bit Allocation under Arbitrary Budgets           18/05/2026  EXACT
2608.24945  FAMPWQ: Fisher Information-based Adaptive Mixed Precision      24/08/2026  EXACT
2508.08192  Efficient Speculative Decoding for Llama at Scale              11/08/2025  EXACT
2509.15455  annonce « IMPQ »  ->  CoopQ: Cooperative Game Inspired         18/09/2025  NOM FAUX
```

Le contenu décrit pour « IMPQ » — sensibilité par valeur de Shapley — correspond
bien à CoopQ. **Le nom est faux, la substance juste.**

**Avis 1 : aucun identifiant numérique**, seulement des titres et des URL de
recherche, avec « non chiffré dans la source consultée » répété et « date exacte
à confirmer ». **Ses citations ne peuvent pas être fausses parce qu'il n'en
affirme aucune** — le moins immédiatement utilisable, et le seul qui ne demande
pas d'être vérifié avant d'être lu.

## Une erreur factuelle commune aux avis 2 et 3

Les deux traitent **H100 comme sm_120**. H100 est **sm_90** (Hopper) ; sm_120
est Blackwell. C'est sur cette confusion que repose leur argument de
transférabilité — « évalué sur SM 120 donc applicable à votre carte ». L'avis 1
ne commet pas l'erreur : il écrit « Hopper (sm_90) → sm_120 » et qualifie le
transfert d'extrapolation.

## Ce que les trois disent ensemble, et c'est le vrai résultat

**Aucune mesure publique n'existe pour nos trois questions.** Trois sources
indépendantes le disent, dont deux explicitement :

- pas de mesure du coût de lancement et de synchronisation sur **sm_120** en
  décodage paginé ;
- pas de comparaison publiée du **désaccord de classement** entre SNR par octet
  et perplexité par octet ;
- pas de mesure en **joules par jeton** pour MTP, EAGLE-3 ou Medusa, encore
  moins sur GPU sous-occupé.

L'avis 1 va jusqu'à écrire que notre facteur d'écart est « probablement un
résultat original et publiable ». **Une absence de réponse récente est un
renseignement** — c'était l'exigence de forme, et elle a payé.

## Ce qui est actionnable SANS rien d'extérieur

L'avis 1 donne la formule que notre chantier attendait :

```
valeur par octet(u) = ( ΔL(u en basse precision) − ΔL(u en haute precision) )
                      / octets supplementaires(u)
```

soit une **perte marginale par octet** au lieu d'un SNR marginal par octet.
**C'est exactement la grandeur que notre glouton n'utilise pas**, et c'est aussi
exactement ce que `--grille-erreurs` collecte déjà : l'erreur de sortie de
chaque tenseur à chacune des 21 valeurs de la grille AWQ. Il ne manque que de
substituer la clé de tri.

Trois substituts nommés, par coût croissant, et deux d'entre eux ont un article
vérifié :

```
erreur ponderee par l'activation / octet   faible    (SmoothQuant, id a retrouver)
Fisher ou Hessien diagonal / octet         moyen     FAMPWQ 2608.24945 (verifie)
Shapley avec interactions / octet          eleve     CoopQ  2509.15455 (verifie)
perte directe par unite / octet            eleve     l'oracle pratique
```

Et l'avis 1 nomme les métriques de comparaison à publier : **corrélation de
Spearman des classements, taux de désaccord top-k, perplexité à budget fixe**.
Nous avons déjà le troisième.

## Ce que je refuse de reprendre

- **Les prédictions chiffrées sur nos propres nombres.** L'avis 3 annonce que
  ses méthodes feraient passer notre pénalité NVFP4 « de +3,62 % à ≈+1 % ». Ce
  chiffre est fabriqué : aucune source ne peut prédire un résultat sur notre
  modèle et notre corpus.
- **« La spéculation est paradoxalement votre meilleure chance »** (avis 2), par
  réduction de la puissance statique. L'avis 1 donne la décomposition qui
  l'interdit :
  `J_spec = (E_draft + E_verif + E_rejets) / A`, favorable seulement si
  `E_draft + E_verif + E_rejets < A × E_base`. **Le taux d'acceptation seul ne
  suffit pas**, et aucune des deux affirmations n'est mesurée.
- **Le résultat Kog à 3 000 jetons/s** : il est sur AMD MI300X, et l'avis 1 le
  signale lui-même comme non transférable. À garder comme preuve qu'un décodage
  entier dans un noyau persistant est implémentable, pas comme un chiffre.
