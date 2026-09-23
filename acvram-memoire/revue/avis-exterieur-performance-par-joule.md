# Vecteurs de performance par joule : ce que nous avons mesuré, ce que la littérature dit

Réponse à la question-directive de Manon (10/09/2026). **`duck.ai` est
injoignable depuis cette machine** — voir la fin du document — donc la partie
extérieure vient de recherches web, et la partie interne de nos mesures.

## D'abord ce qui est MESURÉ CHEZ NOUS, parce qu'aucune IA extérieure ne l'a

Ces chiffres viennent de nos propres bancs et ils tranchent plusieurs des
treize dimensions sans avis extérieur :

```
l'exil d'une couche                 -61 a -70 % du debit des la premiere
la tranche adaptative               +88 a +104 % au decodage, qualite inchangee
PA_WARPS                            -72,9 % sur l'attention, +13,0 % de debit
graphes CUDA, MoE hybride           x2,17 ; dense +0,4 % ; et -11,5 Gio de pic
                                    si on les coupe sur un MLA
attention paginee                   6,1x sa borne memoire, donc PAS limitee
                                    par la bande passante — 49,5 % du temps GPU
fusion des projections              +2,60 % en bf16 pur (100 % des groupes),
                                    +0,19 % en int8 calibre (7,8 % des groupes)
quota (budget d'octets)             la courbe N'EST PAS plate ; meilleur
                                    rendement en HAUT, 15,77 bits par point
                                    de PPL contre 22 a 25 en bas
duel du 9/09 contre llama.cpp       1,96x au decodage et 1,46x en jetons/kJ
                                    CONTRE nous, a source et cache egaux
```

**La conclusion la plus actionnable est négative** : notre premier poste n'est
pas la bande passante. L'attention paginée tourne à 6,1 fois sa borne mémoire
et occupe la moitié du temps GPU. Tout vecteur qui achète de la bande passante
— quantification plus agressive, cache compressé — se paie donc en qualité sans
rien rendre sur ce poste-là. **C'est l'occupation et la forme des lancements qui
décident**, ce que confirment les trois gains les plus gros ci-dessus : tranche
adaptative, PA_WARPS, graphes.

## Ce que la littérature apporte, et qui reste À VÉRIFIER chez nous

**Quantification.** NVFP4 est annoncé à ×1,6 de débit sur bf16 pour **−41 %
d'énergie**, et le FP8 natif Blackwell à ±0,5 % de perplexité pour ×1,8 de
débit. Nos propres chiffres donnent nvfp4 à +3,62 % de perplexité contre
l'étalon extérieur — donc **l'annonce « FP8 quasi gratuit » n'est pas
transportable au FP4**, et nous avons la mesure pour le dire.

**Spéculation.** MTP plutôt qu'EAGLE, pour une raison qui pèse sur notre
créneau : **pas de second modèle en VRAM**, et les têtes MTP partagent le cache
KV du tronc au lieu d'en allouer un second. À des lots de 32-64, où l'acceptation
d'EAGLE-3 s'effondre, MTP tiendrait ×2. Réserve capitale, et elle est écrite
dans les sources : **la spéculation échange du calcul contre de la latence, le
calcul total par jeton restant à peu près le même.** Pour un objectif en
jetons par joule, ce n'est donc pas un gain acquis — c'est un gain *seulement
si* le GPU était sous-occupé, ce qui est justement notre cas au décodage. À
mesurer, pas à croire.

**Noyaux.** Deux voies concrètes : Flash Attention v2 pour sm_120 existe en
CuTeDSL (PR CUTLASS 3030), et un Top-K épars pour le décodage Blackwell est
implémenté en **noyau à un seul CTA, 512 fils, sans synchronisation globale**,
toute la communication passant par la mémoire partagée. Cette forme-là parle
directement à notre résultat : notre attention n'est pas limitée par la bande
passante, donc supprimer les synchronisations globales et les frontières de
lancement est exactement le levier qui nous reste.

**Génération de noyaux par agent.** CUDA-L1 annonce ×3,12 de moyenne, et un
agent atteint 92 à 100 % de « plus rapide que torch.compile » sur KernelBench.
À considérer comme outil, pas comme résultat : nos noyaux ne sont pas des
noyaux de benchmark.

## Les pièges, tirés de nos propres fautes de la journée

1. **Un gain mesuré dans un régime ne se transporte pas.** Le +2,60 % de la
   fusion valait pour 100 % de couverture ; en int8 calibré il vaut +0,19 %
   parce que 5 groupes sur 64 fusionnent seulement.
2. **Un chiffre qui ne peut être obtenu que d'une seule façon voyage
   hors de son régime**, parce que personne ne peut le contredire ailleurs.
3. **Ne pas diviser par `model.nbytes`** : il compte les vues de fusion en
   double, ×1,38 sur un bf16. Tout Go/s qui en dérive est faux dans le sens
   flatteur.
4. **La borne mémoire de la 5090 est ~1050 Go/s en lecture seule**, soit 58 %
   du pic annoncé. Un vecteur justifié par le pic théorique est justifié par un
   chiffre que la carte ne rend pas.
5. **Un banc qui refuse est une information, pas un obstacle.** Trois refus
   explicites de notre banc — plan rejoué, préfill servi par le cache, EOS
   avant la fin — existent précisément parce que chacun a déjà fabriqué un
   faux résultat.

## RECTIFICATION — la règle du groupe tient, et mon alerte était fausse

**J'ai écrit que `duck.ai` était injoignable et que la consigne permanente avait
perdu sa précondition. C'est faux, et la correction de Jérôme l'était aussi.**

Il a mesuré `duck.ai` à 200 en 0,145 s et `duckduckgo.com` en échec, donc
l'inverse de moi. J'ai remesuré : `duck.ai` en échec deux fois,
`duckduckgo.com` à 200 en 0,32 s. **Nos deux mesures étaient justes à
l'instant où nous les avons faites** — et nos deux conclusions étaient fausses,
parce que nous avons tous les deux attribué l'échec au **nom d'hôte**.

Le taux le dit, et deux anecdotes ne pouvaient pas :

```
duck.ai            4 reussites sur 8
duckduckgo.com     4 reussites sur 8
40.114.177.156     2 reussites sur 6   <- l'adresse NUE, sans nom d'hote
```

Les deux noms résolvent vers **la même adresse**, et l'échec suit l'adresse, pas
le nom. C'est **intermittent, autour de 50 %, et indépendant du nom d'hôte** :
la variable à laquelle nous attribuions l'effet ne discrimine rien.

**Conséquence pratique, et c'est tout ce qui compte : la règle tient, il faut
réessayer.** Une tentative a une chance sur deux ; trois tentatives donnent
87 %. Un échec unique sur `duck.ai` ne prouve rien et ne doit pas être rapporté
comme une indisponibilité.

**Ce qu'il faut retenir de la façon dont nous nous sommes trompés** : deux
mesures contradictoires, toutes deux exactes, sur une propriété qui n'est pas
stable. Chacun a publié la sienne comme un état du monde. Le remède n'est pas
de mesurer mieux — c'est de mesurer un **taux** dès qu'un résultat binaire
pourrait être intermittent, et de tester la variable qu'on croit responsable
en la retirant : ici, interroger l'adresse nue a suffi à disqualifier le nom
d'hôte en une commande.

## Ce que j'avais écrit, conservé pour que la faute reste lisible

**`duck.ai` et `duckduckgo.com` sont injoignables depuis cette machine.**

```
VERIFIE   DNS resout les DEUX vers la meme adresse : 40.114.177.156
VERIFIE   TCP 443 accepte la connexion
VERIFIE   TLS n'aboutit JAMAIS (10 s de delai, aucun certificat)
VERIFIE   example.com : TLS complet, certificat valide, 200 — depuis la
          meme machine, a la meme seconde
VERIFIE   le navigateur rend « error page » sur duck.ai
NON VERIFIE  la cause : filtre local, interception DNS, ou service en panne.
             Je ne peux pas la distinguer d'ici.
```

La consigne permanente « consulter `duck.ai` à chaque impasse » a donc perdu sa
précondition, et **chaque session la rencontrera**. La recherche web fonctionne
et sert de repli ; ce n'est pas la même chose — on n'y pose pas une question, on
y cherche des pages.
