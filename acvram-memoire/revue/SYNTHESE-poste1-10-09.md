# Synthèse d'poste1, 10/09/2026 — tout ce qui est établi, et ce qui ne l'est pas

Consolide onze documents de revue. **Chaque chiffre porte son régime** ; ce qui
n'est pas mesuré est marqué comme tel. Ordre : ce qui décide d'abord.

---

## 1. ÉTABLI — l'ordre du sac à dos budgétaire est mal orienté

A/B à budget égal, même code à un signe près, prédiction écrite avant la mesure.

```
                    promus     depense      PPL    vs etalon
bras A, ordre actuel  198/225  5,9859 Gio  5,4918   +1,435 %
bras B, ordre inverse 149/225  5,9979 Gio  5,4482   +0,630 %
ecart                                     -0,0436   (-0,794 %)
```

Prédit : `5,4144 <= PPL(B) < 5,4918`. Obtenu **5,4482**, dans la fourchette.

**Le bras inverse promeut 49 tenseurs de moins et fait mieux.** 56,3 % du retard
au plafond comblé en renversant un signe. Le glouton n'est pas seulement
sous-optimal : sur la tête de son classement il est **activement
contre-productif**.

Cause, `convert.py` : la clé est `-gain_db / cout`, soit un **gain de SNR par
octet**, quand ce qui décide est la perplexité par octet.

**Ce n'est PAS une borne.** Un ordre inverse est un ordre parmi 225 !, choisi
parce qu'il est facile à nommer. C'est un **plancher** sur le gain accessible
par le seul ordre — rien n'interdit qu'un meilleur ordre récupère 80 %.

### La clé fondée, et ce qui la distingue

`ACVRAM_ORDRE_SAC` accepte `snr` (défaut **inchangé**), `erreur`, `inverse`. Un
mode inconnu **lève** au lieu de retomber en silence. La clé `erreur` —
l'erreur de sortie évitée par octet — n'est **pas** une transformation monotone
de `snr` : elle applique `10^(-snr/20)` aux **deux** SNR avant la soustraction.

```
A   SNR 20 -> 30 dB   erreur 0,1000 -> 0,0316   gain 0,0684
B   SNR 40 -> 50 dB   erreur 0,0100 -> 0,0032   gain 0,0068
```

Vérifié sur nos données : corrélation **−0,6645** entre SNR de base et
déplacement de rang ; 20,83 dB de base moyen pour ceux qui montent contre 29,66
pour ceux qui descendent ; désaccord de tête **90 % au top-10, 3 % au top-100**.
Une exception signalée : l'un des six qui descendent le plus a 20,84 dB, ce qui
contredit le mécanisme pour lui. −0,66 est une tendance, pas une loi.

### Prédiction scellée du troisième bras

Deux témoins, **écart zéro tous les deux** : `snr` simulé rend 198 (= bras A
converti), `inverse` simulé rend 149 (= bras B). Deux comptes reproduits sur
deux ordres très différents.

```
bras erreur prevu    197 promus
FOURCHETTE   5,4144 <= PPL(erreur) <= 5,4918
ESPEREE      < 5,4482      la cle fondee bat l'instrument
NEUTRE       5,4482 a 5,4918
QUI ME GENE  >= 5,4482 ET base_croissant aussi bon
REFUTE       > 5,4918
```

**Une explication concurrente est déjà morte, sans mesurer** : `erreur` promeut
**197** quand `inverse` en promeut 149. Si `erreur` améliore l'actuel en
promouvant le même nombre que lui, « ça marche parce qu'on promeut moins » tombe
de soi-même. Le confondant restant — « on promeut les mal quantifiés » — est
testé par un quatrième bras nommé d'avance, `base_croissant` (163 promus).

---

## 2. ÉTABLI — la courbe du quota, six points, homogène

```
point                   bits/p      PPL   vs etalon   promus   bits/PPL
plancher tout-nvfp4     4,7297   5,6102    +3,622 %       —          —
budget 4,50 Gio         5,7396   5,5643    +2,774 %  110/225      22,00
budget 5,00 Gio         6,3737   5,5394    +2,314 %  147/225      25,47
budget 5,50 Gio         6,9905   5,5125    +1,817 %  172/225      22,93
budget 6,00 Gio         7,6320   5,4918    +1,435 %  198/225      30,99
plafond tout-int8       8,3453   5,4144    +0,006 %       —       9,22
```

**Monotone, sans plateau : l'hypothèse des « octets gratuits en bas » est
réfutée.** Le coût par point de perplexité monte dans l'intérieur.

Le rendement par tenseur promu, dans l'unité qui décide — **rectification, mon
premier chiffre était par tenseur et surestimait de trois fois** :

```
segment              n    bits   milli-PPL   /tenseur   milli-PPL PAR BIT
  0 ->   6           6  0,0551      1,10      0,183          19,96
172 -> 198          26  0,6415     20,70      0,796          32,27
198 -> 225          27  0,7133     77,40      2,867         108,51

facteur global par TENSEUR   15,64x   <- publie a tort
facteur global par BIT        5,44x   <- l'unite juste
```

Les 27 derniers sont **1,071 fois plus gros** et rendent **3,6 fois plus** : si
le rapport de tailles avait valu 3,6, il n'y aurait eu aucune inversion.

**Le témoin plafond ferme la réserve de mécanisme** : le sac à dos à budget égal
au plafond produit le même dossier que la conversion par format — **données
identiques à l'octet** (7 029 149 696 des deux côtés), 996 clefs identiques, et
les 128 octets d'écart entièrement dans les en-têtes, dus à un découpage de
fragments différent (600/396 contre 639/357).

Chaque point est auditable depuis son manifeste : budget dépensé à 0,2 et
2,3 Mio près.

---

## 3. ÉTABLI — `model.nbytes` compte les vues de fusion en double

```
                   nbytes (numel)   stockages uniques   dupliques par les vues
Llama-2-7b-int8      7 485 530 112       7 024 074 752              461 455 360
Qwen3B (4 formats)   2 914 429 472       2 722 240 512              192 188 960
Agents-4B bf16      12 499 507 200       9 060 177 920            3 439 329 280
```

`nbytes` somme des `numel()`, or les empileurs remplacent les originaux par des
**tranches** de la pile. Le compte ferme à **4 096 octets** sur fp16pur :

```
base + vues q,k,v + vues gate,up + embed compte deux fois = 22 731 034 624
mesure                                                      22 731 030 528
```

Les 4 096 restants sont les 2 048 paramètres que le manifeste compte en trop.
**Le 26,987 bits/poids n'avait ni cache, ni tampon, ni doublon accidentel :
c'était l'unité de mesure.** Facteurs mesurés : ×1,38 bf16, ×1,066 int8,
×1,071 quatre formats.

**Vérifié : aucun Go/s du dépôt ne divise par lui.** Les quatre usages sont de
l'affichage et deux champs de relevé ; les bancs divisent par `q.nbytes` d'un
tenseur fraîchement quantifié ou par un compte analytique. Les 634 Go/s et la
borne de 1050 Go/s sont intacts. `nbytes_detail()` porte désormais
`octets_stockage_uniques`.

---

## 4. ÉTABLI — deux empileurs sur quatre dupliquaient les poids

```
stack_plain_linears    (bf16)      vues   ok
stack_nvfp4_linears                vues   ok
stack_int8_linears                 vues du BIAIS seulement   non
stack_int4_awq_linears             idem                      non
```

Le commentaire de l'int8 affirmait « comme pour les poids » au-dessus du seul
code qui repointait le biais. Corrigé ; épreuve CPU qui **discrimine** (échoue
contre l'ancien code, passe sur bf16 déjà juste). Vérifié sur la carte : le
stockage unique tombe **exactement** sur la valeur prévue 7 024 074 752.

**Mais la portée est bien plus petite que mon titre** : arithmétique sur les 18
modèles du parc portant de l'int8 ou de l'int4_awq, tailles réelles contre
capacités réelles — **zéro bascule de placement, sur 5090 comme sur 3080 Ti**,
même à la borne supérieure. La chaîne doublon → dépassement → exil → −70 % **ne
se referme sur aucun modèle**. Le correctif reste bon parce qu'il est gratuit.

---

## 5. ÉTABLI — la fusion ne rapporte que +0,19 % sur un int8 calibré

A/B ABBA, une valeur par processus, 256 jetons :

```
avec fusion  172,55 et 172,64 j/s   5 groupes fusionnes
sans fusion  172,28 et 172,26       0 groupe
GAIN +0,19 %,  dispersion intra-bras 0,05 %  (ecart = 3,8x la dispersion)
```

Proportionnel à la couverture, vérifié à deux ancrages : `2,60 % × 7,81 % =
0,203 %` attendu contre 0,19 % mesuré, 6,5 % d'écart, sur deux modèles
différents.

**Pourquoi 5 groupes sur 64** : la recherche AWQ pose `s = act.pow(alpha)/mean`
où `act` ne dépend que de l'entrée — donc `gate` et `up` la partagent — mais
`alpha` est choisi en minimisant l'erreur de **ce** tenseur. `_scaler_commun`
exige `torch.equal` : deux alpha différents, refus. **Les 5 groupes fusionnés
sont exactement les 5 paires dont la recherche a choisi le même exposant.**

Confirmé **sans GPU** par une route indépendante — les `act_scale` du dossier :
`gate/up` 5 échelles égales sur 32, `q/k` 4 sur 31, `k/v` **0** sur 31. Aucun
qkv ne fusionne parce que k et v ne s'accordent jamais.

**Ce qu'un alpha commun vaudrait** : passer de 5/64 à ~50/64, soit de +0,19 % à
~+2,4 %, pour **zéro octet** depuis que les quatre empileurs rendent des vues.
Prix mesuré sur deux points : **45 groupes récupérables sur 64** à 4,50 Gio,
**40** à 5,00, **38** à 6,00, sous 2 % de surcoût relatif. Décroissant avec le
nombre de promus — donc **le gain est plus grand sur les dossiers les moins
promus**, ceux serrés en mémoire, contrairement à mon attente.

**Réserve maintenue** : 2 % est une erreur de **sortie** par tenseur, pas une
perplexité.

---

## 6. ÉTABLI — le compte exact des trois étalons, fermé à +0 octet

`P = 6 738 417 664` poids, part 16 bits **1,9491 %** = embedding 32 000 × 4 096
+ 65 normes × 4 096.

```
                fichiers        en-tetes      donnees   bits/poids
fp16pur     13 476 872 072        36 744  13 476 831 232    16,0000
int8         7 029 266 352       116 656   7 029 149 696     8,3452
nvfp4        3 983 834 740       117 488   3 983 717 252     4,7296
```

Excès sur la densité analytique, **identique à quatre décimales** pour les deux
formats : `0,3174` = part 16 bits `0,31186` + act_scale `0,00539` + inv_freq et
en-têtes `0,00014`.

**Trois copies de la densité existaient** ; celle qui décidait (`BPW_NOMINAL`,
écrite en dur) **ignorait `--group-size`** : à groupe 32 l'int8 réel vaut 8,750
et la constante 8,250 sous-estimait de 5,71 %, soit 401 Mio — le sens qui fait
croire qu'un modèle tient. Remplacée par `bpw_nominal(fmt, group_size)`.

Étalon extérieur `transformers`/GPTQ : **5,4141**. Notre chaîne : fp16 5,4142,
int8 **5,4144** sur le binaire courant, nvfp4 5,6102 (+3,622 %).

---

## 7. LE CORPUS — deux textes, deux références, 2,67 % d'écart

```
wiki-gptq.txt   e52922746ad09bac...  344 402 jetons  168 segments  PPL ref 5,4141
wiki.test.raw   173c87a53759e020...  335 688 jetons  163 segments  PPL ref 5,5625
```

Vérifié **au jeton** : tokenizers rapide et lent rendent des **suites**
identiques ; BOS présent des deux côtés (`[1, 29871, 13]`) pour `use_fast` vrai
et faux, donc **aucune des 168 frontières n'est décalée** ; la normalisation
GPTQ est une identité (le 2048 s'annule) ; le reste est jeté des deux côtés.

Notre chaîne lit `wiki-gptq.txt` — identifié par l'arithmétique avant que le
champ n'existe : **343 896 = 168 × 2047**, impossible avec 163 segments. Le
relevé porte désormais `corpus_chemin`, `corpus_octets`, `corpus_sha256`.

Et l'avertissement des trois relevés archivés (« un jeton compté deux fois ou
pas du tout ») est **FAUX** : mon invariant comptait 167 segments au lieu de
168, le nombre de segments étant le **plafond** de la division.

---

## 8. LES AVIS EXTÉRIEURS — triés, et le tri vaut plus que les avis

```
avis   identifiants   exacts   pointant ailleurs
 1          0            —            —
 2          4            0            4
 3          5            4            0  (+1 nom faux)
```

**Avis 2 : quatre sur quatre pointent vers des articles réels et sans rapport**
— la forme la plus dangereuse, parce que les liens s'ouvrent. **Avis 3** donne
les seuls liens utilisables : Concordia `2606.23521`, GAMMA `2605.18475`,
FAMPWQ `2608.24945`, spec-decoding `2508.08192` ; le cinquième annonce « IMPQ »
là où l'article est **CoopQ** `2509.15455`.

**Erreur factuelle commune aux avis 2 et 3 : H100 traité comme sm_120.** Il est
sm_90. C'est sur cette confusion que repose tout leur argument de
transférabilité. À porter en règle : **vérifier l'architecture de mesure avant
l'argument de transférabilité.**

**Le vrai résultat est négatif et convergent** : aucune mesure publique n'existe
pour nos trois questions — sm_120 en décodage paginé, désaccord SNR/octet contre
PPL/octet, joules par jeton en spéculation.

**Refusé comme utilisable** : les prédictions chiffrées sur nos propres nombres
(« de +3,62 % à ≈+1 % » est fabriqué) ; « la spéculation est votre meilleure
chance » par réduction de puissance statique — la décomposition
`J = (E_draft + E_verif + E_rejets) / A` l'interdit sans mesure ; le résultat
Kog, qui est sur AMD MI300X.

---

## 9. CE QUI N'EST PAS ÉTABLI, et pourquoi

- **le désaccord réel entre SNR et perplexité.** Le Spearman de +0,9163 que j'ai
  calculé est **presque tautologique** : les deux clés sont des fonctions des
  mêmes deux nombres. Seul le désaccord de tête est informatif. Répondre
  vraiment demande un ΔL par tenseur sur une perte de calibration ;
- **le facteur résiduel de `model.nbytes` sur un modèle non identifié.** Le
  couple 26,99/20,04 que j'avais publié : le 20,04 **n'a aucune source**, c'était
  mon arithmétique. Le seul chiffre mesuré est 26,987, et il est expliqué ;
- **le prix en perplexité d'un alpha commun.** 2 % de surcoût est une erreur de
  sortie, pas une perplexité ;
- **`duck.ai`** : accès **intermittents, une chance sur deux, indépendants du nom
  d'hôte** (4/8, 4/8, et 2/6 sur l'adresse nue). Mon alerte « injoignable » était
  fausse ; la correction « c'est duckduckgo qui est bloqué » l'était aussi. **La
  règle tient : réessayer**, trois tentatives donnent 87 %.

### Chiffres RETIRÉS, à ne pas laisser survivre

```
15,64x   facteur de rendement du sac a dos    ->  5,44x   (etait par TENSEUR,
                                                            l'unite est l'OCTET)
20,04    bits/poids « en bf16 »               ->  aucune source, mon arithmetique
+0,9163  desaccord SNR contre perplexite      ->  presque tautologique, les deux
                                                  cles derivent des memes nombres
6,4 %    « de VRAM reprise » comme argument   ->  0 bascule de placement sur 18 modeles
100x     etalement de l'echelle de sortie     ->  2,1x entre les quartiles 25 et 95
                                                  (le 98,7x tient a UN tenseur)
+2,60 %  gain de la fusion, transporte        ->  +0,19 % sur un int8 calibre
9,500    densite analytique de l'int8         ->  8,1875 a groupe 128
```

Le `100x` était une illustration de mécanisme présentée comme une estimation
d'ampleur ; elle a circulé dans trois messages du soir. **Le mécanisme reste
vrai** — et il est même plus intéressant que le facteur : le biais n'est pas
dispersé, il est **structurel**.

```
gate/up/down   123 a 131      les MLP
q/k             96 a  97
v/o                 65        deux fois moins
```

**La clé relative sur-promeut systématiquement les projections d'attention par
rapport aux MLP, d'un facteur deux, par construction.** Un biais orienté sur
225 tenseurs se cumule là où du bruit s'annulerait.

### Et l'agrégat qui suit la perplexité mesurée

Un sac à dos glouton suppose que l'objectif est **additif**. Aucune de nos clés
ne l'est : les décibels ne s'additionnent pas, un rapport ne s'additionne pas
entre tenseurs, et une erreur absolue s'additionne en **quadrature**. Testé
contre la seule vérité dont nous disposons — six perplexités mesurées :

```
                                              correlation avec la PPL mesuree
somme des erreurs RELATIVES                              +0,9223
somme des erreurs ABSOLUES (approchees)                  +0,9539
QUADRATURE des erreurs absolues                          +0,9931
```

**La quadrature suit la perplexité mieux que les deux autres**, ce qui est
cohérent avec une erreur qui se propage. Mais six points ne distinguent pas des
écarts fins, et les trois agrégats dérivent des mêmes `out_snr_db` : ce test
peut **écarter** un agrégat qui ne suit pas, il ne peut pas couronner le
meilleur des trois. Et il ne teste **pas** l'additivité elle-même — pour cela il
faudrait comparer la perplexité d'un ensemble de promotions à la somme de celles
de chaque promotion seule.

---

## 10. LES RÈGLES QUI SORTENT DE LA JOURNÉE

1. **Un chiffre peut être exact, dans son régime, avec son témoin, et rester
   faux comme aide à la décision** parce qu'il ne dit pas ce qu'il change.
2. **Un chiffre qui ne peut être obtenu que d'une seule façon finit par voyager
   hors de son régime**, parce que personne ne peut le contredire ailleurs.
   (Formulation de poste4 ; remède : un second chemin de mesure.)
3. **Quand un correctif ne se voit pas, soupçonner l'instrument avant le
   correctif** — surtout quand l'instrument mesure la grandeur que le correctif
   déplace.
4. **Dès qu'un résultat binaire peut être intermittent, mesurer un TAUX** et
   jamais un état ; et **tester la variable qu'on croit responsable en la
   retirant**.
5. **Un correctif qui change une convention doit être suivi jusqu'à ses
   appelants.** Quatre fois aujourd'hui.
6. **Un correctif non éprouvé ressemble à un correctif absent** — la seule
   différence est qu'on croit le problème résolu.
7. **Un témoin par habitude, pas par flair** : une règle qui dépend de
   reconnaître qu'un résultat nous flatte échoue quand il nous flatte le plus.
8. **Un outil livré avant sa campagne doit être inerte par défaut.**
9. **Trois niveaux de détachement** : les bras, le pilote, **et celui qui
   regarde**. Le superviseur « ne tue pas le coupable, il tue ce qui lui
   appartient » (poste4).
10. **Un seul `carte.sh`, et c'est le plus intérieur.** L'envelopper *et* le
    laisser dans les bras crée un interblocage avec soi-même qu'aucune attente
    ne résout.
