# Le seuil qui tranche ETABLI.md:2476 — écrit avant la mesure

Objet : le protocole du profil par noyau (Laure) doit trancher « 26,09 Gio / 28,14 ms
= 995 Go/s effectifs ⇒ le décodage est mémoire-borne, pas l'ordonnancement ».
Trois exigences de recevabilité, dont la troisième neutralise la réserve `ncu`.

## 1. Le modèle se nomme, il ne se déduit pas

Un profil sur *un* dense ne dit rien de *ce* dense. Deux voies pour le nommer, dans
cet ordre :

- **par les octets** : 26,09 Gio de poids lus par jeton est une signature. Chercher au
  parc le converti dont la somme des octets de poids (hors tête liée, cf.
  `tete-liee-projection-sortie`) vaut 26,09 Gio ± 1 %. Si un seul candidat sort, il est
  nommé ; si plusieurs, la ligne n'est pas identifiable par là.
- **par le journal** : le passage a été produit par une commande, elle a laissé une
  trace datée.

Si aucune des deux ne nomme le modèle, **la ligne est retirée, pas re-mesurée**. Une
mesure sur un substitut ne réhabilite ni ne réfute un chiffre dont on ignore l'objet.

## 2. Le seuil ne s'invente pas : il se dérive du chiffre contesté

« Part faible » / « part comparable au MoE » ne sont pas des critères. Mais il n'y a pas
à choisir un seuil arbitraire — le tableau en contient un.

Soit `p` la part du temps du pas occupée par des noyaux qui **ne lisent pas de poids**
(attention paginée en tête). La bande passante réellement exigée des GEMM est

    B_corr = 26,09 Gio / (28,14 ms × (1 − p))

La borne mesurée de la 5090 en lecture est **1050–1095 Go/s** (`borne-memoire-5090`).
`B_corr` franchit 1050 Go/s dès que

    p > 1 − 995/1050 = **5,24 %**

**Au-delà de 5,24 %, le calcul ne s'affaiblit pas : il se contredit.** Il exige des GEMM
une bande passante que la carte n'atteint pas, donc l'une de ses deux prémisses est
fausse (les 26,09 Gio ne sont pas tous lus depuis la VRAM, ou les 28,14 ms ne sont pas
tous du transfert de poids). Dans les deux cas il ne soutient plus « ce qui limite est la
bande passante, pas l'ordonnancement ».

C'est le seuil à inscrire avant de mesurer. Il a la propriété que réclame la règle :
il est chiffré, et il est **dérivé de la conclusion qu'il doit trancher**.

## 3. La réserve `ncu` est close par la marge, pas par une correction

`ncu` surestime les noyaux courts d'environ 45 % (`ncu-surestime-les-noyaux-courts`),
et l'attention paginée est un noyau court : le biais pousse `p` vers le haut, du côté
qui fait tomber la conclusion. C'est bien la configuration où l'on se trompe volontiers.

Mais la question posée n'est pas « combien vaut `p` », elle est « `p` dépasse-t-il
5,24 % ». Et la marge est de **8 à 9×** : même en attribuant au biais la totalité de son
amplitude, 42,9 % deviennent 29,6 % et 49,5 % deviennent 34,1 % — toujours 6× au-dessus
du seuil. **Aucune correction du biais ne peut sauver le calcul.** Comme pour
l'`act_scale` en fp16, la question se clôt par la marge sans avoir à mesurer le terme
suspect avec précision.

Ce qui reste néanmoins obligatoire, et qui est un contrôle et non une correction :

- **Le contrôle positif du profileur.** Publier `Σ(temps ncu des noyaux) / t_mural
  non profilé`. Si le rapport n'est pas ≈ 1, l'excès est réparti sur toute la
  population — 549 lancements côté dense, dont beaucoup de noyaux courts qui ne sont
  pas l'attention. Sans ce rapport, aucune part n'est publiable, quelle que soit sa
  valeur. Avec lui, la part devient un rapport de deux quantités biaisées dans le même
  sens, ce qui est précisément ce qui la rend utilisable.
- **Le dénominateur réparé.** Le bon calcul de bande passante est
  `(octets de poids + octets de cache KV + activations) / t_pas`. Les octets de KV se
  comptent (`n_ctx × n_kv_head × d_head × 2 × 2` octets), ils ne se profilent pas. Ce
  compte est indépendant de `ncu` et il donne à lui seul un plancher de temps pour
  l'attention : `octets_KV / 1050 Go/s`. Publier ce plancher **à côté** de la part
  profilée : deux instruments qui ne partagent pas leur biais.

## 4. Si `p` était mesuré sans profileur du tout

Voie de secours, à garder en réserve et non à exécuter d'emblée : mesurer `t_pas` par
événements CUDA sans profileur, puis remplacer `paged_attn_partial` par un noyau inerte
de même signature de lancement, et reprendre `t_pas`. En décodage le graphe de lancement
ne dépend pas des valeurs, donc la substitution ne change que le noyau visé, et
`Δt / t_pas` est `p` **mesuré sur le montage réel, sans instrument distordant**. Le
résultat est faux numériquement : c'est acceptable ici, mais seulement si le montage
refuse explicitement de publier la sortie du modèle — sinon on tient le prochain chiffre
plausible qui se transporte.

---

# Suite du 10/09 — le modèle est nommé, le minorant est calculé, il ne clôt pas

Rectification acceptée (1c) : les 42,9 et 49,5 % viennent de
Qwen3-Coder-30B-A3B (MoE), pas de ce dense. **J'ai transporté une part d'une
population à une autre** — la faute que Laure a refusé de commettre une heure plus
tôt, et c'est ce refus qui rend son résultat utilisable. Le seuil de 5,24 % tient ;
la conclusion « il est franchi » était empruntée. Les deux chiffres sortent de la
marge.

## 1. Identification : fermée, par deux voies concordantes

**`Qwen2.5-Coder-14B-Instruct`, dense, bf16 pur, `ETABLI.md:1527`** (le duel du
9/09 ; le passage `nsys` porte sur le même montage).

Reconstruction depuis `config.json` (48 couches, `hidden 5120`, 40 têtes Q,
**8 têtes KV**, `d_head 128`, `vocab 152064`, `tie_word_embeddings: false`) :

    poids lus par jeton   26,06 Gio      dossier : 26,09    ecart 0,1 %
    fichier complet       27,51 Gio      dossier : 27,5     ecart 0,04 %

L'écart entre les deux est exactement la table d'embedding (1,45 Gio), dont une
seule ligne est lue au décodage — le dossier avait donc raison de distinguer les
deux chiffres. Contrôle indépendant : `1 / 28,14 ms = 35,5 t/s`, contre 35,3 t/s
au banc. Deux voies, la signature des octets et le débit impliqué, désignent le
même modèle et le même régime.

## 2. Le minorant par les octets de KV : 0,68 %, il ne franchit pas le seuil

`KV = 48 × 8 × 128 × 2 × 2 = 196 608 o/jeton` (192 Kio/jeton en bf16).

    ctx 1024  bf16   192,0 Mio   plancher 0,192 ms   0,68 % de 28,14 ms
    ctx 1024  q8_0   102,0 Mio   plancher 0,102 ms   0,36 %
    ctx  660  bf16   123,8 Mio   plancher 0,124 ms   0,44 %

**Seuil 5,24 %, minorant 0,68 % : sept fois en dessous.** La voie ne clôt pas le
dossier, et il faut le dire ainsi plutôt que de la présenter comme prometteuse.
Forme équivalente, même verdict : le dénominateur réparé donne
`(26,06 + 0,19) Gio / 28,14 ms = 1002 Go/s`, encore sous la borne de 1050.

## 3. Ce que le calcul apporte quand même : un contraste, et trois issues

Le cache KV ne pèse que **0,72 % des octets lus par jeton**. Donc si l'attention
paginée était limitée par la bande passante, elle prendrait 0,68 % du temps. Toute
part observée au-delà est du temps qui n'est **ni** lecture de poids **ni** lecture
de KV. Les trois issues, inscrites avant la mesure :

    p ~ 0,7 %          l'attention est memoire-bornee ; la conclusion de 2476 TIENT
                       et la ligne sort de « suspecte »
    0,7 % < p < 5,24 % la conclusion tient encore arithmetiquement, MAIS l'attention
                       est deja hors bande passante : le dossier gagne une cible
    p > 5,24 %         la conclusion tombe par contradiction (GEMM au-dela de la
                       borne), ET l'attention est revelee a plus de 7x son plancher
                       d'octets : une cible d'optimisation, pas un plancher physique

Aucune des trois n'est « le test n'a rien donné ». C'est la condition que je
réclamais pour les deux harnais, appliquée ici.

## 4. Donc la mesure est nécessaire, et c'est la voie sans profileur qui tranche

Le minorant ayant échoué à clore, l'ordre est : `t_pas` par événements CUDA hors
profileur, puis substitution de `paged_attn_partial` par un noyau inerte de même
signature de lancement ; `Δt / t_pas` est `p` sur le bon modèle et sans instrument
distordant. Condition maintenue : le montage refuse explicitement de publier la
sortie du modèle. Un profil `ncu` reste acceptable en second, à condition de
publier `Σ(temps ncu) / t_mural`.

Annotation d'`ETABLI.md` : **suspecte**, pas réfutée. Elle attend son instrument, et
l'instrument est maintenant désigné.

---

# Rectification du 10/09 — mon minorant lisait un cache que nous n'avons pas

**Le cache K/V d'acvram n'est ni bf16 ni q8_0 : il est int8 avec une échelle fp16 par
vecteur.** Lu dans la signature du noyau, `acvram_kernels.cu:749` :

    kc [NB, 16, HKV, D]  signed char        vc  idem
    ks [NB, 16, HKV]     __half            vs  idem

Une échelle par (bloc de 16, position, tête KV), donc une par vecteur de D = 128
valeurs : `(128 + 2) × 2 = 260` octets par position et par tête KV.

    octets KV par jeton   48 × 8 × 260 = 99 840 o = 97,5 Kio   (et non 192 Kio)
    ctx 1024              97,5 Mio   plancher 0,097 ms   0,346 %
    ctx  660              62,8 Mio   plancher 0,063 ms   0,223 %
    part des octets lus                                  0,36 %

**Minorant rectifié : 0,346 %, contre 0,68 % annoncé.** J'avais écrit
`n_ctx × n_kv_head × d_head × 2 × 2` en présentant le dernier facteur comme les
octets — c'était le cache de quelqu'un d'autre. Le verdict ne change pas de sens (le
seuil de 5,24 % est maintenant à **15×** au-dessus), mais le chiffre était faux et
il provient de la même faute que ce dossier traque : une constante prise ailleurs.

## Un instrument par le compte, non profilé et exact

`PA_CHUNK = 512`, `C = ceil(N × 16 / 512)` ; à ctx 1024, `C = 2`, donc **deux noyaux
lancés par couche** (`partial` puis `reduce`) :

    lancements d'attention par pas   48 × 2 = 96
    lancements totaux (nsys)                549,3
    part des lancements                    17,5 %

C'est la mesure « par le compte » que réclamait l'annotation d'`ETABLI.md`, et elle
ne passe par aucun profileur. Elle ne prouve rien sur le temps — mais si ces 96
noyaux avaient le temps moyen des 549, ils feraient 17,5 %, soit **3,3× le seuil**.
L'issue 3 n'a donc rien d'extravagant. À noter au passage, fait de code et non
mesure : la grille du `partial` vaut `1 × 40 × 2 = 80 blocs` pour 170 SM.

## Le défaut de MA voie : elle est biaisée dans le même sens que `ncu`

J'ai vendu « deux instruments qui ne partagent pas leur biais ». C'est faux.
`Δt / t_pas` par substitution porte deux biais de signes opposés :

    le noyau inerte garde son cout de lancement (96 lancements)  ->  Delta t SOUS-estime p
    retirer l'attention libere le L2 et la bande passante des
    GEMM, qui accelerent                                          ->  Delta t SUR-estime p

Le second n'est pas borné a priori, et il pousse du côté qui fait tomber la
conclusion — exactement la configuration que `ncu` nous faisait craindre, revenue
par ma propre voie.

## Remède : trois bras, dont un contrôle positif à valeur prédite

    A   attention reelle
    B   noyau inerte : meme grille, memes blocs, ecrit part / part_m / part_l
    C   noyau qui LIT tout le KV (meme parcours de kc, vc, ks, vs) et ecrit une
        reduction triviale — le trafic memoire de A, sans son calcul

    C − B  =  cout de la lecture du KV seule    PREDICTION : 0,097 ms  (0,346 %)
    A − C  =  le calcul de l'attention
    A − B  =  p, dont le biais L2 est desormais MESURE et non suppose

Le troisième bras convertit le biais L2 d'inconnu en quantité mesurée, et `C − B`
arrive avec sa valeur écrite d'avance : l'instrument peut se tromper de façon
détectable. Sans lui, `A − B` est un majorant dont personne ne connaît la marge.

Mise en œuvre : **un seul interrupteur** `ACVRAM_PA_ARM=A|B|C` lu une fois, pas trois
binaires — et vérification du **contenu** du `.so`, pas de sa date
(`ccache-rend-un-noyau-cuda-perime`). Garde reprise à mon compte par 1c :
hors bras A, le moteur **refuse de servir du texte** — une erreur dans le chemin de
sortie, pas un avertissement.

---

# PREDICTION ECRITE AVANT LA MESURE — 10/09, binaire bbbb85c59e8f

Demandée par 1c avant que le banc tourne, et posée sans rien ajuster à
l'inférence de Manon (64–76 %).

**J'attends `p` entre 11 et 25 %, et je ne crois pas les 64–76 %.**

## D'où vient ma prédiction — de son tableau isolé, pas de son inférence

Son balayage donne, à `chunk = 512` : 78,72 µs par appel dont 11,81 de plancher de
harnais, soit **66,91 µs de travail**. Et son propre constat — le temps double quand
la tranche double — dit que ce travail est **le temps d'UNE tranche**, donc
indépendant du contexte total. Il y a 48 à 96 lancements de `partial` par pas :

    48 appels   3,21 ms   11,4 % de 28,140 ms
    96 appels   6,42 ms   22,8 %

C'est indépendant du profileur ET de son équation. Et c'est compatible avec le
profileur **corrigé de son biais** : 42,9–49,5 % surestimés de 45 % donnent
29,6–34,1 %, du même ordre que ma borne haute.

## Pourquoi les 64–76 % sont probablement trop hauts

L'équation `(1 − p) + p/3,04 = 1/2,044` attribue **tout** le gain de débit à
l'accélération du noyau. Or le chunk adaptatif ne change pas seulement le temps par
tranche : `C = ceil(N × 16 / chunk)`, donc passer de 512 à 64 **multiplie C par 8** —
huit fois plus de lancements, donc huit fois plus de blocs, donc un parallélisme et
un temps mort différents. Un changement qui touche trois grandeurs ne permet pas de
résoudre pour une seule inconnue.

Deuxième réserve, plus fine : son facteur 3,04 est un rapport de **temps mesurés**
(78,72 / 25,47), qui incluent chacun les ~11,5 µs de plancher de son harnais. Le
rapport des **travaux** vaut 66,91 / 10,11 = 6,6. Selon qu'on prend 3,04 ou 6,6,
l'équation rend 76 % ou 57 % : le résultat dépend d'un choix de dénominateur qui
n'est pas discuté. C'est la faute du jour, une troisième fois.

## Ce que chaque issue voudra dire, écrit d'avance

    p entre 11 et 25 %   ma prediction tient. La conclusion d'ETABLI.md:2476 TOMBE
                         quand meme — 5,24 % est depasse d'un facteur 2 a 5 — mais
                         par un facteur 4, pas 13. Les 64-76 % sont alors une
                         inference, pas une mesure.
    p ~ 30-35 %          le profileur corrige du biais avait raison ; ma borne haute
                         est un peu basse et le compte de lancements est plutot 96.
    p entre 60 et 80 %   je me suis trompee et son inference etait juste : alors le
                         temps d'un appel de partial dans le MOTEUR est trois fois
                         celui du meme appel dans son harnais, et il faut expliquer
                         cet ecart avant de publier l'un ou l'autre.
    p sous 5,24 %        les trois instruments se contredisent ; c'est MON montage
                         qu'il faut suspecter d'abord, parce qu'il est le plus jeune.

Aucune de ces quatre issues ne se lit « le test n'a rien donné ». Et la troisième est
la plus intéressante : elle ne dirait pas que j'ai tort, elle dirait qu'un même noyau
coûte trois fois plus dans le moteur que dans le harnais — ce qui serait le premier
renseignement sur la cause du plancher que nous cherchons depuis hier.

Conditions déclarées : `PA_CHUNK = 512` (production, celle du tableau), binaire
`bbbb85c59e8f`, cache par arbre, ctx 1024, ordre A B C C B A.

---

# RESULTAT — 10/09 09:47, binaire bbbb85c59e8f, et DEUX de mes controles echouent

    A (reel)     33,286 ms   [33,113 – 33,578]   deux occurrences a 0,138 ms
    B (plancher) 28,088 ms   [28,068 – 28,170]   deux occurrences a 0,125 ms
    C (KV lu)    30,770 ms   [30,715 – 30,843]   deux occurrences a 0,005 ms

    A - B   p                      5,198 ms
    A - C   calcul de l'attention  2,516 ms
    C - B   lecture du KV          2,682 ms      predit 0,097 ms   x27,65

Témoin passé : trois sorties distinctes, chaque bras reproductible. 198 pas retenus
par bras, ordre A B C C B A, `PA_CHUNK = 512`, 0 couche exilée.

## 1. Ma prédiction tient — et mon script a emprunté un dénominateur

`p = 5,198 ms`. Ma prédiction était **11–25 %** : dans la fourchette, quel que soit le
dénominateur retenu. Mais lequel ?

    p / MON pas (33,286 ms)        15,62 %      <- le seul denominateur mesure ici
    p / pas du tableau (28,140)    18,47 %      <- EMPRUNTE

**Mon script publiait le second.** J'ai écrit `T_GPU_MS = 28.140` en tête du banc et
divisé par cette constante, alors que mon pas mesure 33,286 ms. C'est exactement « le
dénominateur emprunté » que j'ai consigné en mémoire ce matin après l'avoir retiré
trois fois chez les autres — commis dans le banc écrit pour l'éviter, et le contrôle
qui l'a révélé est celui que j'avais moi-même inscrit à la dernière ligne.

## 2. Conséquence : mon `p` ne réfute PAS le tableau

Le seuil de 5,24 % était dérivé du pas du tableau. Sur **mon** pas, la question se
repose et la réponse change :

    GEMM de mon pas : 26,06 Gio / (33,286 ms x (1 - 0,1562)) = 996 Go/s

**996 Go/s, sous la borne de 1050 : aucune contradiction sur mon pas.** Et la raison
est arithmétique, pas physique — mon pas est 15,8 % plus lent, donc il laisse plus de
temps pour lire les mêmes octets. Un montage plus lent affaiblit mécaniquement le
seuil qu'il devait franchir.

Transporter `p` au pas du tableau donne 18,47 %, soit 3,5 fois le seuil, donc l'issue
3. Mais **c'est un transport, sous une hypothèse à nommer** : que le noyau d'attention
coûte le même temps absolu dans les deux moteurs. Elle est plausible — même noyau,
même `PA_CHUNK`, ctx voisin — et elle n'est pas vérifiée. Mon quatrième garde-fou
s'applique : **le contrôle du montage a échoué (+15,8 %), donc c'est mon montage
qu'on suspecte d'abord, parce qu'il est le plus jeune.**

Écarts de conditions non appariés, à corriger avant de publier un verdict : invite
pseudo-aléatoire contre invite réelle, ctx moyen 924 contre 1024, graphes CUDA non
déclarés, cache KV du banc apparié q8_0.

## 3. L'échec du contrôle positif est le résultat le plus solide de la journée

`C − B = 2,682 ms` contre 0,097 prédits. Un contrôle positif qui échoue signifie
l'une de deux choses, et il faut les séparer au lieu de choisir la commode :

- **l'instrument est faux** — mais le bras C est le plus reproductible des trois
  (0,005 ms entre ses deux occurrences), il produit une sortie distincte de A et de B,
  et il se place entre eux dans l'ordre attendu ;
- **la prédiction était fausse** — elle supposait la lecture du KV **à la borne** de
  1050 Go/s. C'est précisément l'hypothèse que ce dossier cherchait à tester.

Le compte :

    KV par pas      88,0 Mio uniques, x5 relectures de GQA = 440 Mio
    en 2,682 ms     172 Go/s avec les relectures, 34 Go/s en octets uniques
                    16,4 % de la borne  ->  6,1x sa borne memoire

**L'attention paginée n'est pas limitée par la bande passante.** Elle est à 6,1 fois
sa borne, et l'explication est dans la grille : `1 x 40 x 2 = 80 blocs` de 4 warps
pour 170 SM — on ne mesure pas un débit, on mesure une latence faute de parallélisme.
Et ce chiffre concorde avec le « 8,6 fois sa borne à 3007 jetons » obtenu par un
chemin indépendant : ni le même montage, ni la même hypothèse.

## 4. Ce qui est acquis, et ce qui ne l'est pas

    ACQUIS      p = 5,198 ms, et l'attention se partage en 2,516 ms de calcul
                pour 2,682 ms de lecture — moitie-moitie
    ACQUIS      la lecture du KV est a 6,1x sa borne memoire : la bande passante
                n'est pas ce qui limite ce noyau
    ACQUIS      ma prediction de 11-25 % tient sur les deux denominateurs
    PAS ACQUIS  que la conclusion d'ETABLI.md:2476 tombe. Sur mon pas les GEMM
                exigent 996 Go/s, sous la borne. Le transport au pas du tableau
                donne 18,47 % et ferait tomber la conclusion, mais mon montage
                est 15,8 % plus lent que le banc et c'est lui qu'on suspecte
                d'abord. La ligne reste SUSPECTE.

Ce qu'il faut pour clore, et c'est peu : rejouer le bras A dans les conditions
exactes du banc — invite réelle, ctx 1024 franc, graphes déclarés, cache apparié —
jusqu'à retrouver 28,7 ms. Alors `p` aura le dénominateur qu'il mesure, et le seuil
redeviendra applicable.
