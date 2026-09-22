# Barrière de qualité : protocole, écrit avant la première mesure

Six correctifs ont atterri le 9 septembre 2026. **Aucun n'a été confronté à
l'étalon.** Cinq se disent numériquement neutres, un change les bits par
construction. Ce document fixe ce qui sera mesuré, comment, et **comment on
constate que l'instrument pouvait rendre autre chose**.

## L'étalon

    corpus   /mnt/AI_GENERATOR/corpus/wiki.test.raw
    sha256   173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
    taille   1 290 590 octets

**Le sha est vérifié avant chaque mesure**, pas supposé. Un corpus différent
rendrait une perplexité différente sans que rien ne le signale.

> **Précision du 10/09, après une divergence relevée par 0a entre ce
> document et la chaîne de mesure.** Ce corpus-ci est le bon POUR CETTE
> barrière, où le texte ne sert qu'à fabriquer une invite : la comparaison porte
> sur la sortie du noyau contre l'attention dense sur les mêmes tenseurs, et le
> corpus n'y change rien. **Il n'est PAS le bon pour une perplexité comparée à
> l'étalon extérieur** : le protocole GPTQ ne lit pas de fichier brut, il
> construit `"\n\n".join(testdata['text'])` sur le split test de
> `wikitext-2-raw-v1`, ce qui donne 344 402 jetons et 168 segments contre
> 335 688 et 163 ici. **Les deux références sont séparées de 2,67 %** (5,4141
> contre 5,5625) et confondre les deux attribuerait cet écart aux bits. Le
> corpus de l'étalon est `/mnt/AI_GENERATOR/corpus/wiki-gptq.txt`, sha
> `e52922746ad09bac`. Voir `acvram-memoire/corpus/FICHE-CORPUS.md` — aucun des
> deux documents ne mentait, ils parlaient de deux mesures différentes.

## Le cadrage

    acvram eval <modele> --corpus wiki.test.raw \
        --window 512 --stride 512 --min-context 256 --max-tokens 16384

**32 fenêtres**, tranché par Océane. Le cadrage fait le chiffre : le même
modèle sur le même corpus donne **7,233** à `min_context` 256 et **9,525** à 0.

**Un modèle par processus.** `acvram eval` chargeait N modèles sans libérer le
premier ; le second était alors mesuré sur les restes — constaté ici même, OOM
à 111 Mio libres sur 31,36 Gio après 32 MLP exilés.

## Le témoin d'instrument — sans lui, une réussite ne démontre rien

**Un dispositif qui rend « conforme » à tous les coups ne mesure peut-être
rien.** Avant de conclure quoi que ce soit, la barrière doit montrer qu'elle
sait dire non :

    temoin NEGATIF   le meme modele, deux executions   ->  ecart nul attendu
    temoin POSITIF   bf16 contre nvfp4, meme source    ->  ecart NON NUL attendu

Le second est une différence **connue et de sens connu** : la quantification
dégrade. **Si la barrière ne distingue pas bf16 de nvfp4, elle ne mesure pas la
qualité** et son verdict sur les correctifs ne vaut rien.

## Le seuil

    ΔPPL / PPL <= 0,1 %

Ancré et non posé : la divergence d'ordre de sommation vaut 5,4e-3 sur des
activations, et une perplexité l'amortit d'au moins un ordre de grandeur.

**Ne pas utiliser le ± de l'estimation.** Il mesure la dispersion **entre
morceaux du corpus**, commune aux deux exécutions puisque c'est le même corpus
dans le même ordre : **elle s'annule dans la différence.** La prendre pour
l'erreur de la différence serait la propriété voisine sur le dernier contrôle
de la journée.

## Les classes

    classe A   identique au bit pres   ->  PPL strictement egale, tout ecart = defaut
               f543e9d, eb4c099, 04d0da7, cache d echelle
    classe B   bits differents         ->  residu numerique attendu
               095b5ef (ordre des sommes), 225aa5f (float32)

**L'ensemble hérite de la classe la plus faible.**

## L'« avant »

**Produit par le code, jamais repris d'un journal.** Un `git stash push` sur
arbre propre ne remise rien et laisse mesurer un correctif contre lui-même —
cinq passages plausibles, +0,9 % annoncé là où le vrai gain valait +6,9 %.

**Vérification de l'état du dispositif avant chaque mesure**, par une propriété
observable du binaire, pas par la confiance dans la commande qui l'a produit.


## Ce que la carte borne : les bf16 de 14B ne s'étalonnent pas contre eux-mêmes

Première exécution, `Qwen2.5-Coder-14B-pur-bf16` :

    carte 30,85 Gio libres sur 31,36 au depart
    plan reajuste : 5 MLP de plus en RAM hote (27,3 Gio pour 29,6 libres)
    arene epinglee : 1,98 Gio
    OutOfMemoryError : 55,88 Mio libres, il manquait 136 Mio

**Ce n'est pas un incident, c'est une borne.** Un bf16 de cette taille exile
déjà cinq MLP au chargement et ne laisse pas de quoi évaluer. **La classe de
modèles que cette barrière peut étalonner contre leur propre bf16 s'arrête donc
en dessous de 14 milliards de paramètres sur cette carte.**

Conséquence pratique : le témoin positif se fabrique sur un modèle **petit**,
converti deux fois depuis la même source.

## État de la barrière au 9 septembre 2026, 20 h

    temoin NEGATIF   PASSE   nvfp4 deux fois : 7,270 et 7,270, deux processus
    temoin POSITIF   ABSENT  le bf16 de 14B ne tient pas a l'evaluation

**La barrière a la reproductibilité et n'a pas la sensibilité.** Les deux sont
nécessaires ; seule la première est acquise. **Statut : suspendue faute de
témoin**, et non « passée ».

Le couple à fabriquer : `Qwen3-0.6B`, même source GGUF, converti en nvfp4
(existe) et en bf16 (à produire). Écart attendu **connu de signe** — le bf16
doit être meilleur, le nvfp4 étant une approximation du même modèle. Deux
chiffres identiques sur ce couple signifieraient un instrument aveugle.

## Ce que chaque mesure exerce — la ligne qui manquait

Un correctif n'est pas éprouvé parce qu'il est présent dans l'arbre : il faut
que la mesure **emprunte son chemin**. Sans cette ligne, « la classe A ne
change pas la perplexité » se lit comme une affirmation sur six correctifs
alors qu'elle en couvre trois.

    revision   fichier                    exerce par une eval de converti ?
    f543e9d    acvram_kernels.cu          oui   noyau gemv nvfp4
    eb4c099    engine/layers.py           oui   fusion nvfp4
    670db7f    quant/calibrate.py         oui   cache d echelle au chargement
    095b5ef    engine/layers.py           oui   36 groupes a biais empiles
    04d0da7    quant/convert.py           NON   chemin de conversion
    225aa5f    quant/calibrate.py         NON   chemin de conversion
    f380b62    kernels/cpu.py, nvfp4.py   NON   chemin d execution CPU

**Les deux « NON » de conversion** ne sont pas une lacune du protocole : un
modèle déjà converti ne peut pas les exercer. `225aa5f` porte sur
`search_channel_scales` et `state_dict` — son propre message le dit, « les
modèles déjà convertis restent en fp16, le lecteur prend le dtype qu'il
trouve ». Les éprouver demande de **reconvertir**, pas de réévaluer.

**Le « NON » de `f380b62` est d'une autre nature, et il vaut d'être connu :
aucun réglage de mesure n'atteint ce chemin sur une carte qui a de la place.**
`ACVRAM_EXIL_COUCHES` pose `mlp_storage="cpu"` mais réaffirme
`mlp_exec="gpu"` (loader.py:1252) : les poids passent en RAM hôte, le calcul
reste sur la carte, et `int4_matmul_cpu` n'est jamais appelée.
`ACVRAM_MLP_HOTE_CPU` n'agit que dans le réajustement sous contrainte de VRAM
(loader.py:1110), qui ne se déclenche pas quand la carte n'est pas pleine. Le
chemin n'est donc atteignable qu'en **saturant réellement la VRAM**.

## Le classement de 095b5ef était faux : c'est de la classe A

Le protocole le rangeait en B au titre de l'ordre des sommes. **L'observable
dit autre chose.** Une sonde sur le modèle, avant et après le correctif :

    avant 095b5ef   72 groupes fusionnes, dont 36 a biais
    apres 095b5ef   72 groupes fusionnes, dont 36 a biais

Le correctif ne change pas **ce qui est fusionné** mais **où le biais est
rangé** : une concaténation, puis des vues `narrow` portant les mêmes valeurs.
La disposition mémoire change, l'arithmétique non — donc classe A.

Mesuré en conséquence : perplexité 9,928 des deux côtés, et 0 amorce sur 12
change à l'empreinte.

## L'empreinte de generation : ce qu'elle prouve, et ce qu'elle ne prouve pas

12 amorces disjointes, 128 jetons, greedy, un serveur neuf par passage — le
serveur en place est **toujours tué d'abord**, faute de quoi `acvram-serveur`
sort en `exit 0` sur un modèle déjà servi et l'empreinte porterait sur le code
précédent.

Éprouvée dans les deux sens avant de servir :

    temoin NEGATIF   0 amorce sur 12 differe entre deux processus du meme code
    temoin POSITIF   2 amorces sur 12 different quand ACVRAM_FUSION_NVFP4=0

**Sensibilité mesurée : 17 %.** Un texte ne bouge que si deux candidats se
croisent. **L'empreinte prouve qu'un changement a eu lieu ; son immobilité ne
prouve pas l'identité** et ne doit jamais être lue dans ce sens. Quand elle ne
bouge pas, c'est une sonde d'état — ici le compte des fusions — qui dit si le
chemin a seulement été emprunté.

Deux gardes ont servi pendant cette campagne, et toutes deux ont attrapé une
erreur avant la mesure, pas après :

* `ACVRAM_SEUIL_FUSION` est un **plafond** (`t <= SEUIL_FUSION`) : le lever
  active la fusion au lieu de la couper, et au décodage `t=1` la franchissait
  déjà. Le premier témoin positif ne testait rien — il rendait l'empreinte de
  référence, à l'identique.
* `pbiais` existait **déjà 4 fois** dans le code d'avant `095b5ef` : pris comme
  motif de garde, il aurait approuvé l'état qu'il devait refuser. Le motif
  retenu compte 1 avant et 3 après, et la garde compare au **nombre exact**.

## Un champ que l'API annonce sans le produire

`logprobs` et `top_logprobs` sont déclarés dans `server/protocol.py:72` et ne
sont jamais remplis : une complétion avec `"logprobs": 5` rend `null`. Ils
auraient donné une empreinte bien plus fine que le texte — toute différence de
bit apparaît dans les décimales d'une log-probabilité, là où un jeton ne
bascule qu'au croisement. Signalé, non corrigé.
