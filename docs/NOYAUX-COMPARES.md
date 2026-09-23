# Où passe le temps GPU : acvram contre llama.cpp, noyau par noyau

> **Note de renommage (9 septembre 2026)** : le modèle témoin cité ici sous
> `Qwen2.5-Coder-14B-bf16-pur` a été renommé **`Qwen2.5-Coder-14B-pur-bf16`**
> lors du rangement de `models_acvram`. Contenu identique ; seul le chemin
> change. Les chiffres publiés sous l'ancien nom restent valides.


Mesuré le 9 septembre 2026 sur RTX 5090, `Qwen2.5-Coder-14B-bf16-pur`, les deux
moteurs sous `ncu` avec les mêmes métriques.

## Dispositif

**acvram** : `Engine.step()` — le chemin du banc, vérifié à 34,38 pas/s contre
34,3 t/s au banc. Quatre pas délimités par NVTX, huit pas de chauffe hors
plage. Garde : les graphes CUDA étant actifs sur ce modèle, les quatre pas
doivent produire quatre rejeux, sinon le chiffre est refusé.

**llama.cpp** : `~/llama.cpp/build/bin/llama-cli`, le build CUDA du dépôt —
celui que la colonne `binaire` du banc désigne, pas le build Vulkan de
`/mnt/AI_GENERATOR/llamacpp/officiel/`. Ils n'ont pas de plage NVTX à nommer :
on profile **deux exécutions séparées**, `-n 4` et `-n 8`, et on soustrait. Le
chargement, le prefill et la sortie sont identiques et s'annulent. Deux
processus distincts : aucun cache de préfixe partagé.

## Trafic mémoire — nous ne lisons pas plus qu'eux

| | acvram | llama.cpp |
|---|---|---|
| DRAM lu | **26,092 Gio/pas** | 26,131 Gio/pas |
| coalescence | **32,0 o/secteur** | **32,0 o/secteur** |
| L2 secteurs lus | **906 585 222** | 955 093 100 |
| rapport L2/DRAM | **1,035** | 1,089 |

Le volume lu est à 0,12 % du théorique (26,06 Gio) et la coalescence est
parfaite des deux côtés. Sur le L2, **c'est leur noyau qui relit 5 % de plus**.
Trois hypothèses tombent : nous ne lisons ni plus, ni moins bien.

## Temps GPU par famille — l'écart est dans le MLP

`gpu__time_duration.sum`, en nanosecondes.

| famille | acvram | llama.cpp | écart |
|---|---|---|---|
| GEMV / matmul | 337 noyaux — 18,55 ms | 289 — 17,98 ms | **+0,56 ms** |
| elementwise / silu | 251 noyaux — 0,54 ms | **2** — 0,006 ms | **+0,54 ms** |
| rmsnorm | 97 — 0,61 ms | 97 — 0,50 ms | +0,11 ms |
| attention | 48 — 0,45 ms | 96 — 0,36 ms | +0,10 ms |
| cache kv | 48 — 0,16 ms | 48 — 0,10 ms | +0,07 ms |
| rope | 48 — 0,12 ms | 96 — 0,22 ms | **−0,10 ms** |
| **total** | **20,45 ms/pas** | **19,17 ms/pas** | **+6,7 %** |

**Deux postes font 86 % de l'écart, et ils désignent le même endroit.**

1. **Sept GEMV par couche contre six.** Ils fusionnent `gate` et `up` en une
   seule multiplication ; nous en faisons deux. 48 GEMV de plus par pas.
2. **251 noyaux élémentaires contre 2.** Silu et le produit `gate ⊙ up` sont
   fusionnés chez eux ; chez nous ce sont des
   `at::vectorized_elementwise_kernel` de PyTorch, un par opération.

**Là où nous les battons** : `rope` (48 noyaux contre 96, et plus rapide) et
l'attention en une passe (`paged_attn_partial`) contre leurs deux
(`flash_attn_ext_vec` + `combine`). L'attention est mieux structurée que la
leur ; c'est le MLP qui traîne.

## Ce que cela corrige

Il avait été écrit ici que la fusion `gate`/`up` « ne rendrait aucun watt »,
parce que les activations font 0,08 % du trafic et ne quittent pas le L2.
**L'argument était juste sur les octets et faux sur le temps.** La fusion
n'économise pas de bande passante — elle économise 48 lancements de GEMV et
~250 noyaux élémentaires. La bonne mesure avait été appliquée à la mauvaise
question.

## Réserve d'instrument

Ces durées sont relevées **sous `ncu`**, qui sérialise les lancements et rejoue
chaque noyau : ce ne sont pas des temps d'exécution libres. Ce qui est solide,
c'est la **comparaison** — mêmes métriques, même instrument, même carte, même
modèle — et le fait que le total retombe à 70 % du pas réel des deux côtés.
Prendre 20,45 ms pour un temps vrai serait une faute.

**Le seul test qui vaudra** : écrire le SwiGLU fusionné et le mesurer au banc,
hors `ncu`. Gain attendu si les deux postes tombent : ~1,1 ms sur 20,45, soit
**5,4 %**.


## Ce que la fusion a rendu, mesuré (9 septembre, après-midi)

    Qwen2.5-Coder-14B   avec graphes   34,38 -> 35,28 pas/s   +2,60 %
    phi4-bf16-pur       sans graphes   34,08 -> 34,78 pas/s   +2,06 %

**Le gain ne dépend pas des graphes CUDA.** Sur `phi4`, un MLP exilé les
désactive tous, et la fusion rapporte quand même : le coût de frontière qu'elle
supprime est un coût **GPU**, pas le coût CPU de lancement que les graphes
effacent déjà.

**Sur phi4, 39 MLP fusionnés sur 40** — celui de la couche 39 est en flux depuis
la RAM hôte, donc refusé. C'est exactement le module que le message de refus des
graphes désigne : deux mécanismes indépendants pointent le même, sans avoir été
coordonnés.

**Réserve : la dispersion est plus grande sans graphes.** 34,50 / 34,78 / 34,93
contre 35,27 / 35,28 / 35,28 sur Qwen. L'écart entre passes (1,2 %) est du même
ordre que le gain (2,06 %) — la médiane de trois passes alternées tient, la
deuxième décimale non.

**Limite de portée, déclarée** : la fusion est éprouvée sur **deux denses à
têtes groupées**, rien d'autre. Le parc n'a pas d'autre bf16 qui tienne sur la
carte (57 et 44 Gio pour les suivants). Ce chiffre vaut pour cette famille.

## Le TTFT : ce que le prefill coûte vraiment

> **Rectification.** La première version de cette section concluait à un « coût
> fixe de 44 ms » dans le forward de prefill. **C'était un artefact de régime**,
> corrigé plus bas : les deux mesures qui l'ont produit tournaient sur un
> chargement où un MLP était exilé en RAM hôte. Le chiffre nominal est 25 ms
> par appel, et c'est le plancher mémoire, pas un défaut.

Mesuré le 9 septembre 2026, `Qwen2.5-Coder-14B-bf16-pur`, serveur lancé sous
`py-spy` (`ptrace_scope = 1` interdit l'attachement à un processus existant),
prompts tirés au hasard pour que le cache de préfixe ne serve rien, première
requête jetée.

    chez Jerome   44,8 ms de forward pour 171 jetons
    chez Manon    44,2 ms de forward pour  36 jetons

**Cinq fois moins de jetons, le même temps : ce n'est pas du calcul, c'est un
coût par appel.** Et le TTFT complet de llama.cpp vaut **33 ms** — **notre seul
coût fixe de forward dépasse leur chaîne entière**.

    TTFT median      60,9 ms
    forward          44,2 ms   73 %
    hors forward     16,6 ms   27 %

Cette répartition est l'inverse de celle mesurée sur des prompts plus longs
(29 % de forward pour un TTFT de 152 ms). Les deux mesures ne se contredisent
pas — le forward est le même, c'est **l'enveloppe** qui varie d'un dispositif à
l'autre. Ce qui la fait varier n'est pas isolé : à éclaircir avant de publier un
chiffre d'enveloppe.

### La contention du GIL est réfutée

Même TTFT, moteur au repos puis en plein décodage, douze mesures chacun :

    repos    mediane 60,9 ms   min 52,9   max 67,9
    charge   mediane 62,3 ms   min 57,4   max 76,3
    ecart    +1,4 ms (+2,4 %) — dans la dispersion

Épreuve choisie **parce qu'elle ne dépend d'aucun profileur** : `py-spy`
échantillonne les piles, donc il montre où le code *est*, pas où il *attend*.
Un thread bloqué sur le GIL lui paraît inactif — et `record` exclut les threads
inactifs par défaut, ce qui rendait le dispositif aveugle à l'attente par
construction. `--idle` corrige la collecte ; la mesure au repos contre en
charge se passe de l'instrument.

### Où va le coût fixe

Sur 32 314 échantillons de service retenus (3 691 de chargement et 5 092 hors
catégorie, comptés et annoncés) :

    17,87 %  _prefill        (model.py:295)   boucle Python par sequence
     3,41 %  _ref_matmul     (kernels:554)    le GEMM lui-meme
     3,22 %  pin             (kvcache.py:423)
     2,77 %  _emit           (runner.py:741)
     1,09 %  attention  +  0,91 % causal_mask
     1,38 %  jinja2 _compile + tokeniter

`_prefill` est une boucle par séquence qui appelle `repeat_kv`, `causal_mask` et
`attention` — un lancement chacun par couche, pour 36 jetons. **Même forme que
le défaut de décodage que la fusion a corrigé, sur un chemin qu'elle ne touche
pas.**

### Un faux défaut, réfuté avant d'être rapporté

`attention()` semblait construire un masque explicite à chaque prefill, ce qui
écarte FlashAttention au profit d'un chemin lent. **`causal_mask` rend `None`
quand `q_offset == 0` et `q_len == kv_len`** : le prefill standard prend donc
bien le chemin rapide. Le défaut n'existe que **lorsque le cache de préfixe a
servi quelque chose** (offset > 0) — cas réel, mais absent des mesures à
prompts aléatoires. Piste, pas trouvaille.


## Rectification : l'exil d'un MLP est non déterministe

Le **même modèle sur la même carte avec le même code** donne deux régimes selon
ce qui tournait juste avant :

    un MLP exile en RAM hote   prefill d'1 jeton : 42,16 ms
    aucun exil                 prefill d'1 jeton : 26,45 ms   (-37 %)

`_reajuster_plan` décide selon la VRAM libre **au moment du chargement**. Un
poids exilé traverse le PCIe à chaque pas **et** désactive les graphes CUDA des
quarante-huit couches. Les deux mesures qui ont produit le « coût fixe de
44 ms » sont tombées dans le mauvais régime — le message d'exil était à
l'écran, il n'a pas été lu comme une invalidation.

**Toute mesure qui ne contrôle pas l'exil est suspecte.** La courbe refuse
désormais de rendre un chiffre s'il reste un poids en flux.

### La courbe, en régime sain

    jetons      1      4     16     64    128    256    512
    forward  26,45  28,79  31,08  30,35  32,86  45,94  84,78 ms

    cout par appel   24,97 ms       cout par jeton   0,1075 ms

**Les 25 ms par appel sont le plancher mémoire** : lire 26,09 Gio de poids à
1 122 Go/s effectifs. Ce n'est pas un défaut, et **llama.cpp les paie aussi** —
aucun moteur ne fait un forward sans lire les poids.

### L'écart réel : le coût par jeton

    nous       171 jetons -> 43,35 ms, dont 25 de poids -> 18,4 ms
                                                  0,1076 ms par jeton
    llama.cpp  TTFT complet 33 ms, dont ~25 de poids -> ~8 ms pour TOUT
                                                  ~0,047 ms par jeton au plus

**Environ deux fois leur coût par jeton**, ce qui recoupe les TFLOP/s mesurés
indépendamment (35,4 contre 78,4, facteur 2,2). C'est un défaut de **débit de
calcul en prefill**, pas de latence.

### Le surcoût du chemin prefill est nul

Même chargement, un jeton par chaque chemin :

    prefill  d'un jeton   41,98 ms
    decodage d'un jeton   42,14 ms      ecart -0,16 ms

Ils lisent les mêmes 26,09 Gio ; le chemin n'ajoute rien. **La piste « la boucle
Python de `_prefill` coûte cinq fois le GEMM » était fausse** : elle comparait
17,87 % (un englobant de profil) à 3,41 % (une feuille) — deux grandeurs qui ne
se comparent pas.

### La contention du GIL est réfutée

Même TTFT, moteur au repos puis en plein décodage, douze mesures chacun :

    repos    mediane 60,9 ms   charge   mediane 62,3 ms   ecart +1,4 ms

> **Ces deux valeurs absolues sont en RÉGIME DÉGRADÉ** (un MLP exilé, voir la
> rectification ci-dessus) : en régime sain le TTFT vaut environ 45 ms. **Ne
> pas les citer comme chiffres de TTFT.** La *comparaison* entre elles reste
> valide — les deux sont prises dans le même régime, sur le même serveur, à
> quelques secondes d'intervalle — et c'est elle seule qui réfute le GIL.

Épreuve choisie **parce qu'elle ne dépend d'aucun profileur** : `py-spy` montre
où le code *est*, pas où il *attend*, et `record` exclut les threads inactifs
par défaut — un thread bloqué sur le GIL lui paraît inactif.

## Le GEMM n'est pas la cible : il fait 85 à 94 % du forward

Mesuré sans charger de modèle — les formes exactes d'une couche de
`Qwen2.5-Coder-14B`, `F.linear` nu, comparé au forward réel :

    jetons   forward   GEMM nu    reste   part GEMM
         1     26,45     22,53     3,92     85,2 %
        16     31,08     26,06     5,02     83,8 %
        64     30,35     25,35     5,00     83,5 %
       128     32,86     29,46     3,40     89,7 %
       256     45,94     41,38     4,56     90,1 %
       512     84,78     79,72     5,06     94,0 %

**Tout ce que notre moteur ajoute au-dessus de cuBLAS — attention, normes,
rope, boucle Python — coûte 4 à 5 ms, quel que soit le nombre de jetons.**
C'est donc un coût par appel, pas un coût de calcul.

### Nous sommes à 65 % du pic, pas à 17 %

    48 couches x 171 jetons  =  4,52 TFLOP
    GEMM nu 33,06 ms   ->  137 TFLOP/s   (pic bf16 5090 : ~210)
    forward 37,3 ms    ->  121 TFLOP/s

Le chiffre de 17 % qui circulait était calculé sur le **TTFT complet** : il
comptait le plancher mémoire, l'enveloppe HTTP, le gabarit et la tokenisation
comme du calcul.

Ce qui limite le GEMM n'est pas notre usage mais **la taille du lot** — le même
appel rend 2,4 TFLOP/s à 1 jeton, 65 à 36, 137 à 171, 175 à 512. La carte ne se
remplit qu'avec les jetons.

### Une convergence qui n'en était pas une

Deux mesures donnaient un facteur ~2,2 par des chemins « indépendants » — coût
par jeton d'un côté, TFLOP/s de l'autre. **Elles comparaient deux grandeurs
différentes** : l'une incluait l'enveloppe, l'autre non. La convergence était
une coïncidence, et elle avait été présentée comme une confirmation mutuelle.

À 171 jetons, notre GEMM seul coûte 33 ms — le TTFT **complet** de llama.cpp en
vaut 33. Or ce GEMM lit 26,09 Gio, soit 25 ms à la bande passante observée :
**ils font leur prefill entier à peu près au plancher mémoire, et nous aussi
sur le GEMM.**

**Le GEMM n'est donc pas la cible.** Laurine l'avait établi par le noyau (son
GEMV Triton est 1,6× plus lent que `F.linear`), on le retrouve par la forme.

## Le temps mort réel : 2,1 %, pas 32 %

Trace `nsys` — qui n'exécute rien deux fois —, 200 pas, graphes actifs et
vérifiés, noyaux **bornés à la plage NVTX** et sommés en **union d'intervalles**
(deux noyaux qui se recouvrent occupent le GPU une fois, pas deux) :

    plage « pas »        28,748 ms par pas
    noyaux dans la plage    549,3 par pas      (544 attendus)
    temps GPU reel       28,140 ms par pas
    TEMPS MORT            0,608 ms par pas  ->  2,1 %
    occupation reelle                          97,9 %

    ce que ncu annoncait  9,0 ms  ->  32 %     (facteur 14,8)

**Le gisement de temps mort n'existe pas.** Même réduit à zéro, il rendrait
**+2,2 %** — et il faudrait supprimer tout l'espace entre 549 lancements.

26,09 Gio de poids lus en 28,14 ms font **995 Go/s effectifs**, 55,6 % du pic
théorique. Le décodage est mémoire-borné, à 98 % d'occupation : ce qui limite
est la bande passante, pas l'ordonnancement.

### Quatre défauts d'instrument traversés pour ce chiffre

1. **`--capture-range=nvtx` n'écrit rien** — « No reports were generated »
   alors que les pas avaient tourné. Tracer tout, filtrer au dépouillement.
2. **`nsys` trace un graphe CUDA « comme un tout » par défaut** : 6 099 noyaux
   comptés pour 200 pas qui en lancent 108 800. **Symétrique exact du défaut de
   `ncu`** — l'un sérialise les nœuds, l'autre ne les voit pas.
   `--cuda-graph-trace=node`.
3. **`nsys stats` somme toute l'exécution**, chargement et prefill compris,
   quand la plage ne couvre que les pas mesurés : 103,7 % d'occupation, un
   numérateur plus large que son dénominateur. Requête SQL bornée aux temps de
   la plage.
4. **`nsys stats … >/dev/null 2>&1`** — les CSV n'ont pas été réécrits, le
   dépouillement a lu ceux de la trace précédente et rendu **97,4 % de temps
   mort** sans que rien ne le signale. *Masquer la sortie de l'outil, c'est
   masquer ce qu'il avait à dire.* Le dépouillement refuse désormais si les CSV
   sont antérieurs à la trace.

### Ce qu'il reste comme cible sur le décodage

Rien. Trafic nominal, coalescence parfaite, L2 meilleur que le leur, occupation
à 98 %, fusion appliquée. **Pour gagner encore sur ce chemin il faut lire moins
d'octets — donc quantifier davantage, pas mieux ordonnancer.**

## Ce que les noyaux maison rapportent, chiffré

Mesuré aux formes d'une couche de `Qwen2.5-Coder-14B`, 88 jetons, sur 48
couches — chaque noyau maison contre son équivalent PyTorch :

    rmsnorm   maison 0,97 ms   PyTorch naif 3,52 ms   facteur 3,6
    swiglu    maison 0,53 ms   PyTorch naif 0,83 ms   facteur 1,6

**C'est la première fois que ce gain est chiffré** plutôt que supposé. Il est
apparu par accident : une première version du dispositif réimplémentait ces
opérations en PyTorch *pour pouvoir les mesurer*, et donnait un résidu de
7,98 ms — **le double du réel**. L'erreur était de mesurer une implémentation
que le moteur n'exécute pas ; en la corrigeant, la comparaison est restée.

## Le résidu de prefill, décomposé

À 88 jetons, hors GEMM :

    attention SDPA          1,82 ms   36 %
    rope                    0,98 ms   19 %   [dispositif naif : ncu donne 0,12]
    rmsnorm maison          0,97 ms   19 %
    residuel                0,77 ms   15 %   [deja fusionne dans rmsnorm_bf16]
    swiglu maison           0,53 ms   10 %
    TOTAL                   5,07 ms

**L'attention domine**, et elle passe déjà par FlashAttention (`is_causal=True`,
sans masque explicite). Les deux lignes marquées sont surestimées par le
dispositif : le résidu réel est sous 4 ms et encore plus concentré sur
l'attention.

**Non affiné davantage** : mesurer au dixième un poste de 4 ms sur un forward de
31 coûterait plus que ce qu'il rapporterait, et les deux corrections connues
vont dans le sens qui réduit la cible.

## État des deux chemins chauds, au 9 septembre 2026

|  | décodage | prefill |
|---|---|---|
| trafic | nominal (+0,12 % du théorique) | — |
| coalescence | 32,0 o/secteur, parfaite | — |
| L2 / DRAM | 1,035 (llama.cpp : 1,089) | — |
| occupation GPU | **97,9 %** | — |
| bande passante | 995 Go/s, 55,6 % du pic | — |
| GEMM | — | 137 TFLOP/s, **65 % du pic** |
| hors GEMM | — | < 4 ms sur 31, dominé par l'attention |
| fusion | +2,60 % | +2,6 à +6,2 % |

**Les deux chemins sont près de leur plancher matériel.** Les gains restants ne
sont pas dans l'exécution mais dans **ce qu'on demande à la machine de lire** —
donc dans le format des poids, ce qui touche la qualité.
