# Où passe le temps GPU : acvram contre llama.cpp, noyau par noyau

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

    chez chef   44,8 ms de forward pour 171 jetons
    chez poste2    44,2 ms de forward pour  36 jetons

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
