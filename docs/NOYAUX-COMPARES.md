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
