# Verdict — pièce 180 : pavage Marlin sur l'alias mixte, b=8 — poste3, 25/09

Suite à 173 (poste1) : Marlin NVFP4 5,75 ms/pas contre 4,7 idéal (≈ 1,0 ms de gain max).

## Chemin réel identifié (correction en route)

Les 112 appels Marlin/pas ne sont NI l'attention NI le GDN — vérifié sur `acvram_manifest.json`
de l'alias : `self_attn.{q,k,v,o}_proj` sont **int8** (64/64 tenseurs, 0 nvfp4) ; `linear_attn.*`
(GDN) est **int8 ou bf16** (144+96, 0 nvfp4). Seul le MLP dense porte du nvfp4 (168/195 tenseurs).
`_SUFFIXES_DOUBLES` (`acvram/engine/loader.py:1863`) fusionne `gate_proj`+`up_proj` en UN appel :
**112 = 56 couches × (gate‖up fusionné + down)**, confirmé par la mesure (ci-dessous, quasi
identique aux moyennes nsys de 173 : 64,9 µs mesuré contre 67,9 µs nsys pour gate‖up ; 33,7 µs
contre 34,8 µs pour down).

Mes premières formes de travail (attn_qo/attn_kv/gdn_*, message précédent) correspondaient à des
tenseurs qui ne passent PAS par Marlin dans cet alias — utiles pour caractériser le noyau en
général, mais sans rapport avec les 112 appels réels. Corrigé.

## Sélection de configuration (`determine_exec_config`, `marlin.cu:280-320`)

Pour M ≤ 16 (notre b=8), `small_batch_thread_configs[]` (`marlin.cu:139-144`) essayées par
priorité ; aucun filtre caché dans `generate_kernels.py` (lu en entier). Mesuré (nom du noyau lu
sous `torch.profiler`, `scratchpad/poste3-p180-25-09/forme_vers_config.py`) :

**Seuil net : N ≥ 6 144 → `{thread_k=128, thread_n=128, threads=256}` ; N ≤ 5 120 →
`{thread_k=64 ou 128, thread_n=128 ou 64, threads=128}`.**

## Débit par forme (sous graphe CUDA, comme le service ; 12 poids distincts pour éviter le cache L2
— sans quoi le débit dépasse le plafond physique de la carte, faute d'instrument corrigée en cours
de route)

| forme réelle | N | K | THREADS | µs/appel | To/s | % du plancher 1,56 To/s (int8 étroit, 173) |
|---|---|---|---|---|---|---|
| **mlp gate‖up (fusionné)** | 34 816 | 5 120 | 256 | 64,9 | 1,72 | **110 %** |
| **mlp down** | 5 120 | 17 408 | 128 | 33,7 | 1,65 | **106 %** |

**Les deux formes RÉELLEMENT servies par Marlin dans cet alias tournent déjà à 106-110 % du
plancher de référence — au-dessus, pas en-dessous.** Le calcul en octets : 56 × (110,5 + 55,7) Mio
≈ 9,28 Gio à travers Marlin par pas ; au plafond physique de la carte (1,79 To/s, `PLAFOND_BPS`) :
5,18 ms — proche des 5,75 ms mesurés (89 % du plafond matériel). Le pavage n'est donc **pas** la
source de l'écart signalé par 173 : les deux configurations choisies automatiquement sont déjà
proches de l'optimum physique. L'écart de 173 (idéal 4,7 ms pour LE PAS ENTIER, pas Marlin seul)
vient d'ailleurs — voir le tableau des postes de 173 (int8 étroit 8,3 ms, glue 3,25 ms, etc.).

## Verdict pavage

**CLOS.** Six formes synthétiques sur sept testées (dont les deux réelles) à 79-110 % du plancher ;
aucune comparaison forcée entre configurations n'a été nécessaire ni faite (les deux formes réelles
n'ont pas de marge à gagner par un autre pavage). Pas de correctif proposé.

## Correction du diagnostic « attn k/v à 20 % »

**Fausse piste** — `self_attn.k_proj`/`v_proj` sont en INT8 dans cet alias, jamais en Marlin ;
zéro appel Marlin pour l'attention. La forme synthétique testée (N=1024,K=5120, 20 % du plancher)
caractérisait le noyau Marlin sur une petite forme HYPOTHÉTIQUE, pas un appel réel de ce service.
Une concaténation q‖k‖v en Marlin n'a pas de sens ici : il n'y a pas de q/k/v Marlin à concaténer.
Si un gain existe sur k/v, il est du côté du noyau INT8 étroit (`_etroit_reduit_kernel`, 8,3 ms/
pas, 42,6 % — le premier poste de 173) — déjà exploré par poste1 (176, `tests/test_gdn_qkv_gate_176.py`
et `scratchpad/poste1-p176-25-09/`, fusion analogue côté GDN/int8), hors du périmètre Marlin de
cette pièce.
