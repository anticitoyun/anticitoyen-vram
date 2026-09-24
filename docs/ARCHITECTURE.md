# Architecture

## Le chemin que suit un modèle

```
  point de contrôle Hugging Face (safetensors, bf16)
        |
        |  acvram plan     ->  où va chaque tenseur ?
        v
  plan de placement  ------------------------------+
        |                                          |
        |  acvram convert  ->  quantifie selon     |
        v                      l'appareil visé     |
  fragments acvram + manifest.json                 |
        |                                          |
        |  acvram serve                            |
        v                                          v
  chargeur ----> modèle placé ----> moteur ----> HTTP OpenAI
```

Le plan est calculé **avant** la conversion et rangé **dans** le manifeste du
modèle converti. C'est ce qui supprime toute une classe de bogues : au
chargement, la question « ce GPU sait-il lire ce tenseur ? » ne se pose jamais,
parce que le tenseur a été écrit pour ce GPU.

## Le planificateur

`memory/tiering.py`. À partir d'un `ModelSpec` et d'un `Rig`, il produit un
`Plan`.

Il raisonne sur un modèle de décodage limité par la mémoire : produire un jeton
revient à lire une fois chaque poids *actif*, donc le temps vaut

```
  somme sur les couches de :
      résident  ->  octets_actifs / bande_passante_vram
      streamé   ->  max(octets_actifs / bande_passante_pcie,
                        octets_actifs / bande_passante_vram)   # le préchargement recouvre
  + un saut par le tampon hôte à chaque changement de GPU
```

`octets_actifs` vaut toute la couche pour un bloc dense, et seulement les
experts routés pour un bloc à mélange d'experts — ce qui explique à lui seul
pourquoi un modèle MoE de 235 milliards de paramètres et un modèle dense de 70
se comportent si différemment sur la même machine.

Quatre décisions, dans cet ordre :

1. **Budget du cache KV.** Pris en premier, parce qu'il croît avec le trafic
   alors que les poids, non.
2. **Tranches du pipeline.** Des plages de couches contiguës par GPU,
   dimensionnées pour que le pipeline ne franchisse qu'une seule fois une
   frontière entre cartes. Les cartes GeForce n'ont pas de NVLink et NVIDIA y
   désactive le pair-à-pair PCIe : un franchissement transite donc par la
   mémoire hôte épinglée — négligeable pour un état caché par jeton, ruineux si
   cela arrivait à chaque couche.
3. **Placement de l'attention.** Épinglée sur le GPU de sa tranche dès qu'elle y
   tient. Elle est petite et se trouve sur le chemin critique de la latence.
4. **MLP et experts.** Remplissent la VRAM de l'avant vers l'arrière ; le reste
   part en RAM hôte. La VRAM qui subsiste devient un cache LRU d'experts
   fréquents.

`auto_plan` explore ensuite le produit (nombre de GPU utilisés) × (fraction de
VRAM pour le cache KV) et classe par débit de décodage estimé, en rejetant toute
configuration qui déborde ou qui ne tient pas un contexte complet.

## Formats

| | NVFP4 | INT4 |
|---|---|---|
| élément | E2M1 (4 bits) | uint4 |
| échelle | FP8 E4M3, une pour 16 | FP16, une pour 128 |
| point zéro | aucun (symétrique) | uint4, un pour 128 |
| échelle globale | FP32 par tenseur | aucune |
| bits par poids | 4,50 | 4,156 |
| déquantification | `niveau × e4m3(bloc) × globale` | `(q − zéro) × échelle` |

L'échelle globale en FP32 du NVFP4 existe parce que l'E4M3 sature à 448 : on la
choisit égale à `amax / (6 × 448)`, de sorte que la plus grande échelle de bloc
atterrisse exactement sur ce plafond, quelle que soit la dynamique du tenseur.

## Noyaux de calcul

Deux points d'entrée par format :

* `*_dequant` — matérialise la matrice dans un type de calcul, puis laisse
  cuBLAS faire le produit. C'est le chemin du **prefill** : le coût de
  déquantification s'amortit sur tout le lot, et cuBLAS bat tout produit écrit à
  la main.
* `*_gemv` — déquantification et multiplication fusionnées, pour un lot de 1 à
  8. C'est le chemin du **décodage**, purement limité par la mémoire : il s'agit
  de lire les poids 4 bits depuis la mémoire globale sans jamais en écrire une
  copie 16 bits.

Le seuil entre les deux est `gemv_threshold=8` dans `kernels/__init__.py`.

`kernels/__init__.py` compile l'extension au premier usage, en n'émettant du
code que pour les architectures présentes. Si la compilation échoue pour une
raison quelconque, il avertit une fois et retombe sur l'implémentation PyTorch
de référence — lente, mais numériquement identique, et suffisante pour que la
suite de tests s'exécute n'importe où.

## Moteur

`engine/runner.py` fait tourner un lot continu : à chaque étape il admet autant
de requêtes en attente que les blocs KV libres le permettent, précalcule chacune
séparément (une longue invite mêlée à un lot de décodage bloquerait tout ce qui
la suit), décode un jeton pour tout ce qui tourne, et libère les blocs d'une
séquence dès qu'elle s'arrête.

Il est monothread à dessein. Les couches du modèle sont réparties sur deux GPU
et la mémoire hôte, et une étape les touche en série ; des fils d'exécution se
disputeraient les mêmes appareils sans ajouter de parallélisme. La concurrence
vient du lot, pas des fils. Le serveur asynchrone fait le lien avec un fil
d'arrière-plan et une file par requête.

**Requêtes closes au bon moment (pièce 146).** Une séquence tronquée par
épuisement du budget KV (chemins pipeline et spéculatif) n'était jamais
livrée à sa requête HTTP : la sortie finie était jetée sans clore la requête,
qui restait en attente indéfiniment. Corrigé : les séquences épuisées sont
livrées par `step` dans le pas même où il les détecte, jamais différé.
Preuve bout en bout (serveur réel, graphes + pipeline) : rouge avant, vert
après. Deux autres défauts du budget KV par défaut corrigés dans la même
pièce — un départage qui laissait tomber la capacité KV à 6 % de la VRAM dès
qu'une seule séquence y tenait, et un OOM à la chauffe par tête liée
convertie en fp32 sans libérer d'abord le cache de l'allocateur. Détail :
revue/verdict-146-kv-defaut-24-09.md.

## Poids streamés

`engine/layers.py:StreamedWeight`. Les poids résidant en RAM vivent en mémoire
**épinglée** — une mémoire paginable forcerait le pilote à sérialiser la copie
et le recouvrement disparaîtrait — et sont copiés sur un flux CUDA annexe vers
un double tampon. `ACVRamModel.forward` lance le transfert de la couche *i+1*
avant d'exécuter la couche *i*, de sorte qu'une couche streamée coûte
`max(copie, calcul)` et non leur somme. C'est exactement ce que suppose le
modèle de coût du planificateur : si l'un change, l'autre doit changer.

## Cache de préfixe

`memory/kvcache.py:BlockAllocator` est à la fois la liste des blocs libres et le
cache de préfixe, parce que les deux se disputent la même ressource.

Le contenu d'un bloc est déterminé par les jetons qui l'ont produit *et* par
tous ceux qui précèdent, d'où un hachage chaîné :

```
h_0 = hachage((0,     jetons[0:16]))
h_1 = hachage((h_0,   jetons[16:32]))
h_i = hachage((h_{i-1}, jetons[16i:16i+16]))
```

Le chaînage n'est pas décoratif. Les mêmes seize jetons apparaissant dans deux
contextes différents ne produisent pas les mêmes clés et valeurs, puisque
l'attention a vu une histoire différente ; hacher la seule tranche servirait
volontiers le cache d'une séquence à une autre.

Seuls les blocs **complets** sont publiés. Un bloc à moitié rempli, retrouvé par
un hachage qui décrit un contenu qu'il ne porte pas encore, livrerait à une
requête ultérieure des clés et des valeurs jamais écrites.

À la libération, un bloc dont le contenu reste identifiable part en fin de file
LRU au lieu de rejoindre les blocs libres, et n'est recyclé que lorsque la
réserve s'épuise — le cache survit donc entre les requêtes sans jamais refuser
une allocation qu'il aurait pu servir.

Un bloc est toujours retenu lors d'une correspondance : une requête dont
l'invite est entièrement en cache a tout de même besoin d'un jeton à faire
traverser le modèle, sans quoi il n'y a rien pour produire des logits.

La publication a lieu **avant** que les blocs ne soient rendus. L'ordre inverse
— celui d'origine — faisait que `_finish` vidait `seq.blocks` quelques lignes
avant que `_register_complete_blocks` ne cherche à publier : le cache ne
recevait jamais rien, et la réserve de cache KV en mémoire hôte, alimentée par
l'éviction des blocs identifiables, restait inutilisée.

### Hybrides à récurrence linéaire

Les blocs KV ne suffisent pas à reprendre une invite sur un modèle dont
certaines couches portent un état récurrent : cet état vit hors du cache
paginé, dans `Engine.gdn_states`. Le cache y était donc simplement coupé.

Il repose désormais sur des **instantanés** de cet état, pris aux frontières
régulières du prefill et rangés en mémoire hôte épinglée. Sur un 27B, un
instantané pèse 150 Mio pour quarante-huit couches ; trois sont conservés, en
éviction par ancienneté.

Le prefill se coupe une fois, à la plus grande frontière multiple du pas
d'instantané strictement intérieure à l'invite. Ce choix ne dépend que de la
longueur de l'invite, donc deux requêtes partageant une amorce tombent sur la
même frontière tant qu'elles restent dans la même tranche.

À l'admission, l'appariement des blocs est **plafonné** à la frontière dont on
tient l'instantané, et un appariement partiel est rejeté en entier : l'état
récurrent et les clés doivent décrire exactement la même position, faute de quoi
la reprise produirait des logits fondés sur deux histoires différentes.

Le découpage du prefill change l'ordre des calculs récurrents ; la sortie
diverge donc légèrement d'un prefill monolithique, au même titre qu'un
changement de taille de lot.

**Fusions du décodage (pièce 156)**, toutes numériquement identiques au
chemin qu'elles remplacent (au bit ou à l'ulp, mesuré sur le modèle servi),
**toutes par défaut** (poste5, 255042e8) : la conv de décodage fusionnée,
l'état GDN mis à jour en place et le résidu différé des couches GDN
(`ACVRAM_GDN_CONV_FUSEE`, `ACVRAM_GDN_ETAT_EN_PLACE`, `ACVRAM_GDN_RES_DIFFERE`,
ensemble −7,8 % de temps de pas à b = 8), la RMSNorm en registres
(`ACVRAM_NORME_REGISTRES`, +8,1 % à b = 8, au bit — l'ordre de sommation d'un
fil sur ≤ 8 carrés bf16 n'est pas observable en sortie), et les portes dans
le noyau fla et la norme gated Triton (`ACVRAM_GDN_PORTES_NOYAU`,
`ACVRAM_GDN_NORME_FUSEE`, ± 1 ulp bf16, KL et PPL tenues contre le témoin).
Détail : revue/poste5-piece156c-verdict-24-09.md, revue/poste5-piece156d-verdict-24-09.md.

## Décodage spéculatif

`engine/speculative.py`. Le lot de vérification est `[dernier jeton produit] +
[K propositions]`, présenté aux positions absolues `n-1 .. n+K-1`. Fournir le
dernier jeton produit n'est pas un surcoût : ses clés et valeurs n'ont jamais
été écrites, puisqu'un jeton n'entre dans le cache qu'au moment où on le
présente. Les K+1 positions donnent donc exactement les K+1 prédictions
nécessaires.

Les positions rejetées laissent des entrées périmées dans le cache au-delà de la
longueur de la séquence. Rien ne lit au-delà de `seq_len`, et l'étape suivante
les écrase.

L'acceptation suit la règle de rejet standard. Pour un propositeur sans
distribution (les n-grammes), q est une masse de Dirac sur la proposition : la
probabilité d'acceptation vaut donc `p(x)`, et un rejet rééchantillonne dans `p`
privé de ce jeton — ce qui est exact, et non une approximation.

### Trois brouillons

- **n-grammes** (défaut) : ne coûte rien et paie quand la sortie recopie
  l'entrée. Adaptatif — il se met en pause quand le gain retombe.
- **modèle brouillon** : un modèle entier, avec son propre cache paginé.
  Mesuré ici, il fait tomber le débit de 152 à 30 t/s malgré 78 % d'acceptation
  sur du code : le brouillon coûte trop cher pour ce qu'il rapporte.
- **tête MTP** (`engine/mtp.py`) : la couche `nextn` que portent Qwen3.5 et
  suivants, DeepSeek compris. Deux conventions de noms de tenseurs sont
  reconnues (`engine/mtp.py:83-96`, `noms_mtp`) : `model.mtp.<n>.` (GGUF
  renommé, DeepSeek) et `mtp.layers.<n>.` + `mtp.fc`/`mtp.norm`/
  `mtp.pre_fc_norm_*` partagés (Qwen3.5, `engine/mtp.py:65-76`, `cles_mtp`) —
  avant la pièce 105, cette seconde convention n'était pas reconnue : les
  tenseurs MTP de Qwen3.8-27B étaient convertis mais jamais chargés, et
  `--speculative auto` retombait sur les n-grammes sans le dire. La tête
  mélange le plongement du jeton émis et l'état caché de la position
  précédente, puis applique un bloc de transformeur :

  ```
  h' = eh_proj( [ enorm(plongement(t)) ; hnorm(h) ] )
  logits = lm_head( shared_head_norm( bloc(h') ) )
  ```

  L'ordre de la concaténation n'est pas indifférent : plongement d'abord donne
  50 % de prédictions justes en forçage enseignant, l'ordre inverse en donne
  zéro. Cette tête est un soixante-quatrième d'un 27B — mais elle n'est pas
  rentable en l'état : son cache se pollue de ses propres états au fil du
  brouillonnage, et chaque jeton proposé traverse `lm_head` en entier, hors
  graphe CUDA. Disponible par `--speculative mtp`, non activée par défaut.
  `--speculative auto` (`cli.py:796-801`, `repli_speculatif`) choisit `mtp` si
  le modèle chargé porte une tête reconnue, sinon retombe sur `ngram` avec un
  repli **nommé**, jamais un `None` silencieux : la raison vient de
  `model.mtp_raison` (posée par le chargeur) et est portée sur
  `speculator.repli` (`cli.py:830`). Reste : rien ne lit `speculator.repli`
  ailleurs dans le dépôt — la bannière de démarrage (`cli.py:886-888`)
  n'imprime que `args.speculative`, pas la raison du repli.

## L'étage hôte comme appareil de calcul

`DecoderLayer` porte deux appareils : l'attention s'exécute sur `self.device`,
le MLP sur `self.mlp_device`. Quand le planificateur juge qu'un MLP résidant en
RAM vaut mieux calculé sur place, seul l'état caché traverse le bus —
`[jetons, dimension]`, quelques kilooctets par jeton décodé face à des
gigaoctets de poids.

La décision se prend dans `plan_placement`, en comparant le lien hôte mesuré du
GPU exécutant à `PlannerOptions.host_compute_gb_s`. `acvram bench --what
bandwidth` affiche les deux nombres et la recommandation qui en découle.

## Disposition Marlin (linéaires denses)

Depuis la pièce 156, les linéaires NVFP4 des modèles **denses** (attention et
MLP hors bloc `MoEBlock`) sont servis par défaut par le port Marlin de vLLM
0.29 (`kernels/marlin_port`), préparé AU CHARGEMENT plutôt que reconstruit à
chaque pas : +57 à +90 % de débit de décodage à b = 8, coût de préfill ramené
à +2 à +4 ms par la réécriture du dépaquetage (pièce 147, ci-dessous).

Cinq variables pilotent le mécanisme (`acvram/regime.py`) :

| variable | défaut | rôle |
|---|---|---|
| `ACVRAM_PROJ_MARLIN` | `1` | disposition Marlin unique aux godets éligibles ; `0` = repli naturel |
| `ACVRAM_PROJ_MARLIN_PORTEE` | `denses` | `denses` : un modèle à `MoEBlock` garde tout son chemin naturel (ses linéaires hors experts sont éligibles mais non mesurés — 32 alias du menu touchés, revue/poste1-piece142-inventaire-denses-24-09.md) ; `global` : tout poids dense éligible, MoE compris |
| `ACVRAM_GEMV_MARLIN_V2` | `1` | GEMV Marlin v2 (tuiles de colonnes, x en mémoire globale) contre v1 (x en mémoire partagée) |
| `ACVRAM_GEMV_MARLIN_TPB` | `0` | tuiles de 64 colonnes par bloc ; `0` = choisi par forme |
| `ACVRAM_GEMV_MARLIN_S` | `0` | split-K forcé ; `0` = règle automatique de v1 |

**Replis nommés**, jamais un `None` silencieux : capacité KV ou mémoire
insuffisante au chargement refuse explicitement (`ACVRAM_PROJ_MARLIN_CAPACITE`,
`loader._verifier_memoire_marlin`) plutôt que d'exiler des poids en silence ;
un poids dont l'échelle ne tient pas dans le format S0E5M3 de Marlin (voir
« échelles sous-normales » ci-dessous) est retiré de la disposition et rendu
au chemin naturel, raison nommée sur la ligne de régime.

### Échelles sous-normales (pièce 157)

Le format d'échelle de bloc de Marlin, S0E5M3, n'a qu'environ 2^14,8 de plage
contre 2^17,8 pour l'E4M3 d'origine du NVFP4. Un poids dont les échelles de
bloc mêlent une valeur proche du plafond (448) et des sous-normales perdait
des blocs de 16 poids entiers, mis à zéro plutôt que représentés — touché au
défaut servi côté MoE (max|Δ logits| = 4,52 sur un modèle réel). Corrigé par
un facteur d'échelle **par ligne** (plutôt que par tenseur ou par pile
d'experts entière) pour les poids denses, et par l'exclusion pure et simple
d'un poids qui écrase encore après ce correctif. Preuve au bit contre le
chemin naturel après correctif. Détail :
revue/verdict-157-marlin-sous-normales-24-09.md.

### Dépaquetage au préfill (pièce 147)

La disposition unique n'a pas de chemin de préfill natif : elle dépaquette en
bf16 puis laisse cuBLAS faire le produit, comme `*_dequant` (voir « Noyaux de
calcul » ci-dessus). La première implémentation (un fil par tuile) coûtait
+26 à +34 ms par requête ; réécrite (un fil par colonne, lignes entières en
deux `uint4`, vues à pas libre sans copie de transposition), elle coûte +2 à
+4 ms à toute longueur d'invite mesurée (512 à 4 096 jetons), au bit contre
les chemins Triton et torch de référence. `ACVRAM_DEPAQUETAGE=auto` choisit
CUDA si l'extension le porte, sinon Triton. Détail :
revue/poste6-piece147-verdict-24-09.md.

### Cache de compilation (pièce 161)

Le `.so` compilé du port Marlin vivait sous un nom de cache FIXE, partagé par
tous les worktrees : deux arbres aux sources différentes alternant sur la
même machine se recompilaient l'un l'autre à chaque changement (25 s de nvcc,
y compris hors du verrou `carte.sh`), sans qu'aucun message ne désigne
l'autre arbre comme cause — le même défaut que corrigeait déjà
`kernels/__init__.py` pour l'extension principale (cache keyé par
`sha256(realpath(...))`), pas encore porté ici. Corrigé (poste6) : cache
keyé par empreinte sha256 des sources, sources copiées dans le cache, le
moteur en service ne relance jamais ninja (charge le `.so` de son empreinte
ou replie au naturel, raison imprimée).

## Répartition des noyaux

Trois implémentations, numériquement identiques :

| implémentation | fichier | quand |
|---|---|---|
| CUDA | `acvram_kernels.cu` | poids sur un GPU |
| processeur | `acvram_cpu.cpp` (ctypes) | poids en RAM hôte |
| référence | `quant/*.py` | tout ce qui n'a pas pu se compiler |

Le GEMV CUDA tire sa vitesse de trois choses : des chargements `uint2` portant
16 poids empaquetés (exactement un bloc d'échelle NVFP4, et un nombre entier de
groupes INT4, si bien qu'un fil ne chevauche jamais une frontière d'échelle) ;
quatre lignes de sortie par bloc, de sorte que la tranche d'activation est lue
une fois et réutilisée ; et un découpage de la réduction sur K quand la matrice
est trop courte pour occuper l'appareil, ce qui est toujours le cas des petites
projections d'un bloc d'attention à requêtes groupées.
