# Régler la machine cible

i9-14900K · ROG Maximus Z790 Dark Hero · 96 Go DDR5 · RTX 5090 Astral LC OC
32 Go · RTX 3080 Ti 12 Go · Linux Mint 22.3

Tout ce qui suit doit être **confirmé avec `acvram detect`** sur la machine
réelle. Le profil intégré porte des valeurs de plaque signalétique, pour que la
planification puisse avoir lieu avant que le matériel ne soit joignable ; la
détection l'emporte toujours sur lui.

## Socle logiciel

| | minimum | pourquoi |
|---|---|---|
| pilote NVIDIA | 570 | premier à gérer Blackwell / `sm_120` |
| CUDA | **12.8** | rien de plus ancien ne sait émettre du `sm_120` |
| PyTorch | 2.7+ compilé pour cu128 ou cu130 | une roue cu126 ne tournera pas sur la 5090 |
| GCC | ≤ 13 pour nvcc | nvcc refuse les compilateurs hôtes plus récents |

`acvram doctor` échoue bruyamment sur l'incompatibilité de version CUDA, parce
que le symptôme est sinon un déroutant `no kernel image is available for
execution`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## L'asymétrie PCIe — à vérifier en premier

Sur la plupart des cartes mères Z790, le premier port x16 est relié au
processeur en PCIe 5.0 et le second passe par le chipset avec bien moins de
lignes. Si la 3080 Ti atterrit dans un port Gen4 x4, son lien hôte tourne autour
de **7 Go/s contre 54 Go/s** pour la 5090 — un facteur huit qui décide laquelle
des deux cartes ira chercher des poids en mémoire vive.

```bash
nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current \
           --format=csv
acvram bench --what bandwidth      # le nombre qui compte vraiment
```

Le planificateur lit la largeur de lien réelle et achemine les couches streamées
vers la carte au lien le plus rapide. Si `bench` contredit `detect`, croyez
`bench` : la valeur Gen4 x4 du profil pour le GPU 1 est une supposition
pessimiste.

## Pas de pair-à-pair

Aucune des deux cartes n'a de NVLink, et NVIDIA désactive le pair-à-pair PCIe
sur les GeForce. Les tenseurs échangés entre GPU transitent donc par la mémoire
hôte épinglée. Ce n'est pas grave — le pipeline ne franchit qu'une frontière par
jeton et n'y transporte qu'un état caché, quelques kilooctets — mais c'est
pourquoi le planificateur impose un unique franchissement au lieu d'entrelacer
les couches entre les cartes.

## Processeur : épingler sur les cœurs P

Le 14900K est hybride : 8 cœurs P avec hyperthreading et 16 cœurs E. Un fil qui
atterrit sur un cœur E fait le même travail bien plus lentement, et pour une
boucle sensible à la latence c'est pire que d'avoir moins de fils.

```bash
acvram detect                      # affiche l'ensemble de cœurs P
taskset -c 0-15 acvram serve ...   # vérifiez la plage sur votre machine d'abord
export OMP_NUM_THREADS=8
```

À avoir également : le microcode `0x12B` ou plus récent, pour l'atténuation de
la dégradation Raptor Lake. `grep microcode /proc/cpuinfo`.

## Mémoire

96 Go, c'est presque certainement 2 × 48 Go. Restez à deux barrettes : quatre
modules sur Z790 imposent une fréquence stable plus basse, et l'utilité de
l'étage hôte est directement proportionnelle à sa bande passante. Activez
XMP/EXPO et vérifiez :

```bash
sudo dmidecode -t memory | grep -E 'Speed|Size'
```

`--host-fraction` (0,85 par défaut) plafonne la part de `MemAvailable` qu'acvram
s'autorise. Montez-la à 0,95 pour un modèle qui tient tout juste ; baissez-la si
la machine fait autre chose.

Les grandes pages aident mesurablement les tampons d'échange épinglés :

```bash
sudo sysctl -w vm.nr_hugepages=8192      # 16 Go en pages de 2 Mo
```

## Alimentation et températures

**Ce poste tourne bridé, et c'est délibéré — mais la limite POSÉE dérive dans
le temps, elle n'est pas un fait permanent.** Relevé le 8/09/2026 (posé) puis
re-relevé le 10/09/2026 (constaté, sans action de personne d'identifié) :

| carte | posé le 8/09 | **constaté le 10/09** | par défaut (usine) | plage réglable |
|---|---|---|---|---|
| RTX 5090 | 400 W | **500 W** | 600 W | 400 à 600 W (le plancher est à 400) |
| RTX 3080 Ti | 275 W | **375 W** (son plafond réglable) | 350 W | 100 à 375 W |

**Le bridage n'est pas persistant au redémarrage** (déjà noté ci-dessous,
confirmé être la cause probable de la dérive) : un reboot ramène chaque carte
à SA limite d'usine (600/350), pas au réglage documenté ici — la 3080 Ti à
375 W n'est même pas au défaut usine (350), quelqu'un ou quelque chose l'a
réglée au-dessus depuis. **Aucune campagne ne doit supposer un bridage
"documenté" sans le vérifier au moment même de mesurer.**

**Ce qui reste vrai malgré la dérive** : les chiffres de jetons/kJ de ce
document et de `FEUILLE-DE-ROUTE.md` sont calculés à partir de la puissance
**mesurée** (colonne « W tirés », échantillonnée pendant la mesure), pas de la
limite posée — vérifié par recalcul direct sur plusieurs lignes le 10/09/2026.
Le balayage ci-dessous (3/09) l'établit lui-même : de 400 à 600 W sur la 5090,
le tirage réel ne dépasse jamais ~340 W, la limite ne mordait donc déjà pas au
moment de la mesure. **La dérive du réglage ne rend donc pas ces chiffres
faux**, mais elle rend fausse l'affirmation « tous nos chiffres sont mesurés à
400 W/275 W » si on la lit comme une limite qui aurait pu mordre — elle ne
mordait pas, quelle que soit sa valeur exacte dans la plage testée.

**Règle qui en sort, pour toute mesure future** : un chiffre de jetons/kJ doit
toujours porter la puissance **tirée mesurée**, jamais la limite posée seule —
et la limite posée doit être relevée à l'instant de la mesure
(`nvidia-smi --query-gpu=power.limit --format=csv`), jamais supposée d'après
ce document.

```bash
sudo nvidia-smi -i 0 -pl 400      # 5090, plancher constructeur
sudo nvidia-smi -i 1 -pl 275      # 3080 Ti
nvidia-smi --query-gpu=power.limit,power.default_limit --format=csv
```

Le bridage n'est **pas** persistant au redémarrage (le mode persistance est lui
aussi désactivé) : à vérifier après chaque reboot avant toute campagne de
mesure.

### Ce que la limite coûte vraiment — mesuré le 3 septembre 2026

Balayage à modèle chargé une seule fois, limite changée à chaud, trois tours
de décodage et deux de prefill par point, meilleur retenu ; puissance tirée
échantillonnée pendant la mesure.

**RTX 5090 — la limite de 400 W ne mord jamais.**

| limite | Qwen3-Coder-30B (MoE) | dense 27B Q6_K | puissance tirée, crête |
|---|---|---|---|
| 400 W | 180,2 t/s | 36,6 t/s | 238 W (MoE) · 332 W (dense) |
| 500 W | 180,6 | 36,8 | 175 W · 338 W |
| 600 W | 180,8 | 36,8 | 212 W · 340 W |

De 400 à 600 W : **+0,3 % sur le MoE, +0,5 % sur le dense** — dans le bruit de
mesure. La carte ne demande jamais plus de ~340 W, prefill compris. Les 200 W
retirés ne coûtent rien, et 400 W est de toute façon le plancher réglable.

**Le levier utile sur la 5090 n'est pas la puissance mais la fréquence.**
Verrouillage par `nvidia-smi -lgc 0,<MHz>`, même protocole :

| fréquence | dense 27B | W | jetons/kJ | MoE 30B | prefill MoE |
|---|---|---|---|---|---|
| 3135 (libre) | 37,1 t/s | 352 | 105 | 184,4 t/s | 8824 j/s |
| 2700 | — | — | — | 175,7 (−4,7 %) | 8631 |
| 2400 | 35,2 (−5 %) | 268 (−24 %) | **131 (+25 %)** | 163,8 (−11 %) | 8271 |
| 2100 | 33,3 (−10 %) | 246 | 136 | 153,6 (−17 %) | 7810 |
| 1800 | 31,0 (−16 %) | 224 | 138 | 140,9 (−24 %) | 7225 |
| 1500 | 27,6 (−26 %) | 202 | 137 | — | — |

Deux enseignements. Le **genou dépend de la charge** : un dense, limité par la
bande passante, accepte 2400 MHz pour 5 % de débit et un quart de la puissance ;
un MoE, qui lit peu de poids par jeton et paie surtout des lancements de
noyaux, perd déjà 11 % au même réglage — son genou est vers 2700 MHz.
Et sous 2100 MHz, l'efficacité **plafonne** (136-138 j/kJ) pendant que le débit
continue de tomber : il n'y a plus rien à gagner.

**L'horloge mémoire n'est PAS un levier sur cette carte** : `nvidia-smi -q -d
SUPPORTED_CLOCKS` n'y liste qu'une seule fréquence mémoire (14001 MHz,
14/09) — `-lmc` accepte n'importe quelle demande mais ne peut rien honorer en
dessous, sans le dire (revue/verdict-horloge-memoire-14-09.md).

**RTX 3080 Ti — là, la limite mord.** Qwen3-4B, décodage et prefill :

| limite | décodage | prefill | W tirés | jetons/kJ |
|---|---|---|---|---|
| 375 W | 59,0 t/s | 7198 j/s | 279 | 212 |
| 325 W | 59,0 | 7116 | 276 | 213 |
| **275 W** (réglage actuel) | 57,4 (−2,7 %) | 6760 (−6 %) | 251 | 229 |
| 250 W | 55,9 (−5,3 %) | 6499 (−10 %) | 235 | 238 |
| 225 W | 53,8 (−8,8 %) | 6094 (−15 %) | 214 | 252 |
| 175 W | 46,5 (−21 %) | **4097 (−43 %)** | 174 | 267 |

Au-dessus de 325 W la carte ne demande rien de plus (~280 W tirés) : les
50 derniers watts de la plage sont inutiles. En dessous de 225 W le prefill
s'effondre bien plus vite que le décodage — c'est lui qui a besoin des cœurs.

### Réglages retenus

| carte | limite | pourquoi |
|---|---|---|
| RTX 5090 | **400 W** (plancher) | la limite ne mord jamais ; rien à gagner plus haut, impossible de descendre |
| RTX 3080 Ti | **275 W** — garder | 275 coûte 2,7 % de décodage contre 375 W. Le point 250 W, mesuré ensuite, rend 238 j/kJ contre 227 : 5 % d'efficacité pour 2,5 % de débit et 4 % de prefill en moins. Le gain ne vaut pas le réglage ; 275 W reste le bon choix |

Le verrouillage de fréquence est **hors banc** : il fausse toute comparaison de
t/s. Ne l'utiliser que pour une campagne d'efficacité énergétique, et remettre
`nvidia-smi -i 0 -rgc` ensuite.

Ne bridez **pas** les fréquences mémoire de la 5090 : le débit de décodage est
proportionnel à la bande passante GDDR7, qui est tout l'intérêt de cette carte
ici.

## Ce qui tient

En NVFP4 sur la 5090 (4,5 bits/poids) et INT4 sur la 3080 Ti (4,16), avec
32 + 12 Go de VRAM et 96 Go de RAM. Les débits sont les estimations du
planificateur pour un lot de 1, pas des mesures.

| modèle | poids | placement | décodage estimé |
|---|---|---|---|
| Qwen3-32B | 18 Gio | 5090 seule, 3080 Ti oisive | ~93 jetons/s |
| Llama-3.3-70B | 38 Gio | les deux GPU, 3,4 Gio en RAM | ~18 jetons/s |
| Mistral-Large-123B | 61 Gio | les deux GPU + RAM | quelques unités |
| Qwen3-235B-A22B | 121 Gio | exige `--host-fraction 0.95`, contexte ~27 000 | ~3,5 jetons/s |

Le cas MoE est celui pour lequel les 96 Go de RAM ont été achetés : 235
milliards de paramètres stockés, mais seulement 22 actifs par jeton, donc la
machine lit environ 10 Gio par jeton au lieu de 121.

## Décider où calcule l'étage hôte

La seule mesure qui tranche :

```bash
acvram bench --what bandwidth
```

Elle affiche le débit hôte-vers-appareil de chaque GPU et le débit de lecture
séquentielle de la DDR, puis désigne le gagnant :

```
  cuda:0   NVIDIA GeForce RTX 5090       h2d    52,8 Go/s   d2h    51,9 Go/s
  cuda:1   NVIDIA GeForce RTX 3080 Ti    h2d     6,6 Go/s   d2h     6,4 Go/s
  hote     lecture sequentielle DDR         71,4 Go/s   (16 fils)
           -> calculer l'etage hote sur le processeur
              (--host-exec cpu --host-gb-s 71.4)
```

> **Relevé du 13/09/2026 — non applicable tel quel.** Les 52,8 Go/s ci-dessus
> ont été mesurés avec la 5090 seule en PCIe 5.0 **x16**. Depuis que la 3080 Ti
> occupe le second port, la Z790 Dark Hero partage le lien en **x8/x8** :
> `nvidia-smi` montre `pcie.link.width.current = 8` (max 16) sur les deux cartes,
> et la bande mesurée (M0, `acvram-topology.json`) est de
> **20,9-22,1 Go/s** — le régime des manifestes (18,7). Décision de
> l'utilisateur : on garde x8/x8 (pas de troisième port, affichage sur l'UHD
> 770). Tout coût d'exil se chiffre à 21 Go/s ; le levier sous exil est le
> calcul des ratés sur le processeur (DDR 71 Go/s, insensible au lien).

Réinjectez ce nombre dans la planification : `acvram plan MODELE --host-gb-s
71.4`. La valeur par défaut de 70 Go/s est une plaque signalétique pour de la
DDR5-6000 en double canal, et devrait être remplacée par la mesure.

Si le chiffre DDR ressort nettement sous 70 Go/s, vérifiez que les deux
barrettes sont dans les bons emplacements et que XMP/EXPO est actif — un kit de
96 Go tournant aux fréquences JEDEC divise à peu près par deux l'utilité de
l'étage hôte.

## Choisir un propositeur

```bash
# gratuit, sans second modèle, rentable quand la sortie recopie l'entrée
acvram serve REP --speculative ngram

# la 3080 Ti est oisive pour tout ce qui tient sur la 5090 : servons-nous-en
acvram convert ~/modeles/Qwen3-1.7B -o ~/acv/brouillon --gpus 1
acvram serve ~/acv/qwen3-32b --speculative draft \
      --draft-model ~/acv/brouillon --draft-device cuda:1 --spec-k 5
```

Surveillez `acceptance_rate` et `tokens_per_step` dans `/metrics`. Sous environ
1,3 jeton par étape, la spéculation ne paie pas les positions supplémentaires :
montez `--spec-k` si l'acceptation est haute, baissez-le si elle est mauvaise.

## Séquence de vérification au premier démarrage

```bash
acvram doctor
acvram detect
acvram bench --what bandwidth        # confirme les deux liens et la DDR
acvram bench --what kernels          # confirme que l'extension compile et va vite
acvram plan ~/modeles/Qwen3-32B      # le plan correspond-il au tableau ci-dessus
pytest -q                            # le chemin de référence doit passer
```

## Compteurs de performance GPU (Nsight Compute)

`ncu` refuse de lire les compteurs matériels tant que le pilote les réserve à
l'administrateur : `ERR_NVGPUCTRPERM`. Le déblocage est un paramètre de module,
donc un redémarrage :

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profilage.conf
sudo update-initramfs -u && sudo reboot
```

**Depuis le 14/09/2026, sans redémarrage** : `/etc/sudoers.d/ncu` autorise
`sudo -n /usr/local/cuda/bin/ncu …` sans mot de passe (à côté de
`/etc/sudoers.d/nvidia-smi`). `sudo` remet l'environnement à zéro : repasser
les variables par `env` (HOME de travail, `ACVRAM_KERNEL_CACHE` sur une copie
du cache compilé, `TRITON_CACHE_DIR`) pour que root n'écrive rien dans les
caches de l'utilisateur — `outils/ncu_instr_par_octet.sh` le fait. Le
paramètre de module ci-dessus reste à poser au prochain redémarrage.

Sans cela, seul `nsys` (chronologie des lancements) fonctionne — il suffit pour
compter les noyaux et voir où va le temps, mais pas pour connaître l'occupation,
la pression de registres ni les conflits de banques, qui sont précisément ce qui
décide de nos noyaux.

**vLLM 0.29 patché localement (14/09)** :
`/opt/ia/vLLM/.venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_decode_attention.py:528`
— garde `num_stages = 1` abaissée de `BLOCK_DMODEL >= 1024` à `>= 512`
(original conservé en `.orig-0.29`). Sans lui, le décodage MLA Triton (Lk 576,
BLOCK_DMODEL 512) demande 102 400 o de mémoire partagée par bloc, la 5090 en
offre 101 376 : capture impossible, vLLM ne décode aucun MLA façon DeepSeek sur
RTX 50. Trouvé le 14/09 (revue/duel-mla-glm-14-09.md) ; à refaire après
toute réinstallation de vLLM. `BANC_MLA_STAGES1=0` dans
`outils/banc_decode_vllm_glm.py` évite le double correctif.

**Plafond de puissance de la 5090 : 400-600 W seulement** (14/09, vérifié
`nvidia-smi -pl 300` → « should be between 400.00 W and 600.00 W »). Le mode
« eco » ne peut donc pas passer par `-pl` : il passe par l'horloge
(`-lgc 2 100` mesuré 13/09 : J −19 %, t/s −20 %). Le refus observé le 14/09
n'était pas sudo, c'était la borne matérielle.

**FlashInfer installé (14/09, décision utilisateur, P1 de Sage)** :
`/opt/ia/flashinfer/.venv` (uv, Python 3.12, torch 2.14.0+cu130, flashinfer-python
0.6.18.post1, nvidia-cutlass-dsl 4.7.1 ; import et 5090 sm_120 vérifiés) ;
sources en lecture seule `/opt/ia/flashinfer/src` (a72f726, GitHub, jamais de
publication) — banc de l'étalon : `src/benchmarks/bench_b12x_mxfp4_moe.py`.
Aucune mesure lancée ; la mesure vient après le pas complet MoE (5), sous carte.sh,
`CUDA_VISIBLE_DEVICES=0`, 20 s.

**Piège de chemin (15/09)** : `/mnt/4TO_SATACMR_2022/Modeles/models_acvram` est un
LIEN vers le SSD `/mnt/2TO_2023_980PRO/Modeles/models_acvram` — écrire « sur le
HDD » par ce chemin remplit le SSD (deux variantes 1aj = 34 Go → 870 Mo libres,
« No space left » alors que `df` du HDD affiche 1,1 To). Les convertis d'essai
vont dans `/mnt/4TO_SATACMR_2022/Modeles/models_acvram_hdd/` (vrai dossier HDD) ;
le SSD reste réservé au parc servi (`ACVRAM_MODELES`).
