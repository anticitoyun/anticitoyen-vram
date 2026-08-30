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

La 5090 Astral LC OC approchera les 600 W en prefill soutenu. Une limite posée
par `nvidia-smi -pl` coûte moins de débit que ne le fait un étranglement
thermique, et le décodage, limité par la mémoire, s'en aperçoit à peine.

```bash
sudo nvidia-smi -i 0 -pl 500
```

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
