# 236 — Flathub pour acvram (poste6, 26/09, à sec) : `packaging/flathub/` livré ; le point dur CUDA tranché — pilote par l'extension GL.nvidia, bibliothèques CUDA embarquées (roues nvidia-* élaguées), noyaux PRÉCOMPILÉS car pas de nvcc dans le bac à sable ; taille ≈ 1,9 Go téléchargés

* instrument : lecture du modèle `rog-flare2-anime-matrix/packaging/flathub/` et de son workflow ; `du` du venv servi (torch 2.14.0+cu130) ;
  `packaging/acvram-gui` (GTK 3 + WebKit2 4.1, `/usr/share/acvram/{galerie,langues}`), `acvram-console`, `.desktop`, `kernels/__init__.py`. Aucune carte.
* commit : poste6-236 = origin/main 88e375bdd + `packaging/flathub/` (manifeste, metainfo en/fr, `sources-torch.sh`, README). Rien n'est soumis :
  l'utilisateur soumet ; poste3 branche le job dans `release.yml`.
* verdict : faisable tel quel côté paquet ; **trois prérequis de code** avant la première construction verte (README § « ce qui reste »),
  dont un dans `kernels/__init__.py` (noyaux précompilés sans ninja/nvcc).

## Le point dur : CUDA dans un Flatpak
| couche | où elle vit dans le Flatpak | décision |
|---|---|---|
| pilote (libcuda.so.1, libnvidia-ml) | **extension `org.freedesktop.Platform.GL.nvidia-<version>`**, posée automatiquement par flatpak d'après le pilote de l'hôte, exposée avec `--device=all` (les nœuds /dev/nvidia*, /dev/nvidia-uvm ; `--device=dri` ne les donne pas) | rien à embarquer ; c'est ce que fait Blender pour CUDA/OptiX |
| bibliothèques CUDA de l'application (cudart, cublas, cusparse, cufft, curand, nvrtc, nvjitlink) | les **roues `nvidia-*-cu13`** que la roue torch tire ; redistribuables (CLUF NVIDIA, redistribution des « runtime libraries » permise) → sources `file` avec sha256, PAS `extra-data` (Flathub réserve extra-data à ce qui ne peut pas être redistribué : torch est BSD, les libs CUDA sont redistribuables) | embarquer, **élaguer** : cudnn (1,1 Go), nccl (0,3), nvshmem, cusparselt ne servent à rien à acvram (attention maison + Triton, une carte) — `cleanup` du manifeste |
| torch 2.14 cu130 (1,2 Go) + triton 3.8 (0,9 Go) | roues du dépôt download.pytorch.org (hors PyPI : `flatpak-pip-generator` ne les voit pas) → `sources-torch.sh` les résout en sources `file` | embarquer |
| **noyaux acvram** (`acvram_kernels.cu` 8 600 lignes, port Marlin) | aujourd'hui compilés à l'import par `torch.utils.cpp_extension.load` (nvcc + ninja, `~/.cache/acvram/kernels-<empreinte>`) — **le bac à sable n'a pas nvcc** (le SDK GNOME non plus ; l'extension nvidia n'apporte que le pilote) ; sans nvcc, `kernels/__init__.py` retombe sur le chemin de référence (copie 16 bits des poids : lent, donc un Flatpak qui « marche » mais à un tiers du débit, sans le dire) | **précompiler** en CI (runner avec nvcc) pour sm_120 (5090 ; + sm_89, sm_86 si publiés), ranger par empreinte sous `/app/lib/acvram/noyaux`, `ACVRAM_KERNEL_CACHE` pointé là — ET ajouter le chemin de chargement direct du `.so` (≈ 30 lignes) ; Triton, lui, compile ses noyaux à l'exécution sans nvcc (il embarque son ptxas) : cache sous xdg-cache |

## Taille et règles de Flathub
* Installé : torch 1,2 + nvidia 3,2 + triton 0,9 = 5,3 Go dans le venv servi ; après élagage cudnn/nccl/nvshmem/cusparselt : **≈ 3,6 Go installés,
  ≈ 1,9 Go téléchargés** (roues compressées) ; le reste (acvram, GNOME) est négligeable.
* Flathub n'écrit **aucune limite dure de taille** dans ses règles (`flatpak-builder-lint` ne la vérifie pas) ; la revue demande de justifier
  les gros paquets et refuse le gonflement évitable — d'où l'élagage et le refus d'embarquer le toolkit CUDA (nvcc, 2 Go de plus) au profit
  des noyaux précompilés. Précédents : Blender (CUDA/OptiX embarqués, ~0,3 Go), les applications ML sur Vulkan restent petites ; un paquet
  à 1,9 Go est exceptionnel mais pas inédit (jeux, IDE). **À écrire aux réviseurs avant la soumission** (issue sur flathub/flathub), avec le
  chiffre et la raison ; si refusé : repli = `extra-data` pour les roues torch/nvidia (accepté au cas par cas pour les très gros binaires)
  ou dépôt Flatpak propre (hors Flathub) servi depuis GitHub Releases.
* Question ouverte, envoyée à poste4 → duck.ai par chef : « Flathub accepte-t-il un paquet de 1,9 Go embarquant torch cu13 ; extra-data
  admissible pour des roues BSD ? » — la réponse ne change pas la structure du répertoire, seulement le module `torch-cu130.json`.

## Ce que le manifeste fait et ne fait pas
* Fait : GNOME 51 (GTK 3 + WebKit2 4.1 présents), `acvram-gui` en commande, `acvram-console` en action du `.desktop`, `.desktop`/icône renommés
  à l'identifiant, galerie et langues sous `/app/share/acvram` (chemins que `acvram-gui` cherche déjà après `/usr/share/acvram`), modèles en
  lecture seule (home, /mnt, /media), config et cache xdg, réseau (API locale, téléchargement de modèles).
* Ne fait pas : la règle `-lgc` (nvidia-smi sur l'hôte, hors bac à sable : `acvram eco` doit le dire), le `.deb` parc, les tests.
