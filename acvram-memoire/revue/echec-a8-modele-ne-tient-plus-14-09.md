# A8 échoue deux fois — Qwen2.5-Coder-14B-pur-bf16 ne tient plus sur cette carte

poste3, 14/09/2026. Deux tentatives de profil nsys (`protocole-verdict-etabli-995gos.md`),
deux échecs OOM, à des points différents mais dans le même régime — à
signaler avant une troisième tentative, pas à retenter à l'identique.

## Ce qui échoue, et où

`Qwen2.5-Coder-14B-pur-bf16` : 27,9 Gio de poids. Sur cette carte
aujourd'hui, la capacité totale rapportée par PyTorch est **31,36 Gio**
(pas 32,6 — écart de ~1,2 Gio déjà pris par le pilote/contexte CUDA avant
tout chargement). Le chargeur détecte la marge trop juste et exile 1 MLP en
RAM hôte (`acvram] plan réajusté`), ce qui **désactive les graphes CUDA
pour tout le modèle** — la condition « graphes actifs » du protocole
d'origine ne tient déjà plus dans ces conditions.

**Tentative 1** (allocateur par défaut) : OOM pendant la chauffe, dans
`layers.py:_ensure` (double-tampon de préchargement du MLP exilé).

**Tentative 2** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
admission restructurée pour éviter le churn de la tentative 1) : OOM
**quand même**, plus tard (au tout début de la chauffe, `_ref_matmul`,
tentative d'allouer 2,90 Gio pour la couche MLP EXILÉE elle-même — la
sortie NVML au moment du crash : 2,12 Gio libres sur 31,36 Gio, 29,21 Gio
déjà utilisés par CE processus). `expandable_segments` a repoussé le mur,
pas supprimé le problème : **le modèle ne tient simplement plus dans la
marge disponible**, fragmentation ou pas.

**Fait notable** : le noyau qui échoue est `_ref_matmul` — la voie de
**référence** (`kernels/__init__.py:768`, `torch.nn.functional.linear`),
pas un noyau optimisé. La couche MLP exilée passe par le chemin lent, qui
matérialise le poids en pleine précision — cohérent avec le coût déjà
signalé (`5,4 ms/jeton de PCIe, 48 % du pas résident`), mais ça veut aussi
dire que ce chemin a besoin d'un tampon de taille pleine (poids complet
d'une couche, ~2,9 Gio) en plus du reste — un coût que le calcul de marge
du chargeur (`1,6 Gio` annoncé) ne semble pas couvrir.

## Ce que ça implique pour `ETABLI.md`

**Ce dont ce protocole a besoin — graphes actifs, pas d'exil — ne se
reproduit plus sur cette carte dans son état actuel.** Deux hypothèses,
non tranchées ici :

1. La carte avait plus de VRAM libre au moment de la campagne originale
   (moins d'autres processus/réservations concurrentes) ;
2. Un changement depuis (pilote, ordre de chargement, un autre poste qui
   consomme systématiquement plus qu'avant) a réduit la marge disponible.

**Dans les deux cas, c'est un fait à publier, pas à contourner en douce** :
le chiffre de `ETABLI.md` (26,09 Gio / 28,14 ms, graphes actifs) décrit un
régime qui n'est **plus reproductible tel quel** sur cette machine
aujourd'hui, indépendamment de la question `p > 5,24 %` qu'il visait à
trancher.

## Ce que je ne fais pas maintenant

Je n'attaque pas une troisième variante de contournement mémoire — deux
échecs à des points différents avec deux stratégies différentes suffisent à
dire que le problème n'est pas une astuce d'allocateur, c'est une marge
réellement insuffisante. Options qui restent, à choisir par chef, pas par
moi seule :

- **Réduire le modèle mesuré à ce qu'il faudrait** (une version quantifiée
  du même Qwen2.5-Coder-14B, si `p` peut se transporter — risque : `p` est
  une part relative aux poids lus, pas absolue, donc pourrait rester valide
  à un autre format, mais ce n'est pas garanti sans le dire) ;
- **Libérer de la VRAM avant de lancer** (vérifier qu'aucun autre résidu de
  session ne tient de la mémoire hors du recensement `nvidia-smi
  --query-compute-apps`, ce que je n'ai pas trouvé jusqu'ici — chaque
  vérification après crash montre une carte propre) ;
- **Accepter la mesure en régime dégradé** (exil actif, pas de graphes) et
  la publier comme telle, en sachant qu'elle répond à une question
  différente de celle posée par `ETABLI.md`.

Carte libérée proprement après les deux échecs (vérifié :
`nvidia-smi --query-compute-apps` revient à un état propre à chaque fois) ;
actuellement réoccupée par poste2 (`poste2-profil-decode`).
