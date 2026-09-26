# Dossier — pièce 141 : GEMV int8 QKVO du Coder à b=1 — masquer la latence (à sec, aucun code, poste6, 24/09)

Entrée : 137 (poste1, ncu) — QKV `int8_gemv<4,1>` 1 280 blocs × 128 fils, 9,46 µs, DRAM 64,7 %, SM 19 %, warps actifs 58 %,
`long_scoreboard` 77,8 % ; O 512 × 256, 8,48 µs, DRAM 57,8 %, SM 17 %, warps 47 %, `long_scoreboard` 73,2 %. 116 quater (poste3) :
les mêmes noyaux sous graphe, poids chauds en L2, 1 000 répétitions : **4,10 µs** chacun, indifférents à ROWS ∈ {2..16}.

## 1. Anatomie du noyau tel qu'il est lancé à b=1 (fichier:ligne)
* Lanceur `int8_gemv` (`acvram_kernels.cu:1640-1700`) : `threads_for(K=2048)` = 128 (`:1236`), `splits_for` = 1 (1 280 ≥ 2 × SM,
  `:1254-1262`), grille (M/4, 1), `int8_gemv_kernel<4, NV=1, bf16, bf16>` (`:569`). QKV à b=1 passe par la fusion (3b)
  `int8_gemv_norme` (`:1585`, prologue RMSNorm dans le noyau ; appelant `kernels/__init__.py:1165-1182` depuis
  `attention.py:363-370`), O par `int8_matmul` → `ext.int8_gemv` (`kernels/__init__.py:1041-1080`, chemin `gemv` confirmé par
  CHEMINS_INT8 dans la 137).
* Boucle du fil (`:620-675`) : `for (i = lo + threadIdx.x; i < hi; i += blockDim.x)` avec `nloads = K/16 = 128` et 128 fils →
  **exactement un tour par fil** : 4 `uint4` (une par ligne du bloc, 64 o), 4 échelles half, 4 zéros, puis 64 FMA, puis
  `block_reduce_rows` (`:677-690`). Aucun recouvrement possible entre chargement et calcul dans un fil : le fil charge, attend
  la DRAM (~600-800 ns), calcule 64 FMA, réduit, sort.
* Occupation réelle : 48 registres, 128 fils → 16 blocs/SM possibles ; la grille n'en fournit que 1 280 / 170 SM = **7,5 blocs
  par SM = 30 warps sur 64** — c'est le « 47-58 % de warps actifs » de ncu : la grille est trop petite pour remplir la carte,
  pas le noyau trop lourd. Une seule vague : tous les blocs sont résidents d'emblée.
* Décomposition du temps (137 + 116 quater) : 4,10 µs poids en L2 (montée de la grille + 128 FMA/fil + réduction + descente,
  L2 à ~2,5 To/s effectif) ; 9,46 − 4,10 = **5,36 µs de plus à froid** pour 10,5 Mo = **1,96 To/s** — c'est-à-dire AU plancher
  HBM (1,79 nominal, 1,52 mesuré sur la tête). **La partie DRAM du noyau n'est pas améliorable** ; ce qui se récupère est dans
  les 4,1 µs « chauds » : montée/descente de la grille et lancement, ≈ 2-3 µs, et le fait que la DRAM ne commence à travailler
  qu'une fois les blocs montés (sérialisation montée → chargement → calcul → descente).

## 2. Les trois leviers, jugés
| levier | ce qu'il change | sortie | graphes CUDA | gain prédit / GEMV | coût |
|---|---|---|---|---|---|
| **A. PDL** (lancement dépendant programmatique) : le GEMV est lancé par `cudaLaunchKernelEx` avec `cudaLaunchAttributeProgrammaticStreamSerialization` ; dans le noyau, **les chargements de poids, d'échelles et de zéros (indépendants du prédécesseur) sont émis AVANT `cudaGridDependencySynchronize()`**, puis x est lu après | la montée de la grille ET la latence DRAM des poids se recouvrent avec la queue du noyau précédent (MoE `down` de la couche d'avant pour QKV ; `paged_attention` pour O) | **au bit** (aucune arithmétique ne bouge : même fil, mêmes sommes) | **oui** : capturé par le stream capture en arête `cudaGraphDependencyTypeProgrammatic` (guide CUDA, « Use in CUDA Graphs ») ; sm_90+ requis, sm_120 ✓, CUDA 13.2 / pilote 595 ✓ ; sans attribut, `cudaGridDependencySynchronize()` est un no-op → opt-in sûr `ACVRAM_PDL=1` | **−1,5 à −2,5 µs** (montée ~1 µs + latence DRAM ~0,7 µs + lancement) sur 9,46 / 8,48 | 1 lanceur (`:1640`), 2 lignes dans le noyau, un `#if __CUDA_ARCH__ >= 900` ; aucun changement des noyaux voisins (déclenchement implicite à la sortie de leurs blocs) |
| **B. GEMV persistant** : grille = 2 × SM blocs, chaque bloc boucle sur des tuiles de 4 lignes avec un préchargement à 2 étages (émettre la tuile t+1 avant de calculer t) | remplit les 64 warps/SM, supprime la montée de 1 280 blocs, garde 2 tuiles en vol par fil | **au bit si** le fil garde le même `i` et la même `block_reduce_rows` par tuile (seule l'affectation tuile → bloc change ; les sommes d'une ligne restent celles d'un bloc de 128 fils dans le même ordre) | oui (grille fixe, formes fixes) | **−0,5 à −1,5 µs** (la 116 quater a montré que ROWS, donc le nombre de blocs, ne change RIEN à chaud : la montée n'est pas le coût dominant ; le gain vient du préchargement, borné par le plancher DRAM déjà atteint) | nouveau noyau (~120 lignes), équivalence au bit à prouver sur 12 formes, 2 variantes (norme/sans) |
| **C. plusieurs chargements en vol par fil** : 64 fils × 2 tours, ou `uint4` × 2 par fil, émis dos à dos | 2 × d'octets en vol par warp | **PAS au bit** : l'accumulation `part[r]` d'un fil couvre deux `i` au lieu d'un → ordre des sommes changé (± 1 ulp bf16) → **opt-in « ± 1 ulp » seulement** (REGLES § 1) | oui | **≤ −0,5 µs** : le total d'octets en vol sur la carte ne change pas (30 warps × 64 o × 32 fils déjà ≥ 10 Mo : tout le tenseur est demandé dans la première microseconde) — la 116 quater le dit déjà (ROWS sans effet) | petit, mais PPL + KL à publier, cellule étiquetée |

## 3. Prédiction chiffrée (à sceller avant tout code)
* Pas b=1 servi : 312,3 t/s (README ³) → **3,20 ms/pas** ; QKVO = 48 × (9,46 + 8,48) = **0,86 ms/pas** (27 % du pas).
* **A seul** : −1,5 à −2,5 µs × 96 GEMV = **−0,14 à −0,24 ms/pas → +4,6 à +8,1 % de t/s** ; sortie au bit (test d'équivalence
  sur les 12 formes + capture godets {1, 2, 8, 16} + KL b=1 sur 5 invites, tous inchangés par construction).
  Seuil : **≥ −0,10 ms/pas** (≥ +3 %) au pas certifié b=1 (`certifie-b12.py 1`, ABBA, même séance) ; FAUX si < −0,05 ms/pas.
  Preuve que l'attribut a pris : ncu `launch__…` ne le montre pas ; le montrer par une mesure isolée (paire de noyaux jouets,
  PDL on/off, sous graphe, 1 000 répétitions : le recouvrement se voit à la microseconde) AVANT le pas complet.
* **A + B** : −0,2 à −0,35 ms/pas (+6,7 à +12 %) ; B seul −0,05 à −0,14.
* **C** : ≤ −0,05 ms/pas, hors bit : je ne le recommande pas.
* Issues nommées : gain A < −0,05 → le pilote sérialise quand même (arête programmatique non honorée sous graphe : à vérifier
  par la paire jouet, qui rend « faux » proprement) ; gain A ≥ −0,3 → la montée était plus lourde que 116 quater ne le dit
  (elle mesurait à chaud), tant mieux ; ce qui me gênerait : A gagne sur la paire jouet et rien sur le pas — alors le
  prédécesseur réel (`paged_attention`, MoE `down`) finit par une queue longue que le GEMV recouvre déjà par simple
  ordonnancement, et il faudrait `cudaTriggerProgrammaticLaunchCompletion()` DANS les prédécesseurs (levier 2, code voisin).

## 4. Compatibilité et pièges
* Graphes : PDL passe par le stream capture (torch.cuda.graph → `cudaStreamBeginCapture`) sans code de graphe ; les godets
  et `warm_graphs` n'ont rien à savoir. `cudaLaunchKernelEx` est l'API runtime déjà liée par l'extension ; aucun `-rdc`.
* Le prologue de norme de `int8_gemv_norme` (3b) lit `res` et `delta` produits par le prédécesseur → il doit rester APRÈS
  `cudaGridDependencySynchronize()` ; les 4 `uint4` + échelles + zéros peuvent partir avant (adresses connues de `blockIdx` et
  `threadIdx` seuls). À vérifier par `cuobjdump -sass` que le compilateur n'a pas déplacé les chargements sous la barrière
  (`griddepcontrol.wait` en SASS).
* Ordre des noyaux à b=1 dans le graphe (à relever par nsys, 1 min, avant le code) : le gain de A dépend de la durée de la queue
  du prédécesseur ; si le prédécesseur est un noyau de 2 µs, le recouvrement est plafonné à 2 µs.
* Le même geste vaut pour TOUS les noyaux courts du pas (1 517 lancements/pas, pièce 41) : PDL généralisé est le vrai levier
  b=1, QKVO n'en est que la première paire — hors de cette pièce.

## 5. Ordre proposé (sur feu)
1. paire jouet PDL on/off sous graphe (à sec puis 1 min de carte) : prouve que l'arête programmatique est honorée ;
2. A sur `int8_gemv` + `int8_gemv_norme`, opt-in `ACVRAM_PDL=1`, ligne de régime `pdl=on|off`, test au bit 12 formes ;
3. cellule b=1 ABBA certifiée, scellé § 3 ; 4. B seulement si A tient et laisse ≥ 0,05 ms.
