# Pièce 116 bis — décomposition du « reste » (58 %), à sec, même trace que la 116

- instrument : à sec, MÊME trace nsys que 116 (aucune nouvelle prise carte) —
  `outils/gpu/mesure/familles-noyaux.py` (`decomposer`/`detailler`, déjà en service) +
  `scratchpad/poste3-piece116-nsys-moe-23-09/classer2_reste.py` (bytes = calcul depuis
  `config.json` du modèle, **plancher théorique, jamais mesuré** — nsys `-t cuda` ne donne
  pas les octets d'un noyau compute, seulement des `memcpy`)
- commit : `18aa76c` (base 116) + ce fichier
- régime : celui de la 116 (i8c, b=1, graphes actifs, décodage établi)
- scellé : aucun neuf posé par chef pour cette pièce (classement par µs perdues demandé,
  pas de seuil de décision)
- mesuré (µs/pas, plafond 1,79 To/s) :

  | noyau | µs/pas | octets (plancher) | Go/s atteints | % plafond | µs perdues |
  |---|---|---|---|---|---|
  | QKVO projections (48 couches, agrégé) | 834,0 | 906,0 MB | 1086,3 Go/s | 60,7 % | **327,9** |
  | attention (`_partiel_reduit_kernel`, 48/pas) | 273,0 | 14,2 MB | 52,0 Go/s | 2,9 % | 265,1 (non pertinent — voir verdict) |
  | normes (`rmsnorm`, 97/pas) | 221,0 | 0,8 MB | 3,6 Go/s | 0,2 % | 220,6 (non pertinent) |
  | rope + kv_write (96/pas) | 176,0 | 0,25 MB | 1,4 Go/s | 0,1 % | 175,9 (non pertinent) |
  | tête (`int8_gemv_kernel[37984x1]`, lm_head int8) | 190,0 | 311,2 MB | 1637,7 Go/s | **91,5 %** | 16,2 |

- verdict : **le classement brut par µs perdues est trompeur — un seul chiffre est réel.**
  - **tête** : classée par erreur dans `proj_etroites_int8` par `familles-noyaux.py` (même
    nom de noyau générique `int8_gemv_kernel`, distinguée ici par sa grille `[37984x1]`,
    seule à 1 lancement/pas). Déjà à 91,5 % du plafond — quasi optimale, rien à gagner là.
  - **QKVO (hors tête)** : SEUL poste réellement sous-bande-passante avec une marge
    crédible — 60,7 % du plafond, **327,9 µs/pas perdues, soit 10,8 % du pas entier**
    (3,037 ms/pas mesuré en 116) et 39 % du temps propre de ce poste. C'est le candidat le
    plus probable pour le retard de 5,8 % sur llama.cpp. Limite de la mesure : les grilles
    des 3 lancements/couche vues dans la trace (`[1280x1]`, `[512x1]`,
    `gemvx::kernel[32x1]`) ne sont pas attribuables à Q/K/V/O individuellement sans lire
    `acvram_kernels.cu` — seul l'AGRÉGAT (bytes totaux connus par la config, temps total
    mesuré) est fiable ; une pièce de suivi devrait lire le noyau avant de viser une forme
    précise.
  - **attention, normes, rope_kv** : les « µs perdues » affichées sont un ARTEFACT du calcul
    — à ce régime (b=1, contexte court ~289 jetons, ≤ 8 Ko à 1 Ko par lancement), ces
    noyaux sont **liés à la latence de lancement, pas à la bande passante** (quelques Go/s
    contre 1,79 To/s : aucun noyau n'atteint ça sur des transferts de cette taille, ce
    n'est pas un défaut, c'est le régime). Le plafond de bande passante est la MAUVAISE
    grille de lecture ici — à chercher plutôt du côté du nombre de lancements (48-97/pas,
    ~2-6 µs chacun, plancher probable = latence de lancement CUDA elle-même sous graphe) si
    une pièce future les vise. Ne pas les confondre avec un gain de bande passante possible.
  - **Ordre par gain RÉEL et actionnable** (bande passante seulement) :
    1. QKVO ≈ 328 µs/pas (10,8 % du pas)
    2. tête ≈ 16 µs/pas (marginal, déjà optimale)
    3. attention/normes/rope : pas un gain de bande passante, autre nature de coût
- durée : à sec, ~10 min (aucune carte reprise)
