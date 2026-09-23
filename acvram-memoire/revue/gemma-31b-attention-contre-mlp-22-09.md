# Pièce 54 — la source de l'erreur du 31B nvfp4 est PARTAGÉE : attention nvfp4 seule → 4/5 (kl_max méd 0,28), MLP nvfp4 seul → 4/5 (méd 0,80) ; les deux moitiés s'ajoutent en quadrature (0,27² + 0,32² ≈ 0,41²) et la couche 52 amplifie dans tous les cas — 22/09 (Gaelle)

* instrument : `scratchpad/gaelle-p54-22-09/construire-hybride.py` (manifeste de `nvfp4-vision`, tenseurs choisis par motif repris de `bf16-vision`, shards liés, 0 conversion, en-têtes safetensors contrôlés) → bras **A** `…-attn-bf16-p54` (350 tenseurs `self_attn.*`, 15,8 Gio bf16 repris, MLP nvfp4) et bras **M** `…-mlp-bf16-p54` (180 tenseurs `mlp.*`, 38,8 Gio repris, attention nvfp4) ; `kl-gabarit.py acvram|hidden|divergence`, mêmes 5 invites, mêmes dumps HF (pièce 52)
* commit : arbre 618574a1 (gaelle = main + instruments)
* régime : **DÉGRADÉ déclaré, comme écrit avant** — A : `couches_exilées=40/60`, M : 40/60 aussi, `graphes=off`, flux hôte ; arithmétique du prefill inchangée (logits aux positions de réponse), vitesse sans objet ; eco 2700 ; compute-apps début = fin ; verrou tenu 33 + 16 + 59 + 44 s
* scellé (gaelle.md 618574a1, avant) : M ≥ 4/5, kl_max ≤ 1,5 sur ≥ 4 invites, err. résiduelle c57 ≤ 0,20 ; A ≤ 3/5, err. c57 ≥ 0,30 ; réfuté si A ≥ 4/5 et M ≤ 3/5 ; issue gênante : A et M ≤ 3/5
* mesuré (KL(hf ‖ acvram) par pas, 154 pas, seuil kl_max ≤ 1,0 par invite) :

| bras (ce qui reste en nvfp4) | kl_max invites 0-4 | méd. kl_max | p90 des pas | pas ≥ 1 | tenu | err. rés. c57 / finale |
|---|---|---|---|---|---|---|
| nvfp4-vision (tout nvfp4, p52) | 3,17 · 0,67 · 1,01 · 0,02 · 2,84 | 1,01 | 0,376 | 5 | 2/5 | 0,41 / 0,25 |
| **A : attention bf16** (MLP nvfp4) | 2,43 · 0,31 · 0,80 · 0,01 · 0,97 | 0,80 | 0,117 | 1 | **4/5** | 0,32 / 0,18 |
| **M : MLP bf16** (attention nvfp4) | 0,66 · 0,28 · 0,16 · 0,02 · 1,14 | **0,28** | 0,100 | 1 | **4/5** | 0,27 / 0,17 |
| hybride 6 couches (p53) | 3,18 · 0,62 · 0,63 · 0,05 · 4,04 | 0,63 | 0,263 | 4 | 3/5 | 0,32 / 0,22 |

  Divergence par couche : le saut de la **couche 52 subsiste dans les deux bras** (+0,080 en A, +0,068 en M, +0,103 tout nvfp4) et la 51 aussi (+0,035 / +0,034) : amplificateurs, confirmé une troisième fois. Erreur propre de chaque moitié (mesurée par l'autre bras) : attention 0,27, MLP 0,32 ; **0,27² + 0,32² = 0,175 ≈ 0,41² = 0,168** — les deux erreurs sont indépendantes et s'ajoutent en quadrature, MLP ≈ 55 % de la variance, attention ≈ 45 %. Invite 0 (3,17 → 2,43 en A, 0,66 en M) est portée par le MLP ; invite 4 (2,84 → 0,97 / 1,14) par les deux.
* verdict : prédiction **partiellement réfutée** : M ≥ 4/5 et A à 0,32 tenus, mais A fait aussi 4/5 (prédit ≤ 3/5) et M reste à 0,27 (prédit ≤ 0,20) — **aucune des deux moitiés seule n'atteint 5/5 ; la source est partagée presque à parts égales**, ce n'est ni « l'attention » ni « le MLP » ni six couches. Bornes supérieures pour une conversion : `--attn-qkvo-int8-canal` (attention int8, MLP nvfp4) donnerait AU MIEUX le bras M : méd. kl_max 0,28, 4/5, kl_max 1,14 sur l'invite 4 — proche du seuil mais pas dedans ; MLP à plus de bits sans toucher l'attention, au mieux le bras A : 4/5, 2,43 sur l'invite 0. **Pour 5/5 il faut réduire les deux moitiés** : attention int8 ET MLP mieux quantifié (calibration sous gabarit — le 31B a été calibré sur texte brut que ce modèle -it ne lit pas, pièce 37 — ou int8 partout avec exil). Aucune reconversion ne vaut ses 6 h avant qu'un bras hybride ait montré ≥ 5/5 avec méd. kl_max ≤ 0,3 : la technique par manifeste peut l'établir en < 2 min par bras (par exemple attention bf16 + MLP nvfp4 recalibré sous gabarit sur 6 couches témoins, quand une telle calibration existera).
* durée : 152 s de carte en 4 prises, 0 conversion ; verrou rendu 23 h 13, carte LIBRE

## Reste
Test à sec de `mode_divergence` et du constructeur d'hybride (manifeste : clés, plan) ; les trois alias hybrides `*-p53/p54` sont des liens symboliques (0 octet) : à effacer quand Jerome le dit.
