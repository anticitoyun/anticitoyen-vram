# p81 — CAKE (arXiv 2608.12629) pour le KDA de Kimi-Linear-35B : chemin actuel, part du pas, Amdahl, équivalence (poste1, 01/10, à sec)

* instrument : lecture du code (fichier:ligne), config du converti servi `Kimi-Linear-35B-kda-nvfp4`, cellule palier 2 (`verdict-palier2-kimi-linear-17-09.md`), note poste4 30/09 (`poste4-duckai-30-09-arxiv.md`), calcul d'octets
* commit : main 3d9e2537a (branche poste1-p81)
* régime : à sec, aucune carte, aucun profil neuf (aucune trace KDA n'existe au dépôt ni en scratchpad)
* scellé : prédictions ci-dessous, écrites avant toute mesure, chacune avec ce qui la rendrait fausse
* mesuré : rien sur carte. Comptes : 27 couches = **20 KDA + 7 MLA** (3, 7, 11, 15, 19, 23, 26) ; KDA 32 têtes × d 128 ; état S fp32 = **2 Mio par couche et par séquence**
* verdict : **CAKE n'est pas le premier levier.** Code et sm_120 non établis (question posée à poste4). Au décodage b=1, la part KDA reste à mesurer, mais notre propre noyau est le suspect n° 1 (voir ci-dessous) et un remplacement « maison » coalescé y gagnerait autant que CAKE. Une mesure nsys de 10 min tranche avant tout code
* durée : 0 min de carte

## Chemin KDA par phase
| phase | chemin | où |
|---|---|---|
| préfill (t > 1, cuda) | `fla.ops.kda.chunk_kda` (Triton, fla 0.5.2), une séquence à la fois | kda.py:154-165 (`_chunk_kda`, kda.py:33) |
| décodage b=1, graphe | **notre** `kda_decode_kernel` (convs + normes + portes + récurrence + norme de sortie, un lancement après les projections) | kda.py:265-295 → acvram_kernels.cu:6447-6512 |
| décodage en lot | `fla.ops.kda.fused_recurrent_kda` | kda.py:238-263 (`forward_batch`, `decode_static_batch`) |
| repli | boucle torch (`ACVRAM_KDA_CHUNK=0`, ou extension absente) | kda.py:166-173, 296-323 |
Les projections q/k/v, f_a/f_b, g_a/g_b, beta et out sont des GEMV hors KDA : CAKE n'y touche pas.

## Part du KDA dans le pas (bornes à sec ; la cellule du 17/09 servait le KDA « en torch de référence », elle ne dit rien de l'actuel)
* Octets incompressibles du cœur au décodage : lire et réécrire S = 20 × 2 × 2 Mio = **84 Mo par séquence et par pas**.
  b=1 : 55 µs au plancher de 1,52 To/s, soit **1,2 %** du pas de 4,579 ms. b=12 : 1,0 Go, 0,66 ms, soit **2,9 %** du pas de 23,08 ms.
* Notre noyau b=1 est probablement loin de ce plancher (acvram_kernels.cu:6447-6512, lancement :6529) : `<<<H=32, D=128>>>`, soit
  4 096 fils pour 170 SM. Chaque fil lit SA ligne de S (pas de 512 o entre fils voisins : accès non coalescés) et la garde en
  `float row[128]`. Prédiction : 10-25 % du débit HBM, soit 0,22-0,55 ms, **5-12 % du pas b=1**. FAUX si la trace donne < 0,1 ms.
  La prédiction de registres (ptxas : `row[128]` + scalaires, risque de déversement) se vérifie par `nvcc -Xptxas -v` (REGLES § 3)
  hors fenêtre de mesure (poste2 mesurait à 07:37, rien n'a été compilé).
* Préfill : FLOP du cœur par blocs ≈ 3 % de ceux des projections et des experts actifs ; part en temps inconnue (fla Triton multi-noyaux).
  Prédiction 5-15 % d'un préfill de 8 k. FAUX hors de 2-30 %.

## CAKE : ce qui est établi
* Note poste4 30/09 : ×2,05 de moyenne géométrique sur Kimi Delta Attention contre **FlashKDA officiel** (pas contre fla), Ampere→Blackwell ;
  l'abstract ne mentionne PAS sm_120a ; code non confirmé. Questions (code, GPU, phase, référence, tolérance) posées à poste4 le 01/10.
* Sans code sous licence compatible et compilable pour sm_120, CAKE n'est qu'une borne : ce qu'un noyau KDA bien écrit atteint.

## Amdahl (gain bout en bout = p × (1 − 1/s), s = 2,05 → 0,51 p)
| cas | p (part KDA) | gain CAKE prédit | gain d'un noyau maison au plancher |
|---|---|---|---|
| décodage b=1 | 1,2 % (plancher) à 5-12 % (prédit) | 0,6 % à 2,5-6 % | 0 à 4-11 % (p − 1,2 %) |
| décodage b=12 (fla) | ≥ 2,9 % | ≥ 1,5 % | dépend de l'efficacité fla, à mesurer |
| préfill 8 k | 5-15 % | 2,6-7,7 % | à mesurer |
Lecture : si la trace confirme 5-12 % au décodage b=1, recoalescer notre noyau (S lu par colonnes de fils, plusieurs blocs par tête,
état en mémoire partagée par tuiles) passe AVANT CAKE et gagne plus, sans dépendance ni licence.

## Équivalence exigée
* Remplacer `chunk_kda`, `fused_recurrent_kda` ou `kda_decode_kernel` change l'ordre des sommes fp32 : jamais au bit en général. Défaut
  servi : seulement si la sortie est au bit sur le chemin actuel (test d'équivalence dans le même commit, REGLES § 7). Sinon opt-in
  « rapide ± 1 ulp » (REGLES § 1) : variable documentée, ligne de régime, KL contre le chemin exact ≤ 2 × témoin, PPL avec erreur-type,
  cellules étiquetées « ± 1 ulp ».
* Un noyau maison au bit est possible au décodage : même ordre de somme par ligne (`pred` puis `o`, j croissant) et même fp32.
  Coalescer la LECTURE de S sans changer l'ordre des sommes par fil garde le bit. C'est l'exigence à poser pour le décodage.

## Suite proposée (au chef)
1. Une prise de 10 min au plus : nsys de Kimi-Linear b=1 et b=12 (64 pas) et d'un préfill de 8 k, part de `kda_decode_kernel`,
   `fused_recurrent_kda*` et `chunk_kda*` dans le temps GPU. Elle tranche les trois prédictions ci-dessus.
2. Hors fenêtre de mesure : `nvcc -Xptxas -v` de `kda_decode_kernel` (registres, déversement).
3. CAKE seulement si (1) donne une part ≥ 10 % ET que poste4 rapporte un code publié qui couvre sm_120.

## Addendum 01/10 matin : CAKE vérifié à la source, ptxas du noyau b=1
* Réponse duck.ai (poste4, `duck-reponses-01-10.md`) : code dans flashinfer #4262 (préfill) et #4279 (décodage) ; ×2,05 = préfill seul
  contre FlashKDA ; décodage ×1,14 contre FlashInfer amont ; B200. **Vérifié moi-même par l'API GitHub** (ordre chef) : les deux PR
  existent et sont fusionnées le 30/07/2026 (« add optimized B200 recurrent prefill/decode backend », +8 742 et +20 663 lignes).
  **Elles sont sm_100a EXACT seulement** : `CheckExactSm100a` (csrc/kda/flashkda_binding_common.cuh et flashkda_decode_binding_common.cuh)
  refuse tout `major ≠ 10 || minor ≠ 0` ; elles s'appuient sur tcgen05/TMEM, absents de sm_120 ; leurs tests vérifient que compute_120 n'est
  pas dans les drapeaux. **CAKE est inapplicable sur la 5090** : ni le binaire, ni le portage direct (instructions absentes).
* `nvcc -Xptxas -v -arch=sm_120` du `kda_decode_kernel<128>` seul (CUDA 13.4, extrait de acvram_kernels.cu:6446-6512, processeur, nice 19,
  pendant une mesure de qualité de poste2, accord chef) : **255 registres par fil, 48 o de pile, 48 o de déversement**, 1 664 o de smem.
  Ce qui confirme la lecture : `row[128]` sature les registres et déverse. Avec 32 blocs de 128 fils, 32 SM sur 170 travaillent, 4 warps
  chacun, sans parallélisme mémoire. La prédiction scellée (5-12 % du pas b=1) reste à trancher par la prise nsys de 10 min.
* Pièce candidate (après la trace) : `kda_decode_kernel` réécrit avec plusieurs blocs par tête (tranches de lignes i), S en mémoire
  partagée par tuiles et lectures coalescées, sans tableau `row` en registres. Même ordre de somme par ligne (j croissant), donc au bit
  exigé, test d'équivalence dans le même commit.
