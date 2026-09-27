# Pièce 276 g — recouvrement de la tour de vision : TTFT moyen −20 % (TENU), mur −2 à −5 % (FAUX contre le seuil), au bit 12/12, solo −1 ms — verdict

poste6, 27/09/2026 17 h 4x. Scellé : `poste6-piece276g-a-sec-27-09.md` (écrit avant le code, inchangé). Code : origin/poste6-276g
(bc93c8e3e moteur + serveur + régime + 3 tests ; scripts ensuite ; prise à 63ee441ad, cellule b = 1 à ec29c0098). Sous carte.sh
(ACVRAM_NOM=poste6-276g, 17:31:17 → 17:32:39, essai 3 : essai 1 tué par un `grep -c` à 0 sous `set -e`, essai 2 par
l'enveloppe `submit` à signature fixe du script de trace — 500 sur `traits=`, les deux conservés dans le scratchpad), arbre
importé contrôlé à chaque bras (`ARBRE`), charge hôte par cellule, échelle **M G G M** : M = main a6268af2e (arbre
travail/main-276g), G = cet arbre (`ACVRAM_TOUR_PREPARATION` au défaut 1). Qwen3-VL-2B vision, une image 448×448, b = 12 × 7 tours
`max_tokens=1`, solo × 5, équivalence 24 jetons b = 4 et 12. Traces (sha256 en tête, M1 G2 G3 M4) : 1d732539 bccf8aae 8fd007b3 e6567809 .
Journaux : M1/M4 « tour (_admit) » 107 lignes, « tour (préparation) » 0 ; G2/G3 l'inverse (107 / 0) — chaque bras a tourné son chemin.

## Ce qui change (bc93c8e3e)
`Engine.encoder_images` : la tour, image par image, MÊME `traits_niveaux` (aucun lot), sur un flux CUDA annexe
(`TourVision.traits_niveaux_sur_flux` : `torch.cuda.stream`, `synchronize` du fil appelant seul, `record_stream` vers le flux
courant), sous `GraphRunner.verrou_capture` (une capture en mode global et un noyau d'un autre fil s'invalident) ; appelée depuis
le gestionnaire HTTP par `asyncio.to_thread` après `preparer_images` ; `add_request(traits=…)` pose `image_embeds`, `image_niveaux`
et les positions M-RoPE — `_admit` ne touche plus la tour pour ces séquences. `ACVRAM_TOUR_PREPARATION=0` = témoin (tour dans `_admit`).

## 1. Identité (critère écrit avant le code)
* Traits et niveaux, flux annexe contre flux courant, 12 images distinctes, deux passages : **AU BIT 12/12 et 12/12** (série = série
  12/12, annexe = annexe 12/12 ; 106 ms les 12 images à chaud dans les deux cas). Test à sec `test_tour_preparation_276g.py` : traits
  de préparation = traits `_admit` (torch.equal), `_admit` n'appelle plus la tour, verrou de capture tenu pendant la tour.
* Jetons (24, temperature 0) : b = 4 : M4 = M1 4/4 ; G2 = M1 3/4, G3 = M1 2/4, G2 = G3 3/4 (invites 0 et 1). b = 12 : M4 = M1 10/12
  (invites 0 et 1 — même non-reproductibilité de main que la 276 f) ; G2 = M1 10/12, G3 = M1 10/12, G2 = G3 11/12. Lecture : à b = 4
  main préfille les 4 en un pas, G en plusieurs pas (admission continue : pas par tour [12, 9, 6, 7, 8, 8, 7] contre [3, 4, 3…]) ;
  la composition des lots de préfill change, donc les arrondis — même mécanisme que main à b = 12. Inapplicable comme critère
  (scellé), publié.
* **b = 1** (aucun effet de composition ; c'est là que « au bit » se juge sur les jetons) : cellule dédiée, 5 invites M puis G —
  RÉSULTAT_B1.

## 2. Tableau (b = 12, une image par requête, 84 requêtes par bras ; ms)
| bras | pas de préfill par tour | `_admit` méd (≥ 2 req.) | TTFT **moyen** | p25 / p50 / p75 | p95 | **mur** max (méd) | solo |
|---|---|---|---|---|---|---|---|
| M1 main | [3, 4, 3, 3, 3, 3, 4] | 44,3 | **219,0** | 154 / 254 / 275 | 284 | **300,4** (277,8) | 31,4 |
| G2 | [12, 9, 6, 7, 8, 8, 7] | **0,1** | **173,0** | 118 / 179 / 231 | 258 | **282,7** (258,8) | 29,9 |
| G3 | [11, 7, 7, 5, 6, 7, 7] | **0,1** | **177,7** | 123 / 180 / 237 | 263 | **288,8** (263,2) | 30,0 |
| M4 main | [3, 3, 3, 3, 3, 3, 3] | 44,3 | **216,7** | 130 / 277 / 279 | 290 | **293,4** (279,6) | 30,9 |

## 3. Scellé, point par point
* **TTFT moyen G ≤ 0,85 M** : 217,9 → 175,4, **0,805 : TENU** (−19,5 %) ; p50 254-277 → 179-180 (−30 %), p95 −9 %.
* **Mur G ≤ 0,85 M** : max 300,4/293,4 → 282,7/288,8 (**0,96 : FAUX**) ; médiane 278,7 → 261,0 (0,94, faux aussi). Cause lue dans
  les traces : les 12 tours restent en SÉRIE (un flux annexe, un verrou : 12 × 9 ms ≈ 108 ms), donc la dernière requête attend
  toujours 11 tours avant la sienne ; ce que G gagne, c'est que les 11 premières n'attendent plus la 12e — d'où le moyen, pas le mur.
  Ma prédiction (≈ 0,75) supposait la tour hors du chemin critique de la DERNIÈRE requête : faux tant que les tours sont sérielles.
* `_admit` médian ≤ 10 ms : 44,3 → **0,1 : TENU**.
* Solo b = 1 : 31,4/30,9 → 29,9/30,0 : −1,0 à −1,2 ms — à la borne de « ± 1 » (dans le bon sens : la tour n'est plus sous `_lock`).
* Issue nommée (mur meilleur, moyen ≥ M) : **non survenue** — c'est l'inverse : moyen très meilleur, mur peu.
* Régression cherchée : aucune (p25, p50, p75, p95, max, solo, tous ≤ M sur G2 et G3).

## 4. Décision proposée
Identité au bit tenue, aucune régression, TTFT moyen −20 % : **défaut 1** comme le scellé le prévoyait, en disant que le mur n'est
pas tenu (−2 à −5 %, seuil 15 %). Levier suivant, nommé par les traces : les tours d'une rafale sont sérielles — deux flux annexes
(deux images en vol) ou une tour qui accepte plusieurs images sans changer de noyau (la 276 f a montré que le LOT n'est pas au bit ;
deux flux, chacun UNE image, le restent) ; prédit : mur −80 ms si deux tours se recouvrent. À chef.
