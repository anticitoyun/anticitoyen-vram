# Verdict — pièce 175 : portes α et β des couches GDN en un appel (poste6, 25/09) — TENU, `auto` candidat au défaut (mot de chef)

* **instrument** : `scratchpad/poste6-p175-25-09/` — `prise-vitesse.sh` (`frontiere-pas.py`, 300 pas, `pas_gpu` médian, ordre
  A C T T C A), `profil-decode-ab.py` (noyaux d'un pas b=8 eager, torch.profiler), `micro-ab.py` (eager et graphe de 48 appels),
  `kl-decode-lot.py` (b=8, 32 pas forcés après préfill 8 × 78, T1 = b=1 seul contre b=8), `ppl-decode-kv.py` (b=1, préfixe 2 048)
* **commit** : 0d5c055d (mesures), ae0f917a (mode `auto` ≤ 8, casts absorbés, bras U) — branche poste6-175 depuis main a615c931
* **régime** : -lgc 2700, ECO=off, cpu-safe 100, alias mixte `Qwen3.8-27B-unsloth-mixte-i8c` (α/β bf16), témoin défaut
  `Qwen3.8-27B-nvfp4` (α/β nvfp4, inerte) ; preuve du chemin sur la ligne de régime `gdn=fla ab=<mode>(48)` · **scellé** :
  `poste6-piece175-scelle-25-09.md` · **mesuré** : oui · **verdict** : TENU (concat et triton), issues (a) et (c) prises
* **durée** : 3 chaînes (prises 1-2 nulles : voir « faute »), chaîne 3 = 05:08 → 05:35 (attentes 2 147 + 967 + … s), ≤ 30 min chacune

## Vitesse (mixte, pas_gpu médian sur 300 pas, µs ; 2 lots par bras, dispersion ≤ 0,1 %)
| b | A separe | concat | triton | prédit concat / triton | seuil | tenu |
|---|---|---|---|---|---|---|
| 8 | 19 359,7 | **16 918 (−2,44 ms, −12,6 %)** | **17 248 (−2,11 ms, −10,9 %)** | −1,3 à −1,6 / −2,5 à −2,8 | ≤ −1,0 / ≤ −2,0 | oui / oui |
| 1 | 15 185,9 | **14 979 (−0,21 ms, −1,4 %)** | 15 427 (**+0,24 ms**) | −0,3 à −0,8 / −0,5 à −1,0 | ≤ −0,2 / ≤ −0,4 | oui (limite) / **non** |
| témoin défaut b=8 | 13 326 | 13 328 (1,000) | — | 1,000 ± 0,005 | — | oui |

Profil du pas b=8 (eager) : separe 96 × `cutlass wmma 16x16 128x1` = 2,95 ms/pas (= 2,99 de la 173) ; concat 48 × `128x2` =
0,19 ms ; triton 48 noyaux hors liste, Σ noyaux 11,83 → 9,14 → 8,92 ms. Micro-banc en graphe (µs/couche, M = 1 / 8 / 16) :
separe 6,0 / 42,8 / 7,9 · concat 3,0 / 3,9 / 4,1 · triton 9,3 / 8,9 / 9,3. Issue (a) : concat dépasse sa prédiction parce que cuBLAS
prend à N = 96 un noyau 128x2 (split-K) ; issue (c) : triton à 8,9 µs (lancement + tl.dot rembourré à 16 lignes), plus lent que cuBLAS à M = 1.

## Qualité (mixte)
| bras | b=8, 32 pas × 8 séq. : KL_max(A‖·) | 2 × max T1 | argmax ·/A | T1/A | PPL b=1 (3 tranches) B/A | tenu |
|---|---|---|---|---|---|---|
| triton | **0 — AU BIT des deux appels** (rejeu 0, max|Δlogit| 0) | 0,00109 | 100 % | 96,9 % | 0,998 · 1,000 · 1,003 | oui |
| concat | 0,00373 (3,4 × le seuil) | 0,00109 | 98,4 % | 96,9 % | non mesurée | **non** (KL) |

Lecture : le wmma 128x1 de cuBLAS (M ≤ 16, N = 48) et `tl.dot` accumulent en fp32 dans le MÊME ordre tensor-core → sortie
identique au bit à b=8 ; le 128x2 de concat (N = 96) combine deux partiels et s'écarte du fp32 (4e-4 sur 4e-3 au test) : plus
rapide, moins juste. À M = 1, concat est au bit (test) et cuBLAS bat triton. Prise 4 : triton == les deux appels au bit à
M = 2, 4, 8, PLUS à 12 et 16 (cuBLAS y change de noyau). D'où **`ACVRAM_GDN_AB=auto`** (ae0f917a) : M = 1 → concat, 2 ≤ M ≤ 8 →
triton, au-delà → les deux appels ; `test_auto_au_bit_des_deux_appels` (M = 1, 2, 3, 4, 6, 8, 12, 16, 24 : `torch.equal` contre
les deux F.linear). Casts bf16→fp32 de β/α absorbés (demande d'poste1, 182 : 96 `direct_copy` = 115 µs/pas) : le noyau écrit
fp32 arrondi par bf16 dans le registre (au bit, `test_les_casts_absorbes_sont_au_bit`), un seul cast sur β‖α sinon. Cellules
`auto` (prise 5, bras A U U A) : voir tableau final ci-dessous. b = 12 ne gagne rien sous `auto` (les deux appels) — nommé.
→ **défaut possible sans critère KL** (au bit à chaque M), sur le mot de chef.

## Faute consignée (prises 1-2, effet nul sur les 6 bras, deux chaînes de carte perdues ≈ 1 h de file)
Un `ab = None` de CLASSE sur la nn.Module masquait le sous-module assigné (`_modules`, lu par `__getattr__` seulement si la
recherche normale échoue) : le chemin fusionné n'était jamais pris, le compteur du chargement disait 48. Les tests passaient
sur un SimpleNamespace. Règle (ICM, MECANISMES à écrire) : attribut d'instance ; tester sur la vraie classe et par le vrai
point d'entrée ; prouver le chemin par un compteur PRIS au forward ou par le profil, jamais par le seul chargement.

## Tableau final — `auto` (prise 5, commit 3f0bd9d0, 06:18 → 06:21, bras A U U A, mixte, pas_gpu médian sur 300 pas, µs)
| b | A separe (A1 / A2) | auto (U1 / U2) | Δ auto − A | attendu (gagnant de la prise 3, moins les casts absorbés ≤ 0,115 ms) | tenu |
|---|---|---|---|---|---|
| 8 | 19 340,3 / 19 370,0 | **17 127,3 / 17 137,7** | **−2 222,7 (−11,5 %)** | triton 17 248 → 17 133-17 248 | oui (−116 µs sur triton = les 96 casts) |
| 1 | 15 184,7 / 15 184,9 | **14 939,2 / 14 939,1** | **−245,6 (−1,6 %)** | concat 14 979 → 14 864-14 979 | oui (−40 µs sur concat) |

Dispersion intra-bras ≤ 0,15 % (b=8), ≤ 0,01 % (b=1) ; graphes on, repli_eager=0, replays=339 sur les quatre bras ; SM « ? » (frontière
de pas sans dmon, comme les prises 3-4). `auto` prend à chaque M le chemin déjà prouvé au bit (prise 4) : aucun critère KL à ajouter.
Tests `tests/test_gdn_ab_175.py` : 50 verts / 1 rouge au commit 3f0bd9d0 — `test_le_chemin_fusionne_est_pris`, faux du test seul
(sa lambda de comptage ignorait `fp32=` après l'absorption des casts) ; corrigé et rejoué sous verrou : voir ligne « rejeu » ci-dessous.
Rejeu (commit d29766b1, 07:25:13, sous verrou `ACVRAM_NOM=poste6-175`, 46 s d'attente, PID hors verrou : llama-server 4242 seul,
avant et après) : `tests/test_gdn_ab_175.py` **28/28 verts** en 1,1 s (`scratchpad/poste6-p175-25-09/prise-tests-rejeu.txt`).
Branche poste6-175 fusionnable ; `auto` au défaut = décision de chef (issue nommée : b = 12 sans gain, les deux appels).
Tests voisins (03087fe0, 07:31, sous verrou) : `tests/test_regime*.py`, `tests/test_cli*.py` et le fichier de la pièce, **156/156 verts**
(`prise-tests-regime.txt`) — la prise de 06:15 comptait 51 tests sur un périmètre perdu avec la session ; celui-ci le couvre.

## Après fusion de main (176 qkv‖gate int8 dans `_projections`, commit 7e6c12c1, 07:43-07:45, sous verrou, cpu-safe 100)
Tests 175 + 176 + regime + cadrage : **177/177 verts** (`prise-fusion-176.txt`). ABBA b=8 mixte (A U U A, µs) : separe 19 317,6 / 19 301,6,
**auto 16 984,1 / 16 990,2 → −2 322 µs (−12,0 %)** ; les deux bras portent la 176 (separe 19 355 → 19 310 avant/après fusion, −45 µs).
Le gain 175 tient sur l'arbre fusionné ; issue « interaction avec la 176 » : aucune (β‖α et qkv‖gate lisent la même entrée, appels distincts).
