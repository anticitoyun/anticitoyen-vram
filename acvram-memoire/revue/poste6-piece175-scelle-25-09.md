# Scellé — pièce 175 : portes α et β des couches GDN en un appel (poste6, 25/09, AVANT compilation et mesure)

* **Constat (poste1, 173, nsys, décodage b=8 alias mixte `Qwen3.8-27B-unsloth-mixte-i8c`)** : α et β sont des poids bf16 [48, 5120]
  (`PlainTensor`, manifeste), servis par `_ref_matmul` → `F.linear` (kernels/__init__.py:1525-1528) : cuBLAS lance 4 blocs × 32 fils,
  31 µs pour 0,5 Mo ; 96 appels par pas = 2,99 ms sur 19,9 (15 %). Sur le défaut `Qwen3.8-27B-nvfp4`, α et β sont nvfp4
  [48, 5120] (GEMV nvfp4, N = 48 < PROJ_MARLIN_MIN_N) : la pièce y est INERTE, l'alias sert de témoin (attendu 1,000 ± 0,5 %).
* **Choix (argumenté)** : deux voies, toutes deux opt-in `ACVRAM_GDN_AB` (défaut `separe`), branche poste6-175 (main a615c931).
  1. `concat` : β‖α concaténés au chargement en UN `QuantLinear` bf16 [96, 5120] (gdn.py `_fusionner_ab`, +0,5 Mo/couche), un
     F.linear par couche au lieu de deux. Au bit des deux appels SI cuBLAS garde le même noyau à N = 96 (une colonne = la même
     réduction sur K) — le test `test_concat_au_bit_des_deux_appels` (M = 1, 8, 16) le dit ; s'il est rouge, concat passe aux
     critères KL. Gain borné : le noyau reste à ~31 µs (K parcouru en série), donc ≈ −1,5 ms (48 appels de moins).
  2. `triton` : GEMM étroite bf16 à accumulation fp32 sur tensor cores (`kernels/gemv_bf16_etroit.py`, tl.dot, un programme par
     32 colonnes, K entier par programme : déterministe, sans atomique), M ≤ 16 ; ~3-6 µs par couche. PAS au bit (ordre de
     réduction ≠ cuBLAS) : critères scellés ci-dessous. 3. Fusion dans un noyau GDN existant (conv fusionnée, récurrence fla) :
     ÉCARTÉE — α/β lisent x (l'entrée de la couche), pas qkv ; les fusionner dans la pile nvfp4 qkv les quantifierait (hors bit
     et hors sujet) ; ajouter un GEMV dans `conv_decode` = un noyau différent pour deux dispositions d'α/β.
* **Instruments** : `tests/test_gdn_ab_175.py` (au bit concat, chemin pris, triton ≤ 1 ulp + déterministe, bras cassant) ;
  `scratchpad/poste6-p175-25-09/prise-vitesse.sh` (`frontiere-pas.py`, b=8 puis b=1, 300 pas, ordre A C T T C A, `pas_gpu`
  médian) ; `kl-decode-lot.py` (b=8 décodage forcé 32 pas après préfill 8 × 78, KL A‖B par position contre T1 = b=1 seul contre
  b=8 sous A, argmax A/B contre T1/A) ; `ppl-decode-kv.py` (b=1, préfixe 2 048, 512 notés, tranches 0-2, A contre triton).
  Régime : -lgc 2700, ECO=off, cpu-safe 100, worktree figé ; prises `poste6-p175-*`.

## Prédictions (écrites avant)
| cellule (pas_gpu médian, 300 pas) | A separe | concat | triton | seuil « tenu » |
|---|---|---|---|---|
| mixte b=8 | ≈ 19,9 ms (173) | **−1,3 à −1,6 ms** | **−2,5 à −2,8 ms** | concat ≤ −1,0 ; triton ≤ −2,0 |
| mixte b=1 | ? (à relever) | −0,3 à −0,8 ms (cuBLAS gemv M=1 ~10-15 µs) | −0,5 à −1,0 | concat ≤ −0,2 ; triton ≤ −0,4 |
| défaut nvfp4 b=8 (témoin, inerte) | ? | 1,000 ± 0,005 | 1,000 ± 0,005 | hors [0,995 ; 1,005] = prise à requalifier |
| triton : KL_max(A‖B) b=8, 32 pas | — | — | ≤ 2 × max(T1, T2) ; argmax A/B ≥ T1/A − 0,5 pt | sinon hors défaut |
| triton : PPL b=1 (3 tranches) | — | — | B/A ∈ [0,995 ; 1,005] | idem |

Issues nommées : (a) cuBLAS change de noyau à N = 96 → concat hors bit (test rouge) → concat jugé aux critères KL comme triton ;
(b) le noyau cuBLAS à N = 96 dure 2 × plus → concat gain ≈ 0 (nommé, concat retiré) ; (c) triton > 10 µs par appel (lancement
sous graphe, tl.dot à M = 16 rembourré) → gain 2,0-2,5 ms seulement ; (d) témoin défaut hors [0,995 ; 1,005] → dérive de séance,
prise à rejouer ; (e) KL triton hors seuil → triton reste diagnostic, concat (au bit) seul candidat au défaut, sur le mot de chef.
