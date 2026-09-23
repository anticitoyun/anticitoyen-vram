# Protocole — re-tampon 0.6.6 narrow OFF, puis cellule narrow 3 × 3 (0.6.7)

Laure, 16/09/2026, avant mesure. Ordre : Sage
[`sage-narrow-verdict-16-09.md`](sage-narrow-verdict-16-09.md) § 1 et § 3,
distribué par Jérôme (`ETAT.md` l. 14-15, 20). Arbre : travail/laure figé
au commit qui suit ce protocole, nommé dans le verdict et dans l'en-tête de
l'unité (code `acvram/` = main **b06e337**, seul `outils/ppl-narrow-b12-coder30b.py`
modifié ; narrow défaut OFF,
`kernels/__init__.py:501` ; MIN_T défaut 5, `model.py:1471` ; route+pack).
Une seule prise de carte, derrière Laurine (awq-unité, verrou pris 07:04).

## 1. Re-tampon OFF — montage

Même script et même protocole que l'officiel
([`verdict-officiel-066-15-09.md`](verdict-officiel-066-15-09.md), arbre
66c7532) : `certifie-b12-15-09.py`, rondes ctx 2048 / invite 256 / ≥ 20 s,
`energie.py` corrigé (b11b8a3) + chrono hôte, une carte, -pl 400, horloge
libre, température en en-tête. ABAB : **A = `ACVRAM_NARROW_GEMM=0`** (le
défaut de l'arbre, le chiffre à re-tamponner), B = `=1` (témoin du −16 %,
pour que la note porte l'écart et non un chiffre seul). Constante relue au
JSON (`_NARROW_GEMM_lu`).

### Attendu (Sage) : A ≈ 14,0 ms / ≈ 0,51 J.

### Ma prédiction (scellée)

Entre 66c7532 et b06e337, deux changements touchent le pas b=12 : MIN_T
9 → 5 (sans effet à b=12, godet ≥ 9 dans les deux cas) et le routeur
fp32 réduit (déjà dans 66c7532). Donc **A = 13,9-14,1 ms, 0,50-0,52 J ; B
= 11,6-11,9 ms, 0,42-0,44 J ; B/A −15 à −17 %**. Réfuté si A sort de
[13,7 ; 14,4] ms : alors l'arbre a bougé ailleurs que là où je regarde,
et le chiffre publié n'est pas un re-tampon mais une nouvelle mesure à
expliquer (bissection entre 66c7532 et b06e337 avant publication).
Bruit publié d'avance : |A1 − A2| et |B1 − B2| ≤ 0,05 ms (officiel : 0,03).

## 2. Cellule narrow 3 × 3 — montage

`outils/ppl-narrow-b12-coder30b.py` (Manon, 16/09), **+ `PPL_DECALAGE`**
(jetons ; tranche `ids[dec : dec + 12 × 2048]`, écrit au JSON avec le
chemin `acvram` importé) — seule modification, commit nommé dans le
verdict. Trois tranches disjointes de `wiki-gptq.txt` : décalages 0 /
24 576 / 49 152. Trois bras par tranche, dans l'ordre A-graphes (NARROW=0),
B-graphes (NARROW=1), A-eager (NARROW=0, `TIES_EAGER=1`) ; plan
`max_concurrent_seqs=12` (assertion `kv_max_tokens ≥ 24 576`) ;
`ACVRAM_MOE_DECODE_MMA=1` ; preuve par bras : `lancements_narrow_gemm`
(A = 0, B = 2 047 × 96 = 196 512 attendu), `n_jetons_notes` = 24 564.

### Scellé (Sage § 3)

**Tenu** ssi moyenne des trois B/A graphes dans 1,000 ± 0,002 **ET**, sur
chaque tranche, |B/A − 1| ≤ 1,5 × |A-graphes/A-eager − 1|. **Réfuté** si
moyenne > 1,002 ou une tranche > 1,004 → Laurine (M capturé / rejoué),
narrow reste OFF.

### Ma prédiction (scellée) — et un trou dans le scellé, dit d'avance

1. **Tranche 0 reproduit Manon à l'identique** (teacher-forcing
   déterministe, même arbre de calcul) : A = 8,498852, B = 8,518279,
   A-eager = 8,511008 → B/A = 1,002286, témoin 0,00143, borne 1,5 × =
   0,00214. **Sur cette tranche la condition « tenu » est fausse par
   0,00015 et la condition « réfuté » (> 1,004) est fausse aussi.** Si
   les tranches 1-2 tiennent et la moyenne aussi, la cellule rend « ni
   tenu ni réfuté » à la lettre. Je le dis avant de mesurer : dans ce cas
   je publie **INDÉTERMINÉ** avec les trois lignes et je ne tranche pas
   (Sage arbitre) ; je ne reformule pas le scellé après coup.
   Réfuté (ma prédiction 1) si tranche 0 ne reproduit pas Manon à
   10⁻⁶ près : non-déterminisme, résultat en soi, à publier avant tout.
2. **Tranches 1-2** : si le +0,0023 de la tranche 0 est du bruit
   d'échantillon (lecture de Sage § 2), les deux autres B/A tombent dans
   1,000 ± 0,0025 avec des signes non concordants, et **la moyenne des
   trois dans [0,9995 ; 1,0020]**. Réfuté si les trois B/A sont > 1,001
   de même signe : coût systématique, la lecture « bruit » tombe, Laurine.
3. Témoin graphes/eager par tranche : |A-g/A-e − 1| dans [0,0005 ;
   0,0025] ; si une tranche rend < 0,0003, la borne 1,5 × devient
   inatteignable pour tout écart et la tranche ne juge plus (à dire, pas à
   contourner).

## Durée et unité

≈ 10 min (4 rondes) + ≈ 27 min (9 passes PPL) ; unité
`retampon-cellule-laure`, `ACVRAM_ATTENTE=3600`, sorties
`scratchpad/retampon-cellule-narrow-16-09/`.
