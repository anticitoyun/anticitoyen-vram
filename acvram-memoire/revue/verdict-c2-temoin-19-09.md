# Verdict — C2 témoin (chemin prefill `c2` : dépaquetage Marlin → bf16 → grouped_mm, poste1 dafb28a) : **7 869 j/s (260 ms/passe) contre 15 990 pour Marlin sur le même arbre — ×0,49**, publié non scellé (poste7 : C2 = infrastructure de C1, seuil retiré)

instrument : `scratchpad/c2-temoin-19-09/chaine.sh` + `prefill-abab-19-09.py` (Coder L=2048, 7 passes, Engine direct sans graphes, J/jeton `energie.py`), arbre C2 `dafb28a` (poste1-c2-prefill-bf16, worktree `poste2-c2`, `arbre exécuté` imprimé), instrument commit `8faffdd` ; prise `carte.sh` 19:19:56-19:23:29 ; bras A aucune variable (Marlin, défaut de cet arbre), bras B `ACVRAM_PREFILL_GROUPED=c2` seule ; le test carte « repack torch ≡ op CUDA » (`tests/test_depaqueter_marlin.py`) a été joué par poste1 dans son arbre (10/11, un échec à lire chez elle)
scellé : aucun (témoin) ; prédiction poste7/poste1 ~15 000 j/s (arithmétique), poste2 13 000-16 000
mesuré : A Marlin **15 990 j/s · 128,1 ms · 0,0238 brut / 0,0199 net J/jeton** (σ 34 ; = 15 987 publié) ; B c2 **7 869 j/s · 260,3 ms · 0,0508 / 0,0423** (σ 78) ; premier bras 3 min (autotune Triton du nouvel arbre)
verdict : **le chemin c2 coûte 2,03 × Marlin en temps et 2,1 × en joules** — loin des ~15 000 arithmétiques : le dépaquetage transitoire vers la DRAM (poste7 : 50,8 ms/prefill) plus le `grouped_mm` déroulé pèsent 132 ms de plus qu'un GEMM Marlin qui lit les tuiles telles quelles ; comme infrastructure de C1 (requant int8 par ligne → GEMM s8), c'est le coût à battre par le noyau C1, pas un produit ; cohérent avec la décision de poste7 de retirer le seuil 19 500
suite : poste1 (C1) : le budget du dépaquetage (≈ 130 ms sur 260) est ce que le noyau groupé doit éviter (tuiles lues en place) ; poste7 : rien à publier hors la ligne témoin ; ma file continue

## Rejouable
`ACVRAM_TYPE=mesure outils/carte.sh bash scratchpad/c2-temoin-19-09/chaine.sh` (3 min 30 avec autotune, ~1 min 30 ensuite).
