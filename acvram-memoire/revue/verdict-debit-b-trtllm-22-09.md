# Verdict — débit(b) TRT-LLM, cause de l'effondrement b=1 (poste3, 22/09)

Qwen3-Coder-30B-A3B, RTX 5090, trtllm-serve (backend pytorch, max_batch_size=12),
client officiel banc-llamacpp-16-09.py decode, 1024 jetons/séq (b≤4) puis 256 (b=8),
fenêtre 20 s. Cellule b=12 (1 997 t/s) mesurée à part le même jour.

## Courbe débit(b) et les 7 lignes

| b | total t/s | t/s/séq | watts | MHz | lots | durée |
|---|---|---|---|---|---|---|
| 1 | 46,3 | 46,3 | 399,7 | 2533 | 1 | 22,1 s |
| 2 | 49,1 | 24,6 | 399,4 | 2462 | 1 | 41,7 s |
| 4 | 50,6 | 12,7 | 399,7 | 2393 | 1 | 80,9 s |
| 8 | 50,6 | 6,3 | 399,8 | 2598 | 1 | 40,5 s |
| 12 | 1 997,8 | 166,5 | 378,2 | 2946 | 4 | 24,7 s |

1. **Le débit TOTAL est PLAT à ~50 t/s de b=1 à b=8**, à 400 W constants ; il n'explose qu'à b=12 (1 997 t/s, ×40). Transition nette **entre b=8 et b=12** (= `max_batch_size`).
2. **Le débit PAR SÉQUENCE chute** régulièrement (46 → 25 → 13 → 6,3 t/s) de b=1 à b=8 : ajouter des séquences les ralentit chacune sans gain de total → **sérialisation pure** (le GPU traite une requête à la fois, réparti).
3. **H1 (GEMM/MoE sous-alimenté, sur-linéaire) est INFIRMÉE** : un régime sous-alimenté monterait dès b=2 ; ici rien ne monte jusqu'à b=8.
4. **Cause** : le scheduler de trtllm-serve (`CapacitySchedulerPolicy GUARANTEED_NO_EVICT`, `max_batch_size=12`) ne recouvre les requêtes qu'à batch **quasi plein** (proche de 12) ; en dessous, il les sérialise. C'est un choix taillé pour le débit à batch plein, pas la latence à faible charge.
5. **Ce n'est pas le client** : acvram, même client, fait 390 t/s à b=1 (×8,4) et recouvre dès b=1 ; c'est propre à trtllm-serve dans cette configuration.
6. **La cellule b=1 = 46 t/s se publie AVEC cette cause** — reproductible sur 3 instruments (débit-b, cellule A/V, charge.py) et confirmée par la courbe plate jusqu'à b=8. Aucune remontée ≥ 300 t/s en dessous de b=12 : la cellule b=1 n'est pas rejouable à la hausse.
7. **Lecture** : trtllm gagne le débit à batch plein (b=12, +22 % vs acvram), acvram gagne la latence à faible charge (b=1 et jusqu'à b=8, où trtllm plafonne à ~50 t/s total contre 390+ pour acvram) — régimes opposés, aucun ne domine l'autre partout.

Note : la 1re tentative (12:30) a échoué (ConnectError) par interférence d'un vLLM
orphelin d'un pair, hors verrou ; refaite 13:20 sur carte propre (contrôle
compute-apps = seul le llama-server de l'utilisateur). Toutes fenêtres fenetre_valide=true.
