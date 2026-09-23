# Pièce 105 — mesure : tête MTP chargée, sorties identiques, a = 0,32 (sous la bande prédite), MTP −20 % à b=1 ; R de vérification MoE : ABANDON du proposeur par lot — 23/09 (poste1)

* **instrument** : `scratchpad/poste1-p105-23-09/accept-mtp.py` (Qwen3.8-27B-nvfp4, 5 invites × 128 jetons gloutons, graphes, A sans spéculation / B `MTPProposer` k = 3, une séquence à la fois) et `r-verif.py` (Coder qkvo-i8c, noyaux CUPTI d'une passe de vérification `_build_spec_batch` à b·(γ+1) jetons contre une passe à b, b ∈ {8, 10, 12}, γ ∈ {1, 2, 3}).
* **commit** : 7e479aeb (branche poste1-mtp, main fusionné). Prise de 21:31:40 à 21:36:16 ; seul llama-server 4627 au début et à la fin ; horloge libre (seul le t/s en dépend, en information).
* **scellé** : `scelle-accept.md` et son addendum R, commités avant.
* **mesuré** :

| contrôle | seuil scellé | prédit | mesuré | |
|---|---|---|---|---|
| tête MTP chargée | model.mtp ≠ None | oui | **oui** (`mtp_raison` vide) | tenu |
| sorties B = A, 5 invites (sha256) | 5/5, obligatoire | 5/5 | **5/5** | tenu |
| acceptation a (MTP3) | faux si < 0,30 | 0,40-0,55 | **0,321** | au-dessus du faux, SOUS la bande |
| t/s B / A (information) | — | +35 à +75 % | **52,8 / 65,8 = −19,8 %** | prédiction réfutée |

  R de vérification (MoE Coder), seuil R < 0,8·τ avec τ = 1 + γ·a = 1,32 / 1,64 / 1,96 :

| γ | 0,8·τ | R b=8 | R b=10 | R b=12 |
|---|---|---|---|---|
| 1 | 1,06 | 1,35 | 2,24 | 2,42 |
| 2 | 1,31 | 2,58 | 2,81 | 3,94 |
| 3 | 1,57 | 3,18 | 4,30 | 4,68 |

* **verdict** :
  1. **Chargeur juste, justesse tenue.** Le premier geste (c39b7073) charge la tête et les sorties spéculatives sont identiques au bit près à la sortie sans spéculation.
  2. **MTP à b=1 : perte de 20 %.** Il faut ≈ 37 ms par pas spéculatif (1,96 jeton), contre 15,2 ms par jeton sans spéculation. Le brouillon eager coûte ≈ 6-7 ms par jeton proposé, contre les 1-1,5 ms estimés au dossier : c'est mon erreur d'estimation. Il est pris en eager (couche entière + lm_head [248 320 × 5 120] + lancements).
  3. **Acceptation basse : hypothèse nommée, non prouvée.** vLLM passe à la tête l'état caché APRÈS la norme finale (`qwen3_next.py:726`, `hidden_states, _ = self.norm(...)`, puis rendu au proposeur). Chez nous, `_garder_hidden(x)` précède `x = self.norm(x)` (`engine/model.py:356-361`), alors que sa propre docstring dit « état caché normalisé ».
     * Contrôle en 10 min de carte : a avec l'état normalisé contre l'état brut, même instrument. S'il remonte vers 0,45-0,50, c'est la cause.
  4. **Proposeur par lot (112) : ABANDON au seuil scellé.** R > 0,8·τ dans les 9 cellules, de 1,3× à 3× au-dessus du seuil. Sur notre MoE, la vérification à b·(γ+1) jetons coûte 1,35 à 4,7 passes, ce qu'annonçait la littérature (MoESD, Cascade), et même pour γ = 1 à b ≥ 8. Le saut de b=8 à b=10 à γ = 1 (1,35 → 2,24) tient sans doute au passage du nombre de jetons au-delà d'un seuil du chemin MoE ; il n'est pas examiné.
* **Suite (au chef)** : ne rien servir en MTP à ce stade. Si la 105 continue :
  * (a) le contrôle de l'état normalisé, 10 min ;
  * (b) le brouillon capturé en graphe, 3-5 h, seulement si (a) remonte a ≥ 0,45.
* **durée** : prise de 4 min 36 s.
