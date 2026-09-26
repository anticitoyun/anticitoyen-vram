# Prédiction scellée — routeur GLM-4.7-Flash en fp32

poste1, 14/09/2026, avant mesure.

## Cause identifiée par lecture de code

HF (`transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:401`) :
`router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))`
— logits, sigmoid et biais calculés en fp32 de bout en bout.

acvram (`acvram/engine/model.py:1038-1050`, `_router_logits`) : le poids est
casté dans le dtype de l'entrée (`x.dtype`, bf16 dans ce banc) avant le
`F.linear` — la sortie est arrondie en bf16 avant sigmoid+biais+top-k, même
si l'accumulation interne de cuBLAS est fp32 (le commentaire en place le
disait déjà, mais l'arrondi de sortie compte, pas seulement l'accumulation).

## Contrôle topi (fait, ce document le clôt)

Diff exact `topi-acvram.json` vs `topi-hf.json` : un seul expert bascule à
chaque position 0, 1, 3, 5 (42↔4, 48↔37, 0↔7, 6↔23), 12/16 positions
identiques bit à bit sur les indices. Position 0 bascule aussi mais son
cosinus logit était déjà bon (0.999548) — écart d'un rang minime entre les
deux derniers experts retenus, sans effet mesurable sur la sortie.

## Prédiction

Calculer `_router_logits` et le sigmoid+biais de `_route` en fp32 (comme
HF) fait tomber le nombre de positions divergentes en topi de 4 (0,1,3,5) à
0 sur les 16 positions, et ramène pire cosinus ≥ 0,999 partout dans
l'équivalence 2 couches.

**Seuil de réfutation** : si après le correctif il reste ≥ 1 position avec
topi différent, OU pire cosinus < 0,999, la prédiction est réfutée — la
cause n'est pas purement l'arrondi bf16 du routeur, il reste une deuxième
divergence à chercher (poids gathered, expert partagé, ou attention MLA).
