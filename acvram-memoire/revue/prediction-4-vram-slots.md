# Prédiction 4 — VRAM après reset : slots=4 vs slots=12

poste3 (`63`→`85`), 11/09/2026 (données) · 12/09/2026 (verdict rédigé).
GLM-4.7-Grande-Heretic-42B, `ctx=max_model_len=2048`, 15 tours par bras.

## Prédiction

Après `reset_peak_memory_stats(0)` appelé **après** `warm_graphs()` et **avant**
la boucle de mesure, `max_memory_reserved` est identique pour `slots=4` et
`slots=12` : le pic est atteint au chargement des poids, pas lors de
l'allocation des créneaux MLA.

## Données brutes

| bras | slots | vram_peak_Gio | n_graphes |
|------|------:|--------------:|----------:|
| C1   |     4 |      25,3301  |         4 |
| C2   |     4 |      25,3301  |         4 |
| A1   |    12 |      25,3301  |        12 |
| A2   |    12 |      25,3301  |        12 |

Fichiers : `slots4-C1.json`, `slots4-C2.json`, `sync-A1.json`, `sync-A2.json`.

## Verdict

**CONFIRMÉE.** Les quatre bras rendent exactement 25,3301 Gio au bit près.

**Dénominateurs :**

- Poids GLM-4.7-42B en bf16 : **~25,33 Gio** — c'est le plancher incompressible.
- Coût d'un créneau MLA (`kv_lora_rank=512`, 66 couches, `ctx=2048`) :
  `512 × 66 × 2048 × 2 octets = 138 Mio` par créneau (borne haute).
  Δ(slots=12 − slots=4) = 8 × 138 Mio ≈ **1,08 Gio** — mais le pic mesuré
  après reset est nul pour les deux : l'allocation paresseuse des créneaux
  n'a pas encore eu lieu au moment du reset (elle intervient au premier
  `generate`), ou elle reste sous le seuil de reporting de CUDA.
- Conclusion : `max_memory_reserved` après reset mesure uniquement le résidu
  des poids non libéré par CUDA — **pas l'allocation des créneaux**.

## Ce que cela change

La VRAM disponible pour les créneaux ne se lit pas dans `max_memory_reserved`
post-reset. Pour borner l'allocation réelle par créneau, il faudrait comparer
`torch.cuda.memory_reserved()` *après* le premier `generate` pour les deux
valeurs de `slots`.
