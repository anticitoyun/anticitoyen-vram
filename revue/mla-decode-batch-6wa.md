# Décodage MLA batché : 12 créneaux en un lancement — bead `anticitoyen-vram-6wa`

Date : 14/09/2026 — Laurine. Spec : Laure,
`acvram-memoire/revue/spec-batching-mla-6wa-13-09.md`. Outil :
`outils/ab-mla-batch.py`. Tests : `tests/test_mla_decode_batch.py`.

## Ce qui est livré

- `mla_scores_batch_kernel` grid (B, H, ⌈L/128⌉) et `mla_reduce_batch_kernel`
  grid (B, H) — même arithmétique que `mla_scores_kernel`/`mla_reduce_kernel`,
  caches par **table d'adresses `[B]` int64** (spec §3-A ; les caches par
  créneau ne sont jamais réalloués, adresses stables).
- `mla_decode_batch(q_eff [B,H,W], cache_ptrs, lens [B], scores, L, rank,
  scale)` → `[B, H, rank]`.
- `MLAttention.decode_static_batch` : préparation (projections, RoPE, norme,
  écriture du latent), einsum `v_b` et `o_proj` **restent par créneau** ; seul
  `ext.mla_decode` est batché — périmètre retenu par Jérôme (« le noyau seul,
  ce que le bead mesure »). `decode_static` et `decode_static_batch`
  partagent le même texte (`_prep_decode`, `_sortie_decode`).
- `MoEBlock/_la_decode` (model.py) : `_mla_lot(b)` construit la table
  d'adresses et le tampon `scores [b, H, max_len]` **une fois par jeu
  d'adresses** ; la création tombe dans l'échauffement eager de
  `GraphRunner._capture` (deux `step()` avant capture), jamais dans la
  capture. `ACVRAM_MLA_BATCH=0` rend la boucle.

## Équivalence (règle 9) — 17 tests

- Noyau : caches non contigus, `len` hétérogènes (0 à L−1), créneaux de
  rembourrage à `len` 0, L > 48 Ko de shared ; `torch.equal` contre la
  boucle de `mla_decode`. Contrôle négatif : permuter deux adresses de la
  table change la sortie.
- Module `MLAttention` complet : boucle `decode_static` contre
  `decode_static_batch` sur deux jeux d'états identiques (caches remplis
  jusqu'à `len`, non triviaux), b = 12 et b = 4 avec 3 rembourrages :
  sorties et états **bit-identiques**.
- Moteur : les jetons générés par les 12 séquences sont identiques entre les
  bras A et B (empreinte des sorties `9d4ffb7e0b66` dans les deux).

## Preuve de chemin sous graphes

Les compteurs Python ne voient que le code exécuté : une capture et ses
échauffements, jamais un rejeu. Ils prouvent donc ce qui a été capturé :
bras A `decode_static_batch = 0` ; bras B `= 396` pendant la capture du
godet b=12 (66 couches × 6 exécutions), `eager = 0` sur tous les pas.

## A/B — GLM-4.7-Grande-Heretic-42B, slots 12, `ACVRAM_CHRONO_SYNC=1`

Prédiction scellée (Laure) : −5,3 à −15,9 % sur `pas_total` ; réfutation :
< 3 %.

Régime mesuré : 12 séquences concurrentes, invite 32 jetons, 150 jetons
décodés, EOS neutralisé, 148 pas par fenêtre, tous en voie graphe ;
b_réel = 12 (12 × 182 jetons = 2 184 ≪ capacité KV 16 384 ; 148 pas pour
150 jetons par séquence). Godets capturés : (12, 1, 8, 128) et
(12, 1, 16, 256).

| bras | fenêtres | pas_total médian | σ | replay médian |
|------|---------:|-----------------:|--:|--------------:|
| A (boucle) | 15 | **76,11 ms** | 0,13 | 75,21 |
| B (`mla_decode_batch`) | 15 | **54,38 ms** | 0,20 | 53,50 |
| A (boucle, relance) | 2 | 76,37 ms | 0,15 | 75,43 |

**−28,5 % sur le pas** (76,11 → 54,38 ms), soit ×1,40 en débit de décodage
concurrent — au-delà de la borne haute scellée (−15,9 %). Ordre A, B, A.

### Ce que le témoin dit du montage

Le bras A rend 76,1 ms, pas les 92,80 ms du 13/09 : le montage diffère de
celui de Laure sur la longueur des séquences (ici 32 + 150 jetons, godet
MLA 128-256 ; le sien courait vers 2 000 jetons, godet 2 048). Vérifié :
le même montage court avec `ACVRAM_MLA_BUCKET=2048` rend 101,6 ms. Le
gain mesuré est donc celui du régime **court**, où les 12 lancements
pèsent le plus ; à godet 2 048 la part de `mla_*` monte et le gain est à
remesurer.

### Une mesure retirée

Un régime « long » (invite 1 536) a rendu A 33,0 / B 39,5 ms — chiffres
**invalides** : la capacité KV (16 384 jetons) ne tient pas 12 × 1 686, le
moteur n'admettait que 1 à 2 séquences (`b_réel` imprimé après coup :
1..2), et la boucle d'admission du banc avait consommé les premiers
décodages. Un pas dont le lot n'est pas prouvé ne compare rien ; l'outil
imprime maintenant `b_min..b_max` par fenêtre. Ce que ce faux régime
montre quand même, à b=1-2 : la variante batchée est plus lente (+20 %) —
attendu, le lancement unique ne rachète rien et `grid.x=b` n'apporte
d'occupation qu'avec des créneaux.

## Reste

- ABBA complet en régime court (le dernier bras B a été tué par le
  garde-fou mémoire du harnais en passant en arrière-plan ; la carte est à
  Océane puis Laure) : à relancer `ACVRAM_TYPE=mesure AB_TOURS=15
  outils/carte.sh … outils/ab-mla-batch.py B`, au premier plan.
- Régime long à b=12 réel : demande un contexte qui tient (12 × ≤ 1 300
  jetons) — `AB_INVITE=1024 AB_JETONS=150`.
- Second levier de la spec (split-K sur L, grid.z) : non commencé ; utile
  si le régime long montre que `mla_reduce` domine.
- Bead séparé : batcher tout `decode_static` (projections/RoPE/norme ×12
  redondants), comme `forward_batch` côté eager.
