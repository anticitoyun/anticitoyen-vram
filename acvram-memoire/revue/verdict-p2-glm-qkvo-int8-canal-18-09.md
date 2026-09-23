# Verdict — P2 GLM : GLM-4.7-Flash-srcbf16-nvfp4-k48-qkvo-i8c converti

Manon, 18/09. Sage (relais Jérôme) ouvre le même chantier P2 que Coder pour
GLM-4.7-Flash (MLA) : projections d'attention int8 → symétrique par canal,
reste identique au classé (`GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA` :
`format_impose=nvfp4`, `awq=True`, `group_size=128`, corpus bras-A
16 seqs/128 jetons, `mixed_precision=auto`, `snr_floor=25.0`,
`max_promotions=0.15`, `quant_device=cpu`). À sec (CPU, comme le classé),
1h estimée par Sage — réel : 6181 s (103 min).

## Défaut trouvé avant la conversion : q/k/v/o ne nomme rien chez GLM

GLM est MLA, pas GQA : aucun tenseur `self_attn.{q,k,v,o}_proj.weight`.
Le filtre à 4 suffixes fixes écrit pour Coder aurait laissé passer les deux
projections réellement candidates du classé
(`kv_a_proj_with_mqa`, `q_b_proj`, toutes deux promues nvfp4→int8).
Généralisé en `_est_projection_attn(name)` (`convert.py`) : tout
`.self_attn.*.weight` qui n'est pas une norme (`q_a_layernorm`,
`kv_a_layernorm`) est une projection candidate — `k_b_proj`/`v_b_proj` y
passent aussi mais restent `bf16` (absorptions MLA 3D, `SENSITIVE_SUFFIXES`),
jamais gênés par le filtre `fmt=="int8"` en aval. 7 tests dédiés, dont un
sur cette généralisation précisément (GQA + MLA, témoins négatifs sur les
normes).

## Corpus vérifié identique au classé

sha256 de `scratchpad/corpus-calib-k48/bras-A-anglais.txt` (déjà présent
dans le dépôt, reconstitué plus tôt cette session) comparé à celui inscrit
dans le manifeste classé : **identique**
(`cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138`).

## Converti

```
model.layers.0.self_attn.kv_a_proj_with_mqa.weight  int8 canal  group_size=2048  snr 39,37 dB  (nvfp4 -> int8)
model.layers.0.self_attn.q_b_proj.weight            int8 canal  group_size=768   snr 43,20 dB  (nvfp4 -> int8)
model.layers.0.self_attn.o_proj.weight              nvfp4       (non promu, snr 20,43 dB — inchange)
model.layers.0.self_attn.q_a_proj.weight            nvfp4       (non promu, snr 20,81 dB — inchange)
```

Contrôle de fuite : 0 tenseur hors `self_attn` ne porte `symmetrique` —
`mlp.down_proj`/`mlp.shared_expert.*` (aussi promus nvfp4→int8 par le même
mécanisme SNR) restent au groupe de 128 affine standard, comme le classé.

- tenseurs : 9751, entrée 58,2 Gio, sortie 18,4 Gio (×3,16)
  (nvfp4 14,9 / int8 1,8 / bf16 1,6 / fp32 0,01)
- SNR sortie moyen 21,5 dB, 335 tenseurs promus nvfp4→int8 (attention MLA +
  `down_proj`/`shared_expert` sous le plancher, mécanisme inchangé par
  rapport au classé)
- source : `/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16`, sha256 dans
  le manifeste (`source.sha256`)
- convertisseur : commit `1039c14` (généralisation MLA incluse)
- sortie : `/mnt/2TO_2023_980PRO/Modeles/models_acvram/GLM-4.7-Flash-srcbf16-nvfp4-k48-qkvo-i8c`,
  sha256 combiné des fragments dans le manifeste
- `attn_int8: "canal"` au sommet du manifeste

## Tests

7 tests dédiés (`tests/test_int8_symetrique_canal.py`, formats.py isolé +
convert_checkpoint bout en bout GQA et MLA + promotion + généralisation
`_est_projection_attn`) : tout vert avant la conversion réelle.

## Clôture (Laure, 18/09 22h55)

**P2 GLM FERMÉ.** i8c/classé = 1,0143 (tranches 1,0146/1,0052/1,0231), les
3 > seuil 1,005. Absolu attendu ≈ 1,029. Converti gardé comme pièce, pas
classé.

Pas un défaut de cette conversion : cause identifiée par Laure, spécifique
aux tenseurs de COMPRESSION MLA (`kv_a_proj_with_mqa` 576×2048,
`q_b_proj`) — l'int8 symétrique par canal y coûte ce que le groupe de 128
du classé préservait (compression vers un latent de rang 512, pas une
projection dense large). Sur Coder (denses larges, mêmes tenseurs
conceptuels mais sans compression latente), le même choix GAGNE +0,5 %
(`verdict-p2-hors-moteur-18-09` : i8c/classé Coder = 0,9947, P2 Coder
OUVERT). Le SNR sain mesuré ici (37-43 dB) ne pouvait pas voir ce coût :
il compare une reconstruction à son poids source, pas au groupe-128
affine qu'elle remplace sur un tenseur de compression.

Rien à refaire côté conversion : le mécanisme (int8 canal sur PROJECTIONS
MLA génériques, `_est_projection_attn`) reste correct et réutilisable,
mais le VERDICT NUMÉRIQUE dépend du rôle du tenseur (dense large vs
compression latente), pas seulement de son SNR de reconstruction —
distinction à garder pour un futur chantier similaire sur une architecture
MLA.
