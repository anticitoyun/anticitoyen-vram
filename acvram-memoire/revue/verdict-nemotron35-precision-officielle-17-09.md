# Verdict — Nemotron-3.5-30B-A3B : comparaison dtype tenseur par tenseur contre le checkpoint officiel, correctif appliqué et reconverti

poste2, 17/09. Suite de `poste7-hybrides-etape1-close-gemm-dense-17-09.md` :
Nemotron 1,0632 vs vLLM officiel 0,987, prédiction poste7 « in_proj/out_proj
Mamba2 ou dt/A_log/conv en bf16 chez eux ».

## Comparaison dtype tenseur par tenseur

Checkpoint officiel : `/mnt/4TO_SATACMR_2022/Modeles/models_vllm/
Nemotron-3.5-Lightning-30B-A3B-NVFP4/hf_quant_config.json`
(`quant_method: modelopt`, `quant_algo: MIXED_PRECISION`, `ignore: []`).
5981 entrées dans `quantized_layers`, regroupées par motif générique :

| motif | nb | algo |
|---|---|---|
| `mixer.experts.N.{down,up}_proj` | 2944 | W4A16_NVFP4 |
| `mixer.shared_experts.{down,up}_proj` | 46 | W4A16_NVFP4 |
| `mixer.in_proj` / `mixer.out_proj` (23 couches Mamba2) | 46 | **FP8** |
| `lm_head` | 1 | W4A16_NVFP4 |

Rien pour `self_attn` (les 6 couches full_attention) : absent de
`quantized_layers` ET de `ignore` (vide) → **bf16, jamais quantifié**.
Rien non plus pour `dt`/`A_log`/`conv1d` — déjà protégés chez nous par
`SENSITIVE_SUFFIXES` (`acvram/quant/convert.py:267`), confirmé identique
côté officiel par lecture directe des dtypes du checkpoint bf16 (déjà
bf16 dans notre propre conversion, rien à changer là).

Prédiction poste7 confirmée pour `mixer.in_proj`/`out_proj` (bien exclus de
NVFP4 côté officiel), mais le format réel est **FP8**, pas bf16 — et un
second écart, absent de la prédiction : **`self_attn` sur les 6 couches
full_attention n'est quantifié dans AUCUN format** côté officiel. Notre
conversion précédente (`verdict-nemotron35-srcbf16-17-09.md`) mettait les
deux familles en NVFP4 comme le reste du modèle.

## Correctif

acvram ne porte pas de format FP8 autonome pour les poids (FP8 E4M3
n'existe ici que comme échelle de bloc À L'INTÉRIEUR de NVFP4/Q3N,
`acvram/quant/formats.py:94`) : bf16 est la meilleure approximation
disponible pour `mixer.in_proj`/`out_proj` — plus fin que le FP8 officiel,
jamais plus grossier — et exact pour `self_attn`.

`TensorRouter.format_for` (`acvram/quant/convert.py`, commit `4c1fd65`) :
nouvelle branche gardée par `model_type == "nemotron_h"`, place ces deux
familles en bf16 avant le routage par format de couche. Gardée par
`model_type` pour ne pas toucher les modèles denses ordinaires — vérifié
par test (`test_hors_nemotron_h_lattention_reste_quantifiee`, l'attention
d'un modèle `llama` simple reste en nvfp4).

Témoin (REGLES §5) : désactiver la branche fait rougir
`tests/test_nemotron_h_precision_officielle.py` (`nvfp4` au lieu de `bf16`
sur `mixer.in_proj`/`out_proj` et `self_attn.*`) ; réactivée, 4/4 verts.
Suite complète : 801 passed, mêmes 7 échecs déjà rouges avant ce
changement (fla, dépôt, menus — voir `verdict-nemotron35-srcbf16-17-09.md`).

## Reconversion réelle

Dry-run puis réel, mêmes 412-437 s de carte chacun, intercalés dans la
chaîne FLA de poste3 (verrou pris/rendu proprement).

```
Nemotron-3.5-Lightning-30B-A3B-nvfp4-precision-officielle
tenseurs 6243 ; entrée 58,8 Gio ; sortie 18,5 Gio (×3,19, contre ×3,44 avant)
  nvfp4 15,8 Gio ; bf16 2,6 Gio (contre 688,9 Mio avant)
SNR sortie moyen 20,5 dB
pires tenseurs : uniquement mlp.experts.*.down_proj (20,4 dB) --
  les self_attn.o_proj, pires tenseurs de la première conversion,
  ont disparu du classement (ne sont plus quantifiés)
```

## sha256 (`.../Nemotron-3.5-Lightning-30B-A3B-nvfp4-precision-officielle/`)

```
acvram-00000.safetensors : 498894ebbe1ec3cba9f8a80468f95a01b6201f7e6ce1eceff8e5c8607f5164a6
acvram-00001.safetensors : c286c9ffae6aa18c856ebd327f82c7f187a25aaf4be46718b6c221d217ea7d5b
acvram-00002.safetensors : 4657154033ae2335046d00cb87cfd34f839026a08576d094581676c418abfbd7
acvram-00003.safetensors : c6e24f322d51243bd41aa77fb39232246b48b4eec642f3dacdff7b02ae1ebd0d
acvram-00004.safetensors : 0c9875a07ff72c0cc2bc64223add4eb70b56bffd2ce7b34c2d64fb9104ab033d
acvram_manifest.json     : 98b0a0ffa682e39733ad82003b26ed34ba3d575afa35592aa65c304694ddefc7
config.json               : a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c
```

Ancienne conversion (`Nemotron-3.5-Lightning-30B-A3B-nvfp4`, sans
l'exclusion) laissée en place sur disque — comparaison PPL avant/après
possible sans reconvertir.

## Suite

Prêt pour poste3 : PPL 3 tranches sur
`Nemotron-3.5-Lightning-30B-A3B-nvfp4-precision-officielle`, comparer
contre l'ancienne conversion et contre 0,987 (officiel). Prédiction poste7
sur ce geste : 1,018-1,021 — hors périmètre ici (c'était la prédiction pour
Coder gateup/GLM, pas Nemotron ; aucune prédiction chiffrée écrite pour ce
modèle avant cette mesure, à demander si poste7 en veut une avant que poste3
mesure).
