# Pièce 73 — nos arguments Marlin contre ceux de vLLM 0.29 : **aucun écart**, la piste des réglages est réfutée à sec — 23/09 (poste1)

## Comparaison argument par argument (à sec, 0 min de carte)

Sources lues : notre `acvram/kernels/marlin_port/__init__.py:415-437` et
`bindings.cpp:12-25` ; vLLM installé,
`site-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:135-162`
et `:325-339`, `quantization/utils/marlin_utils.py:408-420`.

| argument | nous | vLLM 0.29 | écart |
|---|---|---|---|
| `moe_block_size` | `choisir_block_size` : `for b in [8,16,32,48,64]: if M*top_k/E/b < 0.9: break` | **la même boucle, au mot près** (`marlin_moe.py:333-335`) | **aucun** |
| `use_atomic_add` | `False` | `False` (`:159`) | aucun |
| `use_fp32_reduce` | `True` | `True` (`:160`) | aucun |
| `is_zp_float` | `False` | `False` (`:161`) | aucun |
| `is_k_full` | `True` | `is_k_full` (vrai hors act-order) | aucun |
| `thread_k` / `thread_n` / `blocks_per_sm` | `−1, −1, −1` | **non passés** → les mêmes défauts du binding | aucun |
| `workspace` | `espace_travail(blocs_par_sm=4)` = sms × 4 | `marlin_make_workspace_new(device, 4)` = sms × 4 | aucun |
| `b_q_type` | `kFE2M1f` (nvfp4) | idem | aucun |
| **disposition** | gate, up, down : **3 GEMM** | gate‖up interfoliés (`size_n = w13_num_shards * N`), down : **2 GEMM** | **le seul** |

**Et la prémisse tient** : leur modèle est
`Qwen3-Coder-30B-A3B-Instruct-FP4-a16`, `config.json` →
`{"num_bits": 4, "type": "float", "group_size": 16, "input_activations": null}`
— c'est **notre format exact** (nvfp4, échelles tous les 16, activations bf16),
servi par **le noyau dont notre port est la copie**. La comparaison est donc
légitime, et « leurs réglages » n'existent pas comme levier : **la pièce 73
telle que formulée est réfutée à sec, sans une seconde de carte.**

## Ce que la trace de vLLM dit, et le désaccord qu'il faut trancher

`scratchpad/poste5-p59-23-09/familles-vllm.txt` : `marlin_moe_wna16::Marlin`
**96,0 lancements/pas à 26,6 µs** = 2,551 ms/pas = **53,1 µs/couche**, soit
exactement 2 lancements par couche. Nous : 3 lancements à 22,0 = 65,5 µs/couche.

Si nos trois lancements coûtent à peu près le même prix, alors notre `down`
(22,0) est **plus rapide** que le leur et tout l'écart est sur gate+up : 44 chez
nous contre 26,6 pour le même travail, soit **−17 µs/couche = −0,84 ms/pas** à
gagner par la fusion. Or le banc w13 de poste5 (pièce 71 bis) ne rend que
**−4,0 µs/couche**. Les deux ne peuvent pas être vrais. **Ce désaccord est la
vraie question de la pièce**, et il se tranche par une décomposition de NOS
lancements par forme — ce que ni la p63 ni la p71 bis ne donnent.

## Prédiction et issues, écrites AVANT la prise

Instrument : `frontiere-pas.py 12 60` sous nsys puis
`familles-noyaux.py --detail experts_marlin` sur l'alias servi — le détail est
par forme de grille, donc il sépare gate, up et down. ≤ 5 min, aucun code touché.

| issue | condition | ce qu'elle rend |
|---|---|---|
| **I1 — la fusion vaut le détour** | gate et up ≈ 20-23 µs chacun (somme ≥ 40) et down ≤ 26 | l'écart est bien le 3ᵉ lancement ; le banc de la 71 bis sous-estime d'un facteur ≈ 4 et il faut savoir pourquoi avant d'engager la demi-journée |
| **I2 — la fusion ne vaut rien** | gate + up ≤ 30 µs (le gros est dans `down`) | 71 bis confirmée, l'écart est ailleurs, et la piste w13 se ferme |
| **I3 — nos lancements ne sont pas 3** | le compte par pas n'est pas 144 | la prémisse « 3 contre 2 » est fausse et tout le raisonnement tombe |

* **prédiction nominale : I1**, gate ≈ up ≈ 21 µs, down ≈ 23, total ≈ 65.
* **seuil de chef** : une piste ne vaut d'être poursuivie qu'à **≥ 0,3 ms/pas**
  équivalent. I1 donnerait ≈ 0,8, I2 donnerait ≈ 0,19 — sous le seuil, donc
  réfutée.
* **ce qui me gênerait** : I2 confirme poste5 et invalide ma lecture de la
  trace vLLM, après que j'ai écrit qu'elle valait 17 µs. Je le nomme d'avance.
* **alarme** : si le total mesuré s'écarte de plus de 15 % des 65,5 µs/couche
  de la p63, je ne mesure pas le même régime (godet, `MOE_TENSOR_MIN_T`,
  horloge) et je le dis avant toute conclusion.
