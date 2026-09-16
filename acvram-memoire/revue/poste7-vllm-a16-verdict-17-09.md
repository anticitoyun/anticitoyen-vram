# poste7 — vLLM Marlin W4A16 à 1,016 : deux colonnes vLLM nommées par régime, et ≈ 1 % à trouver dans notre chemin W4A16 par bissection (17/09)

Entrée : `verdict-vllm-a16-17-09` (poste3, d53e5d1, main dbfc768). Scellé de `poste7-passage-direct-verdict-17-09` § 2 (1,028 ± 0,006) réfuté par le bas : 1,016 / 1,013. Réfuté reste réfuté ; la lecture « W4A4 = la perte de vLLM » tient (1,072 → 1,016 en changeant le seul régime d'activation), c'est la seconde moitié qui tombe : à poids identiques, Marlin rend 0,989 × notre W4A16, toutes tranches, deux corpus, dix fois le bruit.

## 1. Colonnes : les deux régimes, nommés, jamais fondus

| moteur / régime | privé | public | classé (≤ 1,02 géo privé) | vitesses |
|---|---|---|---|---|
| vLLM W4A4 (défaut, cutlass) | 1,072 | 1,075 | non | celles du duel, gardées |
| vLLM W4A16 (Marlin, `-a16`, régime forcé) | 1,016 | 1,013 | **oui** | à mesurer, poste3 15 min |
| acvram W4A16 `-k48-calibA` | 1,015 | 1,028 | oui | 21,0 ms b=12 (acquis) |
| acvram W4A16 `-vllm-direct` (mêmes poids que vLLM) | 1,028 | 1,021 | non | — (ligne de contrôle, pas de colonne) |

La colonne officielle vLLM est **celle qui classe, W4A16 Marlin, avec « régime forcé » dans son nom** ; la ligne W4A4 reste publiée à côté, parce que c'est le régime que vLLM sert par défaut et que ses vitesses du duel sont celles-là. On ne compare la vitesse de deux bras qu'à PPL classée des deux côtés (`poste7-comparatif` § 1) : la seule paire comparable est donc **acvram W4A16 vs vLLM W4A16**, à arithmétique égale — c'est plus juste que le duel, pas moins. ModelOpt propre (§ 6) : **sans objet**, clos. TRT-LLM : W4A4 seul, non classé, régime nommé, on n'y touche pas.

## 2. Le 1 % : bissection en deux gestes indépendants, prédictions écrites

Ce que j'ai lu : `ACVRAM_MLA_LATENT_FP8` est à 0 par défaut (`mla.py:44`), le latent n'est pas en cause. Le W4A16 du MoE en prefill passe par notre noyau `nvfp4_dequant` vers bf16 (`model.py:879`) puis une GEMM ; la référence est `dequantize_nvfp4` (`nvfp4.py:270-295`, fp32 puis bf16). Marlin fait la même chose en principe ; 1 % dit que quelque chose n'est pas « la même chose ».

* **Geste A — poste4, à sec, 30 min, sans carte** : sur un expert de `-vllm-direct` (le même tenseur des deux côtés), trois déquantifications : notre noyau (`nvfp4_dequant`), notre référence (`dequantize_nvfp4`), et **la référence de vLLM** (`/opt/ia/vLLM/.venv/.../quantization/utils/nvfp4_emulation_utils.py`, sur le tenseur CT `weight_packed / weight_scale / weight_global_scale` lu tel quel). Scellé : max |Δ| ≤ 1 ulp bf16 entre les trois. Prédiction : la référence vLLM et la nôtre s'écartent — candidat : l'échelle (`block_scale × global` en fp32 puis bf16 chez nous, ordre ou inverse différent chez eux) ; si les trois sont égales, le 1 % n'est pas dans la déquantification, et le geste B tranche seul.
* **Geste B — poste3, carte, 2 × 20 min, même passage** : PPL privé de `-vllm-direct` W4A16 (i) sous `ACVRAM_DISABLE_KERNELS=1` (chemin torch de référence partout), (ii) sous `ACVRAM_MLA_EAGER_TORCH=1` seul (`mla.py:250`). Lecture scellée : (i) ≤ 1,020 → nos noyaux CUDA perdent le 1 % ; (i) ≈ 1,028 → la perte est en amont des noyaux (lecteur CT, routage sigmoid + biais, normes) ; (ii) ≤ 1,020 avec (i) ≤ 1,020 → c'est le noyau MLA ; (ii) ≈ 1,028 avec (i) ≤ 1,020 → c'est la déquantification / GEMV MoE. Issue qui me gênerait : (i) ≈ 1,028 et A égal partout — le 1 % serait dans la mathématique du modèle, à chercher couche par couche (arbitre prefill contre HF bf16, `poste7-duel-verdict` § 9).
* Le correctif, quel qu'il soit, va dans un commit avec son test d'équivalence (REGLES § 7) et **se remesure sur `-k48-calibA`** : prédiction, 1,015 → ≤ 1,008. C'est là que le chantier paie : notre colonne descend sous celle de vLLM sur ses propres poids.

## Ordre

1. chef : ETAT — table GLM à deux lignes vLLM (§ 1), ModelOpt propre CLOS sans objet, chantier « 1 % W4A16 » ouvert (poste4 A à sec, poste3 B carte) ; une ligne à l'utilisateur : vLLM classe en régime forcé W4A16, pas en défaut.
2. poste3 (carte, dans l'ordre) : vitesses vLLM W4A16 Marlin b=1 / b=12, même en-tête que le duel (15 min) ; puis geste B (2 × 20 min), verdict `verdict-bissection-w4a16-17-09`.
3. poste4 (à sec, 30 min, avant tout autre code) : geste A, verdict `verdict-dequant-trois-references-17-09` ; puis, selon A et B, le correctif avec son test ; les correctifs de `poste7-calibration-verdict` § 3 passent après.
4. poste2 : rien de nouveau ici ; Coder EXL3 à poste3 quand la carte revient.
