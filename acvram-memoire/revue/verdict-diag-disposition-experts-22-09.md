# diag-disposition-experts (Océane) — CONFIRMÉ, cause du régime naturel isolée — 22/09 (Manon)

* instrument : `CUDA_VISIBLE_DEVICES=0 .venv/bin/python outils/diag-disposition-experts.py <alias officiel -qkvo-i8c> <alias qkv-22-09>`, à sec (sans carte.sh)
* commit : main à jour (060fd8dd)
* mesuré :
  * **alias officiel (-qkvo-i8c)** : `sans_stats=18432` (calibration totalement absente — alias ancien, pré-fixes collect.py), `gate≠up 0/128` sur toutes les couches vues (0, 1, 24, 47), écart médian/max 0,0000 — **toutes les échelles sont à l'identité par accident de non-calibration**, pas parce que gate et up coïncident réellement. Marlin accepté par coïncidence.
  * **alias qkv-22-09 (calibré)** : `sans_stats=5235` (calibration réelle), `gate≠up` jusqu'à 51/128 experts sur une couche, écart médian 0,06-0,19, max jusqu'à 0,98 — **échelles AWQ gate/up genuinement distinctes par expert**, `torch.equal` global casse dès qu'UN SEUL expert diffère → `_construire_marlin` refuse (`moe.py:446`, raison nommée « gate/up à entrées distinctes »).
* verdict : **CONFIRMÉ** — le régime `experts_layout=naturel` de l'alias qkv-22-09 n'est PAS un défaut de conversion ni de format, c'est une conséquence directe et attendue d'une calibration RÉELLE (contrairement à l'alias officiel qui ne l'a jamais eue) : `quant/convert.py:543-544` exclut explicitement les experts MoE de la pré-passe d'alpha commun gate/up. Le correctif proposé (`--alpha-commun-experts`) étend cette pré-passe aux experts.
* durée : ~15 s, aucune carte

## Suite
Reconversion avec `--alpha-commun-experts` (mode service, sortie `Qwen3-Coder-30B-A3B-nvfp4-qkv-alpha-22-09`), puis re-vérifier diag-disposition (doit dire marlin).
