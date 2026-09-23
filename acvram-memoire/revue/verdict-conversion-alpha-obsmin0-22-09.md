# Reconversion Coder --alpha-commun-experts --obs-min 0 — RÉUSSIE, marlin confirmé — 22/09 (Manon)

* instrument : `outils/carte.sh .venv/bin/python -m acvram convert ... --alpha-commun-experts --obs-min 0`, sortie `Qwen3-Coder-30B-A3B-nvfp4-qkv-alpha-22-09`
* mesuré : 18 867 tenseurs, 56,9→16,6 Gio (×3,44), SNR sortie moyen **20,9 dB** (contre 21,1 dB sur l'alias qkv-22-09 sans alpha commun) — **pires tenseurs 13,6-13,8 dB** (contre 20,4 dB avant), dégradation nommée, attendue avec `--obs-min 0` (experts sous-calibrés acceptés explicitement). Durée 543,5 s. `experts_sans_stats=5235` (identique à qkv-22-09, cohérent).
* `diag-disposition-experts.py` rejoué : **`gate≠up 0/128` sur toutes les couches vues** (0, 1, 24, 47) — `--alpha-commun-experts` a bien forcé une échelle commune gate/up par expert. `marlin PRÉDIT : tables AWQ présentes mais gate == up partout` — **confirmé**.
* verdict : **RÉUSSIE** — la cause isolée ce matin (échelles AWQ gate/up distinctes) est corrigée, le repack Marlin devrait maintenant se déclencher au chargement. Coût qualité nommé : SNR pires tenseurs -6,8 dB par rapport à l'alias sans alpha commun (13,6-13,8 contre 20,4 dB), dû aux 5235 experts sous-calibrés acceptés via `--obs-min 0`.
* durée : 543,5 s de carte

## Suite
Vérification runtime (familles-noyaux) pour confirmer `experts_layout=marlin` au chargement réel, puis ABBA débit.
