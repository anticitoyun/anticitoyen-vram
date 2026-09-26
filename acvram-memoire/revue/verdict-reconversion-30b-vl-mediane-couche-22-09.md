# Reconversion Qwen3-VL-30B-A3B avec --repli-experts mediane_couche — RÉUSSIE — 22/09 (poste2)

* instrument : `outils/carte.sh .venv/bin/python -m acvram convert Qwen3-VL-30B-A3B-awq-dequant-bf16 -o Qwen3-VL-30B-A3B-repli-mediane-nvfp4-vision --repli-experts mediane_couche`, calibration par défaut (32 séquences × 512 jetons = 16 384)
* commit : main/poste2 à jour (b08a3d34)
* source : `/mnt/AI_GENERATOR/models_acvram/Qwen3-VL-30B-A3B-awq-dequant-bf16` ; sortie `Qwen3-VL-30B-A3B-repli-mediane-nvfp4-vision`
* mesuré : conversion réussie, 19 218 tenseurs, 57,9 Gio → 17,6 Gio (×3,30, nvfp4 15,8 + bf16 1,6), SNR moyen 21,1 dB, pires tenseurs 20,2 dB (experts down_proj), durée **543,7 s** (proche du 516,5 s de la conversion identite précédente). Manifeste (`acvram_manifest.json`) : `experts_repli = "mediane_couche"`, **`experts_sans_stats = 2487`** — **identique au compte de la conversion identite** (attendu : le repli change ce que ces 2487 experts REÇOIVENT comme échelle, pas s'ils sont comptés comme sans statistique — mêmes indices `experts_sans_stats_liste`).
* durée : 543,7 s de carte (≈9 min)

## Suite
P3 (3) à rejouer sur ce nouvel alias pour juger si le repli médiane réduit l'écart qualité (prédit 6-9 %, réfuté si ≥11 %). **En attente : régime GEL (session 87 %, remise 11 h 10)** — rien de nouveau ne part avant, ce verdict est poussé seul.
