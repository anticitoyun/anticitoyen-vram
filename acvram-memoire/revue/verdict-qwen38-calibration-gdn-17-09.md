# Verdict — Qwen3.8-27B reconverti CALIBRÉ (bras-A), chantier calibration-hybrides-gdn-17-09 clos côté conversion

Manon, 17/09. Suite de `verdict-qwen38-reconversion-17-09.md` (--no-awq,
PPL 1,0306) et `d23f78b` (diagnostic + implémentation GDN/attn_output_gate,
à sec). Sage : une itération, carte 1h20-1h30 au bord d'un bloc palier 1.

## Contrôle avant la carte (dry-run, `--dry-run`, mêmes options)

```
calibration sur 32 sequences (16384 jetons) ...
statistiques relevees pour 496 tenseurs
SNR sortie moyen 22.8 dB   (contre 20,5 dB sans calibration, --no-awq)
0 avertissement de couche indisponible (64/64 couches, contre 0/64 avant ce chantier)
```

Confirmé : le chemin GDN (48/64 couches) et `attn_output_gate` (16/64)
fonctionnent tous les deux — plus aucun repli silencieux. Les pires SNR
(20,1-20,5 dB) sont tous des tenseurs `mtp.*`/`lm_head` : la tête MTP est
un module séparé, hors de la passe de calibration par construction
(comme le module principal), pas une régression.

## Conversion réelle

`outils/carte.sh`, `systemd-run --user`. Source identique à la tentative
`--no-awq` (`/mnt/4TO_SATACMR_2022/Modeles/models/Qwen3.8-27B-bf16`),
sortie `/mnt/2TO_2023_980PRO/Modeles/models_acvram/Qwen3.8-27B-nvfp4-calibA`.

```
acvram convert <source> --name Qwen3.8-27B-nvfp4-calibA -o <sortie> \
  --calib-file scratchpad/corpus-calib-k48/bras-A-anglais.txt
statistiques relevees pour 496 tenseurs
tenseurs 866, sortie 16,1 Gio (x3,16), SNR sortie moyen 22,8 dB
durée 980,9 s
```

Manifeste vérifié : `options.awq: true`, `calib_seqs: 32`, `calib_tokens:
16384`, `calib_source.fichier` pointe le bras-A, sha256 du corpus
`cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138`
(identique à la mesure précédente, fichier inchangé). Carte : 1226 s
tenus (≈ 20 min sur les 1h20-1h30 alloués), libérée (verrou vide,
`nvidia-smi` sans process de calcul).

## sha256 (converti calibré)

```
condensé de la liste : 44c69c2e57daeb2e4fe7be353ddd20704dcde00d2af5eec71ee4a96b6f624f81
acvram-00000.safetensors : b3998ef2d90287e4aaffe9ad7b87330f344a2ac7a91d3554e6bd15c59ee6fa44
acvram-00001.safetensors : 67de3db633bf335cba1fec525a8de9ec5c96956a7105dfb0ea5dff110bb19181
acvram-00002.safetensors : 8438822d2352fa48fc58ee60c390ddc0d102521701a019f415ee622bebd667f7
acvram-00003.safetensors : 0d5049cb9dece6d9016bcfedbc997cf52efdafe6b9a758e48c648866ab059bbb
acvram-00004.safetensors : b52ff2cd09a9549c581267b04f2137607e6edf3ea79ba2e3b3ff37fb0144761c
acvram_manifest.json     : c1c6e003f3c0c512546807a10a7645c75e42176dcf961b18720ccac21d3cd741
```

Note : `acvram-00004.safetensors` (38,5 Mio, petits tenseurs/tête) a le
MÊME sha256 que la version `--no-awq` précédente — attendu, ces tenseurs
ne passent pas par le chemin AWQ calibré, cohérent avec la structure du
convertisseur, pas une anomalie.

## Prochaine étape — Laure

Carte rendue à Laure (Llama-70B derrière), pour la PPL privée bras-A,
même montage que `verdict-qwen38-v2-mesure-17-09.md` (f24d11e, tranches
`scratchpad/palier2-qwen38-17-09/`).

Scellé Sage, une itération, sur cette mesure :
- PPL ≤ 1,020 → classée, la calibration était le levier manquant.
- 1,020-1,025 → gain réel mais non classé, à noter.
- > 1,025 → la calibration n'est pas le levier sur cette famille (GDN +
  attn_output_gate), chantier clos.
