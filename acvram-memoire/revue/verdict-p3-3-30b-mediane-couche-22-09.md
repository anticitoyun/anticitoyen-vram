# P3 (3) Qwen3-VL-30B avec repli-experts mediane_couche — RÉFUTÉ, LÉGÈREMENT PIRE que identite — 22/09 (Manon)

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-3.sh` (override local `ALIAS=`/`NOM=` ajouté, script sans git — scratchpad), référence transformers bf16 régénérée (le 1er essai avait réutilisé un .pt dégénéré 0 image/0 description faute de chemin `IMG` correct dans ce worktree — corrigé en pointant `MANON` vers `travail/manon-w-21-09` où vit le corpus `images-20`, 20/20 présentes)
* commit : main/manon à jour (af0ab06f)
* régime : `ctx_tenu=non-chauffe deepstack=3 eco=2700(2692) graphes=on(hybrides≤4) kv=int8 masque_images=causal mrope=[24,20,20](interleaved) vision=bf16(eager,transformers=5.17.0)`, alias `Qwen3-VL-30B-A3B-repli-mediane-nvfp4-vision`
* scellé (Océane/pièce 25) : haut_2se_pct prédit **6-9 %** (contre 12,6 % identite) ; réfuté si **≥ 11 %** ; alarme si < 3 %
* mesuré : `greedy8_identiques=2/3`, `premier_jeton_identique=3/3`, `ppl_desc_geo_pct=-10,012 %`, **`haut_2se_pct=13,212 %`**, détail par image [img00:2, img01:8, img02:8] (sur 8 comparaisons greedy chacune)
* verdict : **RÉFUTÉ** (13,212 % > seuil de réfutation 11 %), et surtout **légèrement PIRE que l'alias identite** (13,21 % contre 12,61 %, `verdict-p3-3-30b-collectpy-21-09`) — le repli médiane par couche ne réduit PAS l'écart qualité sur ce test, contrairement à la prédiction (6-9 %). Cohérent avec `verdict-reconversion-30b-vl-mediane-couche-22-09` : le nombre d'experts sans stats (2487) est identique entre les deux alias, seule leur échelle diffère (médiane des experts calibrés de la couche au lieu de l'identité/arrondi-seul) — cette statistique de repli ne semble pas être le facteur dominant de l'écart P3(3), ou son effet est noyé par la variance du test (n=3 images, sd_pct=19,9 %, large).
* durée : ~2,5 min de carte (référence 126,3 s + mesure 20,8 s), + 1er essai invalide (référence dégénérée, 0 s de carte, repéré avant jugement)

## Suite
Le levier « repli médiane_couche » ne tient pas sur ce test — à Océane : soit la statistique de repli n'explique pas l'écart de 12-13 % (autre cause dominante : plafond de la source AWQ double-quantifiée, déjà nommé dans P3(3) identite), soit n=3 images est trop peu pour trancher entre 12,6 et 13,2 (bruit sd=19,9 %). Carte gardée pour le KL acvram/bf16 (Coder, decode-pas), demandé ensuite par le groupe.
