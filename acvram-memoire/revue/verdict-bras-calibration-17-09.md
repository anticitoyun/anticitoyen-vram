# Verdict — bras de calibration A (anglais) et B (mixte) sur `-k48` : l'inversion privé/public disparaît avec A, B fait pire sur le privé

instrument : `scratchpad/ppl-bras-calib-17-09.sh` → `ppl-acvram-17-09.py` (fenêtres `encode_brut`, préfixe `[gMASK]<sop>`, cibles 1024..2047, géo ET médiane, 4 ids) ; référence bf16 HF de `ppl-refonte-17-09` — journaux `scratchpad/ppl-bras-calib-17-09/`
commit : arbre laure 1e9cac9 (= main) ; convertis Manon a1da690 (A) et f2b4e4e (B), `-k48` sans rotation, AWQ ; protocole : `sage-hadamard-verdict-17-09.md` § 3-4
régime : W4A16 prefill, corpus privé 5909d27 et public wiki-gptq, 3 tranches × 12 fenêtres ; calibration A `bras-A-anglais.txt` sha256 `cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138` ; B `bras-B-mixte.txt` sha256 `6a96eda740147ac2687c91ba9fbe63321b4a13d13a324e41b31dc9e24aa1adb7` ; **jetons de calibration réellement lus : 16 × 128 = 2 048** (`calib_seqs 16, calib_tokens 128`, options par défaut du convertisseur ; les fichiers font 738 / 857 Ko, `load_calib_ids` en prend 16 tranches de 128 jetons) — pas les ≥ 16 k demandés par Sage
scellé (Sage § 3, sur l'écart privé − public, référence `-k48` 0,032 en médiane) : A ≤ 0,020 → la taille suffit · A ≈ 0,032 et B ≤ 0,020 → la langue · A et B ≥ 0,028 → la calibration n'explique pas l'inversion · juge = moyenne géométrique (§ 4), médiane en colonne
mesuré (× bf16 HF, moyenne géométrique des 3 tranches) :
    converti          géo privé   géo public   écart géo   | méd privé   méd public   écart méd   | classé privé (géo ≤ 1,02)
    -k48 (défaut)     1,0156      1,0040       +0,012      | 1,0289      0,9972       +0,032      | oui
    -k48-calibA       **1,0150**  1,0284       **−0,013**  | 1,0236      1,0268       −0,003      | oui
    -k48-calibB       **1,0244**  1,0059       **+0,019**  | 1,0316      1,0093       +0,022      | non
    -sansawq          1,0166      1,0209       −0,004      | 1,0207      1,0055       +0,015      | oui
    -hadamard512      1,0289      0,9913       +0,038      | 1,0421      0,9856       +0,056      | non
    0 fenêtre explosée partout ; ids [154822, 154824, …] dans les 12 JSON.
verdict : **A réfute l'inversion : écart −0,013 (géo) / −0,003 (médiane) — sous 0,020, règle de Sage : « la taille suffit » — mais ce qui a changé n'est pas la taille (2 048 jetons lus, contre 230 par défaut, pas 16 k) : c'est le texte, 16 extraits de prose anglaise générale au lieu des 6 phrases de `collect.py`. Le public perd ce que le défaut lui donnait (1,004 → 1,028) : les 6 phrases favorisaient wikitext. B (mixte FR/EN/code) fait PIRE sur le privé (1,024, non classé) que A (1,015) et que le défaut (1,016), sur toutes les fenêtres (B/A > 1 sur 27 des 36), pas sur une seule : la langue française dans la calibration n'aide pas ce corpus, elle nuit. Sur le juge géométrique, `-k48` par défaut ET A sont classés sur le privé (1,016 / 1,015) ; A l'est sans l'artefact wikitext.**

## Bornes
- La « taille » n'a pas été testée : 2 048 jetons lus dans les deux bras. Un bras à 16 k exige de passer `--calib-seqs 32 --calib-len 512` (à Manon, si Sage le veut).
- B : 1/3 code du dépôt `acvram/` — le privé (`revue/*.md`) parle de ce code sans en être ; pas un recoupement textuel, mais un registre voisin. B tranche 2 fenêtre 4 : 1,161 × bf16, la plus mauvaise fenêtre de toute la campagne (A : 1,058 sur la même).
- Bruit : ± 0,004 par tranche ; les écarts A/B (−0,013 / +0,019) sont hors bruit ; la différence A − défaut sur le privé (1,0150 vs 1,0156) ne l'est pas.
- Médiane en colonne : elle donne A −0,003 (encore plus net) et B +0,022 (au-dessus de 0,020) — les deux agrégats concordent sur le sens.
