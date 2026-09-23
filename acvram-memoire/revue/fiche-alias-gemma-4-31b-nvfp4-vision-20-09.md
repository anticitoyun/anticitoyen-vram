# Fiche d'alias — gemma-4-31B-it-nvfp4-vision (20/09, 17 h 08, Jérôme ; ordre Sage 17 h 06)

Alias servi par 0.6.33 : dossier `gemma-4-31B-it-nvfp4-vision` (texte nvfp4, tour de vision bf16, régime `vision=bf16(eager)`, KV int8 gardé avec image).

## Chiffres publiés tels quels (régime « 31B nvfp4 en général »)

| mesure | valeur | source |
|---|---|---|
| decode-pas texte seul, 259 jetons + consigne, 8 pas forcés, contre HF bf16 offload | **0 promotion**, KL par pas [0,13 ; 0 ; 0 ; 2,12 ; 0,02 ; 0 ; 0 ; 0,26], **KL max 2,12** | verdict-decode-pas-31b-kv (témoin texte, Manon 17 h 05) |
| decode-pas img00, ordre texte→image, 8 pas | **1 promotion** (pas 4 : acvram 23158 = rang 4 chez HF, 7,50 nat sous le top-1 ; HF quasi-égalité 723/34464 à 0,125 nat), KL 1,11 | verdict-decode-pas-31b-kv-20-09 + Océane 16 h 50 |
| decode-pas img01 | 8/8 égaux, KL ≤ 0,32 | idem |
| PPL relative sur 5 descriptions HF (img00-04), cibles alignées | géo −33,5 % (nvfp4 plus confiant 4/5), sd 47 %, +2 SE = +1,6 % ≤ 3 : « non établi pire », bande non résolue à n=5 | verdict-c-nvfp4-31b-20-09 |
| coût KV int8 avec image (12B bf16, img00-02) | PPL int8/bf16 −1,3 / −16,4 / −3,1 %, géo −7,2 % (int8 plus confiant, publié sans explication) → int8 reste | verdict-decode-pas-31b-kv-20-09 |
| duel vision contre llama.cpp (31B, 20 images) | TTFT 0,058 s contre 0,651 (× 11), J net 204 contre 314 | verdict duel vision 20/09 |

## Réserve, mot pour mot (Sage 17 h 06)

« texte 0 promotion / image 1 promotion sur 8 pas, un échantillon chacun : la question lignes-image reste OUVERTE, non résolue à n=1 ». Le nom du régime vient de la clause écrite avant la prise (≥ 1 promotion OU KL max ≥ 1,0 sur texte seul → « 31B nvfp4 en général ») ; le critère promotion seul aurait dit « lignes image ».

Chantier à part, en file après Qwen3-VL : « qualité nvfp4 du 31B dense » — par bande (PPL texte contre bf16 étagé, promotions avec/sans image sur 5 + 5 invites, puis calibration ou échelle par bloc si > bande). Le 31B dense nvfp4 n'a jamais eu de certification par jeton contre HF, seulement des bandes PPL.
