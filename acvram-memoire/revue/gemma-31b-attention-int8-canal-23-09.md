# Pièce 56 — la vraie attention int8 par canal (230 projections promues, MLP nvfp4) : 2/5, kl_max 6,63, méd. 1,15 — l'int8 W8A8 par canal ne vaut pas le bf16 du bras M, prédiction RÉFUTÉE — 23/09 (Gaelle)

* instrument : `acvram convert gemma-4-31B-it-bf16 -o gemma-4-31B-it-nvfp4-attn-i8c-p56 --snr-floor 99 --promotion-classes q_proj,k_proj,v_proj,o_proj --max-promotions 1.0 --attn-qkvo-int8-canal --quant-device cuda:0 --calib-device cuda:0` (calibration défaut, texte brut ; mode service, `.qui = gaelle-p56-conversion`), **contrôle du manifeste AVANT la KL** (compte par classe), puis `kl-gabarit.py acvram|hidden|divergence`, mêmes 5 invites et dumps HF (pièce 52)
* commit : conversion sous f59660b1 ; KL sous 655d889a (kl-gabarit corrigé : libération de l'Engine par invite — la 3e invite tombait en OOM sur cet alias, +4 Gio)
* régime : conversion 376 s sur carte ; KL/hidden avec `ACVRAM_EXIL_COUCHES=24` (l'alias tient à 30,3 Gio sur 31,4 sans exil : le prefill n'avait plus de place pour ses tampons bf16 — arithmétique inchangée, vitesse sans objet) ; eco 2700 ; compute-apps début = fin (llama-server seul) ; verrou tenu 376 + 8 + 8 + 18 + 8 + 13 + 13 s
* scellé (gaelle.md f59660b1, avant) : contrôle manifeste = 240 int8 / 0 nvfp4 sur q/k/v/o — rendu « FAUX » à tort par mon script (60 v_proj attendus, il y en a 50 : les 10 couches globales ont k = v), corrigé à la main : **q 60, k 60, o 60, v 50 int8, group_size 5376 = par canal, 0 nvfp4** — l'int8 est bien là. Prédiction : méd. kl_max 0,3-0,5, 4/5, kl_max ≤ 1,5 ; réfuté si ≤ 3/5 ou méd. > 0,8
* mesuré :

| alias | kl_max invites 0-4 | méd. kl_max | p90 des pas | pas ≥ 1 | tenu | err. rés. c57 / finale |
|---|---|---|---|---|---|---|
| nvfp4-vision (tout nvfp4) | 3,17 · 0,67 · 1,01 · 0,02 · 2,84 | 1,01 | 0,376 | 5 | 2/5 | 0,41 / 0,25 |
| bras M (attention **bf16**, p54) | 0,66 · 0,28 · 0,16 · 0,02 · 1,14 | 0,28 | 0,100 | 1 | 4/5 | 0,27 / 0,17 |
| **p56 : attention int8 par canal** | **3,68 · 0,28 · 0,75 · 6,63 · 1,15** | **1,15** | 0,219 | 3 | **2/5** | **0,35 / 0,22** |

  Par invite : 1, 2, 4 nettement meilleures qu'en tout-nvfp4 (0,67 → 0,28 ; 1,01 → 0,75 ; 2,84 → 1,15), l'invite 0 pire (3,17 → 3,68), et **l'invite 3 (arithmétique), parfaite dans TOUS les autres bras (0,02), saute à 6,63 sur un seul pas (le 21e)** : l'attention int8 W8A8 par canal introduit une erreur ponctuelle catastrophique que ni le nvfp4 ni le bf16 de l'attention ne produisent — signature d'une activation quantifiée par jeton qui écrase un canal (A8), pas d'une erreur de poids (le SNR des q/k/v/o int8 est meilleur que leur nvfp4). Erreur résiduelle c57 0,35 : l'int8 n'enlève qu'un tiers de l'erreur d'attention (0,41 → 0,35, le bf16 fait 0,27) ; profil et amplificateurs (52, 51, 57, 56, 39) inchangés.
* verdict : **prédiction RÉFUTÉE** (2/5, méd. 1,15 > 0,8) — l'attention int8 par canal telle qu'elle existe ne remplace pas le bf16 du bras M sur Gemma 4 : elle gagne sur trois invites et perd sur deux, dont un pas isolé à 6,6 nats. La chaîne « int8 attention + MLP nvfp4 » ne donne pas un 31B servable ; **la borne du bras M (attention bf16) n'est pas atteignable par l'int8 W8A8 actuel**. Ce qui manque au projet, en une ligne : un format d'attention entre int8-A8 et bf16 (int8 poids seuls / activations bf16, ou fp8 W8A8 — Q(13)) et, pour le MLP, plus de bits ; la calibration n'est pas le levier (pièce 55). Règle 6 livrée : `_bilan_attn_int8` (convert.py) — le manifeste dit `groupe` + avertissement quand le drapeau n'a rien promu ; 4 tests dont un bout en bout qui aurait cassé le 22/09.
* durée : 376 s de conversion + 68 s de KL/hidden (4 prises dont 2 OOM) = 444 s de carte ; verrou rendu 01 h 26, carte LIBRE pour Laure

## Reste
Nommer le pas 21 de l'invite 3 (quel jeton, quelle couche) avec les hidden p56 contre bras M — à sec, 10 min ; A8 par jeton contre par canal sur les projections d'attention si Jerome veut creuser l'int8. Effacer les alias p53/p54 (liens) et p55 (16 Gio inutile) sur ordre.
