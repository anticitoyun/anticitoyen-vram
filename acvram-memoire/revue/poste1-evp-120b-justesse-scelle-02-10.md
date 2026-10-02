# evp — scellé de justesse gpt-oss-120b (acvram contre llama.cpp, top-20)

poste1, 02/10, écrit à sec, avant tout instrument et toute mesure. Critère accepté par chef le 02/10. Converti :
`/mnt/AI_GENERATOR/models_acvram/gpt-oss-120b-srcmxfp4-nvfp4` (code 78e621ef5).

## Pourquoi llama.cpp, et ce que cela coûte

La référence HF du 20b (MXFP4 déquantifié en bf16, sur processeur) est impossible pour le 120b : elle demanderait
234 Go en bf16 pour 93 Gio de RAM. La référence devient donc llama.cpp sur le GGUF MXFP4 officiel
(`ggml-org/gpt-oss-120b-GGUF`, `gpt-oss-120b-MXFP4.gguf`, 63 387 346 208 o, dans
`/mnt/2TO_2023_980PRO/Modeles/models_gguf/gpt-oss-120b-mxfp4/`). C'est un autre moteur, inexact à sa façon : un
écart mesuré mêle nos fautes et les siennes. D'où une étape 0 qui chiffre l'écart propre de llama.cpp là où une
référence exacte existe, sur le 20b.

## Instrument (à écrire, éprouvé à sec contre un faux serveur avant la carte)

* Invites : les 11 dont la référence HF du 20b existe (`scratchpad/poste1-evp-01-10/dumps`, sha256 dans
  `dumps.sha256` et `dumps-b.sha256`). On reprend leurs **ids**, pas leur texte : le vocabulaire du 120b est celui
  du 20b (o200k harmony), ce qui écarte toute différence de tokenisation.
* llama.cpp (`llama-server` officiel 4c9233c) : `/completion` avec `prompt` = la liste d'ids, `n_predict` 32,
  température 0, `n_probs` 20 (probabilités avant échantillonnage), `cache_prompt` false. On garde, pas à pas,
  le jeton émis et ses 20 premiers (ids et log-probabilités). Pour le 120b : `-ngl 99 --n-cpu-moe N`, N étant le
  plus petit nombre de couches qui tient (lu au chargement), `-c 4096 -np 1 --flash-attn on`.
* acvram : le converti chargé selon le plan A (5090 + RAM), forcé sur la trajectoire de llama.cpp (ids de
  l'invite, puis les 32 jetons émis), en une passe de préfill. On prend les logits aux 32 positions.
* Métriques par position : top-1 = l'argmax d'acvram est le jeton émis par llama.cpp. KL(P_llama ‖ P_acvram) sur
  les 20 ids de llama.cpp, les deux distributions renormalisées sur ces 20. Puis moyenne par invite et sur les
  352 positions.
* Contrôle de l'arrondi : un crochet compte, sur les 352 positions, combien de fois l'expert 43 de la couche 13
  est routé (top-4).

## Étapes et prédictions

| étape | ce qui est comparé | prédiction (écrite avant) |
|---|---|---|
| 0. étalonnage | llama.cpp **20b** (GGUF 20b officiel) contre la référence HF exacte des 11 dumps, mêmes métriques | top-1 97-99,5 %, KL top-20 0,001-0,010 |
| 1. contrôle positif, à sec | jouet des tests, deux experts permutés dans une couche, contre le jouet intact (même instrument de métrique) | KL > 10 × celle du jouet intact, sinon l'instrument est aveugle aux fautes d'indexation : ÉCHEC d'instrument, pas de carte |
| 2. **verdict** | acvram 120b contre llama.cpp 120b | top-1 96-99 %, KL top-20 = étalon(0) + 0,000-0,005, pire invite ≤ 0,03 ; expert 13.43 routé 0 à 20 fois sur 352 × 4 choix |

## Seuils (acceptés par chef)

* **TENU** si top-1 ≥ 95 %, KL moyenne ≤ 0,05 et aucune invite > 0,20.
* **Diagnostic, en plus du seuil** : si KL(2) > 3 × (étalon(0) + 0,0012, la KL du 20b contre HF), l'écart ne
  s'explique plus par llama.cpp. Il faut alors chercher dans notre chargement, même si le seuil tient.

## Issues nommées

1. **Tenu et dans la prédiction** : le 120b est servi juste. L'écart à llama.cpp est du même ordre que celui de
   llama.cpp à HF.
2. **Tenu mais au-delà du diagnostic** (KL > 3 × étalon) : un défaut probable du chargement (indices sur 128
   experts, exil, rangement des piles) caché par des seuils lâches. Pièce à ouvrir, pas de publication « juste ».
3. **Résultat limite** (top-1 entre 95 et 97 %, ou KL entre 0,03 et 0,05). Trois causes possibles, nommées dans
   cet ordre :
   * (a) l'imprécision de llama.cpp, lue sur l'étalon (0) ;
   * (b) notre chargement du 120b ;
   * (c) **l'arrondi de `layers.13.experts.43.down`** : 26 blocs de 16, erreur au plus 4,77e-7 du maximum du
     tenseur, sur une rangée 2⁻¹⁸ sous le maximum. Prédit invisible (< 1e-6 nat). Le compteur de routage le
     départage : si l'expert 13.43 n'est jamais routé, (c) est exclu ; s'il l'est, on regarde l'écart aux
     positions où il l'est contre les autres.
4. **Échec net** (top-1 < 95 % ou KL > 0,05) : on compare d'abord à l'étalon. Un étalon lui-même mauvais
   invalide la référence llama.cpp, et le verdict devient INDÉCIDABLE, pas FAUX. Sinon, défaut de chargement :
   même code que le 20b, donc on cherche ce qui change avec l'échelle (128 experts, 36 couches, exil, arrondi).
5. **Celle qui me gênerait** : l'étalon (0) donne llama.cpp plus loin de HF (KL > 0,02) qu'acvram du 20b
   (0,0012). Notre référence serait alors la moins juste des deux, et le seuil de 0,05 serait à revoir avant
   tout verdict.
6. Instrument : llama.cpp refuse les ids en entrée, ou `n_probs` rend des probabilités après échantillonnage →
   corrigé à sec avant la carte, jamais pendant.

## Coût et condition

Carte : environ 15 min en une prise (étalon 20b, environ 2 min ; llama.cpp 120b, environ 2 min de chargement + 3 ;
acvram 120b en exil, environ 4 + 2), sous `carte.sh` avec `ACVRAM_POSTE=poste1`. Pas de mesure de temps : le type
« service » suffit, mais pas pendant la mesure d'un autre poste (lecture de `.qui` avant). RAM : llama.cpp (mmap de
63 Go) et acvram (≈ 36 Go épinglés) ne coexistent pas, d'où l'arrêt de l'un avant le chargement de l'autre. Après
evp (2), au feu de chef.
