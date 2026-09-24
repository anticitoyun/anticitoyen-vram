# Scellé — pièce 125 : préfill ou décodage, lequel est le plus juste ? (poste6, 24/09, écrit AVANT la mesure)

Ordre de chef. poste5 (`revue/poste5-rust-prefill-kl-24-09.md`) : sur Qwen3-4B-srcgguf-nvfp4, le préfill servi
(GEMM, attention flash sur K/V bf16) et le décodage forcé (GEMV, attention paginée sur KV int8) remplissent le KV de
l'invite différemment — 32-51 % des codes K, jusqu'à 19 % L2 sur V dès la couche 1, KL de forçage jusqu'à 0,12 nat.
Lequel a raison n'est pas mesuré. Ici chaque chemin est jugé contre une référence bf16 HF, sur processeur.

## Instrument
`outils/gpu/mesure/kl-chemins-p125.py` (fichier suivi, commit dans le verdict) ; prises `scratchpad/poste6-p125-24-09/prise-*.sh`
sous `carte.sh` (`ACVRAM_TYPE=mesure`, `ACVRAM_DUREE_MAX=1800`), HEAD asserté (rc 65). Le bras HF (processeur, `CUDA_VISIBLE_DEVICES=""`,
8 fils, nice 19) tourne SOUS le verrou aussi : une charge processeur pendant la fenêtre d'un pair contamine sa mesure (REGLES § 2).
* 5 invites de `kl-gabarit.py` (pièce 52), gabarit de conversation de HF, `enable_thinking=False` pour Qwen3 ; réponse
  gloutonne de HF, 32 jetons ; **aucun bras ne retokenise** (ids de HF partout).
* Par chemin et par invite : log-probs aux 32 positions de réponse (forçage par l'enseignant sur la réponse de HF) ;
  flux résiduel fp32 après CHAQUE couche (crochets sur `layers[i]`, même sémantique côté HF : sortie du bloc, avant la
  norme finale) à toutes les positions.
* `prefill` : un forward de préfill (`_build_batch(prefill=True)`, `logits_positions`), sans graphes. `force` : 1er jeton
  préfillé, puis chaque jeton (invite ET réponse) forcé un par un dans `_sample_only`, comme `vidage_decode_force.py`
  (poste5) ; eager (les crochets ne voient pas un graphe rejoué) ; **témoin `force graphes`** = même chose, graphes servis,
  logits seuls.
* Mesures : KL(HF ‖ chemin) par position, moyenne et max par invite ; KL(préfill ‖ forcé) ; L2 relative par couche
  ‖chemin − HF‖/‖HF‖ sur les positions d'invite (le KV que poste5 compare) et sur les positions de réponse ; argmax égaux.

## Alias et références
| alias acvram | référence HF bf16 | rôle |
|---|---|---|
| `Qwen3-4B-srcgguf-nvfp4` | `Qwen3-4B-bf16-hf` (hub) | le cas de poste5 |
| `Qwen3-4B-bf16-acvram` (398 tenseurs bf16, poste5 23/09) | idem | **contrôle** : poids non quantifiés, seul le chemin change |
| `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c` (servi) | `models/Qwen3-Coder-30B-A3B-Instruct` (bf16, 61 Go, processeur) | le verdict qui compte |

Réserve écrite d'avance : la source du 4B nvfp4 est un GGUF (`Qwen3-4B-gguf` au manifeste), pas le hub — un plancher
commun aux deux chemins est possible ; il ne change pas la comparaison préfill/forcé, qui est APPARIÉE (même alias,
même référence, mêmes ids). Le contrôle bf16-acvram dit si la référence et les crochets sont justes.

## Seuils, fixés ici
* **S0 — l'instrument peut rendre faux** : (a) sur `Qwen3-4B-bf16-acvram`, préfill contre HF : KL moyenne ≤ 5e-3 nat et
  L2 d'invite à la dernière couche comparée ≤ 0,02 sur 5/5 ; sinon la référence (gabarit, ids, sémantique des crochets)
  est en cause : **arrêt, dit tel quel**, aucun verdict. (b) témoin `force graphes` = `force` eager à **0 ulp** sur les
  log-probs (max |Δ| = 0) sur 5/5 ; sinon les résultats du forçage sont étiquetés « eager », pas « servi ».
* **S1 — verdict par alias** : le chemin X est « le moins juste » si KL_moy(X) > 1,25 × KL_moy(Y) sur ≥ 4/5 invites
  ET L2 d'invite (dernière couche comparée) X > 1,25 × Y sur ≥ 4/5. Dans les deux sens. Sinon « équivalents à 25 % près »
  (indécidable au sens de ce scellé : l'écart de poste5 serait un bruit symétrique, aucun chemin n'a « raison »).
* **Défaut servi** = S1 rend « préfill le moins juste » sur l'alias SERVI (Coder i8c). Le 4B seul ne suffit pas.

## Prédiction et issues (toutes nommées)
* Prédit : le **préfill est le plus juste** sur les trois alias — il écrit le KV de l'invite depuis des K/V bf16 exacts en
  un GEMM, le forçage le réécrit jeton par jeton en int8 par jeton (2⁻⁷ relatif) et relit ces codes à chaque pas.
  Chiffres : 4B nvfp4 KL_moy forcé / préfill = 1,5-3 ; bf16-acvram préfill 1e-4-1e-3 nat, forcé 1e-3-1e-2 ; Coder i8c
  préfill 0,05-0,20 nat (KL b=1 ≤ 0,74 qualifiée, 100 B), forcé 1,5-3 ×.
* FAUX si KL_moy(forcé) ≤ 1,25 × KL_moy(préfill) sur ≥ 2/5 invites d'un alias.
* Issue qui gêne chef (nommée dans l'ordre) : **préfill le moins juste** sur i8c → défaut servi, à corriger (pièce).
* Issue qui me gêne : **« équivalents »** sur i8c — alors les 32-51 % de codes de poste5 sont le prix symétrique d'un
  cache int8, il n'y a rien à « corriger », seulement un plancher à documenter ; je le dirais tel quel.
* Issue qui invalide : S0(a) faux → référence en cause, pas le moteur.

## Durée et régime
Prise 1 (4B) : HF ≈ 3 min + 2 alias × 3 bras ≤ 12 min → ≤ 15 min. Prise 2 (Coder) : HF ≈ 10 min (30B-A3B bf16 sur
processeur, 5 × 33 jetons) + 3 bras ≤ 8 min → ≤ 25 min. `nvidia-smi --query-compute-apps` et `max_perf_pct` relevés
au début et à la fin de chaque prise ; `cpu-safe=off` (bridage levé à 00:54:35) ; aucun chiffre de temps n'est publié ici.
