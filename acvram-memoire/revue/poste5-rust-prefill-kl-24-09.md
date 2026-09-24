# acvram_rust (2) : préfill Rust par KL — FAUX au scellé (poste5, 24/09)

instrument : `moteurs/acvram_rust/src/bin/porte_kl.rs` (debug) contre le vidage Python aux logits complets (`VIDAGE_LOGITS=1`) ; prise `scratchpad/poste5-rust-a-23-09/prise-kl.sh` (`kl.txt`)
commit : d698833a (branche poste5) ; extension `kernels-86bf52a9f2f9` des deux côtés
régime : Qwen3-4B-srcgguf-nvfp4, b=1 ; préfill Rust v1 = jetons de l'invite un par un dans le pas prouvé au bit ; Python = préfill servi
scellé : `revue/poste5-rust-prefill-kl-scelle-24-09.md` (8591b3b9, écrit avant) — 1er jeton argmax égal et KL ≤ 1e-3 ; forçage 128 pas KL moy ≤ 1e-3, max ≤ 1e-2 par invite ; argmax ≥ 99 %
mesuré : 1er jeton **5/5 tenu** (KL 9,6e-10 · 1,5e-7 · 1,4e-8 · 6,9e-7 · 2,7e-4 ; argmax 5/5) ; forçage **FAUX 5/5** — KL moy 1,1e-2 · 5,8e-4 · 1,2e-3 · 4,1e-3 · 1,3e-3, max 0,119 · 0,012 · 0,029 · 0,090 · 0,037 ; argmax 339/344 = 98,55 % ; porte au bit rejouée sur ce vidage : 5/5
verdict : FAUX
durée : prévu ≤ 900 s ; tenu ≈ 5 min (vidage compris)

## Lecture

* Le préfill Rust rend le PREMIER jeton presque au bit (KL ≤ 2,7e-4, souvent ~1e-8) : les ids, la rope, les positions
  et le remplissage du cache sont justes — un défaut de préfill aurait fait sauter ce chiffre de plusieurs ordres.
* L'écart NAÎT ensuite : pendant le forçage, la KL par pas monte jusqu'à 0,12 nat alors qu'elle était ~1e-8 au jeton
  précédent. Mon seuil supposait un bruit bf16 sur K/V ; il ignorait que le cache est INT8 par jeton : un écart d'un
  ulp bf16 avant quantification peut faire basculer un code int8 (pas de amax/127, soit 2⁻⁷ relatif), et les lignes
  du préfill sont relues à chaque pas suivant. Hypothèse, NON vérifiée.
* Le seuil ne se rouvre pas (REGLES § 3). Issue nommée au scellé : bisection par couche — codes K/V int8 du préfill
  Rust contre ceux du Python (`kv-<invite>.safetensors`), part des codes différents et amplitude, par couche et par
  position. Ce qui trancherait l'hypothèse : des écarts d'au plus ± 1 code, répartis sur toutes les couches → prix
  du préfill par GEMV sur un cache int8, et le remède est un préfill Rust par les mêmes noyaux que le Python (option
  B du 23/09) ; un écart concentré (une couche, une position, > 1 code) → défaut du préfill Rust, à corriger.

## Bisection (issue nommée au scellé) — 24/09 00 h 47 → 01 h 15, cpu-safe=off (max_perf_pct 100 au début et à la fin des deux dernières prises)

| comparaison du KV de l'invite (5 invites, 36 couches) | codes K différents | codes V différents | fichier |
|---|---|---|---|
| préfill Python servi ↔ préfill Rust | 33-51 % | 71-82 % | `kv.txt` |
| même couche 0 | **0** | **0** | `ecriture.txt` |
| préfill Python ↔ ses propres K/V en vol (entrée de `cache.write`) | quantification seule, 1-2,4 % L2 | 0,8-1,1 % L2 | `ecriture.txt` |
| **décodage forcé Python** (même moteur servi, jetons de l'invite imposés un par un) ↔ préfill Python | 32-51 % | 70-81 % | `decode-force.txt` |
| décodage forcé Python ↔ préfill Rust | 14-18 % | 42-50 % | `decode-force.txt` |
| décodage forcé Python ↔ préfill Rust **avec la ligne 0 du Python** | **0 — au bit 5/5** | **0** | `ligne0.txt` |

* Pas une permutation (appariement ligne à ligne = identité, `comparer_kv.py`).
* **Le préfill Rust reproduit AU BIT le chemin de décodage d'acvram** dès que la ligne 0 est la même ; la seule
  différence restante vient de la ligne 0, que le Python calcule par son préfill (un jeton) et le Rust par son pas.
* **L'échec du scellé mesure donc un écart INTERNE à acvram** : son préfill (GEMM, attention flash sur K/V bf16) et
  son décodage (GEMV, attention paginée sur KV int8) remplissent le cache de l'invite différemment — jusqu'à 19 %
  L2 sur V dès la couche 1, et une KL de forçage jusqu'à 0,12 nat. Une perturbation de 3 codes sur la seule ligne 0
  suffit à déplacer 14-18 % des codes K de toute l'invite : le cache int8 amplifie les écarts d'arrondi.
* Lequel des deux chemins est le plus juste n'est PAS mesuré ici (il faudrait une référence bf16 non quantifiée).

## Décision demandée au chef (deux options, une ligne chacune)

* **B** : préfill Rust par les MÊMES noyaux que le préfill Python (cuBLASLt du venv, attention flash de libtorch) —
  seul moyen d'atteindre le scellé tel qu'écrit ; + 1 à 2 jours, fragile (option B du 23/09).
* **A'** : nouveau scellé, écrit avant, relatif au témoin : KL(Rust ‖ préfill Python) ≤ KL(décodage forcé Python ‖
  préfill Python) + marge — le préfill Rust est déjà AU BIT du décodage forcé, ligne 0 comprise mise à part.
* À signaler en plus : l'écart préfill/décodage d'acvram (KV de l'invite) mériterait sa propre pièce (qui est le
  plus juste contre une référence bf16 ; effet sur la PPL et la KL servies).

## Porte A' (décision du chef 24/09, **scellée APRÈS la mesure**, ne vaut QUE pour la question (a), le langage hôte)

Porte : le préfill Rust est identique AU BIT à un chemin d'acvram — le décodage forcé du moteur Python servi —, ligne 0
commune : **TENUE 5/5** (`ligne0.txt`, 0 code différent sur 36 couches, 5 invites). Scellée après coup : elle ne se
cite jamais comme un résultat de justesse du préfill Rust contre le préfill servi (celui-là reste FAUX au scellé
d'origine), seulement comme preuve que l'hôte Rust exécute le calcul d'acvram. B refusé (chef). L'écart
préfill/décodage d'acvram devient la pièce 125 (poste6).
