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
