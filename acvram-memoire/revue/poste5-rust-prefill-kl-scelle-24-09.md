# Scellé — acvram_rust (2) : préfill Rust jugé par KL (poste5, 24/09, écrit AVANT le code et la mesure)

Ordre du chef (option A) : la porte au bit couvre le décodage à préfill injecté ; le préfill Rust, lui, ne peut pas
être au bit (le Python préfille par cuBLAS/cuBLASLt et l'attention flash de libtorch, `poste5-rust-p1-releve-23-09.md`
§ 2) et se juge par KL.

## Ce que fait le préfill Rust (v1)

Les jetons de l'invite passent UN PAR UN dans le pas de décodage déjà prouvé au bit (mêmes noyaux, q_len = 1). Le
KV qu'il écrit diffère du KV Python au dernier bit bf16 près (GEMV au lieu de GEMM, attention paginée au lieu de
flash) : c'est cet écart, et lui seul, que la KL mesure.

## Instrument

`porte-kl` (Rust) contre le vidage Python refait avec les logits fp32 COMPLETS de chaque pas (`VIDAGE_LOGITS=1`).
Par invite (les 5 de `tests/invites.json`, ids d'invite déjà prouvés égaux) : préfill Rust → logits du 1er jeton
contre ceux du Python ; puis 128 pas en FORÇAGE PAR L'ENSEIGNANT (Rust reçoit les jetons que le Python a émis) → KL
de chaque pas. KL(P_python ‖ Q_rust) en nats, calculée en f64 sur les 151 936 logits.

## Seuils, fixés ici

1. Premier jeton : argmax égal 5/5 ET KL ≤ 1e-3 nat sur chaque invite.
2. Forçage 128 pas : KL moyenne ≤ 1e-3 nat ET KL max ≤ 1e-2 nat sur chaque invite ; argmax égaux ≥ 99 % des pas
   (sur l'ensemble des 5 invites).

Pourquoi ces nombres : un écart d'un bit bf16 sur K/V (≈ 2⁻⁸ relatif) déplace les logits de l'ordre de 1e-2 ; une KL
de Δ²/2 sur les logits qui comptent donne ≈ 5e-5 nat. Les seuils laissent un facteur 20 : ils rendent « faux » un
défaut de préfill (position, rope, cache mal rempli : KL ≫ 1e-1, argmax faux dès le 1er jeton), pas le bruit bf16.

## Prédiction et issues

* Prédit : KL 1er jeton 1e-5 à 1e-4 nat ; KL moyenne de forçage 1e-5 à 2e-4 ; argmax 1er jeton 5/5.
* FAUX si un seul seuil est dépassé → bisection par couche (KV Rust contre KV Python de l'invite, par couche).
* Ce qui me gênerait : une KL du 1er jeton ≈ 0 exactement (au bit) — cela voudrait dire que le Python ne préfille
  PAS par GEMM ici (invites courtes, T ≤ 32 : `_dense_etroit_kernel` Triton ?) et l'instrument ne mesurerait pas ce
  qu'il prétend ; je le dirais au lieu de m'en féliciter.
