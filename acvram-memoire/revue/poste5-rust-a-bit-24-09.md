# acvram_rust suite (1) : porte AU BIT des logits — TENUE 5/5, cassure démontrée (poste5, 24/09)

instrument : `moteurs/acvram_rust/src/bin/porte.rs` (debug) ; vidage `outils/vidage_reference.py` avec espion de `Engine._sample_only` (sha256 des logits fp32 de chaque pas, là où le moteur servi échantillonne) ; prise `scratchpad/poste5-rust-a-23-09/prise-bit.sh`
commit : prise 1 9f3ab93f (`bit-1.txt`), prise 2 f31996bb (`bit.txt`) ; extension `kernels-86bf52a9f2f9` (recompilée avec les puits, la même pour les deux moteurs)
régime : Qwen3-4B-srcgguf-nvfp4, b=1, glouton, 128 jetons, préfill injecté (option A) ; Python graphes on
scellé : logits fp32 de chaque pas de décodage == Python au bit sur 5/5 ; la cassure prévue d'avance doit la faire tomber
mesuré : **porte au bit 5/5** (344 pas, deux prises) ; ids 5/5 ; empreintes 605/605. Cassure 1 (tranches un pas plus tôt) : 5/5 au bit — **ne cassait rien** ; cassure 2 (échelle d'attention + 1 ulp fp32) : **logits FAUX 5/5 dès le pas 0**, ids 3/5 (code diverge au pas 24, anglais au pas 77)
verdict : TENU, et la porte sait rendre faux
durée : prévu ≤ 600 s par prise ; tenu ≈ 90 s (prise 1, vidage compris) et ≈ 60 s (prise 2)

## Ce que disent les deux cassures

* Cassure 1, prédiction RÉFUTÉE : à la position 127 la séquence tient dans 8 pages ; passer de 2 à 4 tranches de 64
  jetons ajoute deux tranches VIDES, qui pèsent exactement 0 dans la réduction (m = −∞, l = 0). Le découpage n'agit
  sur les bits que par la taille de tranche, pas par leur nombre : un fait sur notre noyau, utile pour la suite (le
  godet `nblk` n'a pas à être reproduit au pas près tant que la taille de tranche reste 64, soit N ≤ 128 blocs =
  2 048 jetons à b=1 ; à N = 256 elle passe à 128 — effet sur les bits non mesuré).
* Cassure 2 : un ulp sur UN paramètre suffit à faire échouer la porte au bit à chaque pas — et à faire basculer un
  argmax en 24 pas sur « code ». La porte des ids (scellé d'origine) l'aurait attrapée sur 2 invites sur 5
  seulement ; celle des logits sur 5/5. C'est la porte au bit qui fait foi désormais.

## Suite (ordre du chef)

(2) Préfill Rust jugé par KL — seuil écrit avant la mesure dans la note de la pièce ; (3) débit b=1 en release, ABBA
≥ 5 lots contre acvram, J/jeton, prédiction scellée 180-215 t/s.
