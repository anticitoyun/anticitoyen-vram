# acvram_rust étape 1, option A : porte du décodage à préfill injecté — TENUE 5/5 (poste5, 23/09)

instrument : `moteurs/acvram_rust/src/bin/porte.rs` (debug) contre `outils/vidage_reference.py` + `vidage_tables.py` ; prises `scratchpad/poste5-rust-a-23-09/prise-vidage.sh`, `prise-porte.sh`
commit : vidage ca2cebbe, porte 229cea22 (branche poste5) ; extension servie `kernels-86bf52a9f2f9` des deux côtés
régime : Qwen3-4B-srcgguf-nvfp4, b=1, glouton, 128 jetons ; Python = moteur construit comme `serve` (graphes on, chauffe, KV int8) ; Rust = eager, nos noyaux (fatbin du .so servi) + cubins Triton vidés
scellé : sha256 des ids générés == Python sur 5/5 (chef, option A), écrit avant
mesuré : **5/5 TENU** — court 12/12, code 128/128, systeme 51/51, anglais 128/128, tours 30/30, aucune divergence ; empreintes des poids envoyés : 605 égales, 0 différente (37 sans pendant Python, noms non vidés côté Python)
verdict : TENU
durée : prévu ≤ 600 s ; tenu ≈ 25 s (attente carte 341 s)

## Ce qui est prouvé, et ce qui ne l'est pas

* Prouvé : l'hôte Rust enchaîne les 432 lancements du pas dans l'ordre servi — empilements, normes fusionnées
  à la rope, tranches de l'attention (2 → 4 à la position 128, les deux variantes Triton lancées :
  10 368 + 2 016 appels), tête int8 fp32, argmax — et rend les MÊMES JETONS que le Python sur 5 invites.
* NON prouvé : l'égalité AU BIT des logits. Un sha d'ids ne rend « faux » que si un argmax bascule ; un écart
  d'un ulp qui ne change aucun jeton passerait. Contrôle plus fort proposé (§ suite), avec sa cassure.
* Les durées affichées par `porte` (0,48 s pour 128 jetons) viennent d'un build debug, sans horloge posée, avec
  copie des logits vers l'hôte à chaque pas : ce ne sont PAS des chiffres de débit, rien n'en est tiré.

## Suite proposée (ordre du chef attendu)

1. Porte au bit : le vidage Python enregistre le sha256 des logits fp32 de chaque pas ; `porte` compare pas à pas.
   Cassure prévue d'avance (doit rendre FAUX) : décaler d'un pas le passage 2 → 4 tranches de l'attention
   (ordre de sommation changé à la position 127 seulement, calcul valide) — si les ids restent égaux et les
   logits diffèrent, cela montre exactement ce que la porte actuelle ne voit pas.
2. Préfill Rust (option A : jugé par KL, seuils à écrire avant), puis `/v1/chat/completions` bout en bout.
3. Débit b=1 : build release, argmax sur la carte (plus de copie des logits), ABBA contre acvram, -lgc 2700 des
   deux côtés ; prédiction scellée 180-215 t/s contre ≈ 227 (étape 0).
