# Scellé — acvram_rust (3) : débit b=1, Rust contre acvram (poste5, 24/09, écrit AVANT le banc et la prise)

Question (a) du contrat : le coût du langage HÔTE. Mêmes noyaux (fatbin servi, porte au bit 5/5), même modèle
(Qwen3-4B-srcgguf-nvfp4), même extension, même client HTTP, même prise.

## Bras

* **A — acvram** : `acvram serve <modèle> --max-batch 1 --max-model-len 2304 --speculative none` (graphes CUDA,
  pipeline recouvert, éco 2700 posé par le serveur — régime servi par défaut).
* **B — acvram_rust** : `acvram-rust serve` en build **release**, eager (un lancement par noyau, ~432 par pas, pas
  de graphe), argmax sur la carte ; `-lgc 2700` posé par la prise sur la carte 0 avant B, rendu après.

## Instrument

`scratchpad/poste5-rust-a-23-09/banc-rust.py` = copie de `banc-llamacpp-16-09.py` (mode décodage, bras « serveur
déjà lancé », `/v1/completions` en flux SSE, invite de 256 ids tirés, 1 024 jetons, `ignore_eos`), à UN ajout près :
l'horodatage du premier et du dernier fragment SSE de chaque requête. Colonnes :
* **décodage seul (t/s)** = (jetons − 1) / (t_dernier − t_premier) par lot — LA grandeur jugée ;
* `jetons_s` du banc (préfill compris) et J/jeton net du banc (NVML, repos soustrait) — publiés, jugés pour J
  seulement. Le préfill Rust v1 passe jeton par jeton (256 pas) : `jetons_s` Rust est pénalisé par construction.

Ordre ABBA × 2 (A B B A A B B A), un serveur neuf par fenêtre, fenêtre 20 s, chauffe du banc ; ≥ 16 lots par bras.
En-tête : `max_perf_pct` au début et à la fin (cpu-safe), `nvidia-smi --query-compute-apps` au début et à la fin,
horloge SM moyenne de chaque fenêtre (résumé NVML du banc), commit, sha du fatbin.

## Prédiction (scellée à l'étape 0, 23/09, `poste5-rust-etape0-23-09.md` § 6) et seuils

* Décodage seul Rust **180-215 t/s** ; acvram ≈ 227 (remesuré ici, bras A). FAUX vers le bas si Rust < 150 ; FAUX
  vers le haut si Rust > 240 ou si Rust ≥ acvram au-delà de 2 σ (Welch) — alors l'hôte Python coûte à b=1 un temps
  que nos graphes ne rattrapent pas, et je le dirais.
* J/jeton net : Rust **+5 à +25 %** au-dessus d'acvram (même puissance à peu près, moins de jetons par seconde).
* `jetons_s` (préfill compris), prédit Rust 145-175 t/s : publié, pas jugé.

## Issues nommées

* Tient (180-215) : l'hôte Rust eager coûte ce que coûtent ~432 lancements sans graphe ; la suite logique est les
  graphes CUDA côté Rust (étape 2 de l'étape 0, prédit 220-235).
* Ce qui me gênerait : Rust ≈ acvram à ± 2 σ (les graphes d'acvram ne gagneraient rien à b=1 sur ce modèle) — ou une
  horloge SM moyenne différente de plus de 30 MHz entre bras, qui rendrait la cellule non comparable (et je le dirais
  au lieu de comparer).
