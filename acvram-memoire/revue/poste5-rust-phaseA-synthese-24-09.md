# acvram_rust — synthèse de la phase A (question (a) : le langage HÔTE) — poste5, 24/09

Contrat : `revue/moteurs-rust-mojo-23-09.md`. Modèle : Qwen3-4B-srcgguf-nvfp4, b=1, carte 0 (RTX 5090). Mêmes noyaux
que le service (SASS du fatbin servi `acvram_kernels.so` sha256 21f54a69a37f6fe1, cubins Triton vidés), mêmes poids,
même tokeniseur (`tokenizers` 0.23.2). Crate `moteurs/acvram_rust`, branche poste5.

**Réponse : à b=1, le langage hôte ne change rien.** Hôte Rust eager (432 lancements par pas, sans graphe CUDA)
307,7 t/s contre acvram (hôte Python, graphes CUDA, pipeline recouvert) 311,7 t/s : −1,29 %.

| pièce | verdict | fichier |
|---|---|---|
| étape 0 (à sec) : FFI vers le SASS servi, pas de recompilation, pas de libtorch | choix posé | `poste5-rust-etape0-23-09.md` |
| relevé des noyaux servis : 432 lancements/pas, période exacte | relevé | `poste5-rust-p1-releve-23-09.md` |
| porte option A : décodage Rust sur préfill Python injecté, mêmes ids | **TENUE 5/5** | `poste5-rust-a-porte-23-09.md` |
| porte au bit des logits fp32, cassure +1 ulp démontrée (logits FAUX dès le pas 0) | **TENUE 5/5** | `poste5-rust-a-bit-24-09.md` |
| (2) préfill Rust contre préfill servi, par KL | **FAUX** au scellé (1er jeton 5/5, forçage 5/5 faux) | `poste5-rust-prefill-kl-24-09.md` |
| porte A' : préfill Rust = décodage forcé d'acvram au bit (ligne 0 commune) | TENUE 5/5, scellée APRÈS la mesure | idem § Porte A' |
| (3) débit b=1 release : décodage seul, prédit 180-215 | **FAUX vers le haut** : 307,7 contre 311,7 | `poste5-rust-debit-24-09.md` |

## Ce que la phase prouve

* L'hôte Rust exécute le calcul d'acvram : logits identiques au bit sur 5 invites (décodage), cache KV identique au
  bit au décodage forcé d'acvram (préfill). La porte au bit a été rejouée sur le build release juste avant la
  cellule de débit.
* À b=1 sur ce modèle, le coût hôte est recouvert : 432 lancements eager par pas coûtent 3,25 ms/pas contre 3,21 ms
  pour les graphes d'acvram (0,04 ms, 1,3 %). Même horloge (2 670/2 671 MHz), même bridage de puissance (~318 W).

## Énergie : J/jeton +25 %, dû au préfill jeton par jeton

J/jeton net Rust 0,9837 contre 0,7869 (+25,0 %, borne haute de la prédiction). Le décodage coûte pareil (même vitesse,
même puissance) ; l'écart vient du préfill Rust v1, qui passe l'invite en 256 pas de décodage au lieu d'un préfill
groupé : 250 contre 310 t/s préfill compris, 110 contre 304 t/s sur les passes courtes de 128 jetons.

## Limites (à ne pas dépasser en citant ces chiffres)

* Une seule taille (4B dense), b=1, un seul format de poids (nvfp4 + int8), contexte ≤ 1 280. Rien n'est dit de b>1,
  des MoE, ni des hybrides : là où l'hôte ne se recouvre plus (petits pas, lots), la conclusion peut s'inverser.
* Le préfill Rust n'est PAS égal au préfill servi (FAUX au scellé, KL de forçage jusqu'à 0,12) : il égale le chemin
  décodage d'acvram. L'écart préfill/décodage interne à acvram est la pièce 125 (poste6).
* Serveur Rust minimal : un lot de 1, glouton seul, pas de graphe, pas de spéculation, pas de /metrics réel.
* Non suite : graphes côté Rust (≤ 1,3 % à gagner ici), b=8, préfill groupé — aucune sans décision de l'utilisateur.

## Défauts trouvés et corrigés pendant la phase (avant la prise jugée)

`.gitignore` `*token*` excluait `tokeniseur.rs` (crate non compilable depuis git) ; PTX NVRTC 13.4 refusé par le pilote
595 → SASS sm_120 ; fil serveur sans contexte CUDA (INVALID_CONTEXT, 1re prise invalide et rejouée) ; banc aveugle au
flux d'acvram (`usage` sur chaque fragment) ; garde « 8 jetons » avant chaque fenêtre.
