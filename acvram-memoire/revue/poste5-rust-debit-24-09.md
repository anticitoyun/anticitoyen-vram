# Verdict — acvram_rust (3) : débit b=1, Rust release contre acvram (poste5, 24/09)

* **instrument** : `scratchpad/poste5-rust-a-23-09/banc-rust.py` (copie de `banc-llamacpp-16-09.py` + horodatage des fragments SSE), `prise-debit.sh`, `resume-debit.py` ; NVML `energie.py`, carte 0 seule
* **commit** : 73c02b20 (branche poste5) ; fatbin `acvram_kernels.so` sha256 21f54a69a37f6fe1 ; acvram importé depuis l'arbre de la branche
* **régime** : b=1, Qwen3-4B-srcgguf-nvfp4 (alias q4b), invite 256 ids tirés, 1 024 jetons, `ignore_eos`, `-lgc 2700` (A : posé par `acvram serve` ; B : par la prise), plafond 400 W, horloge moy. 2 670 / 2 671 MHz, bridage « puissance » des deux côtés (~317-320 W), max_perf_pct 100 début/fin (cpu-safe désactivé), apps début/fin = llama-server 8081 seul, load1 ≤ 2,1
* **scellé** : `revue/poste5-rust-debit-scelle-24-09.md` — décodage seul Rust 180-215 t/s ; FAUX < 150 ou > 240 ou Rust ≥ acvram au-delà de 2 σ ; J/jeton net Rust +5 à +25 %
* **mesuré** : décodage seul Rust **307,7 t/s** (20 lots, σ 0,04) contre acvram **311,7** (28 lots, σ 0,08) : **−1,29 %** ; `jetons_s` (préfill compris) 250,1 contre 310,1 ; J/jeton net 0,9837 contre 0,7869 (**+25,0 %**, borne haute)
* **verdict** : **FAUX vers le haut** — Rust 307,7 > 240. L'hôte Rust eager (~432 lancements par pas, sans graphe) décode à 1,3 % des graphes CUDA d'acvram : c'est l'issue « qui me gênerait » nommée au scellé (Rust ≈ acvram), à 1,3 % près
* **durée** : prévu ~15 min / tenu 4 s (porte) + 221 s + 221 s (`carte.sh`)

## Table (ABBA × 2, un serveur neuf par fenêtre)

| k | bras | lots | décodage seul t/s | jetons_s | J/jeton net | horloge MHz | W |
|---|---|---|---|---|---|---|---|
| 1 | A acvram | 7 | 311,73 | 310,0 | 0,7878 | 2 670 | 316,6 |
| 2 | B Rust | 5 | 307,74 | 250,1 | 0,9814 | 2 671 | 318,7 |
| 3 | B Rust | 5 | 307,71 | 250,0 | 0,9853 | 2 671 | 320,3 |
| 4 | A acvram | 7 | 311,79 | 310,1 | 0,7890 | 2 670 | 318,3 |
| 5 | A acvram | 7 | 311,72 | 310,1 | 0,7852 | 2 671 | 316,4 |
| 6 | B Rust | 5 | 307,74 | 250,1 | 0,9854 | 2 671 | 320,1 |
| 7 | B Rust | 5 | 307,72 | 250,0 | 0,9828 | 2 671 | 320,7 |
| 8 | A acvram | 7 | 311,74 | 310,1 | 0,7857 | 2 671 | 318,0 |

Passes courtes (128 jetons, préfill compris) : Rust 109,9 contre acvram 304 t/s.

## Lecture

* La prédiction 180-215 supposait qu'un pas eager paie ~432 lancements hôte non recouverts. Mesuré : 3,25 ms/pas
  Rust contre 3,21 acvram — les lancements sont recouverts par l'exécution (file asynchrone, la carte ne
  s'arrête pas) ; à b=1 sur ce modèle, le graphe CUDA ne rapporte que 0,04 ms/pas (1,3 %). La suite prévue (graphes
  côté Rust, 220-235 t/s) n'a plus d'objet sous cette forme : il reste au plus 1,3 % à gagner.
* Welch t = −257 : σ par lot ~0,05 t/s, l'écart de 4 t/s est systématique, pas du bruit.
* Le +25 % de J/jeton ne vient PAS du décodage (même vitesse, même puissance) mais du **préfill jeton par jeton**
  du Rust v1 (256 pas au lieu d'un préfill groupé) : il pèse dans chaque lot (250 contre 310 t/s préfill compris)
  et domine les passes courtes (110 contre 304). Levier Rust suivant, s'il y en a un : le préfill groupé.
* Bridage « puissance » (plafond 400 W, ~318 W moyens lus) commun aux deux bras à horloge égale : comparable.

## Défauts trouvés en route (corrigés, commités avant la prise jugée)

1. `.gitignore:17` `*token*` excluait `moteurs/acvram_rust/src/tokeniseur.rs` : la crate commitée ne compilait pas
   (c1bd0ad0).
2. Argmax NVRTC en PTX 13.4 refusé par le pilote 595 (CUDA_ERROR_UNSUPPORTED_PTX_VERSION) → SASS sm_120 (b94c2085).
3. Serveur : `generer_flux` appelé d'un fil `spawn_blocking` sans contexte CUDA courant → INVALID_CONTEXT sur tout
   le bras B (1re prise, 01:34-01:49, invalide, archivée `prise1-invalide/`) → `bind_to_thread` (73c02b20).
4. Banc : acvram met `usage` sur chaque fragment SSE, la branche horodatée n'était jamais atteinte (décodage seul = 0
   au bras A) → horodatage hors de la chaîne de comptage (73c02b20). Garde « 8 jetons » ajoutée avant chaque fenêtre.
