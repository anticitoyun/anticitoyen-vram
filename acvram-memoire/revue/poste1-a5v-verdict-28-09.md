# a5v — MLP dense par tranches : Devstral 24B servi à 34 816 sans exil, 94,5 tok/s au lieu de 7,9 (poste1, 28/09)

* instrument : `scratchpad/poste1-a5v-28-09/` — `egalite.py` (logits de la dernière position + 16 jetons, Engine direct),
  `nll-kl.py` (d19 + seuil MLP forcé après le chargement), `prise-1.sh`, `prise-2.sh` (serve + `banc-moteurs --gen 128` × 3,
  instrument de la pièce 294), `plan-a-sec.py`
* commit : poste1-a5v 53466ec68 (prise 2), b17901de4 (prise 1c ; seul `egalite.py` change) ; main bc13a2bb8 (rien sous `acvram/` depuis 63e9ca0dc)
* régime : carte 0, venv de dev 3.12 pour les deux arbres, maxperf 100, compute-apps vides au début et à la fin ; serve
  `--speculative ngram`, graphes on ; équivalences et KL : Engine direct, sans graphes ni cache de préfixe
* scellé : `scratchpad/poste1-a5v-28-09/scelle.md` (écrit avant les prises)
* mesuré : tables ci-dessous
* verdict : **VRAI pour le service** (0 exil à 34 816 et 32 768, débit ×12, au bit sous le seuil sans exil) ; qualité au-delà
  du seuil : KL 2,5e-4, top-1 99,28 %, dans le critère FAUX (KL ≤ 1e-3, ΔNLL < 1 SE), mais **hors de la bande prédite**
  (KL ≤ 1e-4, top-1 ≥ 99,5 %). Trois autres chiffres tombent hors de leur bande, tous dans le bon sens (voir la table).
* durée : prise 2 prévue ~15 min, tenue 433 s ; prise 1 (1c) prévue ~15, tenue 300 s (1 et 1b : défauts d'instrument, voir plus bas)

## Correctif
* `config.py:ModelSpec.mlp_prefill_plafond` + `activations_prefill_bytes` : le terme MLP dense (3·I·2 + I·4 o/jeton,
  86 % de la réserve de Devstral) s'arrête au plafond. Sans plafond, la réserve est identique à l'octet (test).
* `loader.py:_plafonner_mlp_prefill` : seulement pour un modèle DENSE pur (ni MoE, ni GDN) dont la réserve d'un seul tenant
  exile des MLP. On y cherche le plus grand plafond (multiple de 1 024, ≥ 4 096) qui n'exile pas plus qu'un plafond minimal.
  Tout modèle qui tenait garde son plan, sa réserve, son KV et sa sortie. Le seuil MLP est posé au chargement (Engine sans
  chauffe).
* `attention.py:MLP.forward` : tranches de `ACVRAM_MLP_MORCEAU` (4 096 ; 0 = témoin) au-delà de `_MLP_SEUIL`.
* `contexte.py` : la chauffe d'un modèle dense pur remplace le seuil par le tenu d'un seul tenant qu'elle a PROUVÉ (d19).
* Tests : `tests/test_mlp_tranches_a5v.py` (7 tests, rouges sur main) ; 136 verts sur les fichiers concernés.

## Prise 2 — service (Devstral, `banc-moteurs --gen 128`, port 8090)
| arbre, contexte | plan | chauffe | régime | tok/s (× 3) | prédit |
|---|---|---|---|---|---|
| branche 34 816 | seul tenant : 13 MLP exilés → plafond 21 504, **0 exilé**, activations 8,60 Gio au lieu de 12,66 | tenu 34 816, `tranches>32768` | NOMINAL, graphes on | 94,5 / 94,7 / 94,7 | 0 exil, tenu ✔ ; 88-96 ✔ ; N 17-24 k ✘ (32 768, plus haut) ; 9-11,5 Gio ✘ (8,60, plus bas) |
| branche 32 768 | 10 → plafond 21 504, **0 exilé**, 8,50 Gio au lieu de 11,94 | 32 768 d'un seul tenant (aucune tranche) | NOMINAL | 94,5 / 94,6 / 94,6 | ✔ ; N sans objet (aucune tranche servie) |
| main 32 768 (témoin) | **10 MLP exilés**, 13,41 Gio | 32 768 | DÉGRADÉ, graphes off | 7,9 × 3 | ≥ 6, < 20 ✔ |

La réserve estimée est prudente : la chauffe prouve un seul tenant jusqu'à 32 768, au-delà du plafond de 21 504. En service,
seules les invites entre 32 768 et 34 816 prennent les tranches ; en Engine direct (sans chauffe), c'est au-delà de 21 504.

## Prise 1 — équivalences et qualité (Engine direct)
| contrôle | mesuré | scellé |
|---|---|---|
| (1 bis) main = branche, mml 16 384, invite 16 000 (pas de plafond) | logits et 16 jetons ÉGAUX, max 0,0 | TENU |
| (1) main = branche, mml 32 768 (main exile, pas la branche), invite 16 384 < seuil | logits ≠ (max 1,9e-6), 16 jetons égaux | écart attribué à l'exil de main (poids en flux), pas aux tranches : sans exil (1 bis), au bit. Preuve par exclusion, non mesurée à invite égale |
| (1 ter) mml 32 768, invite 30 720 > seuil (tranches dans la branche) | max \|Δlogit\| 0,138, 16 jetons égaux | information ; prédit ≤ 1 ulp : ✘ |
| (2) MLP forcé 4 096 contre seul tenant, 2 × 16 384, corpus 5909d27 (76088761d41b2abc) | PPL 10,0599 → 10,0587, ΔNLL −0,00012 ± 0,00013 ; KL moyen **2,46e-4** (max 0,85) ; top-1 **99,28 %** ; témoin KL 0 exact | FAUX non atteint (KL ≤ 1e-3, ΔNLL < 2 SE) ; bande prédite ✘ (KL 0 ou ≤ 1e-4, top-1 ≥ 99,5 %) |

## Lecture
* Les GEMM du MLP dense ne sont PAS invariantes en M à 4 096 contre 16 384 (sinon KL 0) : mon analogie avec le routeur de
  d19b était fausse. L'écart reste 30 fois plus petit qu'en d19 (7,9e-3), et le top-1 passe le seuil de 99 % que d19 manquait.
  Cela confirme d19b : c'est le routage top-k du MoE qui amplifie un ulp ; un modèle dense l'absorbe.
* Devstral sous kimi (CTX_CLIENT_MIN 34 816) : servi sans exil. Invites ≤ 32 768 : au bit de main sans exil. Au-delà : ± ulp,
  étiqueté `tranches>32768` sur la ligne de régime.

## Défauts d'instrument (prises 1 et 1b, aucune mesure perdue)
* prise 1 : `egalite.py` tirait des ids jusqu'à 150 009 (Qwen), au-delà du vocabulaire de Devstral (131 072) : assert CUDA
  dans les DEUX bras. prise 1b : ma correction avait mis un commentaire au milieu de la compréhension (SyntaxError). 1c : juste.
* Attente de carte : `carte-libre.sh` compte la campagne 290 en pause comme « une mesure qui démarre » ; elle attendait mon
  drapeau et moi `carte-libre` → interblocage de 12 h 05 à 13 h 47 (bd g2c, poste3). Remède : essayer `carte.sh` lui-même
  (rc 4 = service, on réessaie).

## Leçons
* Un instrument copié d'une famille de modèles porte ses constantes (vocabulaire) : les lire avant de changer de modèle.
* « Invariance en M » ne se transporte pas d'une GEMM à une autre (le routeur 2 048 → 256 l'était, pas le MLP 5 120 → 32 768).
  Mesurer la KL avant de prédire « au bit ».
