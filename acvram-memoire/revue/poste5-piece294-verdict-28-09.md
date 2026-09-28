# poste5 — pièce 294 (bd 8zy) : verdict — Devstral 24B à 7,9 tok/s = exil de 10 MLP à 32 768, pas une régression de code

* instrument : `~/.local/bin/banc-moteurs --gen 128 --seul 8090` × 3 (celui de la campagne 290), b=1, ngram ; `scratchpad/poste5-p294-28-09/prise.sh`
* commit : d2529ad70 (poste5-294) ; moteur = paquet servi `/usr/bin/acvram` (celui que poste2 a mesuré)
* régime : carte 0 seule, horloge du poste, compute-apps vides au début et à la fin ; campagne 290 en pause par drapeau
* scellé : `revue/poste5-piece294-scelle-28-09.md` (c04fe5cf) — VRAI si 16 k et 8 k sans exil et ≥ 4 × témoin
* mesuré : 16 384 → 94,6 / 94,6 / 94,5 tok/s, NOMINAL, graphes on, 0/40 exilées ; 8 192 → 94,5 × 3, NOMINAL ;
  32 768 (témoin) → 7,9 × 3, DÉGRADÉ, 10/40 MLP en RAM hôte, graphes off
* verdict : **VRAI** (12,0 ×). Aucun alarme (témoin 7,9 < 30).
* durée : prévue ≤ 15 min ; tenue 10:29:58 → 10:32:57 (3 min)

## Cause (fichier:ligne)
`loader.py:1604` — la marge de `_reajuster_plan` comprend `reserve` = `_reserve_prefill` (`loader.py:1981-2007`) →
`ModelSpec.activations_prefill_bytes(32768)` (`config.py:355`) : préfill d'un seul tenant d'un modèle DENSE, MLP
(3·I·2 + I·4 octets/jeton, I = 32 768) = 86 % du terme → 13,41 Gio réservés → 10 MLP exilés (`loader.py:1659`), un
seul poids en flux coupe les graphes de tout le modèle → 7,9 tok/s.
Pas de régression : réserve 13,43 Gio au 19/09 (d566ba14e) contre 13,74 sur main (calcul à sec, même manifeste) ;
lanceur à `--max-model-len 32768` déjà le 20/09. Le « 50* » de la fiche n'a pas été mesuré sous ce régime ; ce que
Devstral sert sans exil : 94,5 tok/s.

## Correctifs possibles (décision chef, rien n'est codé)
* **A — contexte servi** : colonne 3 de `~/TSV/acvram-chemins.tsv` 32 768 → 16 384 (prouvé 94,5, 0 exil) ou le plus
  grand sans exil (≈ 26 k estimé à sec, non mesuré). Sortie inchangée ; invites > ctx refusées (400 nommé).
* **B — moteur** : MLP dense du préfill par tranches (même règle que d19 GDN/MoE : engagé seulement au-delà du tenu
  d'un seul tenant) et réserve calculée sur la tranche → 32 768 sans exil (réserve ≈ 4,5 Gio). Au bit sous le seuil,
  ± ulp au-delà (comme d19) ; test au bit + test cassant sur la réserve (un dense à 32 k ne doit pas exiler).
* **C — garde générale** : au chargement, exil annoncé > 20 % du pas (« falaise », déjà calculé et imprimé `loader.py:1718`) → contexte
  réduit au lieu d'exiler, sauf opt-in. Touche tous les modèles : décision.

## Hors mandat, relevé
Port Marlin du paquet : compilation échouée (nvcc cu13, venv python 3.14), « REPLI au naturel » à chaque chargement ;
sans effet sur l'exil ni sur 94,5 ; pièce à part (non compilé ici pour ne pas charger le processeur pendant la 290).
