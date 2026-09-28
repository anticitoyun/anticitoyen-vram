# poste5 — pièce 294 (bd 8zy) : scellé, Devstral 24B 7,8 tok/s — écrit AVANT la prise

## Relevé à sec (fait)
* `/tmp/acvram-serveur.log`, 8 démarrages de `devstral-24b-srcawq-nvfp4` à 32 768 : tous
  « plan réajusté : 10 MLP de plus en RAM hôte (poids réels 14,5 Gio pour 29,5 libres, activations
  de préfill réservées 13,41 Gio) » → « régime DÉGRADÉ — graphes=off (poids en flux) couches_exilées=10/40 »,
  exil annoncé 128,1 ms/jeton de PCIe (1 061 % du pas résident 12,1 ms). Message : `loader.py:1659`.
* Réserve = `_reserve_prefill` (`loader.py:1981`) → `ModelSpec.activations_prefill_bytes` (`config.py:355`) :
  préfill non découpé pour un modèle DENSE, 3·I·2 + I·4 octets/jeton de MLP (I = 32 768) = 86 % du terme.
* « Bissection » à sec (`scratchpad/…/reserve.py` hors dépôt, CUDA_VISIBLE_DEVICES="") : réserve à 32 768 =
  **13,43 Gio au 19/09 (d566ba14e)** contre 13,74 sur main ; 8 192 : 4,71 / 5,02 ; 16 384 : 7,61 / 7,93.
  L'exil à 32 768 existait donc déjà au 19/09 ; le lanceur du 20/09 passait déjà `--max-model-len 32768`.
  Le « 50* » de la fiche (présent dès `notes-modeles.tsv.bak-19-09`) n'a pas pu être mesuré dans ce régime.
* À côté : port Marlin en échec de compilation dans le venv du paquet (nvcc cu13, python 3.14) → « REPLI au
  naturel » ; pas la cause de l'exil (démarrage l.1054 exilé sans ce message).

## Prédiction (prise `scratchpad/poste5-p294-28-09/prise.sh`, paquet servi, b=1, banc-moteurs --gen 128 × 3)
* ctx 16 384 et 8 192 : aucun « plan réajusté », régime sans « DÉGRADÉ », graphes on ; ≥ 40 tok/s (médiane).
* ctx 32 768 (témoin, même prise) : exil 10/40, DÉGRADÉ, 6-10 tok/s.

## Seuil (fixé avant)
* VRAI si 16 384 ET 8 192 sans exil et médiane ≥ 4 × témoin.
* FAUX (a) si 16 384 exile : mon modèle de capacité est faux, la limite de contexte est à chercher plus bas.
* FAUX (b) si pas d'exil mais médiane < 2 × témoin : l'exil n'est pas la cause (Marlin naturel, ngram, autre) → bissection sur carte.
* Alarme : témoin ≥ 30 tok/s → je ne mesure pas le régime de poste2, cellule invalide.

## Suite si VRAI (décision chef)
(1) Correctif lanceur : contexte servi de Devstral 32 768 → le plus grand sans exil (à borner par l'outil, pas à la main) ;
(2) Correctif moteur : MLP dense du préfill par tranches au-delà du tenu d'un seul tenant (même règle que d19 GDN/MoE),
réserve 32 768 ≈ 13,4 → ≈ 4,5 Gio ; test au bit sous le seuil + test cassant. (3) Marlin du paquet : pièce à part.
