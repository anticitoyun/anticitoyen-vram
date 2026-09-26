# Verdict — 123-quater : critère relatif S1b/i8c sur invites neuves (dernière variante) — 24/09 01 h 3x (poste1)

* **instrument** : `scratchpad/poste1-123-24-09/prise-123q.sh` — `kl-123.py` (S1b témoin `ACVRAM_MOE_TENSOR=0`), `verif-123ter.py` (crochet même entrée, pire rapport par invite et par godet) sur S1b et i8c servis ; références HF neuves `hf-lot-123q.py` (prise 01:15:09 → 01:27:56, 6/6 invites valides, retenues invite5 à invite9, sha256 dans `dumps-neufs.sha256`)
* **commit** : 2ebdda72 (poste1-mtp)
* **régime** : S1b `…-assemble-S1b-proj-tete-i8-23-09`, i8c `…-nvfp4-qkvo-i8c` ; horloge libre (justesse) ; cpu-safe=off (max_perf_pct 100 au début et à la fin)
* **scellé** : `scratchpad/poste1-123-24-09/scelle-123quater.md` (commit 707a958a, écrit avant les références et la mesure) — (1) rapport S1b ≤ 1,25 × i8c par invite et par godet ; (2) ΔKL ≤ +0,025 par invite à b=1 et b=12 ; (3) T=1 au bit ; (4) aligneur@T8+ atteint ; FAUX si un seul tombe ; aucune troisième variante
* **mesuré** :
  * **b=12** (décodage à 12, T = 12) : rapports S1b / i8c par invite = 0,01282 / 0,01399 (**0,92**), 0,01053 / 0,01000 (**1,05**), 0,00880 / 0,00933 (**0,94**), 0,01741 / 0,01267 (**1,38**), 0,01290 / 0,00851 (**1,52**) ; ΔKL par invite 0 / **+0,0371** / +0,0003 / 0 / −0,0018 ; aligneur@T8+ atteint.
  * **b=1** : T = 1 au bit (tous les appels) ; ΔKL 0 sur 5/5 ; mais **aucun appel T ≥ 8 sur le chemin tensor** — les invites neuves font plus de 32 jetons, et leur préfill passe par `_forward_prefill_grouped` (au bit dans les deux alias, rapport 0) ; aligneur@T8+ absent à b=1. Les invites de la 123-ter (T = 30/32) y passaient.
  * Agrégat du script : « NON MESURE (configuration) », à cause du critère (4) appliqué globalement alors qu'il manque seulement à b=1.
* **verdict** : **FAUX** — à b=12, là où le chemin AWQ-tensor est prouvé atteint, le critère (1) tombe sur 2/5 invites (1,38 et 1,52 > 1,25) et le critère (2) sur 1/5 (+0,0371 > +0,025). La partie b=1 n'apprend rien (tensor non exercé) et ne peut rien racheter. Selon l'ordre du chef, il n'y aura pas d'autre variante : **la 123 reste hors défaut ; S1b est servi par le GEMV** comme sur main. Le code de la 123 (branche poste1-mtp) ouvre le tensor aux tables AWQ PAR DÉFAUT : il ne fusionne pas tel quel. Au plus, il entre en opt-in (variable documentée, défaut = refus actuel), avec le correctif du pas de ligne (123-bis, juste) et ses tests — au chef.
* **durée** : prise 01:28 → 01:30:59 (verrou) ; compute-apps début = fin (llama-server 4627 seul)

Note d'instrument : le critère (4) du scellé aurait dû être posé PAR GODET (un godet où le chemin n'est pas atteint = non mesuré pour ce godet, pas pour la prise). Ça ne change pas le verdict, qui tombe sur le seul godet mesuré.
