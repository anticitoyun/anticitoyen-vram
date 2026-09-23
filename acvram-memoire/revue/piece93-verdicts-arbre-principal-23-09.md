# Pièce 93 — verdicts citant les cinq outils de outils/gpu/mesure depuis le 19/09 (à sec)

Instrument : `git log --before=<horaire du verdict>` sur `main` + `git diff --stat <branche> <main>` restreint à `acvram/` et `kernels/`. Le déplacement des outils dans `outils/gpu/mesure/` (commit `b6966060`, 19/09 18:27) date le début du périmètre : toute citation antérieure ne peut pas relever du bogue `0a5fc353` (racine à deux niveaux) puisque le fichier n'existait pas encore à ce chemin.

| Verdict | Outil cité | Horaire | Branche mesurée | Main à cet horaire | Écart de code (acvram/, kernels/) | Conclusion |
|---|---|---|---|---|---|---|
| `verdict-c14c-19-09.md` | capture-godets | 19/09 23:19 | oceane-c14c-grille-170 `a6e6a98b` | `28441096` | **oui** — `mla.py` +122/−, `model.py` +98, `acvram_kernels.cu` 257 l., `route_prep.py` | **à rejouer** (C14c porte justement sur la MLA touchée par l'écart) |
| `verdict-c17-19-09.md` | capture-godets | 19/09 22:32 | manon `791be07` | `cadfb52b` | **oui** — `mla.py` 33+/19− (chemin décodé mesuré), `runner.py`, `regime.py` | **à rejouer** |
| `verdict-c15-prefill-19-09.md` | capture-godets | 20/09 07:55 | manon-prefill `14d66909` | `3e1215a9` | **oui** — `model.py` −230 l., `kernels/__init__.py`, `acvram_kernels.cu`, `gemm_w8a8.py` (chemin prefill directement touché) | **à rejouer** |
| `verdict-c14-etape5-19-09.md` | capture-godets | 19/09 21:53 | oceane C14 `c810614` | `9983bec1` | **oui** — `model.py` +62, `acvram_kernels.cu` +132, `marlin_port/disposition.py` nouveau | **à rejouer** |
| `verdict-c5b-19-09.md` | capture-godets | 20/09 01:10 | manon-c5b `a77eb722` | `d2ab8e62` | **oui, majeur** — `kv_canal.py` supprimé (270 l.), `kvcache.py`, `attn_paginee.py`, `gemm_etroit.py` : exactement le format KV canal que C5b mesure | **à rejouer** (le plus exposé) |
| `verdict-c4-godets-19-09.md` | outils/gpu/mesure/capture-godets.py | 19/09 19:50 | manon-c4 `fffd17c` | `11604367` | **oui** — `regime.py` +133/−, `kernels/__init__.py` +67, `runner.py`, `evaluate.py` | **à rejouer** |
| `verdict-capture-parc-19-09.md` (addendum 20/09 07:08) | capture-godets.py | 20/09 07:08 | oceane-gemma-capture `c48c2b2c` | `f6038992` | **oui** — `graphs.py` −53 l., `model.py` −46, `mla.py` : porte sur le correctif `empty_cache()` justement mesuré | **à rejouer** |
| `verdict-niveau2-tf32-decode-19-09.md` | capture-godets | 19/09 21:41 | `01749c0` | `a54b5fce` | **non** — seul `acvram/__init__.py` diffère (`__version__` 0.6.20→0.6.21, chaîne seule) | **tient** |
| `verdict-c15-niveau3-coder-19-09.md`, prise 01:18-01:29 | capture-godets | 20/09 01:29 | manon-n3 `d2ab8e62` | `48b2b693` | **oui** — `kernels/__init__.py`, `acvram_kernels.cu` 278 l., `attn_paginee.py`, `route_prep.py`, `kvcache.py` | **à rejouer** |
| `verdict-c15-niveau3-coder-19-09.md`, prise 04:49-05:01 | capture-godets | 20/09 05:01 | manon-n3 `6d8f2ff0` | `6d8f2ff0` | **non — même commit** (le worktree était déjà à jour sur main) | **tient** |

## Hors périmètre (non tablé)
* `verdict-c11-19-09.md`, `verdict-p2-equiv-19-09.md`, `verdict-glm-b12-19-09.md` : citent `capture-godets-17-09.py` (scratchpad), pas `outils/gpu/mesure/capture-godets.py` — script différent, antérieur au déplacement.
* `verdict-capture-parc-19-09.md`, prise du corps (03:11-03:23) : horaire antérieur à 18:27, avant l'existence du chemin `outils/gpu/mesure/` — la citation du chemin dans le texte est rétrospective ; script réellement exécuté non confirmable ici, à vérifier par qui a la chaîne (`scratchpad/capture-parc-19-09/chaine.sh`) avant de la classer.
* `chantier-c11-19-09.md`, `chantier-c14c-19-09.md`, `chantier-c5b-19-09.md`, `chantier-eco-defaut-19-09.md`, `sage-eco-2700-defaut-19-09.md` : notes de plan ou de politique, aucune conclusion chiffrée ne repose sur une sortie de l'outil — pas des verdicts.
* `qualite-seuil-gemv`, `seuil-gemv-bout-en-bout`, `tensorcores-contre-gemv`, `defauts-ecrits-comme-mesures` : aucune citation trouvée dans `revue/` en dehors des documents d'inventaire et de la note d'origine (`oceane-piece82ter-w13-decodage-23-09.md`) — pas de verdict à rejouer pour ces quatre.

## Bilan
8 verdicts sur 10 prises à rejouer (code mesuré ≠ code prétendu, sur le chemin même de la mesure) ; 2 tiennent (même commit ou écart non fonctionnel).
