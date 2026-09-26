# Verdict — pièce 178 : OOM du chemin nu (chunk fla, gdn.py:253) — poste3, 25/09

Constat de poste5 (172) : sur Qwen3.8-27B, `max_model_len=8192`, `model(batch)` nu tombe en OOM
dès 4 096 jetons dans le chunk fla, alors que le moteur servi tient 8 000.

## (1) Chemins « nus », à sec

* `acvram/evaluate.py:335` — `h = model(batch, return_hidden=True)` (`acvram eval`, PPL par fenêtre
  glissante). `load_model(..., max_model_len=window + BLOCK_SIZE, max_concurrent_seqs=1)` (:243).
* `scratchpad/poste1-p129-24-09/kl-chemins.py:34` — `loaded.model(eng._build_batch(seqs, prefill=True))`.
* `scratchpad/poste5-p172-25-09/pic.py` — l'instrument de poste5 lui-même, `model(batch)` nu,
  **`load_model(..., max_model_len=8192, max_concurrent_seqs=8)`** (ligne 23).

## (2) Mesure — pic mémoire par étape, nu contre servi

Prédiction (avant mesure) : écart > 5 Gio dès T=4096 entre les deux chemins, sinon aucune des
pistes nommées (no_grad/inference_mode, return_hidden, allocation KV, tête, dtype) n'explique
l'OOM. Script : `scratchpad/poste3-p178-25-09/pic-nu-vs-servi.py`.

| T | chemin NU (`max_concurrent_seqs=1`) | chemin SERVI (`Engine`, prefix cache) |
|---|---|---|
| 2048 | 17 799,4 MiO | 17 872,8 MiO |
| 4096 | 18 683,1 MiO | 18 756,2 MiO |
| 8000 | 20 374,5 MiO | 20 524,3 MiO |

**Prédiction FAUSSE** : écart < 100 Mio à chaque longueur, aucun OOM des deux côtés. Ma
reconstruction du chemin nu (`max_concurrent_seqs=1`, comme `evaluate.py` réel) ne reproduit PAS
l'OOM de poste5.

## Hypothèse de chef, vérifiée à sec : confirmée

`scratchpad/poste5-p172-25-09/pic.py:23` : `load_model(chemin, dtype=torch.bfloat16,
max_model_len=8192, max_concurrent_seqs=8)` — SON instrument de diagnostic (pas `evaluate.py`,
pas `kl-chemins.py`) charge le modèle avec un budget KV dimensionné pour **8 séquences
concurrentes** à 8 192 jetons chacune, alors qu'un seul `model(batch)` nu est ensuite appelé. Le
commentaire du fichier le dit lui-même (ligne ~53) : « 1re prise (01:4x) : 1 × 8 192 en OOM dès le
bras A (sans B'), fla chunk (w = k.new_empty, 192 Mio, 152 libres) ». `evaluate.py`, lui, pose
`max_concurrent_seqs=1` (`:243`) — le vrai chemin nu de PPL ne réserve jamais ce KV surdimensionné.

## Verdict

**Artefact de montage, ni bug du moteur ni limite d'instrument.** L'OOM observé par poste5 vient
du réglage `max_concurrent_seqs=8` de SON script de diagnostic (`pic.py`), pas d'un défaut du
chemin nu réellement utilisé par les instruments de mesure (`evaluate.py`, `kl-chemins.py`, tous
deux à `max_concurrent_seqs=1`). Rien à corriger dans `acvram/` — le chemin nu se comporte comme
le chemin servi à budget KV égal (tableau ci-dessus, écart < 100 Mio à toutes les longueurs
testées).
