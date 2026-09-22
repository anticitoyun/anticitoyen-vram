# Diag — 2e serveur trtllm « jamais prêt » (T2), chaîne énergie (poste3, 22/09)

Symptôme (poste2) : dans la chaîne énergie, le 2e serveur trtllm-serve n'est jamais
prêt (T2 absente aux passages 3, 4, 5). Hypothèses de la chef : reste de VRAM
du 1er (kill trop court), port encore lié, cache de compilation.

## Cause réelle (ce n'est ni la VRAM ni le port)

Preuve par les logs `scratchpad/poste4-b12-21-09/sorties-energie/` :
- `fenetre-trtllm-b12-1.log` (1er bras) : RESULTAT présent, `moteur=acvram` (bras
  llama.cpp/gguf) — réussi, aucun AttributeError.
- `fenetre-trtllm-b12-2.log` (bras trtllm) : **2× `AttributeError: 'list' object has
  no attribute 'get'`**, aucune mesure.

C'est le CLIENT qui crashe, pas le serveur : **`banc-llamacpp-16-09.py:107`** —
`client.get(f"{HOTE}/metrics").json().get("cartes")`. Le serveur trtllm-serve a bien
répondu `/v1/models` (sinon `serveur()` bouclerait sur `/v1/models` sans jamais toucher
`/metrics`), mais son `/metrics` est une **LISTE Prometheus**, pas le dict
`{"cartes": …}` d'acvram → `.get()` sur une liste = `AttributeError` non rattrapée
(`except (httpx.HTTPError, ValueError)` seulement). Le client meurt → le harnais
conclut « serveur jamais prêt ». C'est le 4e défaut déjà corrigé dans la copie
`scratchpad/trtllm-cellules-22-09/banc-llamacpp-16-09.py`.

Rien à voir avec la VRAM, le port ou le cache : le 1er bras (acvram) marche parce que
`/metrics` d'acvram est un dict ; tout bras trtllm crashe, quel que soit le passage.

## Correctif (borné, appliqué)

`banc-llamacpp-16-09.py:110` (copie racine `scratchpad/`) : `except` élargi à
`(httpx.HTTPError, ValueError, AttributeError, TypeError)` → un `/metrics` non-acvram
est ignoré, `CUDA_VISIBLE_DEVICES` reste 0. Aligné sur la copie déjà patchée de
`trtllm-cellules-22-09/`. **Une seule copie patchée** désormais dans ce worktree ;
les autres worktrees (chaîne énergie) doivent reprendre cette version.
Sortie inchangée sur les bras acvram (dict → chemin d'avant).
