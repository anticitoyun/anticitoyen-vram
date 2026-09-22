# diag-eval-nll, bras hf sur Gemma 4 31B — ÉCHEC OOM, acvram non libéré avant le sous-processus hf — 22/09 (poste2)

* instrument : `outils/carte.sh env HF_PYTHON=/opt/ia/vLLM/.venv/bin/python MAXMEM=26GiB,80GiB .venv/bin/python outils/gpu/mesure/diag-eval-nll.py gemma-4-31B-it-nvfp4-vision --jetons 300 --source .../gemma-4-31B-it-bf16`
* commit : main à jour
* mesuré : bras eval/serve identiques au rejeu précédent (PPL 227232,509/227215,62, `|Δ| max 0,0062`) ; **le bras hf plante** : `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.51 GiB. GPU 0 has ... 2.89 GiB is free. Process 1668923 has 27,96 GiB memory in use.`
* cause exacte : `diag-eval-nll.py:122` lance le bras hf en **sous-processus** (`subprocess.run([py, s, source, i, maxmem, o], ...)`), correctement isolé côté processus — mais le **processus principal garde le modèle acvram (28 Go) chargé en VRAM** pendant que ce sous-processus tente de charger le 31B bf16 offload sur la MÊME carte physique. `MAXMEM=26GiB,80GiB` était pourtant posé, mais il ne reste que 2,89 Gio de libre réel — le calcul d'offload du bras hf suppose 26 Gio disponibles qui n'existent pas tant qu'acvram occupe la carte. Aucune libération (`del`/`torch.cuda.empty_cache()`) du modèle acvram avant l'appel au sous-processus hf.
* verdict : **ÉCHEC — cause nommée, non corrigée**. Le bras hf ne peut pas tourner après eval/serve sans libérer d'abord la VRAM tenue par acvram. P1/P3 restent non tranchés sur Gemma 4.
* durée : ~2 min de carte (échec après le chargement eval/serve)

## Suite
Correctif suggéré, non appliqué (script frais d'poste1, pas mon domaine) : libérer le modèle/moteur acvram (`del` + `torch.cuda.empty_cache()`) juste avant `subprocess.run(...)` du bras hf, ou réduire `MAXMEM` à ce qui reste réellement libre après acvram (~2-3 Gio, insuffisant pour un 31B même très offload — il faudrait de toute façon libérer acvram d'abord). Sans ce correctif, P1/P3 sur Gemma 4 restent indécidables par cet instrument.
