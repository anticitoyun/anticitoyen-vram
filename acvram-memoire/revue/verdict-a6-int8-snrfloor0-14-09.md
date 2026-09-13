# Verdict — item A6 (audit poste7) : reconversion à snr_floor=0 RÉFUTÉE sur Qwen3-Coder-30B-A3B

poste2, 14/09/2026. Suite de `revue/prediction-a6-int8-snrfloor0-14-09.md`.
**Prédiction réfutée, et plus sévèrement que la pire issue envisagée.**

## Régime

Source bf16 téléchargée avec accord explicite (`Qwen/Qwen3-Coder-30B-A3B-Instruct`,
57 Gio, HDD `/mnt/4TO_SATACMR_2022/Modeles/models/`). Reconvertie sans
flag `--snr-floor` (défaut CLI actuel = 0) vers
`models_acvram/Qwen3-Coder-30B-A3B-sf0` (HDD, le SSD est à 41 Gio libres) —
16,5 Gio, 18867 tenseurs, SNR sortie moyen 22,2 dB. Comparée au dossier
existant `Qwen3-Coder-30B-A3B-nvfp4` (SSD, `snr_floor=25`, 193 tenseurs
promus int8).

PPL (`acvram eval`, wiki-gptq.txt, window 2048, min_context 256) et
décodage b=12 (`acvram.engine.runner.Engine`, même protocole que poste3 —
12 séquences, ctx 2048, 200 jetons, `energie.py` NVML monotone) — chaque
mesure dans son PROPRE processus (piège déjà rencontré hier avec le duel
vLLM/acvram : deux `load_model()` dans le même processus laissent le
premier modèle résident et faussent le second — vérifié ici : le
résultat est IDENTIQUE isolé ou non, donc ce n'est pas la cause).

## Résultat

| | PPL | débit b=12 | écart |
|---|---|---|---|
| avant (snr_floor=25) | 9,3369 | 717,6 t/s | — |
| après (snr_floor=0) | 10,1574 | 47,1 t/s | PPL **+8,79 %**, débit **−93,4 %** |

Seuil scellé (PPL ≤+2 % ET débit ≥+5 %) : **DÉPASSÉ**, sur les deux
critères, par une marge énorme. Ni le sens (le débit devait AUGMENTER,
il s'effondre), ni l'ampleur (×15 plus lent, pas ±10 %) ne correspondent
à la prédiction.

## Cause : les graphes CUDA se coupent, pas un effet mémoire-bande-passante

Log du chargement : `[acvram] graphes CUDA désactivés : pile d'experts
hétérogène` (`acvram/engine/graphs.py:259`) — `MoEBlock._try_build_stacks()`
échoue pour au moins une couche, désactivant les graphes CUDA pour tout
le modèle (pas seulement le chemin par expert), forçant un chemin eager
bien plus lent.

**Le message est trompeur** : un balayage complet du manifeste (48
couches × 3 projections × 128 experts, format/shape/padded_in/group_size)
ne montre **aucune hétérogénéité structurelle** — chaque groupe de 128
experts est parfaitement uniforme (nvfp4, même forme, même group_size).
`_try_build_stacks()` peut aussi rendre `False` sur `torch.OutOfMemoryError`
pendant la construction transitoire d'une pile (`convert.py`/`model.py` :
« la pile d'une projection double transitoirement sa mémoire ») — même
message de log dans les deux cas, cause non distinguée. Le dossier `-sf0`
(16,5 Gio, sans les 193 tenseurs int8 qui occupaient une forme différente
et peut-être un espace mémoire différemment fragmenté) semble déclencher
ce chemin, le dossier `-nvfp4` (avec ses promotions int8) ne le déclenche
pas — **cause exacte non isolée davantage, hors du temps imparti à cette
mesure**.

## Conséquence pour le chantier

**Le mécanisme d'A6 ne généralise pas aux MoE** — exactement l'« issue
qui me gênerait » de la prédiction scellée, mais via un chemin non prévu
(pas un simple effet mémoire-bande-passante moins favorable au MoE, une
vraie régression de mécanisme : perte de la capture de graphes). Ne PAS
reconvertir Qwen3-Coder-30B-A3B au parc à snr_floor=0 dans son état
actuel — ce serait un régression x15 en production, pas une optimisation.

**Kimi-Linear et Ornith-1.0-35B restent à tester** (hybrides linéaires,
pas MoE à experts groupés — le mécanisme de pile groupée ne les concerne
probablement pas de la même façon), mais avec une confiance réduite tant
que la cause exacte de cette régression n'est pas comprise : elle
pourrait aussi les toucher si `_try_build_stacks`-like logic ou un
mécanisme voisin existe pour les couches GDN.

## Ce qui reste

1. Isoler la cause exacte (OOM transitoire vs autre) : instrumenter
   `_try_build_stacks()` pour distinguer les deux branches de retour
   `False`, ou reproduire avec plus de VRAM libre / `torch.cuda.memory_stats()`
   autour de l'appel.
2. Si OOM confirmé : le problème n'est pas snr_floor=0 en soi mais un
   pic mémoire lors du chargement — potentiellement réglable (construire
   les piles projection par projection avec libération intermédiaire,
   déjà en partie fait par le `try/except` existant mais qui abandonne
   au lieu de réessayer avec moins de parallélisme).
3. Décider si Kimi-Linear/Ornith valent la peine d'être testés avant de
   comprendre ce mécanisme, ou si A7 (alpha AWQ commun) redevient
   prioritaire étant donné qu'A6 échoue plus cher que prévu sur la cible
   principale.
