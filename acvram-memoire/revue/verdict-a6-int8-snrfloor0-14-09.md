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

## Addendum 14/09 — cause isolée : ce n'est PAS l'OOM

Consigne de chef après le premier verdict : séparer les messages de
`_try_build_stacks()` et confirmer la cause. Fait (30 min) :

- `acvram/engine/model.py` : chaque branche de retour `None` dans `one()`
  pose désormais `self._raison_repli` avant de sortir (échelle AWQ
  hétérogène / forme-padding différents / formats mélangés / INT4 exilé
  sans table / OOM transitoire — cinq causes distinctes, un seul message
  générique avant ce correctif). `_try_build_stacks()` imprime
  `[acvram] repli lent (pas de graphes, boucle par expert) sur cette
  couche : <raison>` dès qu'une couche échoue, plutôt que de rester
  silencieuse jusqu'à un éventuel message de `graphs.py` — un utilisateur
  qui sert un modèle et tombe à 47 t/s sans le savoir, c'est pire qu'un
  refus (chef).
- `acvram/engine/graphs.py` : `self.raison` reprend `mod._raison_repli`
  au lieu du texte générique fixe.
- **Vérifié empiriquement en rechargeant `-sf0`** (pas de reconstruction
  nécessaire, juste le chargement) : le message précis rendu est
  **« gate_proj : échelle AWQ posée sur certains experts seulement (pas
  tous — repli par expert) »**, répété sur les 48 couches. Le message
  OOM dédié (« mémoire GPU insuffisante ») n'apparaît dans AUCUN des deux
  journaux (avant ou après le correctif) — **ce n'est pas l'OOM**, confirmé
  deux fois.

**Mécanisme complet** : sans l'échappatoire int8 (`snr_floor=0`), le
convertisseur AWQ pose une échelle par canal sur CERTAINS experts
seulement (ceux qui en ont besoin pour rester sous le seuil de qualité),
pas sur les autres — cassant l'hypothèse d'homogénéité que la pile
groupée exige (`p.scaler is not None and not p.scaler.is_identity`,
model.py). Le dossier `snr_floor=25` n'a pas ce problème parce que les
tenseurs difficiles partent en int8 (uniformément absents de la
condition NVFP4) plutôt que de recevoir une échelle AWQ non-uniforme.

**Piège méthodologique trouvé en chemin** : `outils/campagne-a6-
snrfloor0.py` n'insérait pas la racine du dépôt dans `sys.path` — un
`import acvram` y résolvait vers le dépôt DE BASE (installation editable
partagée entre worktrees) au lieu de CE worktree. Sans effet sur les
mesures PPL/débit déjà publiées (le moteur n'avait pas encore changé au
moment de ces mesures), mais aurait rendu invisible tout correctif
ultérieur sans qu'aucune erreur ne le signale — corrigé.

Nettoyage : dossier `-sf0` (17 Gio) supprimé du HDD. Source bf16
(`models/Qwen3-Coder-30B-A3B-Instruct`, 57 Gio) conservée pour l'instant
— à confirmer si elle doit disparaître aussi.

## Ce qui reste

1. **A7** (alpha AWQ commun gate/up, +1,7-1,9 pt, 0 octet) — priorité
   suivante, à sec.
2. Kimi-Linear/Ornith non testés — le mécanisme trouvé (échelle AWQ
   hétérogène) est spécifique aux architectures MoE à experts groupés ;
   les hybrides linéaires n'ont pas cette structure de pile, donc rien
   n'indique qu'ils seraient touchés de la même façon, mais rien ne le
   garantit non plus sans mesure.
3. Piste d'amélioration signalée par chef si le mécanisme se reproduit
   ailleurs : construire les piles par sous-groupe d'experts homogènes
   plutôt que tout-ou-rien, pour ne PAS perdre les graphes CUDA sur une
   poignée d'experts à échelle AWQ non-uniforme.
