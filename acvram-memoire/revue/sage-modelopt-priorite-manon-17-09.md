# Sage — ModelOpt denses (Qwen3.8-27B, Gemma-4-31B) : oui, en tête pour Manon — mais le premier geste est à sec et il tue ou sauve les deux conversions (17/09)

Entrée : Jérôme — conversions jamais lancées, TRT-LLM (Laure) en attente ; 70B clos, exil disculpé (`verdict-exil-ppl-17-09`, 02b316d : R = E1 = E2 = E3 = E4 au bit — **ma prédiction E1 ≠ R est réfutée**, à garder dans mon carnet ; le +89 % du 70B reste ouvert côté conversion srci1 ou calcul dense, pas côté exil).

## Décision : oui, tête de file de Manon. Trois raisons

1. C'était déjà la ligne 7 de `sage-ordre-carte-17-09` (Manon 2 × 1 h, puis Laure 2 × 40 min), dernière du plan : tout ce qui la précédait est clos.
2. **TRT-LLM est dans l'objectif du projet et n'a aujourd'hui aucune entrée classée** (Coder : checkpoint communautaire non classé ; ModelOpt propre = refus documenté). Une revendication « plus performant que TRT-LLM » sans une seule ligne mesurée n'est pas une revendication.
3. La carte est libre : rien d'autre en file.

## Mais : le blocage connu s'applique aux deux denses — à lever à sec d'abord (Manon, ≤ 2 h, zéro carte)

`verdict-modelopt-coder-tf4-17-09` : la calibration a tourné (64/64, 14 min 44 s, `mto.save()` 24,3 Gio écrit) et **l'export a cassé à `nvfp4_tensor.py:84` (`get_weights_scaling_factor`) parce que les couches déportées sur CPU par `--use_seq_device_map` y sont restées** — le modèle bf16 ne tenait pas dans 32 Gio. Qwen3.8-27B bf16 = 54 Go, Gemma-4-31B = 62 Go : **même déport, même ligne, même échec**, prédit avant de lancer. Lancer 2 × 1 h de carte pour le revoir serait la faute que REGLES § 3 interdit.

Geste 1 (à sec) — **rendre l'export indifférent au déport** : calculer l'échelle tenseur par tenseur en déplaçant chaque poids sur `cuda` le temps du calcul (correctif local dans le venv ModelOpt, ou boucle d'export qui `.to("cuda")` avant `get_weights_scaling_factor` et rend après ; fichier:ligne dans le verdict). Contrôle qui rend faux, sans carte : le jouet du verdict précédent, chargé avec un `device_map` qui met explicitement la moitié des couches sur `cpu` (le déport ne s'est pas déclenché parce que le jouet était petit — on le force). Export réussi ⇒ geste 2 ; échec ⇒ « refus ModelOpt, 3e blocage fichier:ligne », TRT-LLM reste « non mesuré » et Manon passe à autre chose. Bonus si ça passe : **l'état Coder de 24,3 Gio est encore sur disque** — un export sans recalibration (Coder MoE reste bloqué par `unified_export_hf.py:419-422`, donc seulement si transformers 4 ne l'atteint pas ; pas prioritaire).

Geste 1 bis (à sec, 15 min, en parallèle) — **TRT-LLM connaît-il ces architectures ?** `qwen3_5` (GDN hybride) et `gemma4` dans le registre de modèles de la version installée par Laure, fichier:ligne. Absente ⇒ pas de conversion pour ce modèle : une conversion ModelOpt sans moteur qui la lit est un fichier de 15 Go pour rien. Ma prédiction : Gemma-4 oui, Qwen3.5-GDN incertain (l'ordre ci-dessous en tient compte).

## Geste 2 (carte, seulement après geste 1 vert)

Ordre : **Qwen3.8-27B d'abord** (acvram y a une ligne à 1,0253 géo, la comparaison a un sens), Gemma-4-31B ensuite. Scellé par conversion : chargement + calibration 64 échantillons + export **≤ 45 min** (Coder : 5 min 40 + 14 min 44 + export ; dense plus petit) ; `mto.save()` avant l'export, toujours ; sha256 du corpus de calibration et du checkpoint dans le verdict ; corpus = le bras A (EN 16 k) de `sage-calibration-verdict`, pas les six phrases de `collect.py`. Faux si > 60 min ou export rouge : refus documenté, pas de 2e tentative sans cause nommée.

Puis Laure, TRT-LLM (2 × 40 min, protocole du comparatif : PPL géo 3 tranches + b=1 + b=12 + prefill + J) — ligne des menus le jour même.

## Gemma-4-31B côté acvram

`verdict-palier1-bloc3` : acvram nvfp4 = « OOM réel » — **avant les quatre correctifs du budget d'exil** (1aa767b … 7ca777d). Une colonne TRT-LLM sans ligne acvram en face ne compare rien : rejouer le chargement Gemma-4-31B acvram (Laure, 5 min, `bloc.sh` entrée seule) avant son tour TRT-LLM. Prédiction : charge en exil ≤ 10/62, b=1 publié. Faux ⇒ OOM à nouveau, cause à nommer (pas un 5e correctif à l'aveugle).
