# Ce qui est fait et ce qui ne l'est pas

Écrit sans complaisance, parce que le projet que celui-ci remplace annonçait une
extension de mémoire GPU fonctionnelle et livrait un module noyau qui se
contentait d'afficher les nombres passés en paramètres.

## Fait depuis la version 0.1.0

| | ce que ça fait | vérifié par |
|---|---|---|
| Décodage spéculatif | propositeurs n-grammes et modèle brouillon, acceptation exacte | identité des sorties + test de distribution sur 40 000 tirages |
| Cache de préfixe | réutilisation de blocs par hachage chaîné, éviction LRU | identité des sorties, comptage des jetons de prefill |
| Calcul hôte sur processeur | GEMV 4 bits AVX2/scalaire, ctypes, sans dépendance de compilation | numérique face à la référence, compilé et exécuté ici |
| GEMV CUDA réécrit | chargements vectoriels 8 octets, 4 lignes/bloc, découpe sur K | décodage des quartets validé sur processeur ; **noyau jamais compilé** |
| GEMM FP4 tensor cores | chemin `torch._scaled_mm` sondé, avec repli | la sonde rapporte elle-même sa raison dans `acvram doctor` |
| Précision mixte | promotion pilotée par le SNR, plafonnée à 15 % des tenseurs | la promotion élève le SNR mesuré, plafond respecté |
| Harnais de perplexité | `acvram eval`, fenêtre glissante | s'exécute, valeur finie, classe |
| Masque causal décalé | attention correcte pour un prefill par morceaux ou avec cache | démontré différent du drapeau intégré |
| Attention de décodage groupée | un seul appel SDPA par étape au lieu d'un par séquence | égale le résultat séquence par séquence |

Le masque causal décalé mérite une note. `F.scaled_dot_product_attention(
is_causal=True)` aligne son triangle en haut à gauche, ce qui n'est correct que
si la requête couvre toute la séquence. Le cache de préfixe et le prefill par
morceaux produisent tous deux un bloc de requêtes décalé, et le drapeau intégré
aurait masqué les mauvaises cellules — silencieusement, avec une sortie
plausible. Un test affirme que les deux diffèrent.

## Toujours jamais exécuté sur la machine cible

Tout a été écrit et testé sur un portable à i5-3230M et GeForce GT 740M en
pilote 470 : ni CUDA, ni Blackwell, ni Ampere.

* **Les noyaux CUDA n'ont jamais été compilés.** `acvram_kernels.cu` respecte
  par construction la numérique de référence PyTorch, et sa partie la plus
  délicate — le décodage des quartets — a été validée en compilant cette seule
  fonction sur processeur, mais nvcc n'a jamais vu le fichier. Attendez-vous à
  corriger des erreurs de compilation au premier build. Le chemin de référence
  étant numériquement identique, rien n'est cassé entre-temps : seulement lent.
* **Aucun débit n'est mesuré.** Chaque valeur en jetons/s est une estimation du
  planificateur à partir de plaques signalétiques. `acvram bench` les remplace.
* **Le chemin AVX2 processeur n'a jamais tourné.** Il compile, et le chemin
  scalaire qui partage sa structure est testé, mais le processeur de ce portable
  est antérieur à AVX2.
* **Le chemin FP4 tensor cores ne s'est jamais lié.** Il sonde
  `torch._scaled_mm` à l'exécution et explique son échec ; sur une vraie 5090
  avec un torch récent il se liera peut-être, ou bien la disposition des
  échelles devra être ajustée dans `_swizzle_scales`.

## Reste à faire, par ordre de valeur

1. **Un noyau CUDA d'attention paginée.** Le décodage rassemble encore le cache
   de chaque séquence dans un tenseur dense avant de calculer l'attention, ce
   qui déquantifie tout le contexte à chaque couche et à chaque étape. Un noyau
   lisant directement la table de blocs supprimerait cela. C'est désormais la
   plus grosse inefficacité structurelle restante.
2. **Cache LRU d'experts fréquents.** `tiering.py` lui réserve de la VRAM et
   modélise son taux de succès ; rien ne l'implémente. Un modèle MoE va encore
   rechercher chaque expert routé à chaque jeton.
3. **Prefill par morceaux.** La machinerie existe — le masque décalé et le
   prefill partiel fonctionnent — mais l'ordonnanceur ne découpe pas une longue
   invite, si bien qu'un très long prefill bloque encore le décodage.
4. **Spéculation à la EAGLE.** Le propositeur n-grammes est gratuit mais n'aide
   que sur des sorties répétitives ; une tête de brouillon entraînée élèverait
   l'acceptation sur du texte libre sans exiger un modèle séparé.
5. **GEMM groupé pour les MoE.** La boucle sur les experts est en Python, un
   appel par expert routé distinct.
6. **Compensation d'erreur à la GPTQ.** La calibration est la variante bon
   marché d'AWQ ; il n'y a ni mise à jour du second ordre, ni propagation de
   l'erreur d'une couche dans la calibration de la suivante.

## Limites connues

* **Ni authentification, ni limitation de débit.** Écoutez sur `127.0.0.1`
  (valeur par défaut).
* **Ni appel d'outils, ni sortie structurée.** `tools` et `response_format` sont
  acceptés et ignorés.
* **`n > 1` ne produit qu'une seule complétion.**
* **Couverture de modèles** : famille llama, dense et MoE — RMSNorm, RoPE,
  attention à requêtes groupées, SwiGLU. Cela couvre Llama, Mistral, Qwen2/3,
  Mixtral, DeepSeek. Non couverts : attention à fenêtre glissante, blocs
  Mamba/hybrides, MLA, tours visuelles.
* **La perplexité d'un modèle non entraîné ne veut rien dire.** Le harnais
  renvoie une perplexité proche de l'uniforme pour des poids aléatoires, ce qui
  est correct et rappelle aussi qu'il ne discrimine que sur un vrai modèle.

## Première heure sur la machine réelle

```bash
acvram doctor                    # ce qui s'est compilé, ce qui non, et pourquoi
acvram bench --what bandwidth    # les deux liens PCIe et la DDR ; alimente --host-gb-s
acvram bench --what kernels      # noyaux CUDA et processeur face à la référence
pytest -q                        # le chemin de référence doit toujours passer
acvram plan ~/modeles/Qwen3-32B  # le plan correspond-il à docs/MATERIEL.md
```
