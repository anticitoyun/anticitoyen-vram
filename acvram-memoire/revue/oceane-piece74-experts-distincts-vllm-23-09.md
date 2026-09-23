# Pièce 74 — combien d'experts distincts vLLM touche-t-il par couche ? (préparé à sec, prédiction avant la carte) — 23/09 (Océane)

## Pourquoi cette pièce existe

La pièce 73 conclut que vLLM tire 1,67 To/s de son MoE contre nos 1,37 sur
**les mêmes octets**. Ces octets valent 88,6 Mo/couche **à condition** que leur
routage touche autant d'experts distincts que le nôtre — 33,4 par couche,
nombre mesuré **chez nous seulement** (`gaelle-p60`). S'il était de 28 chez
eux, ils liraient 16 % d'octets en moins et leur avance serait une illusion
d'instrument. Ce contrôle décide si la 48 vaut encore d'être débloquée.

## Protocole — et pourquoi je recompte les DEUX côtés

Le 33,4 vient du client de cellule (invites en texte, chaîne HTTP). Le comparer
à un comptage vLLM obtenu autrement ferait porter l'écart sur le protocole
autant que sur le moteur. Je recompte donc **les deux moteurs dans la même
prise, sur les mêmes ids d'invite déterministes** (`invite(k, n) = (k·104729 +
i·7919) mod (vocab−100) + 10`, celle des bancs du dépôt), b = 12, 40 pas de
décodage. Le 33,4 de Gaelle reste en référence secondaire ; s'il s'écarte de
mon propre comptage acvram, c'est le protocole qui parle et je le dirai.

* **définition, identique des deux côtés** : pour chaque appel de routage
  (une couche, un pas) avec M = 12 jetons et top_k = 8, le nombre d'experts
  **distincts** parmi les 96 paires ; moyenne sur tous les appels à M = 12.
* **vLLM** : crochet dans le processus, **sans toucher à l'installation** —
  `fused_topk` et `fused_topk_bias` (`vllm/model_executor/layers/fused_moe/
  router/`) enveloppées dans le script, qui lit `topk_ids` en sortie. vLLM est
  lancé **hors ligne** (`LLM(..., max_num_seqs=12)`) : un serveur séparé ne
  verrait pas le crochet, et l'offline suffit puisque le routage ne dépend que
  des jetons, pas du transport.
* **acvram** : `ACVRAM_TRACE_ROUTAGE_PT` (déjà présent, `moe.py:1165`), mêmes
  ids, même godet.
* carte : ≤ 5 min, une prise, `-lgc 2700`, compute-apps début et fin.

## Prédiction et issues, écrites AVANT la mesure

Les deux moteurs servent le **même modèle** avec un routeur **non quantifié**
des deux côtés (leur `quantization_config.ignore` liste tous les
`model.layers.*.mlp.gate`). Pour les mêmes jetons, le routage doit donc être
presque identique.

| issue | condition | ce qu'elle rend |
|---|---|---|
| **I1 — les octets sont bien les mêmes** | écart ≤ 5 % (je prédis ≤ 2 %) | le 1,67 To/s tient : l'écart de la 73 est un **débit de noyau**, et la 48 reste la seule voie |
| **I2 — vLLM touche moins d'experts** | écart > 5 % en leur faveur | **mon verdict 73 tombe** : leur avance est en partie une lecture d'octets, à requalifier avant toute conclusion sur le noyau |
| **I3 — vLLM touche PLUS d'experts** | écart > 5 % dans l'autre sens | leur débit réel est **encore meilleur** que 1,67 et l'écart à expliquer grandit |
| **I4 — les protocoles ne coïncident pas** | mon comptage acvram s'écarte de plus de 5 % des 33,4 de Gaelle | le nombre dépend des invites plus que du moteur ; aucune des trois ci-dessus ne conclut, et il faut refaire les deux sur les invites du client de cellule |

* **prédiction nominale : I1**, 32,5-34,5 experts distincts des deux côtés.
* **ce qui me gênerait** : I2 — j'ai écrit hier que l'écart était un débit de
  noyau, et I2 dirait que je me suis trompée en comparant des octets inégaux.
  C'est précisément pourquoi ce contrôle passe avant la suite.
* **alarme** : si le crochet n'est appelé qu'avec M ≠ 12 (lots fusionnés,
  préfill mêlé au décodage), la moyenne ne porte pas sur le régime servi et je
  le dis au lieu d'agréger des régimes différents.

## État à la pause du 23/09 08 h 2x — un bras sur deux, et ce qu'il dit déjà

* **acvram : 32,46 experts distincts par couche** (moyenne sur **1 968 appels à
  M = 12** ; médiane 32, min 22, max 56 ; 96 appels hors b comptés à part et
  non agrégés, comme prévu). Alias `Qwen3-Coder-30B-A3B-nvfp4`, b = 12, 40 pas,
  ids d'invite déterministes. Carte rendue, compute-apps de fin = llama-server
  seul (autre carte).
* **I4 écartée** : mes 32,46 contre les 33,5 de Gaelle (`gaelle-p60`, invites du
  client de cellule) = **−3,1 %**, sous mon seuil de 5 %. Les deux protocoles
  coïncident, donc mon comptage est comparable au sien et le chiffre d'octets
  de la 73 (88,6 Mo/couche) reposait sur une base saine.
* **vLLM : pas de chiffre.** Deux mises en route ratées, toutes deux nommées :
  1. le crochet ne voyait rien — vLLM lance un sous-processus `EngineCore` et
     le monkeypatch du processus parent ne l'atteint jamais. Corrigé par
     `VLLM_ENABLE_V1_MULTIPROCESSING=0` ;
  2. avec les graphes, la capture échoue (`cudaErrorStreamCaptureInvalidated`) ;
     avec `enforce_eager=True`, vLLM exige **FlashInfer**, absent de ce venv.
     Non corrigé à la pause.

## Prochaine étape, à la reprise (dans l'ordre)

1. Bras vLLM : garder `VLLM_ENABLE_V1_MULTIPROCESSING=0`, **retirer**
   `enforce_eager` et poser `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (ou installer
   FlashInfer, ce qui touche l'installation : à ne pas faire sans ordre) ; si la
   capture échoue encore, baisser `gpu_memory_utilization` à 0,70.
2. Conclure entre I1, I2 et I3 avec le seuil de 5 % déjà écrit. **Aucune
   conclusion sur la 73 avant ce chiffre** : tant que le bras vLLM manque, le
   « 1,67 To/s » reste non requalifié, ni confirmé ni infirmé.

## Verdict — 23/09 08 h 42 (Océane)

* **instrument** : `outils/gpu/mesure/experts-distincts-p74.py` (crochet `fused_topk` dans le processus, vLLM non modifié), chaîne `scratchpad/oceane-p74-23-09/bras-vllm.sh`
* **commit** : 1385e4a2 (arbre mesuré) ; correctif du bras : `attention_backend` (ce commit)
* **régime** : b = 12, 40 pas, ids d'invite déterministes `invite(1000+k, 256)`, glouton, graine 0 ; -lgc 2700 posé puis -rgc ; eager des deux côtés (le crochet ne tourne pas au rejeu d'un graphe) ; vLLM 0.29.0, venv de la cellule `/opt/ia/vLLM/.venv`, alias `Qwen3-Coder-30B-A3B-Instruct-FP4-a16`, attention **TRITON_ATTN** (FlashInfer xqa absent, voir plus bas)
* **scellé** : I1 si |écart| ≤ 5 % (prédit ≤ 2 %) ; I2 si vLLM < 30,84 (−5 % contre 32,46) ; I3 si > 34,08
* **mesuré** : **vLLM 32,80** experts distincts/couche (1 920 appels à M = 12, 240 hors b non agrégés ; médiane 33, min 9, max 53) contre **acvram 32,46** (1 968 appels ; médiane 32, min 22, max 56) : **+1,0 %** du côté vLLM
* **verdict** : **I1.** Les octets sont les mêmes, et vLLM en lit même 1 % de plus. Donc la pièce 73 est **un vrai écart de débit du noyau**, pas une illusion d'octets : Marlin MoE vLLM 1,67 To/s (≈ 1,69 corrigé des +1 %) contre notre 1,37, sur les mêmes arguments (73) et les mêmes experts (74). Suite : la pièce 48 (compteurs ncu, bloquée sur `NVreg_RestrictProfilingToAdminUsers`, sudo utilisateur) est la seule voie pour nommer le mécanisme.
* **durée** : prévue ≤ 5 min ; tenue ≈ 2 min 42 de carte (08:39:54-08:41:55 échec FlashInfer, 08:42:13-08:42:34 mesure), compute-apps début et fin = llama-server 4627 seul

Deux faits d'instrument, sans effet sur le verdict :

1. **Mauvais venv le matin** : le premier bras tournait sous `/mnt/AI_GENERATOR/vLLM/venv` ; la cellule vLLM utilise `/opt/ia/vLLM/.venv` (`chaine-cellule-b12-vraie-vllm.sh:8`). Les deux ont vLLM 0.29.0 et FlashInfer 0.6.18, mais en eager, avec le KV fp8 du checkpoint, vLLM choisit le décodage FlashInfer **xqa** que ni l'un ni l'autre ne fournit (`RuntimeError: FlashInfer backend is not available`). D'où TRITON_ATTN. L'attention ne change le routage qu'à l'arrondi près ; au pire, elle déplace des départages de top-k, pas la moyenne de 1 %.
2. **min 9 chez vLLM** contre 22 chez nous : au moins un pas où les douze suites gloutonnes ont convergé vers les mêmes jetons. Ce cas pèse sur le minimum, pas sur la médiane (33 contre 32), et il irait contre vLLM (moins d'octets) : il ne peut pas avoir fabriqué le +1 %.
