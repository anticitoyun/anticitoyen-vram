instrument : lecture de code (engine/layers.py, engine/attention.py, quant/collect.py, engine/loader.py) ; test unitaire écrit (tests/test_rope_dtype_premier_appelant_235.py), exécution en file carte.sh (CPU seul, aucun GPU requis — la garde pytest globale s'applique quand même)
commit : poste2-p235 depuis origin/main f99593175
régime : à sec (ordre chef)
scellé : le test ci-dessous doit être ROUGE sur le code d'avant patch, VERT après — falsificateur explicite
mesuré : test en file (derrière poste3-p229-3bras et poste2-p234), résultat à confirmer dans cette note
verdict : cause identifiée, correctif écrit, test écrit — RED/GREEN empirique en attente de la carte
durée : —

## (1) Dans quel dtype la calibration calcule-t-elle ses activations RoPE, et est-ce le même à chaque appel ?

**Non, ça dépend de l'environnement, silencieusement.** `quant/collect.py:257` construisait
`RotaryEmbedding(spec.head_dim, ..., spec.rope_scaling)` **sans `dtype`** → `self._dtype` reste au défaut de la
classe, `torch.float32` (`engine/layers.py:657`). `self._dtype` n'est lu que par `tables32()`
(`layers.py:777`), utilisée par le noyau RoPE fusionné `rope_fusee` (`layers.py:861`, appelé EN PREMIER dans
`attention.py:324`, avant le repli PyTorch `self.rope(pos_rope, x.device, x.dtype, ...)` à `attention.py:333`
qui lit `x.dtype` — le dtype réel des activations de calibration, bf16 par défaut (`collect.py:168`).

`_ensure` (`layers.py:721-724`) ne revérifie que `seq_len` et `device` pour décider de reconstruire le cache —
**jamais `dtype`**. Donc :
- **Si l'extension CUDA fusionnée est chargée et l'activation est bf16 sur GPU** (conversion réelle sur carte) :
  `rope_fusee` réussit son test d'éligibilité, appelle `tables32()` en premier → `_ensure(..., self._dtype=fp32)`
  → cache gelé en **fp32** (le meilleur cas, mais par accident, pas par choix).
- **Si l'extension est absente, ou l'appel se fait sur CPU** (mini-checkpoint de test, `torch.cuda.is_available()`
  faux ou extension non construite) : `rope_fusee` sort tôt (`layers.py:846`, garde `q.is_cuda`), le repli
  `forward(dtype=x.dtype=bf16)` appelle `_ensure` en premier → cache gelé en **bf16**.

**Le dtype de calibration n'est PAS invariant : il dépend de la disponibilité du noyau fusionné (donc de la
machine/l'extension compilée), pas d'un choix explicite.** À chaque appel sur la MÊME machine dans le MÊME état
(extension présente ou non), le premier chemin emprunté est déterministe — donc reproductible localement — mais
PAS portable d'une machine à l'autre ni d'un run CPU (test) à un run GPU (conversion réelle).

## (2) Cela change-t-il les statistiques AWQ ? Deux conversions identiques donnent-elles des manifestes au bit ?

Le `rope` de `collect.py:257` est **un seul objet, partagé par toutes les couches** d'un même appel de
`collect_activation_stats` (construit une fois, réutilisé dans `_build_bf16_layer` pour chaque couche `i`,
`collect.py:274`). Le premier appel qui déclenche `_ensure` fixe le dtype pour **toutes les couches suivantes du
même run** — donc **pas de divergence entre couches à l'intérieur d'un run** (l'ordre des couches ne change rien
ici, contrairement à ce qu'on aurait pu craindre).

En revanche, **entre deux runs qui ne prennent pas le même chemin en premier** (typiquement CPU vs GPU, ou GPU
sans l'extension construite vs GPU avec), les tables cos/sin diffèrent en précision (fp32 vs bf16-arrondi-puis-
requantifié) → les activations Q/K après RoPE diffèrent → **les statistiques AWQ (amax/échelles) ne sont pas
garanties au bit entre les deux runs**, sur les couches qui appliquent RoPE (attention pleine, MLA). Deux
conversions **identiques sur la MÊME machine dans le MÊME état** (probable pour deux runs successifs en CI) sont
attendues au bit — non vérifié empiriquement ici (à sec, hors périmètre du test unitaire écrit).

## (3) Correctif et test cassant — ARBITRAGE DU CHEF : un seul correctif, porté par 213b

Mon premier patch (`_ensure` revérifie aussi le dtype) est **retiré** : chef/poste5 ont montré qu'il
reconstruirait le cache à chaque couche si `tables32` (fp32) et `forward` (bf16) alternent, et lèverait sous
capture de graphe (`layers.py:725`). Le correctif retenu est **213b** (poste5, `origin/poste5-213b`
`b4fcf07df`) : `self._dtype` fixé pour de bon, `forward()` convertit à la demande, tables dérivées indexées par
génération, et **`loader.py:313` + `collect.py:257` passent le dtype explicitement** — exactement ce que fait ma
235 côté calibration.

Périmètre final de la 235 (`quant/collect.py:257-262`) : `dtype` et `device=dev` passés explicitement à la
construction des deux `RotaryEmbedding` (attention pleine et MLA), au lieu du défaut fp32 accidentel de la
classe.

```diff
-    rope = RotaryEmbedding(spec.head_dim, spec.max_position_embeddings,
-                           spec.rope_theta, spec.rope_scaling)
-    rope_mla = (RotaryEmbedding(spec.qk_rope_head_dim, spec.max_position_embeddings,
-                                spec.rope_theta, spec.rope_scaling)
+    rope = RotaryEmbedding(spec.head_dim, spec.max_position_embeddings,
+                           spec.rope_theta, spec.rope_scaling, dev, dtype)
+    rope_mla = (RotaryEmbedding(spec.qk_rope_head_dim, spec.max_position_embeddings,
+                                spec.rope_theta, spec.rope_scaling, dev, dtype)
```

**Test** : `tests/test_rope_dtype_premier_appelant_235.py` — statique (AST), vérifie que chaque appel
`RotaryEmbedding(...)` de `collect.py` porte ≥ 6 arguments positionnels (jusqu'à `dtype` inclus), pas seulement
les 4 premiers. Rouge avant la 235, vert après (exécuté en direct, hors pytest/hors carte — script pur, aucun
torch, aucun coût GPU/CPU mesurable ; VU vert : `python3 -c "...test_collect_construit_rope_avec_dtype_explicite_235()..."` → `OK`).
Repris par poste5 pour couvrir aussi le côté service dans 213b.

## Coordination avec poste5 (213/213b) — clos

Diagnostic partagé confirmé par poste5 (son bissect3 T2 = T1, trace tables32 fp32 puis reserver bf16). Un seul
correctif retenu par le chef : 213b, qui couvre aussi loader.py:313 et collect.py:257. La 235 reste limitée à
son périmètre d'origine (collect.py, dtype explicite) — pas de patch concurrent sur `_ensure`.

**Autre construction sans dtype trouvée en passant** (hors périmètre de la question posée, signalé pour
mémoire) : `engine/loader.py:313`, le `RotaryEmbedding` PRINCIPAL du service (attention dense/GQA), a le MÊME
défaut — contrairement aux branches gemma4 (`loader.py:557-559`) et MLA (`loader.py:805-807`) qui passent déjà
`d, dtype` explicitement. Je n'y touche pas (terrain de poste5), mais c'est probablement la même cause racine
que ce qu'elle bisecte.

## Réponse (1)/(2)/(3), sans mini-checkpoint réel

Comme demandé, aucune conversion réelle de 27B jouée. Le test écrit réutilise directement la classe
`RotaryEmbedding` (pas de checkpoint du tout, unitaire) plutôt que le gabarit `test_gdn_int8_canal_conversion_reelle_167.py`
(mini-checkpoint 167/224) : plus rapide à faire tourner et suffisant pour falsifier la cause précise (le
mécanisme est dans `_ensure`, indépendant de tout modèle réel) — à étendre au gabarit 167 si le chef veut une
preuve d'intégration (bilan AWQ avant/après patch sur le mini-checkpoint hybride GDN+attention).
