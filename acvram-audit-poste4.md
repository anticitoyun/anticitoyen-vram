# Audit du motif dominant — poste4, 8/09/2026

Lecture seule, aucun GPU/RAM utilisé. Périmètre : `acvram/`. Pas de correctif ici,
liste d'abord, comme demandé.

## 1. `except` trop larges — silencieux (aucune trace, rien ne le dit)

### `acvram/engine/loader.py:936-941` — `_borner_kv_par_la_vram`
```python
try:
    d = dev(t.name)
    libre, capacite = torch.cuda.mem_get_info(d)
except Exception:                       # noqa: BLE001
    continue
```
Fait : si `dev(t.name)` ou `mem_get_info` lève (device invalide, appel CUDA en
échec, bug), le budget KV de cet étage **n'est pas borné**, et la boucle passe
au suivant sans aucun message.
On croirait : que la fonction dit toujours pourquoi elle n'a pas pu borner —
elle a ce comportement 6 lignes plus bas en cas de succès (`print` détaillé),
donc l'absence de message est indiscernable d'un budget déjà correct.
Ça casserait : retour exact du bug que cette fonction corrige — OOM ou
décodage dégradé faute de budget KV réellement borné — sans aucune ligne de
journal pour le diagnostiquer. C'est la même famille que `_KV_MARGE`
documentée dans `ETABLI.md`.

### `acvram/engine/loader.py:1036-1042` — réplanification / capacité d'étage
```python
try:
    # `dev` est ici la chaîne du device, pas la fonction du module :
    # elle est masquée par la variable locale au-dessus.
    libre = torch.cuda.mem_get_info(torch.device(t.name))[0]
    capacite = min(capacite, libre)
except Exception:                           # noqa: BLE001
    pass
```
Fait : le commentaire documente déjà un piège de nommage résolu (`dev` masqué
par une variable locale) — mais le `except: pass` qui l'entoure est
exactement le mécanisme par lequel un correctif de ce genre peut redevenir
inerte sans que rien ne l'annonce : une régression future sur `t.name` ou
`torch.device(...)` (TypeError, AttributeError) ferait taire tout le
bornage par la VRAM réellement libre, et `capacite` resterait celle du
manifeste (potentiellement périmée, cf. commentaire lignes 1028-1034 sur les
15,8 → 152,2 jetons/s selon ce qui est réellement libre).
On croirait : que le bornage a eu lieu, puisqu'aucune erreur ne remonte.
Ça casserait : le déplacement de couches vers l'hôte (`while utilise() >
capacite - marge`) se ferait sur une capacité fausse — trop généreuse — et on
retrouverait l'OOM au chargement que ce code existe pour éviter.
**C'est le cas le plus proche de l'exemple de chef** (TypeError avalé,
correctif rendu inerte) : à vérifier en priorité si on veut le fiabiliser
(logger l'exception au lieu de `pass`).

### `acvram/quant/gguf.py:565-568` — conversion gemma4
```python
try:
    rf = self.load("rope_freqs.weight")
    cfg["partial_rotary_factor_full"] = int((rf < 1e6).sum()) / float(rf.numel())
except Exception:                    # noqa: BLE001
    cfg["partial_rotary_factor_full"] = 0.25
```
Fait : en cas d'échec (tenseur absent, format inattendu), substitue une
valeur **devinée** (0.25) sans le dire.
On croirait : que 0.25 est une valeur mesurée comme celle du cas normal, au
même titre.
Ça casserait : un modèle gemma4 dont le facteur réel diffère de 0.25 serait
servi avec un RoPE partiel faux, silencieusement — écart de qualité difficile
à attribuer, dans l'esprit du résidu de 0,367 nat non expliqué.
**À vérifier** : est-ce que `rope_freqs.weight` peut légitimement manquer sur
un modèle valide, ou l'échec signale-t-il toujours une anomalie ?

### `acvram/hardware/detect.py:397-402` — `_probe_p2p`
```python
try:
    import torch
    if torch.cuda.is_available():
        return [...]
except Exception:
    pass
return [[i == j for j in range(n)] for i in range(n)]
```
Fait : sur échec (n'importe lequel), retourne la même matrice qu'un vrai
« pas de P2P entre ces cartes » — indiscernable d'une détection qui a
réellement tourné et conclu négativement.
On croirait : que `[[i==j...]]` signifie toujours « ces cartes n'ont pas de
P2P », alors qu'il peut vouloir dire « la détection a échoué ».
Ça casserait : rien de mesurable dans l'immédiat sur la machine cible
(GeForce, P2P attendu faux de toute façon, docstring le dit) — **sévérité
basse ici spécifiquement**, mais le motif (échec = valeur par défaut
plausible, jamais signalé) est le même que les cas au-dessus.

### `acvram/server/chat.py:58-61` — `apply_chat_template`
```python
try:
    return self._render_jinja(messages, add_generation_prompt, extra)
except Exception:                        # noqa: BLE001
    pass
return _chatml(messages, add_generation_prompt)
```
Fait : un template Jinja cassé (erreur de syntaxe introduite par un futur
modèle, ou bug de rendu) retombe silencieusement sur un formatage ChatML
générique.
On croirait : que le modèle a été servi avec SON template.
Ça casserait : rien ne crashe, les réponses restent cohérentes en apparence,
mais le format de prompt réellement utilisé diffère de celui attendu par le
modèle — dégradation de qualité invisible sans comparer les deux sorties.
**À vérifier** : sévérité dépend de la fréquence des templates non
standards dans le parc de modèles testés.

## 2. Correctifs pouvant ne rien faire (no-op silencieux)

Cherché systématiquement `min(...capacité/libre...)` dans `acvram/` : un seul
site, `loader.py:1040`, déjà couvert au point 1 (le no-op vient du `except`
qui l'entoure, pas de la borne elle-même — ici `min()` est correct et
symétrique, contrairement à l'exemple `min(capacity, libre)` où `libre`
dépassait toujours `capacity`).

Aucun autre site trouvé où une borne serait structurellement inopérante
(ex. `min(a, b)` avec `b` toujours ≥ `a`) dans le périmètre audité. **Cette
partie de l'audit est donc : rien trouvé de nouveau**, au-delà du cas déjà
signalé en 1.

## 3. Noms/commentaires promettant autre chose que ce qui est fait

Le cas `self.host` / `_emballer` (`acvram/engine/layers.py:73`) déjà établi
dans `ETABLI.md` est toujours présent tel quel dans le worktree `poste4` —
je ne le re-signale pas comme nouveau, juste : **pas encore corrigé** à ce
commit (`4679d87`), pour mémoire si quelqu'un cherche à le patcher.

Aucun autre écart nom/comportement trouvé avec un niveau de confiance
suffisant dans le temps imparti. Le périmètre restant (`memory/tiering.py`,
`server/app.py` en dehors des `except`, `engine/model.py`) n'a pas été
passé au peigne fin — à signaler si quelqu'un veut que je continue.

## Ce que je n'ai PAS vérifié (à ne pas prendre pour "absent")
- pas de lecture du journal d'exécution réel pour aucun de ces cas — analyse
  de code seule, comme demandé pour cette tâche.
- couverture partielle : je me suis concentrée sur `except` (grep exhaustif,
  28 occurrences toutes lues) et sur `min(...)` (grep exhaustif) ; la
  recherche de noms/commentaires trompeurs (point 3) n'est PAS exhaustive.
