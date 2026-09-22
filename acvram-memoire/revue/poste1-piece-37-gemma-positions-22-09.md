# Pièce 37 — pourquoi Gemma 4 31B est faux à TOUTES les positions et juste à la dernière (22/09, poste1, à sec)

* faits : `diag-eval-nll` (poste2 429cc2c7) — **eval ≈ serve à 0,006 nat près, PPL 227 232 sur les deux** ; **Coder sain (8,434, eval = serve)** ; la génération (dernière position) est cohérente, le scellé E passe (KL 0,87 sous gabarit). Donc : ni la boucle d évaluation, ni le BOS, ni le budget mémoire — **le forward de Gemma 4 lu à toutes les positions**.
* ce que le chemin fait (fichier:ligne) : `ACVRamModel.forward` avec `logits_positions=arange(n)` ne change RIEN au calcul des couches — il ne change que l indexation finale (`model.py:2559-2561`). Le préfill est donc le même que celui du service ; ce qui diffère entre « dernière position » et « toutes », c est seulement ce qu on LIT. **Si les logits de la dernière position sont bons et ceux des autres faux, l erreur est dans une grandeur qui dépend de la position et qui n est juste qu à la fin de la séquence.**

## 1. Hypothèse principale : les couches LOCALES (fenêtre 1024) au préfill non découpé
50 des 60 couches sont `sliding_attention`, fenêtre 1 024 (`manifest.model.layer_types`, `sliding_window`). Au préfill, `Attention._prefill` (attention.py:465-508) appelle `attention(..., window=self.window, q_offset=offset)` (layers.py:981-987) :
```
qpos = arange(q_len) + q_offset ; kpos = arange(kv_len)
mask = (kpos <= qpos) & (kpos > qpos − window)
```
`mask` est un **booléen** passé tel quel à `F.scaled_dot_product_attention(attn_mask=mask)` (layers.py:996) : correct (un masque booléen = « autorisé »). **Mais** dans la branche `ouvert is not None` (images), le masque devient additif (`masked_fill(-inf)`) : deux conventions dans la même fonction, et la fenêtre en booléen **n est jamais testée à q_len > window** — c est le cas exact d un préfill de 2 048 jetons sur une fenêtre de 1 024, et JAMAIS celui du service, qui décode jeton par jeton (q_len = 1, la fenêtre y est appliquée par le noyau paginé, pas ici). À la DERNIÈRE position, la fenêtre [pos−1024, pos] est pleine et « correcte par accident » même si le masque était trop large ou trop étroit aux positions du début ; aux positions < 1 024, une fenêtre mal bornée laisse voir des clés qui n existent pas encore ou en cache d autres — d où des logits faux partout SAUF à la fin. **C est la seule grandeur du chemin qui dépend de la position ET qui est propre à Gemma** (Coder n a pas de couche locale : `layer_types` vide, `sliding_window` 0 → `window=0`, branche `causal_mask`, saine — ce que la mesure de poste2 confirme).
Deux sous-cas à distinguer par la mesure (§ 3) : (a) la fenêtre est fausse d un rang (`>` contre `>=` : 1 023 clés au lieu de 1 024, ou l inverse) — erreur petite, ne donnerait pas 227 232 ; (b) le masque booléen et le dtype : `F.sdpa` avec `attn_mask` **booléen** et `enable_gqa=True` (gqa activé dès `n_rep > 1`, layers.py:977) — chemin peu exercé ; si la fusion GQA ignore ou transpose le masque, toutes les positions sauf la dernière deviennent fausses. **(b) est mon hypothèse n° 1.**

## 2. Écartés, avec la raison
* **softcap final** (30, `model.py:2646-2648`) : appliqué dans `_logits_finaux`, le même pour toutes les positions et pour le service — s il était faux, la dernière position le serait aussi.
* **position_ids / RoPE** : `positions = arange(n)` au préfill (evaluate.py et `_build_batch`), deux RoPE Gemma (locale `rope_theta_swa` 10 000, globale 1 000 000, loader.py:531-537) choisies par `layer_types` — une erreur de thêta rendrait la dernière position fausse aussi.
* **cache vide** : `_prefill` prend `kk, vv = k[start:end], v[start:end]` quand `offset == 0` (attention.py:487-490) — pas de lecture de cache, donc pas de fantôme.
* **`embedding_multiplier` 73,32** et **normes** : appliqués une fois pour toute la séquence.
* **`logits_positions`** lui-même : ne touche que l indexation (`model.py:2559`), démontré par le fait que `eval` (toutes positions via `return_hidden` + tête par tranches) et `serve` (un forward avec `logits_positions`) donnent la **même** PPL fausse à 0,006 nat près.

## 3. Ce qui tranche (poste2, ≤ 10 min, sans carte pour (i))
1. **(i) à sec, 0 min de carte** : `attention(q, k, v, causal=True, window=1024)` sur des tenseurs jouets [2048, H, D] contre une référence torch explicite (masque additif construit à la main) — si les deux diffèrent, c est (b) et le correctif est d une ligne (masque additif partout) ; si elles coïncident, l hypothèse tombe et on regarde les couches globales (partial_rotary_factor 0,25, `global_head_dim` 512, `num_global_key_value_heads` 4 : un chemin GQA 8×).
2. **(ii) carte, 2 min** : `diag-eval-nll --jetons 300` sur le 31B avec `ACVRAM_FENETRE_LOCALE=0` (à ajouter si (i) accuse la fenêtre) ou sur un alias Gemma sans couche locale — PPL attendue 8-20 si la fenêtre est la cause.
3. **(iii) le bras hf** (10 min, après 9a1c06d9 qui libère la VRAM) : si hf ≈ eval ≈ serve, ce n est pas nous mais le converti (P3) ; si hf est sain, c est le forward — et (i) dit où.

## 4. Prédictions écrites (les deux branches de la mesure du Coder, déjà tombée)
* **Coder sain (observé)** → le défaut est propre à Gemma 4 : couches locales ou têtes globales, et l évaluation PPL reste **invalide pour tout modèle à fenêtre locale** (Gemma 4 27B/31B, Ministral…), valide pour les autres — à écrire dans la fiche de `acvram eval` tant que ce n est pas corrigé.
* (Coder faux — non observé) → le défaut aurait été dans la lecture à toutes positions (tête par tranches, `_pertes_par_tranches`), et le correctif aurait porté sur `evaluate.py`.
**Conséquence immédiate** : aucune PPL de Gemma 4 publiée par `acvram eval` n est utilisable (y compris les comparatifs 4sur6 du 22/09) ; le scellé E (decode-pas, dernière position) reste la seule mesure de qualité valide sur ce modèle — ce que la chef avait déjà tranché, pour une autre raison.
