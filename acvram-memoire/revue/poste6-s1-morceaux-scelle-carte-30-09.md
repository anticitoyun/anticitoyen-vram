# kv31b levier 2, étape 1 — scellé de la preuve carte S1 : préfill par morceaux 4 096 au bit du seul tenant, gemma-4-31B (poste6, 30/09, écrit AVANT la prise)

Feu chef (30/09 18 h) : fenêtre commune avec poste1 après la campagne de poste2, ACVRAM_POSTE=poste6, ≤ 15 min par prise.
Code : branche poste6-reserve-attention (contient poste6-prefill-morceaux 301b60449 ; fusion origin/main 87d8bfe0a → 5454071ce) ; le paquet
0.7.16 ne porte pas le levier → moteur depuis l'arbre du worktree (`PYTHONPATH`, venv du dépôt principal).
Scripts : `outils/gpu/mesure/prise-s1-morceaux-kv31b.sh <bras>` (une prise par bras, reprenable, HEAD asserté, relevés début/fin, journal lu par
ses lignes `[acvram]` seulement) et `outils/gpu/mesure/s1-morceaux-comparer.py` (à sec, ne lit jamais le texte : égalités et sha256).

## Pourquoi pas `echo` (contrôle qui ne peut pas rendre faux)
`Engine.logprobs_invite` (runner.py:2407, chemin `echo=true`) fait UN `self.model(batch)` direct : il ne passe ni par `_prefill_morceaux`
(runner.py:1438) ni par la boucle de préfill (runner.py:1916). Des logprobs d'invite identiques entre A et B ne prouveraient rien.
La preuve passe donc par le chemin réel de service : une génération gloutonne de 32 jetons — le premier jeton sort des logits du dernier
morceau, les 31 suivants lisent le KV écrit par les morceaux — avec `logprobs=10` sous `ACVRAM_SAMPLER_LENT=1` (top-10 servis).

## Bras (un processus par bras : `_PREFILL_MORCEAU` est lu à l'import, runner.py:93)
| bras | ACVRAM_PREFILL_MORCEAU | rôle |
|---|---|---|
| A1 | 0 | seul tenant, référence |
| A2 | 0 | témoin : bruit d'un processus à l'autre (REGLES § 3 : le témoin avant le seuil) |
| B | 4096 | morceaux ; invite de ~7 950 jetons → 2 morceaux (4 096 + ~3 860 ≥ 128) |
Communs : `acvram-gemma-4-31b-it-nvfp4-4sur6-vision-nvfp4` (dense, 60 couches, fenêtre glissante 1 024 sur 50 couches, globale sur 10),
`--max-model-len 10240`, `--speculative none`, `--max-batch 1`, graphes par défaut, `ACVRAM_SAMPLER_LENT=1`, port 8093.
Invite : `git show 87d8bfe0a:README.md` (23 683 o, sha256 `0640377149ab8d46…`), 7 953 jetons gemma comptés à sec (sans BOS) — > 5 fenêtres
de 1 024 et > 1 morceau. Requête : `/v1/completions`, `temperature=0`, `max_tokens=32`, `logprobs=10`, `seed` absent. Un seul appel par
processus (pas de cache de préfixe entre bras). Sortie par bras : `scratchpad/poste6-s1-<bras>/{completion.json,metrics.json,serveur.log,releve-*.txt}`.

## Prédiction et seuils
| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| P1 | prise : `/metrics` `engine.prefill_morceaux` | B = 1 ; A1 = A2 = 0 ; régime B porte `(morceaux@4096)` | B = 0 (morceau ignoré ou invite ≤ morceau) |
| P2 | témoin A1 vs A2 : 32 ids, 32 token_logprobs, 320 top-10 | tous identiques au bit (Δ = 0) | un Δ ≠ 0 → « au bit » injugeable d'un processus à l'autre, dit ; seuil de P3 = 2 × max Δ témoin |
| P3 | B vs A1 : mêmes 384 valeurs | identiques au bit (Δ = 0, comme le jouet sous MLP fusionné) | un Δ ≠ 0 avec témoin à 0 : morceaux NON au bit sur carte |
| P4 | `usage.prompt_tokens` | 7 953-7 955 (BOS éventuel), identique dans les 3 bras | < 5 120 ou différent entre bras |
| P5 | exil MLP (`plan réajusté … RAM hôte`) | même compte dans les 3 bras (0 à carte seule ; quelques-uns si le llama-server 4436 est là) | comptes différents → comparaison contaminée, prise B à rejouer |
| P6 | durée par prise / total | ≤ 8 min (chargement 19 Gio + chauffe 10 240 + 1 requête) / ≤ 25 min | > 15 min : carte.sh tue |
Issues nommées, y compris celle qui me gêne : (a) P3 faux et P2 tenu → `forward_tranches` (couche par couche) n'est pas au bit de `forward`
sur ce modèle, ou la coupe des K/V en deux morceaux change une réduction — le levier 2 ne repose plus sur « au bit » et le dit ; bras suivant
à sec : `tranches_possibles` forcé faux (boucle `forward` par morceau) pour séparer les deux causes ; (b) P2 faux → la carte n'est pas
reproductible au bit entre processus (atomiques, ordre des blocs) : on ne conclut ni « au bit » ni « pas au bit », on publie l'écart et l'on
juge B dans 2 × l'écart témoin ; (c) P1 faux avec P3 tenu → rien n'a été prouvé (deux seuls tenants) ; (d) P5 faux → rejeu ; (e) la sortie
du modèle n'est jamais lue (§ 6) : ids, logprobs et sha256 seulement.
