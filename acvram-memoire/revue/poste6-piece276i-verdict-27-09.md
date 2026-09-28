# Pièce 276 i — b = 1 non reproductible entre serveurs neufs : c'est le CACHE DE PRÉFIXE qui ressert des K/V calculés dans un autre lot (GEMM cuBLAS dépendant de M, couche 0, 1 ulp) ; les noyaux, eux, sont déterministes — verdict

poste6, 27/09/2026 18 h 3x. Scellé : `poste6-piece276i-a-sec-27-09.md` (prédiction avant la mesure, inchangée). Q37 poste4 (duck.ai,
86d4b76a2) : cause proposée = autotuning Triton au premier appel (triton#9368, vllm#58899) — départagée ci-dessous. 213/213 b relue
(RoPE fp32 → bf16 à la première forme, corrigée) : pas le même défaut. Sous carte.sh (poste6-276i, 18:18 → 18:20, 9 serveurs neufs,
arbre main a6268af2e contrôlé, charge hôte par cellule), scripts origin/poste6-276i (d7fb58880 + instruments de ce verdict).

## 1. Reproduire (trois configurations × 3 serveurs neufs, 24 jetons temperature 0, invites 0-4 identiques à celles du b = 12)
| configuration | A1 = A2 (neufs, b = 1 seul) | après b = 12 : C1 = C2 | A = C (même serveur) | S3 `--no-prefix-cache` : C' = A |
|---|---|---|---|---|
| VL-2B + image | **5/5** | 5/5 | **4/5** (invite 1, S1 et S2) | **5/5** |
| VL-2B texte seul | **5/5** | 5/5 | **3/5** (invites 0 et 3, S1 et S2) | **5/5** |
| Qwen3-4B texte | **5/5** | 5/5 | 5/5 | 5/5 |
Lecture : (a) deux serveurs neufs sans trafic rendent les MÊMES jetons, dans les trois configurations → **pas de non-déterminisme
de noyau entre processus** (autotuning Triton, cuBLASLt, num_splits : non observés ici — la Q37 nommait une cause possible,
ce protocole la rend fausse pour notre chemin) ; (b) la bascule apparaît SEULEMENT après qu'un lot a préfillé les mêmes invites,
elle est la même sur S1 et S2 (déterministe à histoire égale), et **disparaît avec `--no-prefix-cache`** → ce sont les blocs KV de
tête, calculés dans le préfill de 12 séquences puis resservis à b = 1, qui diffèrent de ceux qu'un préfill seul calcule ; (c) le
défaut touche le VL en texte seul : il est au **moteur** (forme du lot de préfill), pas à la vision ; (d) la bascule arrive tôt
(4e-8e jeton : « interpréta le rôle » → « starred as William ») : à temperature 0 un jeton à égalité près bascule sur 1 ulp.
La 276 g (5/5) et la 276 h (9/10) sont expliquées : la cellule b = 1 de la 276 h courait après les cellules b = 4/12.

## 2. Localiser (en process, `diverge-276i.py`, arbre main, cache de préfixe coupé, crochets sur les 337 modules, invite 0 seule contre
invite 0 en tête d'un lot de 12, mêmes prompt_ids)
* VL-2B + image (369 jetons) : **premier site `layers.0.self_attn.k_proj`** : max 1,00 ulp bf16 sur 38 % des éléments (q_proj au bit —
  cuBLAS choisit son noyau selon (M, N) : à N = 1024 il change avec M, à N = 2048 non) ; puis v_proj 1 ulp 43 %, k_norm 3 ulp,
  o_proj, attention, gate_proj 2,75 ulp… 306/337 modules divergent ensuite ; **seul contre seul (deux passages) : 0/337** — déterministe.
* Qwen3-4B texte (165 jetons) : le chemin b = 1 passe par des modules FUSIONNÉS (`qkv_proj`, `gate_up`) que le lot n'emprunte pas
  (q/k/v et gate/up séparés) — deux chemins selon la forme du lot ; sur les 145 modules communs, **premier site `layers.0.mlp.down_proj`**
  1,00 ulp sur 34 % (l'attention de la couche 0 est au bit), puis o_proj 1,5 ulp, down_proj 3 ulp… 142/145 divergent ; seul/seul 0/217.
  Les 24 jetons n'ont pas basculé sur ces 5 invites (marge d'égalité plus grande) : le mécanisme est le même, la bascule est affaire de marge.
Prédiction 3 du scellé (première projection de la couche 0, 1 ulp, à entrée identique) : **TENUE** (k_proj plutôt que q).

## 3. Ce que ça change pour nos critères « au bit »
Un GEMM cuBLAS bf16 n'est pas invariant à M : les K/V d'une séquence dépendent des séquences qui l'accompagnaient au préfill. Tant que
le cache de préfixe ressert ces blocs, deux passages à b = 1 ne se comparent qu'à historique égal. Règles à poser (à chef) :
1. **Toute cellule d'équivalence de jetons tourne sur serveur neuf OU `--no-prefix-cache`**, et le dit dans son en-tête ; les
   comparaisons « jetons identiques » de la 276 f/g/h à b = 4/12 restent inapplicables (composition), celles à b = 1 de la 276 h
   étaient contaminées par le b = 12 qui précédait (la 276 g, serveurs neufs, tenait).
2. Rendre le préfill invariant à M coûterait le lot (préfill séquence par séquence) — pas proposé ; l'alternative est de **hacher les
   blocs de préfixe avec la forme du lot**… qui casse le cache. Le bon compromis : documenter (REGLES § 4, « les façons dont une mesure
   ment ») et exiger le serveur neuf ou le cache coupé dans les scellés ; `outils/` : un test cassant qui rejoue A/C à sec ? Non
   (cuBLAS n'est pas à sec) — le contrôle reste la prise sous carte, `prise-276i.sh`, rejouable en 3 min.
Hors pièce : pour poste4, la Q37 est départagée (pas d'autotuning en cause sur ce chemin : A1 = A2 15/15).
