# Sage — fla à 128 t/s : le scellé est réfuté tel qu'écrit, et le profil est la suite qu'il prescrivait — les deux à la fois, pas l'un contre l'autre (17/09)

Entrée : Jérôme — JIT fla sm_120 OK ; Qwen3.8-27B b=12 **128 t/s** (torch 97 ; scellé ≥ 400, faux si < 250). Régression 0c91092 (réserve de préfill lisait `kv_max_tokens` au lieu de `max_model_len`) : notée, cause probable des faux « dégradé » du palier 2, à rejouer seulement si une cellule en dépend.

## Verdict sur l'hypothèse

**Réfutée** : « la récurrence torch est le goulot, fla seul rend ×4 » est fausse — +32 % (97 → 128), pas ×4. On l'écrit comme tel dans le verdict, sans adoucir. Ce n'est pas F (poste b=1, clos) : c'est le chantier hybrides, étape 1.

## Suite, telle que le scellé la prescrivait : « faux ⇒ profiler avant de continuer »

Laurine a raison sur la lettre et sur le fond : le profil n'est pas une contestation du verdict, c'est la ligne suivante du même scellé. 94 ms/pas pour 12 séquences d'un dense 27B W4A16 : le plancher de bande (≈ 15 Go de poids / 1,79 To/s) est **8,4 ms** — on est à 11× le plancher, ce n'est pas un noyau lent, c'est de la colle. Profil torch.profiler, même instrument que `profil-pas-coder-17-09.py`, **deux régimes dans la même fenêtre** (`ACVRAM_GDN=torch` témoin, `=fla`), 40 pas, par catégorie : noyau récurrent (fla / torch), conv causale à état, projections (GEMV), **colle d'état** (stack / clone / `.to(float32)` / gather), attention des couches non-GDN, et **lancements par pas** avec la part hors graphe.

Suspects nommés, fichier:ligne, à confirmer ou écarter par le profil, pas par lecture :
* `gdn.py:244` — `torch.stack` des états conv par séquence à chaque pas ; `gdn.py:255` — `b` clones + `b` conversions fp32 par couche et par pas → 48 × 12 × 2 ≈ 1 150 petites opérations par pas, si elles tournent hors graphe.
* `gdn.py:137-142` — conv1d avec `cat` de l'état, un `clone` de plus par couche.
* `graphs.py:159` — `ACVRAM_HYBRID_SLOTS` (défaut 4) et `godet_hybride` : à 12 séquences, la voie « formes fixes » (`gdn.py:182`) est-elle bien celle qui tourne, et sous graphe ? Une seule ligne du JSON le dit : lancements par pas sous profileur contre lancements en rejeu de graphe.

## Scellé du profil (écrit avant)

Prédiction : lancements ≥ 3 000 par pas ; colle d'état + conv ≥ 50 % ; noyau fla récurrent ≤ 15 % ; GEMV ≤ 20 %.
* colle ≥ 50 % ⇒ le remède est **l'état par lot** (un tenseur [b, …] par couche, mis à jour en place, capturé dans le graphe ; zéro stack/clone par pas) — Laurine, à sec, et le ≥ 400 reste l'objectif, rescellé après le profil sur les ms mesurées.
* noyau fla ≥ 40 % ⇒ fla est lent ici (version, formes, `fused_recurrent` appelé par séquence au lieu du lot) — on lit le nombre d'appels : 48 (lot) ou 576 (par séquence).
* GEMV ≥ 40 % ⇒ le goulot est le dense 27B lui-même, et 400 t/s n'était pas atteignable (12 × 27B : plancher 8,4 ms ⇒ ≤ 1 400 t/s théoriques, donc si, mais pas avec cette GEMV) — on revient à la GEMV ≥ 85 % de `sage-lecture-profils`.
Faux pour tout ça : lancements < 1 000 et aucune catégorie > 30 % — alors je n'ai pas compris le pas et on lit le top 25 noyaux avant toute proposition.

Fenêtre : ≤ 30 min (Laure). Aucun noyau, aucune correction avant le JSON.
