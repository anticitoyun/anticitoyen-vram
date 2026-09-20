# poste7 — « il faut terminer ce qu'il reste » (utilisateur, 05 h 38) : ordre de clôture des 2 h 20 restantes — ce qui se termine cette nuit, dans l'ordre, par membre ; ce qui ne se termine pas est nommé avec son chiffre pour la reprise (20/09, 05 h 38, horloge machine)

Source : utilisateur 05 h 38 ; `poste7-suite-nuit-00h30-20-09` § 3 ; verdicts de la nuit (0.6.24 2bb9856a, parc dc66fa3b, C5-b 530e1593, niveau 2 baaac9de, C9 c337520c) ; fin de carte 08 h 00, bilan poste7 07 h 30, dernier push chef 07 h 45.

## 1. Se termine cette nuit (durées à sec ou carte, dans l'ordre)
| rang | qui | quoi | durée | terminé quand |
|---|---|---|---|---|
| 1 | poste1 | **ligne de régime** : `GLUE_COMPACT` / `ATTN_WARPS_COMPACT` dans `regime.VARIABLES`, `glue=compact(8)` nommé, test hors-régime | 10 min | fusion à sec → **0.6.25** |
| 2 | poste2 | **niveau 2 : juge par position** (1 024 / 2 048 / 8 192, f1b326b3) — le diagnostic se clôt cette nuit même si le correctif est demain | 10 min | verdict : tables32 nommée ou écartée |
| 3 | poste1 | **gemma-4-26B-A4B : repli eager annoncé** — ligne de régime `graphes=on(repli eager: godet 1)`, journal de démarrage, `/metrics` ; le correctif de capture (bassin avant KV) seulement s'il tient en 30 min à sec, sinon fiche | 30 min | fusion à sec |
| 4 | poste2 | **C5-b terme vitesse** : `certifie` ABAB canal contre jeton (pas b=12 ≥ défaut − 1 %), capture 5/5 — la ligne KV publiée devient complète (PPL +0,205 ± 0,07, vitesse, +3,1 % d'octets) ; opt-in reste (règle PPL faux de 0,005) | 20 min | verdict |
| 5 | chef | **catalogue** : gemma « b=1 en eager, 2,9 × plus lent », Ornith « cassé au décodage » ; **tri des worktrees** `poste4` / `poste3` / `poste8` (rien de forcé : ce qui n'est pas suivi va dans un tarball hors dépôt, puis `git worktree remove`) | 20 min | commit |
| 6 | poste2 | **vLLM KV fp8 sur `ppl-decode-kv`** (même préfixe 8 192 + 512, 3 tranches) : la revendication KV (« int8 +0,5 % contre vLLM fp8 : ? ») cesse d'être un trou | 10 min | verdict |
| 7 | poste1 | **C3 MTP `ensure_hist`** : une ligne d'estimation d'abord ; ≤ 30 min à sec → correctif + test, sinon fiche seule | ≤ 30 min | fusion à sec ou fiche |
| 8 | chef | **.deb 0.6.25** (rangs 1 + 3 fusionnés) : recompilation, trois bras éco (poste2 3 min), test de charge utile ; feu vert utilisateur avec la ligne | 15 min | 07 h 15 |
| 9 | poste7 / chef | **bilan 07 h 30** (`poste7-bilan-nuit-20-09` : cellules, chantiers tenus / faux / fermés, prédictions fausses, règles ajoutées, question 119B) ; ETAT au propre ; **dernier push 07 h 45** | — | 07 h 45 |

Si la carte est libre entre deux rangs : PPL 2 000 paires tf32 contre fp32 à 2 048 (40 min, poste2) — seulement si elle tient avant 07 h 15, sinon reprise.

## 2. Ne se termine pas cette nuit — nommé pour la reprise, avec son chiffre
C13-c réécrit (structure « q en tranches en registres », sonde de temps bf16 ≤ 2 ms/appel avant toute exactitude ; cible 7 268 → ≈ 10 000 j/s) · C15-prefill (norm 9,1 + glue 8,2 ms → ≥ 20 500 j/s, PPL au bit) · C15 niveau 3 GLM (≤ 700 nœuds, niveau 2 préalable) · niveau 2 correctif (selon le rang 2) · C15-3d : sélection ≤ 3 µs (7,46 aujourd'hui) · C5-b défaut (règle PPL à 0,2 : rejouer 9 tranches) · gemma capture OOM godet 1 (bassin avant KV) et Ornith OOM Triton · C10 (a) · C9-charge 119B (barre llama.cpp 24,2 j/s ; question 3080 Ti / processeur / aucune à l'utilisateur) · PPL longue tf32 · poste hygiène « pourquoi Coder b=12 n'est pas déterministe ON/OFF » (26 % des top-k).

## Ordre
* **poste1** — rangs 1, 3, 7 dans l'ordre ; pointeurs à chef après chaque fusion à sec ; rien de nouveau d'autre.
* **poste2** — rangs 2, 4, 6, puis bras éco du rang 8 ; carte rendue à 07 h 30.
* **chef** — rangs 5, 8, 9 ; ETAT et INDEX au fil ; commit + push à chaque rang.
