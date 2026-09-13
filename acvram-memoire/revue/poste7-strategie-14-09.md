# poste7 — stratégie 14/09 : rattraper vLLM à b=12, ou recentrer ?

Sources : audit-a2-duel-vllm-14-09.md, vis-a-vis-decodage-b12-14-09.md (poste4), prediction-4-3ms-hote-decodage-14-09.md, mesure-qui-tue-autopin-table-14-09.md, prediction-masquage-fantomes-14-09.md, verdict-cout-verification-ngram-b12-14-09.md (poste3), audit-a1-ttft-profil-14-09.md, verdict-horloge-memoire-14-09.md, poste4.md:3736-3752,4335-4358, beads 0si/1aj/6wa, chef.md. Aucun chiffre de moi : chacun renvoie à sa source. Ni carte, ni commit.

## 1. Coder-30B à b=12 : rattrapable à −15 % en ~5 jours-sessions, pas battable ; l'énergie est un autre problème

Le pas (12 jetons), ms — nous (vis-à-vis, eager) / vLLM (profil ÷ 8) / borne octets à 1 050 Go/s / levier / gain prédit / coût :

| poste | nous | vLLM | borne | levier | prédit | jours |
|---|---:|---:|---:|---|---:|---:|
| MoE gate·up + down | 6,0 | 5,95 | 4,9 (≤ 68 experts distincts/couche) | **aucun** : mêmes octets, 82 % de borne ; M2 ncu ne fera que confirmer le dénominateur | 0 | 0,5 |
| projections attention (`int8_gemv` ×192) | 2,62 | 1,83 | 0,86 int8 / 0,49 FP4 | 1aj W4A4 MMA (0,51 Go au lieu de 0,91) | −1,0 | 2 (+0,5 PPL) |
| `lm_head` int8 | 0,88 | 0,44 | 0,30 | 1aj | −0,4 | inclus |
| attention paginée | 0,72 | 0,66 | 0,22 | rien à ctx court | 0 | — |
| routage + glue + normes | 1,7 | 2,2 | — | fusion norme+projection, q/kv/o en un lancement (192 × ~13 µs ≈ plancher de lancement) | −0,5 à sceller | 2 |
| hors rejeu | 826 interne → 717 HTTP (−13 %) ; 0,3 ms interne (poste1) | vLLM mesuré **hors ligne** | — | voie HTTP (2 `decode()`/jeton, SSE) ; régime à apparier d'abord | −1,5 | 1 |

Rejeu 14,5 → ~12,0-12,5 ms → 960-1 000 t/s interne, soit **−17 à −20 % sous 1 198**. Le plancher est commun : mêmes octets de poids par pas, et leur GEMM groupée CUTLASS est déjà à 100 % de sa borne (bead 0si) — il n'y a pas de ×1,45 à prendre dans le MoE, qui est 48 % du pas. Battre vLLM à b=12 sur ce modèle exigerait de lire moins d'octets (experts distincts réels < 68, prior de routage) : non chiffré, pas avant M2.
**Prefill** (19 148 vs 34 788, ×1,82) : sur le pas L=2048 (~113 ms), `int8 dense` pèse 30 ms (poste4:4346) contre 3,3 ms de GEMM dense FP4 chez eux — c'est **1aj appliqué au prefill (M=2048)**, pas 0si, qui rapporte le plus : 30 → ~7 ms prédit, pas → ~90 ms, ~24 000 j/s ; 0si TMA ensuite (24 → 17 ms) → ~26 000. Reste ×1,3.
**Énergie — l'anomalie qui vaut plus que tous les noyaux** : vLLM 0,202 J × 1 198 t/s = **242 W nets** (~310 W bruts) ; nous 0,55 J × 717 = **394 W nets**, plafond 400 W saturé (bridage « puissance » relevé). Temps ×1,45, **puissance ×1,6, énergie ×2,9 pour les mêmes octets et le même format.** Les leviers de temps ci-dessus rendent au mieux ×1,2 en J/jeton ; le ×1,6 de puissance n'a aucune cause attribuée — ni ici, ni pour le −8,5 % vs llama.cpp dense (jamais attribué, ETABLI:1707). Hypothèse la plus probable, réfutable en 1 jour : nos noyaux dépensent ≥ 3× plus d'instructions par octet lu (déquantification E2M1→bf16 et int8 sur cœurs CUDA dans `nvfp4_gemv_grouped`/`int8_gemv`/LUT, `paged_attn` à 8,6× sa borne) là où la MMA block-scaled de CUTLASS consomme l'E2M1 sans ALU. **Mesure** : ncu `sm__inst_executed.sum` et `dram__bytes_read.sum` par noyau, les deux moteurs (`--target-processes all` sur l'EngineCore vLLM), plus J par poste (chaque noyau du pas en boucle 10 s, compteur d'énergie NVML). Réfutation : instr/octet ≤ 1,5× vLLM → la puissance vient d'ailleurs (horloge sous plafond, bulles, 3080 Ti non comptée) et il faut le dire avant tout noyau de plus.
**Verdict (1)** : débit rattrapable à −15 % (1aj + lancements + HTTP, ~5 jours-sessions : poste4 4, poste2 0,5, poste3 1) ; pas battable à b=12 sur ce modèle sans lire moins d'octets ; énergie : ne rien promettre avant instr/octet.

## 2. Où battre vLLM/llama.cpp de façon défendable sur cette carte — et la mesure d'une journée

| # | terrain | ce qu'on sait | prédiction scellée | mesure (1 j) | ce qui la réfute |
|---|---|---|---|---|---|
| a | **énergie brute à faible lot (b=1-4)** | repos 17 W (nous) vs 64 W (vLLM), même carte, instants différents (audit-a2, réserve) ; résident b=1 = 242 j/s : 47 W de repos = 0,19 J/jeton, soit tout le net de vLLM ; llama.cpp b=1 : 1 297 vs 1 432/1 161 j/kJ, indécidable (poste4:3755) | J/jeton **brut** b=1 : nous < vLLM d'au moins 30 % ; b=4 : ≥ 15 % ; vs llama.cpp : ≥ 10 % | même modèle, HTTP des deux côtés, carte exclusive, repos 30 s, compteur NVML, colonnes brut/net (duck-chef), 10 ABBA, ≥ 512 jetons | repos vLLM ≤ 25 W isolé, ou brut vLLM ≤ nôtre à b=1 |
| b | **MLA NVFP4 sur sm_120** (GLM-4.7-Flash / GLM-42B) | ×1,40 vs concurrent après 6wa (54,38 ms b=12, poste4:4335) ; vLLM : FlashInfer absent, FA refuse KV fp8, MLA SM120 = FP8 seulement (poste7-veille §2.4), aucun checkpoint NVFP4 GLM connu | vLLM ne sert pas GLM-4.7-Flash en NVFP4 sur cette carte, ou ≤ 0,8× notre débit b=12 avec 2-4× nos octets | `vllm serve` sur le checkpoint GLM le plus proche, b=1 et 12, t/s + J, octets lus/jeton publiés | vLLM ≥ nous à octets ≤ 1,5× les nôtres |
| c | **llama.cpp à b=12 sur MoE** | ×1,91 débit / ×1,43 J le 13/09 — binaire e34f042 (avril) ; poste8 a reconstruit 4a899373 le 14/09 : noyaux MoE CUDA réécrits depuis | l'écart tombe à ×1,3-1,5 mais tient | même protocole que le 13/09, binaire 4a899373, Q4_K_M vs NVFP4, b=1/4/12 | ≤ ×1,1 → plus rien à revendiquer contre llama.cpp |
| — | **à ne pas revendiquer** | modèles qui ne tiennent pas (exil par expert ×9 b=1, ×23 b=12 ; ggrun 37 t/s côté llama.cpp/CPU) ; prefill/TTFT (×1,8-2 derrière) ; b=12 Coder-30B vs vLLM | — | — | — |

## 3. Ce qu'il faut arrêter

* **Exil par expert au décodage b=12 comme chantier de débit** : ×23 réel et reproductible sous carte exclusive, trois gestes sans effet (table, AUTOPIN, graphes) — geler jusqu'à la mesure « experts manqués sur CPU » (A10, 2 h) ; garder b=1 (×9).
* **TabbyAPI** : ×4-8 derrière, outil mono-utilisateur — clos, ne plus le rebancher.
* **Rouvrir ce qui est clos par mesure** : hôte/pipeline (+0,2 %), horloge SM, `-lmc` (une seule horloge mémoire), tuile 128, A6, A8 (le 14B bf16 ne tient plus) — remplacer A8 par le profil nsys sur Coder-30B directement.
* **Duels non appariés** : vLLM hors ligne contre acvram HTTP (1 198 vs 569/717) — un seul transport des deux côtés, régime porté par le nom du banc (règle 6).
* **`--speculative auto` sans condition de lot** : −50 % débit, +44 % J à b=12 mesurés — aucun chiffre officiel tant que ce n'est pas conditionné.
* **Godet 16 pour b=12** : ne pas y toucher, vLLM capture 1,2,4,8,16,24 et paie le même.
* **Quantification côté carte** (A5 KLD, témoin MoE bf16) : geler tant que l'énergie n'est pas attribuée ; A7 (conversion sans carte) continue.

## 4. Ordre proposé

1. Énergie : instr/octet par noyau + J par poste, deux moteurs (1 j, poste4 ncu + poste3 NVML) — décide si les noyaux sont le bon chantier.
2. 1aj W4A4 projections + `lm_head`, décodage **et** prefill (poste4 3 j, poste2 PPL 0,5 j).
3. Mesures 2a (brut b=1/4) et 2c (llama.cpp 4a899373) — les deux chiffres publiables de la semaine (poste2/poste8 1 j).
4. 2b MLA vs vLLM (1 j). 5. 0si TMA (2 j), après 1aj, pas avant.

## Correction 14/09 soir — 2a réfutée en ampleur à b=1, en sens à b=4

Source : `energie-brute-faible-lot-14-09.md:14-21` (poste3, branche `poste3`, 42545de). Ma prédiction scellée en §2a — « b=1 : nous < vLLM d'au moins 30 % ; b=4 : ≥ 15 % » — est **fausse** : b=1 −3,8 % (sens juste, ampleur ×8 trop grande), b=4 **+20 %** (sens faux). Cause : la prémisse « repos 17 W vs 64 W » était un repos **froid** (rien chargé, audit-a2) appliqué à un régime chaud ; mesuré modèle chargé : 68-76 W nous, 89-97 W vLLM. Règle 4, « chiffre exact hors de son régime » — la mienne.

Ce que le tableau de poste3 contient et qu'elle n'a pas déplié (W brut = J/jeton × t/s ; net = brut − repos ; pas = b / t/s) :

| b | acvram W brut / net / pas | vLLM W brut / net / pas | net W nous/vLLM |
|---|---|---|---|
| 1 | 326 / 258 / 4,41 ms | 296 / 207 / 5,05 ms | 1,25 |
| 2 | **401** / 325 / 6,13 | 252 / 155 / **10,05** | 2,09 |
| 3 | **426** / 353 / 7,04 | 275 / 182 / **10,05** | 1,94 |
| 4 | **435** / 361 / 7,40 | 267 / 172 / **10,05** | 2,10 |

À b=1 : net 1,139 J/jeton nous vs 1,046 vLLM (**+9 % pour nous**) ; repos 0,301 vs 0,451. **Le −3,8 % brut est entièrement le plancher de repos (−0,150 J) moins la perte des noyaux (+0,093 J).** Le créneau 2a existe, mais c'est le repos et le pas plus court qui le paient, pas les noyaux — à publier tel quel.

Deux contrôles à rendre avant de citer le crossover, tous deux tirés du tableau lui-même :

1. **Plafond 400 W dépassé** : 401 / 426 / 435 W brut à b=2/3/4, plafond 400 W posé (chef.md, sudoers). Soit le plafond n'était pas actif (régime ≠ des duels b=12 à 400 W, à porter par le nom du dossier), soit `energie.py` intègre un `power.draw` instantané qui dépasse la moyenne contrainte, soit J et t/s n'ont pas la même fenêtre. Contrôle : `nvidia-smi -q -d POWER` (limite appliquée) relevé pendant, et W×s recalculé depuis le JSON (`scratchpad/resultat-energie-brute-vllm-14-09.json`). Quelle que soit la cause : **dès b=2 nous sommes au plafond, vLLM ne dépasse jamais 300 W** — le débit b=2-4 est borné par la puissance, pas par les noyaux.
2. **vLLM change de régime entre b=1 et b=2** : pas 5,05 ms à b=1, 10,05 ms à b=2, 3 et 4 — **identique à son pas b=12** (10 ms, vis-a-vis-decodage-b12-14-09) ; par séquence 99,5 t/s constant dès b=2. Hypothèse : rembourrage au godet capturé avec lignes factices routées vers les experts (mêmes octets qu'un lot plein), ou repli eager. Contrôle 15 min : tailles capturées dans le journal vLLM, ou relance avec `--cudagraph-capture-sizes 1,2,3,4`. Si le pas b=2 tombe vers 5,5-6 ms, vLLM brut b=2 ≈ 0,70 J/jeton et **le crossover passe sous b=2 : 2a ne tient qu'à b=1**.

Pour le tableau instr/octet de poste4 : le rapport de puissance nette est 1,25× à b=1 et ~2× dès b=2 (plafond touché, donc ≥ 2 hors plafond). 1,25× à b=1 ne réfute pas « instr/octet ≥ 3× » : à b=1 le pas contient les bulles du plancher de lancement (4,41 ms pour 48 couches) et la puissance moyenne les intègre ; seul J par noyau en boucle tranche (protocole §1). Prédiction ajoutée : **si instr/octet explique la puissance, le rapport par noyau sera ~2× et ne dépendra pas de b** ; s'il varie avec b, la cause est dans le pas (bulles, plafond), pas dans les noyaux.

Ordre inchangé. Ce qui se publie de 2a aujourd'hui : « b=1 brut −3,8 %, payé par le repos » ; le crossover et b=2 attendent le contrôle 2.
