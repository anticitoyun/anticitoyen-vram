# poste7 — P2 ligne 2 : la tête a changé de régime sous `cublas`, pas P2 ; file de carte fermée jusqu'à 07 h 00 (19/09, 17 h 45)

Source : `verdict-p2-ppl-19-09` (poste2, c05871d) ; `c62e2ef` (poste1 : `tete_int8_entree_bf16`) ; utilisateur 17 h 40 : **tout doit être fait avant demain 07 h 00**.

## 1. Pourquoi la PPL i8c déborde, et pourquoi ce n'est pas une fragmentation

`c62e2ef` fait rendre `True` à `tete_int8_entree_bf16(n)` pour tout `n` dès que `PREFILL_INT8 ∈ {a8, cublas}` : la tête INT8 g128 (151 936 × 2 048) prend alors, **pour les 2 048 positions d'une fenêtre de PPL**, le chemin W8A8 (`gemm_w8a8.py:126`, `CHEMINS_INT8['a8'] = 2` = la tête, deux fenêtres) — celui qui déquantifie le poids en fp32 puis le requantifie en E4M3 **à chaque appel** (REGLES § 4, 17/09 : 56 % plus lent que bf16 pour cette raison). Arithmétique du pic, tête seule :

```
déquant fp32 du poids   151 936 × 2 048 × 4 o = 1,245 Gio
copie E4M3                                       0,311 Gio
sortie fp32 [2 048 × 151 936]                    1,245 Gio
log-softmax fp32                                 1,245 Gio
                                                ≈ 4,0 Gio  (défaut : bf16 0,62 + fp32 1,245 ≈ 1,9 Gio)
poids Coder + copies i8c                        ≈ 17 Gio
                                                ≈ 21-22 Gio alloués  → « 22,1 Gio alloués » de poste2
```

Le pic **alloué** est expliqué par la tête ; les 8 Gio entre alloué et réservé sont le cache de l'allocateur sur des blocs de 1,2 Gio qu'il ne réemploie pas pour des demandes plus petites — `expandable_segments` déplace le problème, il ne le retire pas. Ce n'est pas P2 : **en service, la tête n'est jamais calculée à 2 048 lignes** (décodage n = 1, prefill = dernier jeton → GEMV). Le correctif a donné à la PPL un régime de tête que le moteur ne sert pas — l'inverse de ce qu'il voulait (« le juge emprunte le chemin servi »).

Contrôle qui peut rendre faux : après le correctif § 2, `CHEMINS_INT8['a8'] == 0` sur une fenêtre, alloué max < 20 Gio, réservé − alloué < 2 Gio, fenêtre 2 passe. Si la fenêtre 2 déborde encore avec la tête au défaut : c'est bien dans `_int_mm`/`_i8c_poids` (copie reconstruite par appel ?), et le diag 12 s de poste2 le dira.

## 2. Décision

* **La tête garde le chemin du défaut sous `cublas`** : `tete_int8_entree_bf16` redevient `n ≤ _INT8_GEMV_MAX` (P2 = q/k/v/o, rien d'autre ; une PPL « P2 » qui change aussi la tête mesure deux choses). Le test de c62e2ef qui l'assertait est retourné : sous `cublas`, à n = 2 048, la tête prend le chemin **défaut**.
* **Instrument durci, régime-neutre** : `perplexity()` calcule la tête et le log-softmax **par tranches de 256 positions** et appelle `torch.cuda.empty_cache()` entre fenêtres ; test à sec : PPL identique à 10⁻⁶ à la version non tranchée sur le petit modèle de test (l'arithmétique par ligne est indépendante). Cela ramène le pic tête de 1,9 à 0,25 Gio et vaut pour tous les régimes.
* P2 **reste opt-in tant que la ligne 2 n'est pas mesurée** ; elle se mesure cette nuit (15 min) ; ≤ 1,020 → P2 au défaut avant 07 h, > 1,020 → opt-in, dit.

## 3. File de carte fermée jusqu'à 07 h 00 (poste2 tient la file ; poste1 par pointeur)

| créneau | fenêtre | qui | verdict |
|---|---|---|---|
| 17 h 45 → 18 h 20 | E1-bis b=1 (`-lgc 2700`, `2400`) + genou b=12 (`1 800`, `1 950`) | poste2 | `verdict-eco-lgc-b1-genou-19-09` |
| 18 h 20 → 18 h 30 | `test_gemv_marlin.py` | poste1 | — |
| 18 h 30 → 19 h 15 | **G1** GLM b=12 t/s + J + prefill défaut | poste2 | `verdict-glm-b12-19-09` |
| 19 h 15 → 19 h 30 | **P2 ligne 2** PPL i8c 3 tranches (après le commit § 2) | poste2 | `verdict-p2-ppl-19-09` addendum |
| 19 h 30 → 19 h 50 | **porte A8** (après `fausse_quant_a8`) | poste2 | `verdict-porte-a8-19-09` |
| 19 h 50 → 20 h | `CUDA_VISIBLE_DEVICES=1 acvram doctor` (0.6.14) + `test_paquet_charge_utile.py` | chef | dans `ETAT` |
| 20 h → 21 h | **ncu M1** : une passe bornée (`--launch-count`, 3 noyaux nommés : GEMV projections b=12, `nvfp4_gemv_marlin`, `int8_gemv`) — instructions par octet DRAM et W par noyau, régime défaut | poste1 | `verdict-ncu-m1-19-09` |
| 21 h → 21 h 45 | **ncu M2** : `dram__bytes_read.sum` du pas b=12, cache froid et `--cache-control none` | poste1 | `verdict-ncu-m2-19-09` |
| 21 h 45 → 06 h | carte libre ; toute fenêtre supplémentaire = pointeur à poste7 d'abord | — | — |
| 06 h 30 | bilan : revendication finale, comparatif, `ETAT`, `REPRISE` § 2/§ 10 | poste7 puis chef | `poste7-bilan-nuit-20-09` |

## 4. Livrables à 07 h 00 (rien d'autre n'est promis)

1. Comparatif : cellules **éco** b=12 (deux réglages) et b=1 (E1-bis), GLM b=12, prefill P2 ; revendication réécrite sur les seules cellules mesurées.
2. P2 tranché (défaut ou opt-in) ; porte A8 tranchée (chantier W4A8 ouvert ou fermé) ; M1/M2 : le poste énergie du décodage **nommé** avec ses chiffres.
3. `.deb` 0.6.14 installé, `acvram doctor` sans alerte sur la 3080 Ti ; `acvram eco {2700|2100|off}` en `main` (poste1, à sec).
4. `REPRISE.md` § 2 et § 10 réécrits (une fois), `ETAT.md` ≤ 40 lignes, `INDEX` à jour, tout poussé sur « oui » utilisateur.

## Ordre

* **poste1** — après ses 6 min : (1) `tete_int8_entree_bf16` → `n ≤ _INT8_GEMV_MAX` seul, test retourné ; `perplexity()` par tranches de 256 + `empty_cache()` entre fenêtres, test d'égalité à 10⁻⁶ à sec ; commit `poste1-11`, pointeur à poste2 (avant 19 h 15) ; (2) `fausse_quant_a8` (note `poste7-w4a4-clos-w4a8-porte`) avant 19 h 30 ; (3) `acvram eco {2700|2100|off}` (note `poste7-e1-eco-tenu`) ; (4) 20 h-21 h 45 : ncu M1 puis M2, une passe par prise, `--launch-count` borné, durée annoncée = mesurée sur 3 noyaux avant d'annoncer (REGLES § 6), régime dans un wrapper (`sudo -n` efface l'environnement).
* **poste2** — file § 3 dans l'ordre ; six lignes fixes par verdict ; un pointeur par verdict.
* **chef** — 19 h 50 : doctor 0.6.14 sur la 3080 Ti + `test_paquet_charge_utile.py` ; fusions après chaque fenêtre (jamais pendant) ; comparatif et `ETAT` à chaque verdict ; `REPRISE` § 2/§ 10 après P2 ligne 2 ; 06 h 45 : tout prêt pour le push automatique (utilisateur 17 h 30).
* **poste7** — relit chaque verdict dans l'heure ; 06 h 30 : `poste7-bilan-nuit-20-09.md`.
