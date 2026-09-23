# Sage — clôture à 23 h 59 : ce qui se termine ce soir, ce qui ne peut pas, la file de carte à la minute (19/09, 17 h 35)

Source : utilisateur 17 h 35 (« tout doit être terminé avant 23 h 59 ce soir, tout le groupe ») ; remplace l'échéance 07 h 00 de `sage-p2-ppl-instrument-file-7h-19-09` (dont les § 1-2 restent valables). Push automatique (utilisateur 17 h 30).

## 1. Ce qui est terminé à 23 h 59 — et ce qui ne peut pas l'être, dit avant

| piste | à 23 h 59 | pourquoi pas plus |
|---|---|---|
| 1 éco | **cellules éco b=12 et b=1 publiées, `acvram eco` en main** ; gouverneur par lot **si** E1-bis montre que 2700 coûte à b=1 (le réglage passe par `sudo -n nvidia-smi -lgc`, déjà autorisé : pas de service à écrire) | — |
| 2 W4A8 | **porte A8 tranchée** (PPL fausse-quant, 3 tranches) | le noyau (MMA k=16 + échelles, capture, équivalence, PPL) est un chantier de 3-5 jours : **pas ce soir**, on ne l'annonce pas |
| 3 énergie décodage | **ncu M1 + M2 rendus** (instr/octet, W et octets DRAM par noyau à b=12) ; **régime `NARROW_GEMM=1` mesuré à b=12** (projections en GEMM étroit Triton au lieu des GEMV, `regime.py:60`, défaut 0 ; passe de capture godets obligatoire, REGLES § 3 : ce régime a planté en service le 17/09) | un nouveau noyau de projections : pas ce soir |
| 4 spéculation | **déjà au défaut** : n-gram actif à b ≤ 2 (`runner.py:429`, `ACVRAM_SPECULATION_LOT_MAX=2`), taux 1,61 mesuré 13/09 — la cellule b=1 le contient ; à dire dans REPRISE | MTP GLM : 2-3 jours, pas ce soir |
| 5 GLM FP8-MLA | **porte tranchée si le temps** : fausse-quant E4M3 par jeton sur l'entrée de q_b/kv_a/o de GLM (même arithmétique que `fausse_quant_a8(fmt=e4m3)`, branchée dans `layers.py` sur les projections MLA), PPL 1 tranche dans la fenêtre G1, scellé ratio ≤ 1,005 | le chemin `_scaled_mm` : demain |
| 6 godets b · KV int8 · GPTQ | **non** — chacun 2 jours et une reconversion | dits dans REPRISE § 10 avec leurs chiffres |
| 7 cache d'experts / 119B | **non** — sur le oui de l'utilisateur | — |

Un résultat « faux » sur une porte est un livrable ; un noyau annoncé sans mesure n'en est pas un.

## 2. File de carte à la minute (Manon tient la file ; carte 1 = 3080 Ti libre pour le doctor)

| créneau | fenêtre | qui |
|---|---|---|
| → 18 h 05 | E1-bis b=1 (`-lgc 2700`, `2400`) + genou b=12 (`1 800`, `1 950`) — en cours | Manon |
| 18 h 05 → 18 h 15 | `test_gemv_marlin.py` (carte 0) ‖ `CUDA_VISIBLE_DEVICES=1 acvram doctor` + `test_paquet_charge_utile.py` (carte 1) | Océane ‖ Jérôme |
| 18 h 15 → 19 h 05 | **G1** GLM b=12 t/s + J + prefill défaut ; + porte FP8-MLA (1 tranche) si le commit d'Océane est là | Manon |
| 19 h 05 → 19 h 20 | **P2 ligne 2** PPL i8c 3 tranches (tête au défaut, PPL par tranches — commit Océane avant 19 h) | Manon |
| 19 h 20 → 19 h 40 | **porte A8** `PREFILL_A8=both|gateup`, int8 puis e4m3 | Manon |
| 19 h 40 → 20 h 20 | **`NARROW_GEMM=1` b=12** : capture godets {1, 2, 8, 16} `warm_graphs` puis ABAB contre défaut, libre et `-lgc 2100` ; scellé : t/s ≥ 0,98 × défaut ET J net ≤ 0,95 × → entre au défaut ; sinon reste 0 | Manon |
| 20 h 20 → 20 h 50 | gouverneur par lot (si construit) : lot mixte 1 → 12 → 1 dans une fenêtre, J/jeton ABAB contre libre ; scellé J ≤ 0,90 × libre ET t/s b=1 ≥ 0,98 × | Manon |
| 20 h 50 → 22 h 30 | **ncu M1** (60 min, une passe, `--launch-count` borné, 3 noyaux nommés) puis **M2** (45 min, `dram__bytes_read.sum`, cache froid et `--cache-control none`) | Océane |
| 22 h 30 → 23 h | bilan Sage `sage-bilan-19-09-23h.md` : revendication finale, ce qui est tenu / faux / non fait | Sage |
| 23 h → 23 h 45 | comparatif, `ETAT`, `REPRISE` (une ligne P2, ligne éco b=1, n-gram), `INDEX`, push | Jérôme |

Toute fenêtre qui déborde de 15 min se coupe et se dit ; rien ne se reprend après 23 h 45.

## Ordre

* **Océane** — à sec, dans cet ordre, un sous-agent par point : (1) tête au chemin défaut sous `cublas` + `perplexity()` par tranches de 256 + `empty_cache()` (note 7 h § 2), commit avant 19 h ; (2) `acvram eco {2700|2100|off}` (`sudo -n nvidia-smi -lgc/-rgc`, refus explicite sans droit, `regime_ligne()` lit `nvidia-smi -q -d CLOCK`) ; (3) porte FP8-MLA : `fausse_quant_a8(fmt=e4m3)` sur l'entrée de q_b/kv_a/o (`layers.py`), variable `ACVRAM_MLA_A8=e4m3` dans `regime.VARIABLES`, test à sec, commit avant 18 h 45 sinon la porte saute ; (4) gouverneur par lot **seulement si** E1-bis rend 2700 < 340,1 à b=1 : dans `runner.py`, même forme que la garde de spéculation (lot réel sur 32 pas, libre b ≤ 2 / 2700 / 2100 b ≥ 8), appel `sudo -n nvidia-smi -lgc`, `regime_ligne()` porte l'état, test à sec (simulateur) ; (5) 20 h 50 : ncu M1 puis M2, wrappers de régime, durée mesurée sur 3 noyaux avant d'annoncer.
* **Manon** — file § 2 ; un verdict six lignes par fenêtre ; scellés ci-dessus, aucun resserré après.
* **Jérôme** — 18 h 05 doctor 0.6.14 sur la carte 1 + `test_paquet_charge_utile.py` ; fusion après chaque fenêtre, push automatique ; comparatif et `ETAT` à chaque verdict ; `REPRISE` § 2 : ligne « n-gram au défaut à b ≤ 2 » maintenant ; 23 h → 23 h 45 clôture.
* **Sage** — relit chaque verdict dans le quart d'heure ; 22 h 30 bilan.
