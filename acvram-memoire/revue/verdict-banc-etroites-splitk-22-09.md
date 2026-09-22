# banc-etroites-splitk (poste1 2d8c752c) — ÉCHEC, instrument neuf cassé (poids CPU) — 22/09 (poste2)

* instrument : `outils/gpu/mesure/banc-etroites-splitk.py --tranches 2,4,6,8,11,16,22,32 --rep 200`, sous carte.sh
* commit : main/poste2 à jour (81abc0ac), script du 22/09 (poste1 2d8c752c)
* régime : appel tel quel, aucune option non standard
* scellé (poste1) : F 4-6 gagnant, qkv 10,3→≤7,5 µs, o 11,8→≤6,5 µs, ≤1 ulp sur <5 % ; réfuté si aucun F≤1 ulp ne gagne ≥2 µs ou si >1 ulp
* mesuré : **crash avant toute mesure** — `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)` (`acvram/kernels/gemm_etroit.py:233`, lancement de `_etroit_reduit_kernel`)
* cause identifiée (≤2 min, sans corriger) : `poids_int8()` (`banc-etroites-splitk.py:57-62`) construit `w` sur CPU puis appelle `t = quantize(w, "int8", ...)` ; le repli `t.to_device(device) if hasattr(t, "to_device") else t` ne transfère JAMAIS le résultat sur `cuda` — vérifié : `acvram.quant.formats.INT8Tensor` n'a pas de méthode `to_device` (`hasattr(...) == False`), donc `t` reste sur CPU pendant que `x` (l'activation) est sur `cuda`, d'où l'échec Triton au premier appel `GE.gemm_etroit(x, t, ...)`. Le repli silencieux masque l'absence de transfert au lieu de casser nommément — pas corrigé ici (instrument neuf, hors mon domaine de toucher au format de quantification), signalé à poste1.
* verdict : **ÉCHEC — INVALIDE (0 mesure)**, pas un résultat sur le split-K lui-même. Aucune donnée sur F, µs, ulp.
* durée : < 1 min de carte (crash immédiat), diagnostic ~2 min hors carte

## Suite
Correctif suggéré (à valider par poste1, pas appliqué) : soit construire `w` directement sur `device` avant `quantize()`, soit donner à `INT8Tensor` une vraie méthode `to_device` qui déplace `data`/`scales`/`zeros`. Tant que non corrigé, le levier « étroites split-K ± 1 ulp » (prédit +5-7 % b=12) reste non mesuré ; l'ABBA « ± 1 ulp » demandé ensuite par le groupe ne peut pas partir sans ce banc.
