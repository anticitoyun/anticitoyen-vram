# Revue finale de la pièce 104 (poste5, 20dad4b7) — à sec — 23/09 (poste1)

* **Chemin int8 intact : oui.**
  * Le corps de `kv_write_int8_kernel` est identique à celui de main (sha256 du texte de la fonction bf7ecb3e… des deux côtés).
  * La quantification int8 torch est inchangée (`kvcache.py:479-481`) : elle n'est plus partagée avec k8v4, qui a désormais `kv_k8v4.quantifier_k`.
  * La répartition dans `kernels/__init__.py` ne change que par l'ajout de la branche k8v4, gardée par `cfg.k8v4`.
* **Défaut 1 de la revue v1 : corrigé.** `graphs.py:435` admet `("int8", "k8v4")`, et le repli sans noyau paginé est désormais nommé (`graphs.py:641`, `_eager(...)`).
* **Défaut 2 de la revue v1 : corrigé.** `__fdiv_rn(amax, 7.f)` pour l'échelle de V. K divise aussi en IEEE (`__fdiv_rn(m, 127.f)`, `__fdiv_rn(x, sc)`), au bit de `quantifier_k` (division tenseur par tenseur).
* **L'int8 servi a la MÊME divergence d'un code aux demi-entiers : oui.** Non corrigé ici.
  * `acvram_kernels.cu:4217` : `m / 127.f`, approchée sous `--use_fast_math`.
  * `:4220` : `inv = 1.f / sc`.
  * `:4223` : `x · inv`.
  * Contre le jumeau torch (`kvcache.py:480-481` : `amax / 127.0`, puis `x / scale`).
  * Le chemin servi (le noyau) et son repli torch rendent donc ± 1 code aux demi-entiers ; c'est l'écart que C5-b avait mesuré.
  * Le corriger changerait la sortie int8 servie par défaut : c'est une pièce à part, avec test d'équivalence et KL.
