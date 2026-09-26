# Pièce — acvram_mojo, étape 0 : faisabilité sm_120 (poste4, bead anticitoyen-vram-3yf.2)

* instrument : `pixi` 0.81.0, `mojo` 1.2.0.dev2026092305, `MAX` 26.7.0.dev2026092305 (canal `max-nightly`),
  venv isolé `~/mojo-isole-poste4/` (hors `.venv` du dépôt, PATH exporté par commande, jamais dans `.bashrc`)
* commit : `0748d676` (contrat), driver 595.91.07, RTX 5090 `compute_cap` 12.0 (`ACVRAM_CARTE=0`)
* régime : `carte.sh` type `mesure` par défaut, ≤ 300 s, deux prises en file après poste1/poste3 (134 s puis 237 s
  d'attente, détenteur nommé la seconde fois — le verrou tient)
* scellé : Mojo compile-t-il et exécute-t-il un noyau GPU sur sm_120 ? seuil = oui/non, chiffré et daté dans
  les deux cas — pas de seuil de débit (question de faisabilité, pas de performance)
* mesuré : `noyau_trivial.mojo` (`max.gpu.host.DeviceContext`, `max.gpu.{thread_idx,block_idx,block_dim}`,
  1024 threads, écrit `2*idx` par case) compilé (`mojo build`) puis exécuté (`carte.sh /tmp/noyau_trivial_bin`)
  → `noyau trivial :  OK  n= 1024`, code 0
* verdict : **OUI, Mojo/MAX compile et exécute sur sm_120** (RTX 5090). API très éloignée de la doc publique
  connue (nightly 26.7 du 23/09) : `fn` retiré (`def` partout, `raises` explicite), `Pointer`/`UnsafePointer`
  exigent une origine explicite (`MutUntrackedOrigin` marche, `_` non-lié échoue à l'instanciation de
  `enqueue_function`), les types passés à un noyau doivent être `DevicePassable` (`Int`/`UInt` refusés, `Int32`
  accepté) ; `gpu.host`/`gpu.id` de la doc publique sont en réalité sous `max.gpu.host`/`max.gpu` ici — cinq
  itérations de compilation (gratuites, sans carte) avant la première prise utile
* durée : faisabilité 0 → OK, deux prises carte ≤ 300 s chacune (tenu), reste (itération de syntaxe) à sec

## Formats disponibles / ce que MAX sert nativement pour Qwen3 (lu dans le paquet, pas mesuré)

* Qwen3 **dense** (`max/pipelines/architectures/qwen3/model_config.py`) : `SUPPORTED_ENCODINGS` =
  **bfloat16** (défaut), **float32**, **float8_e4m3fn** — **aucun nvfp4/mxfp4** pour cette architecture.
* nvfp4/mxfp4 existent dans MAX mais réservés à d'autres architectures (Gemma4, Kimi-K2.5, **Qwen3.5**-MoE
  `architectures/qwen3_5/quantization.py`), pas à Qwen3 dense — cohérent avec l'étape 1 du contrat (Qwen3
  DENSE petit d'abord) : bf16 est le format natif à comparer, fp8_e4m3fn est le seul repli quantifié.

## Reste

Étape 1 (squelette `moteurs/acvram_mojo`, contrat identique à Rust) : bloc suivant, un message par étape.
