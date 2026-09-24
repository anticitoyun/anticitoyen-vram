# Verdict — pièce 161 : le .so du port Marlin était PARTAGÉ entre worktrees (poste6, 24/09) — prouvé, corrigé

* **instrument** : `scratchpad/poste6-p161-24-09/preuve-cache-partage.sh` (deux worktrees aux sources CUDA identiques, cache
  isolé, guetteur du .so à 5 ms) → `preuve.txt` ; `tests-161.sh` → `tests-161b.txt`
* **commit** : preuve sur 676cdf59 (avant) ; correctif 778b6d08 (branche poste6) · **régime** : à sec, sous mon verrou (CPU)
* **scellé** : ordre de chef (3 points) · **mesuré** : oui · **verdict** : mécanisme PROUVÉ ; absence du .so NON observée
  (réécriture en place) ; correctif : même empreinte à sec et sur carte, 60 tests Marlin verts · **durée** : 30 s + 34 s (attentes 583 et 440 s)

## 1. Mécanisme (code d'avant, `preuve.txt`)
| étape | arbre | `charger(compiler=…)` | durée | sha .so | `-I` de build.ninja |
|---|---|---|---|---|---|
| 1 | poste6 | True | 0,9 s (ccache) | 5eddf6d1e690 | poste6/… |
| 2 | poste6-147 | **False** | **25,0 s de nvcc** (ccache aveugle au chemin) | **241f87891841** | **poste6-147/…** |
| 3 | poste6 | False | 0,9 s, rebâti (ccache) | 5eddf6d1e690 | poste6/… |
| 4 | poste6 | False | 0,7 s, rien à rebâtir | 5eddf6d1e690 | poste6/… |
| 5 | .so retiré, poste6-147 | False | 0,7 s → **None** (repli « port non compilé ») | absent | — |

* `charger(compiler=False)` ne refusait que le .so ABSENT ; présent, il appelait `torch.utils.cpp_extension.load` → ninja ; les
  commandes portent le chemin du worktree (`-I…/travail/<arbre>/…`, sources absolues) → tout changement d'arbre appelant
  RECOMPILE (25 s, 16 cœurs) sous le verrou de la prise en cours, ou hors verrou pour un contrôle à sec (mon incident 19:34:26
  pendant poste3-p159-pipefix). Deux arbres alternants = ping-pong de deux binaires de sha différents pour les MÊMES sources.
* Absence : NON observée à 5 ms — le lien réécrit le .so EN PLACE (~0,3 s d'ELF partiel : erreur pour un chargeur concurrent,
  pas repli ; contenu non mesuré). Le repli exige un .so absent : compilation interrompue, ou premier chargement d'un poste.
* 160 : mon profil 8 × 78 (19:11) n'a montré AUCUN noyau Marlin ni `depaqueter_marlin` → disposition NATURELLE dans mon
  processus, cause non établie (bilan non imprimé). Erratum au verdict 160 : « profil sur la naturelle ; le défaut servi
  (Marlin unique + L3') dépaquette au même coût par construction : “les lancements ne dominent pas” tient, le poste dominant
  (dépaquetage + GEMM bf16) est à confirmer sur la disposition servie ».

## 2. Correctif (778b6d08)
* `empreinte_sources()` : sha256 (12 hex) des 23 sources/en-têtes (chemins relatifs + octets) ; `dossier_cache()` =
  `~/.cache/acvram/marlin_port-<empreinte>` (`ACVRAM_MARLIN_CACHE` = racine). Sans les archs : à sec `_arch_flags()` ne voit
  pas la carte → deux empreintes (1re prise : 9e4f5791 à sec contre 2eb083f8 sur carte, .so « absent », 14 tests sautés).
* Sources COPIÉES dans `<cache>/src/` (marqueur `.complet`), ninja compile de là : commandes sans chemin de worktree → un
  second arbre aux mêmes sources ne rebâtit rien ; `-DACVRAM_MARLIN_EMPREINTE=…` interdit à ccache un objet d'une autre version.
* Le moteur (`compiler=False`) ne lance JAMAIS ninja : `torch.ops.load_library` du .so de son empreinte (0,8 s), ou None et
  repli nommé, désormais IMPRIMÉ dans tout journal (`[acvram] disposition Marlin : REPLI au naturel — …`) en plus de la ligne
  de régime du service (`+marlin(repli:…)`, runner.py:916). `tests/test_marlin_cache_empreinte_161.py` : 5 tests à sec
  (empreinte dans le nom, jamais l'ancien dossier partagé, autre source → autre cache, moteur sans ninja, copie complète).
* Chaque poste, avant sa prochaine prise : `CUDA_VISIBLE_DEVICES="" python outils/banc-marlin-p1-18-09.py --compiler-seulement`
  (24 s, une fois par empreinte) — sinon repli nommé, visible, jamais une compilation sous verrou.

## 3. Prises du jour où B a pu tourner sans Marlin (journal 24/09 : 335 prises ; journaux du jour)
* Services : 87 lignes `marlin(doubles=…,seuls=…)`, 20 `marlin(off:ACVRAM_PROJ_MARLIN=0)` (témoin voulu), **0 `marlin(repli:…)`**.
* Indécidables : 85 prises en PROCESSUS (kl/profil/diag/tests/suites : 134 kl-sources, 150 bis kl-lot, 156/157 KL et diag, 160)
  n'imprimaient pas le bilan ; leur B n'est attesté que par le résultat (150 bis mixte +13,7 % conforme à Marlin). Seul le
  160 est ÉTABLI sans Marlin. Depuis 778b6d08 le repli s'imprime : toute prise en processus rejouée commence par cette ligne.
