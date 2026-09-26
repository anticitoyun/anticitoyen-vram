# Verdict — pièce 183 : seuil GEMV int8 abaissé dans la portée de B' (poste5, 25/09)

* **instrument** : `scratchpad/poste5-p183-25-09/prise.sh` (`tests` + cassant, `kl` = `kl183.py` × 3 alias) →
  `prise-tests.txt`, `prise-tests-2.txt`, `prise-kl.txt`, `kl-<alias>.json`
* **commit** : 67b2818c (1re prise `tests`), eb778653 (correctif, 2e prise `tests`, prise `kl`)
* **régime** : défaut + `ACVRAM_INT8_GEMV_MAX_PARTAGE` basculé à chaud (A = 0, B = 16) ; cpu-safe 100 au début et à la fin
* **scellé** : `revue/poste5-piece183-scelle-25-09.md` (critère de chef, repris de la 165)
* **mesuré** : tests 121 verts, cassant ROUGE ; KL : mixte-i8c NON TENU, GLM NON TENU, Qwen3.8 nvfp4 TENU (au bit)
* **verdict** : **FAUX** (deux alias NON TENUS, et le GLM qu'on attendait au bit ne l'est pas) → 183 reste opt-in (0 = coupé),
  banc non lancé
* **durée** : prévu ≤ 30 min de carte ; tenu 07:22:11-07:25:06, 07:28-07:31:00, 07:31:26-07:33:38 (≈ 8 min)

## Tests
1re prise : `test_les_defauts_de_la_table_sont_ceux_du_code` rouge. Le code valait `None`, `_format` rendait « None »
(regime.py:406-419), et la table disait « ». Correctif : 0 = coupé dans le code, la table, le test, kl183 et le cassant.
2e prise : 121 verts ; cassant (condition de portée retirée) → `test_dans_la_portee…` ROUGE, comme attendu.

## KL (critère : KL max B ≤ 2 × témoins ; argmax B ≥ T1 − 0,005 ; ΔPPL B ≤ 2 × témoins ; rejeu 0 ulp)
| alias | C1 | C2 | C3 | déquant int8 B |
|---|---|---|---|---|
| mixte-i8c | tenu (KL 0,028 / T1 0,027) | tenu | **argmax 0,968 < 0,995 − 0,005** | 3 729 / 3 729 / 1 875 |
| GLM-4.7-Flash nvfp4 | tenu | tenu | **argmax 0,926 < 0,937 − 0,005** | 4 890 / 4 890 / 2 514 |
| Qwen3.8 nvfp4 | au bit | au bit | au bit | 0 |
Le rejeu est à 0 ulp partout. Sur le mixte, KL et ΔPPL restent sous les témoins : ce qui échoue, c'est seulement l'argmax de C3
(−2,6 points contre T1). Empreintes sha256 (16 premiers caractères) : GLM e9a864babbd7a0e4, Qwen3.8 nvfp4 8d1b5eab1d833b38,
mixte a254537debec8317.

## Prédiction fausse : GLM
J'avais écrit « MLA, pas de récurrence → au bit ». Faux. `glm4_moe_lite` est `est_mla` (loader.py:771-773) et se charge en
`DecoderLayerGDN` (loader.py:841). Sa boucle par séquence ouvre la portée de B' (couches.py:150). B' couvre donc le MLA et
aussi le conv LFM2 (loader.py:765), pas seulement les couches à récurrence. Le commentaire de la 183 (kernels/__init__.py:806)
et celui de la 172 disent « couche à récurrence linéaire » : c'est trop étroit.

## Leçon
Avant d'écrire la portée d'un chemin dans un scellé, lire qui construit la classe (`grep ClasseX(` dans le loader), pas le
nom de la classe.

## Reste
Levier du banc (préfill 1,19 s à L = 78) toujours ouvert, mais pas par une arithmétique qui change la sortie. Pistes au bit à
chiffrer : un GEMV int8 par tranches plus rapide à M 16-80, ou un GEMM int8 à sortie identique au GEMV. Décision : chef.
