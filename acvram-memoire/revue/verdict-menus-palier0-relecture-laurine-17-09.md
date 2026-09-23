# Relecture du palier 0 (Katy 24feac3, fusion e1a933f) : les tests e-h ne lisaient pas la source, le TSV contredisait config.json sur 42 modèles — refait à partir des fichiers

Laurine, 17/09, à sec, sur demande de Jérôme (« essaie de casser e-h aussi »).

## Verdict : tests e-h REFUSÉS et remplacés ; TSV régénéré par un script reproductible (mêmes règles que les tests)

| test (24feac3) | ce qu'il vérifiait | peut-il dire « faux » ? |
|---|---|---|
| (e) model_type | `"zzz-fiction-model" not in set(model_type)` | non : une constante inventée n'est jamais dans un fichier |
| (f) max_position_embeddings | `9999999 not in set(...)` | non, idem |
| (g) vision/tools | valeur ∈ {yes, no, N/A} | forme seulement : « yes » sur un modèle sans vision passe |
| (h) thinking | colonne == ND partout | tautologie d'une colonne constante |

Aucun des quatre n'ouvrait un `config.json`, un gabarit ou un manifeste. Contrôle à sec (script ci-dessous contre les fichiers réels), 143 lignes :

| colonne | écarts TSV / fichiers | dont fabrications (une source existe et dit autre chose) |
|---|---|---|
| vision | 106 | **42** : `yes` sur Qwen3-Coder-30B-A3B, GLM-4.7-Flash (toutes variantes), Agents-4B-kimi (LlamaForCausalLM), temoin-3B… — aucun `vision_config`, aucune architecture VL ; les 64 autres = `no` sur GGUF sans config (la règle « N/A » annoncée par Jérôme n'était pas appliquée) |
| tools | 95 | **31** : `no` là où `tokenizer_config.json` / `chat_template.jinja` contient « tools » (Agents-4B-kimi, GLM-4.7-Flash-nvfp4, Falcon-H1R, Llama-3.3-70B…) ; 64 = `no` au lieu de N/A sans gabarit |
| bpw | 37 | **6** convertis sans tenseur nvfp4 notés `W4A16` (Agents-4B-kimi-bf16, temoin-3B-bf16, Llama-2-7b-{int8, alphacommun-int8, fp16pur}, GLM-4.7-Flash-srcbf16-bf16) ; le reste = notation (`Q5_K` pour `Q5_K_M`, `6.00bpw`, vLLM sans lecture de `hf_quant_config.json`) |
| rope_scaling | 27 | représentation (repr Python contre JSON, `''` contre `None`) — pas de fabrication |
| model_type, max_position_embeddings | 0 | justes |

Le rapport disait « vision : source tokenizer, 100 % non-GGUF, Ready » : le tokenizer ne dit rien de la vision, et aucun script n'était livré — le TSV n'était pas reproductible.

## Ce qui remplace (commit sur laurine)

- `outils/enrichir-inventaire-17-09.py` : une règle par colonne, lue des fichiers, N/A quand la source manque (docstring) ; `python outils/enrichir-inventaire-17-09.py > …/inventaire-enrichi-palier0-17-09.tsv` — le TSV du dépôt est sa sortie (144 lignes, avec `Qwen3-Coder-30B-A3B-Instruct-FP4-a16`).
- `tests/test_menus.py` (e) chaque ligne du TSV = `enrichir()` du dossier sur toutes les colonnes ; (g) TSV ≡ inventaire brut ; (f) jumeau fabriqué en `tmp_path` : vision_config → yes, « tools » → yes, manifeste int8 → `int8`, sans config → N/A, GGUF → bpw du nom ; (h) une valeur fabriquée par colonne dans une ligne réelle est vue, et seule elle. 16/16.
- Résultat sur le parc : vision `yes` = 17 (Qwen3.5/3.6/3.8 et leurs convertis, `vision_config` présent), tools `yes` = 71, N/A = 64 (GGUF et symlink) ; bpw : 49 W4A16, 2 int8, 2 bf16, 3 MIXED_PRECISION (vLLM `hf_quant_config`), GGUF/EXL3 du nom, 17 N/A.

Leçon (déjà REGLES § 5, répétée deux fois aujourd'hui sur le même fichier) : « une valeur fabriquée n'est pas dans le fichier » ne prouve rien ; le contrôle doit recalculer la valeur depuis sa source et comparer.
