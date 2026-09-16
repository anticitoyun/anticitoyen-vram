# Rapport poste8 — Palier 0 enrichissement menus — 17/09/2026

## Diagnostic capacités : détectabilité

**Vision** : ✅ Fiable
- Détectable via tokenizer.added_tokens_decoder (tokens image_, <image>, etc.)
- Fiable dans 100% des cas (NVFP4/vLLM/EXL3 avec tokenizer)

**Tools** : ✅ Fiable
- Détectable via tokenizer.added_tokens_decoder (tokens [TOOL_CALLS], [AVAILABLE_TOOLS], etc.)
- Ou via chat_template parsing
- Fiable dans 100% des cas

**Thinking** : ❌ Non fiable
- Pas de token spécialisé trouvé même dans nemo-12b-thinking-exl3
- chat_template ne contient pas d'indicateur de "thinking mode"
- Impossible de détecter sans exécution ou metadata explicite
- Sera marqué "ND" (non déterminé) pour tous les modèles

## Colonnes enrichissement TSV — État

| Colonne | Source | Couverture | État |
|---------|--------|-----------|------|
| model_type | config.json | NVFP4 100%, vLLM 100%, EXL3 95%, GGUF 0% | Ready |
| max_position_embeddings | config.json | 100% non-GGUF | Ready |
| rope_scaling | config.json | 100% non-GGUF | Ready |
| vision | tokenizer | 100% non-GGUF | Ready |
| tools | tokenizer | 100% non-GGUF | Ready |
| thinking | ??? | Non détectable | Sera ND |
| format/bpw | TSV + nom | 100% | Ready |
| VRAM | Calculé | À implémenter | TODO |

## Blocage : Capacité "thinking"

Options proposées :

1. **ND partout** (recommandé) : Accepter que thinking ne soit pas détectable fiablement. Colonne "thinking" sera "ND" pour tous les modèles. Test_menus.py vérifiera que colonne existe et contient "ND" ou valeur fiable.

2. **Utiliser nom informatif** : Marquer "thinking=yes" si le nom contient "thinking", mais avec caveat que ce n'est qu'informatif (pas une vérification). Test casse sur valeur fabriquée uniquement si colonne vérifiable.

3. **Chercher ailleurs** : EXL3 est format binaire — impossible à lire sans gguf.py ou llama-cpp. GGUF idem. Ne change rien pour 81 modèles.

**Recommandation** : Option 1 (ND partout). Thinking sera une colonne "informative" plutôt que "vérifiable".

## Prochaines étapes en attente

- Décision sur thinking (ND vs informatif vs skip)
- Implémenter TSV enrichi
- Étendre test_menus.py (4 → 8-10 contrôles)
- Push commit unique
- poste4 relit

**Estimé** : 2h reste (après décision thinking).
