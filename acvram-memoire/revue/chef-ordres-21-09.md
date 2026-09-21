# chef — registre des ordres et de leur respect (21/09, ouvert 12 h 42)

Un ordre = une ligne ; « livré » = sha ou verdict ; « écart » = ce qui n'a pas été fait comme ordonné (nommé, jamais lissé). Mis à jour à chaque passage de surveillance (20 min).

| heure | poste | ordre | livré | écart |
|---|---|---|---|---|
| 09 h 0x | poste1 | commande unique, ordre chauffe, rouges régime, pipeline défaut, crochet GUI, déquant, P3 (4) admission | 7ca88ae2, 1a1bc9f2, 4dba5438, 10d6b4f3, 3a8554ef, 82baa336, 8786c232 | pytest à sec hors trou ×2 (dits) |
| 09 h 0x | poste2 | n=60, 4 alias, § 1b, GLM b=1, NVTX, P3 (3)(4), A/B | n=40 (cb532686), 4 alias TENU (0b43c931), § 1b 2/4 ×2, GLM REFUS puis A/B indécidable, P3 (3)(4) ÉCHEC (instrument/moteur) | prise A/B 41 min > 30 (boucles curl) ; le reste = défauts trouvés, pas des écarts |
| 09 h 0x | poste4 | scellé b=12, M2, H2 instruments, NVTX | f3285301, 476390aa, cb6041d5, b63dee20 | aucun |
| 09 h 0x | poste3 | tests ciblés, colonnes 2b, VM, .deb, menus, lanceurs, buymeacoffee, usage | 0c48d75b…0c9c128a, 92db8334, 9456ce40 | suite complète (261 s) pendant une prise (nommée) ; commits non poussés ×2 (corrigé) |
| 12 h 3x | poste2 | NVTX → 4 alias (8a92ffbc) → trou poste3 5 min → P3 (4) → P3 (3) → A/B | — | — |
| 12 h 3x | poste3 | ComfyUI verrou → § 1a crochet → buymeacoffee → dépersonnalisation → usage ; nvcc au trou | — | — |
| 12 h 3x | poste1 | H2 glue après verdict NVTX de poste4 | — | — |
| 12 h 3x | poste4 | verdict NVTX/H2 dès les sorties de poste2 | — | — |
| 13 h 01 | poste2 | NVTX → 4 alias (8a92ffbc) → … (ordre 12 h 35) | rien depuis 12 h 35 | `sleep 1800` dans son shell depuis 12 h 35 (attente aveugle de 30 min au lieu d'une prise) — rappel envoyé |
| 13 h 01 | poste3 | cinq pièces à sec (12 h 35), rappel 12 h 44 | rien depuis 9456ce40 | **écart répété** : idle sans livrer, signalé à l'utilisateur |
| 13 h 01 | poste1, poste4 | attendent NVTX (dépend de poste2) | — | aucun |
