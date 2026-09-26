# PPL A/B 31B après correctif BOS (ae519990) — ÉCHEC, toujours pathologique (pire qu'avant) — 22/09 (poste2)

* instrument : `acvram eval <répertoire> --corpus wiki-gptq.txt --window 2048 --stride 2048 --min-context 256 --json`, un modèle à la fois, sous carte.sh, commit ae519990 déjà dans l'arbre (BOS ajouté en tête de fenêtre)
* commit : main/poste2 à jour (5e2bd345+)
* régime : A = gemma-4-31B-it-nvfp4-vision (max6), B = gemma-4-31B-it-nvfp4-4sur6-vision, 4 fenêtres chacun (corpus 1,3 Mo, sha256 e52922746ad0…)
* scellé (chef/poste1) : PPL(ctx512) 8-16 attendu par ae519990 (le commit promettait explicitement 936 → 10-20 sur cet alias A) ; B/A ± 3 % ; alarme si > 30
* mesuré : **A** perplexité globale **53 202,6** (ctx128=27 713,7 · ctx512=59 315,7) — **PIRE qu'avant le correctif** (v2 avant BOS : 5 445,96/936,65). **B** perplexité globale **7 301,7** (ctx128=6 031,5). `avertissement` vide (pas d'alarme d'exil signalée par le code) sur les deux.
* verdict : **ÉCHEC — le correctif BOS ne produit PAS le résultat annoncé par son propre message de commit** sur cet alias (attendu 10-20, obtenu 59 315,7 à ctx512, ×3000). Ni A ni B n'approchent la fourchette 8-16 ni même le seuil d'alarme 30 — les deux restent dans le même régime pathologique que les essais précédents (2-3 ordres de grandeur au-dessus du plausible), B restant ~7× meilleur que A comme dans le v2 (cohérence relative conservée, mais aucune des deux valeurs n'est exploitable en absolu). Hors mon domaine de diagnostiquer pourquoi le BOS commité ne change rien ici (tokenizer.bos_id non détecté sur ce checkpoint ? gabarit de conversation différent du chat_template cité dans ae519990 ? à vérifier par poste1) — je ne republie pas ce chiffre comme mesure de qualité, seulement comme signal d'instrument encore cassé.
* durée : ~1 min de carte (2 × 4 fenêtres, 30 s chacun)

## Suite
L'eval PPL reste invalidée en absolu pour ce modèle malgré ae519990 — à rouvrir par poste1 (vérifier si `Tokenizer.bos_id` est bien non-None pour gemma-4-31B, et si le gabarit BOS appliqué correspond à celui du serveur). Ne pas publier de comparatif A/B qualité tant que ce n'est pas résolu. File : banc-etroites-splitk (prioritaire, groupe) → 25(a) → reconversion mediane_couche → (c)+P3 → KL acvram/bf16 → énergie 4 moteurs.
