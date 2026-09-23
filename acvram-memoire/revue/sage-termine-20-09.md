# Sage — point de clôture : « terminé » T1-T4 tenu à 13 h 05 (prédit 13 h 40 à 12 h 08, 21/09 midi à 09 h 22) (20/09, 13 h 20, horloge machine)

## Ce qui est établi (ETAT c53fd588, chaque chiffre a son verdict)
| | acvram 0.6.32, poste 20-09-1030, `-lgc 2700` | llama.cpp | vLLM Marlin |
|---|---|---|---|
| Coder b=1 | **380,8 · 0,442 J** | 323,6 · 0,700 | 290,6 · 0,602 |
| Coder b=12 | 1 378,6 · 0,205 | 929,3 · 0,157 | **1 596,1 · 0,128** |
| Coder prefill 2 048 | **22 707** | 15 734 | 21 054 |
| GLM b=1 servi | **169,0** (0.6.31 : 143,8 ; poste + `=2`) | — | 183,5 |
Revendication inchangée cellule par cellule : devant à b=1 et au prefill Coder ; derrière vLLM à b=12 (vitesse −14 %, J ×1,6) et à GLM b=1 (−8 %) — hors périmètre, cause chiffrée (`sage-tests-rapides-cloture-20-09`, `sage-m4-faux-niveau3-clos-20-09`). T4 : 1 885 passed / 3 rejoués verts / 11 xfail nommés (8 flash_causal + 3 antérieurs) / 38 skipped ; arbre livré 1f1d96bf = main + 2 lignes témoin.

## Prédictions de Sage aujourd'hui, comptées
Tenues : M1 bis (−0,35 ± 0,15 → −0,465), M3 (tenu, dépassé : +13,5 %), affinité (0 à −2 % → +0,6 %), 0.6.32 servi (163 ± 3 → 169,0 : tenue par le bas, dépassée), terminé (13 h 40 → 13 h 05), int8 = instrument (60 % → oui). **Fausses** : M4 (−0,15 ms → +0,271 : noyau non chronométré à sec), (A−B)/ΔN (0,5-1,0 → 1,34 : la loi compte le temps GPU de la glue), 9 xfail (→ 8 : jumeaux devenu servi par M3). Trois règles en sont sorties (REGLES § 2, § 3 ; MECANISMES µs/nœud).

## Deux points d'Océane
1. **Jumeaux, contrat d(=2) ≤ d(=1) élément par élément, marge 0** : accepté tel quel aujourd'hui ; si une autre graine le fait tomber, **ce n'est pas un relâchement mais la règle § 7 qui s'applique** (d ≤ d(réf) + 2 × témoin mesuré, témoin imprimé) — écrit maintenant pour qu'un rouge futur se traite par la règle, pas par une négociation. Trois graines (0, 1, 9) dans le test dès aujourd'hui, à sec, 2 min.
2. **moe_fused : le chemin B servi est à 12 % de l'amplitude de sa référence fp64** — un test qui ne couvre pas B contre sa référence ne couvre pas le chemin servi. Question à Océane, réponse fichier + ligne : quel test juge B (MoE fusionné Marlin/mma) contre une référence indépendante ? Si aucun : test à écrire (à sec sur formes réduites, une prise de 1 min sur carte). Prédiction : **instrument** (référence fp64 construite sans les échelles awq/hadamard ou sur d'autres poids) à 80 % — Coder PPL 1,0094 contre bf16 sur le même chemin ne laisse pas 12 % à un défaut ; 20 % : défaut réel masqué par la PPL (top-k robuste). Faux si la référence corrigée reste à > 2 %. **Reprise priorité 1**, avant tout chantier.

## Après terminé — ordre des chantiers (mot de l'utilisateur pour chacun)
1. moe_fused référence de B (ci-dessus, ≤ 0,5 j) · 2. transfert des 48 alias (Femoceane, trous, `modeles-a-jour` après chaque alias) · 3. **multimodal** Gemma 4 → Qwen3-VL (`sage-acvram-multimodal-20-09`, accord attendu) · 4. `acvram-parc` (accordé, 2 j) · 5. lot poste restant (THP/EPP), memtest86+ une nuit · 6. reprise b=12 Coder (bande Marlin, C15-3d) et GLM b=1 (route_x coopératif ≤ 5 µs à sec).

## Ordre
* **Jérôme** — indexer ce point ; ETAT : les six chantiers dans cet ordre, aucun ne démarre sans le mot de l'utilisateur ; REGLES § 2 : assertion `rev-parse HEAD` dans toute chaîne (13 h 12).
* **Océane** — (1) trois graines jumeaux ; (2) fichier + ligne du juge de B ou « aucun », une ligne à Jérôme.
* **Manon** — carte libre ; rien.
