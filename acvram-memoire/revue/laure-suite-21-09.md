# Laure — suite pytest 21/09 09h01 (261s, à sec, sous accord Manon 08h57-5min ; suite complète non conforme au trou déclaré, cf. laure.md)

Instrument : `pytest -q` complet, arbre anticitoyen-vram, commit ba6a5187 (avant fusion 0c48d75b).
Commit : ba6a5187. Régime : à sec (CUDA_VISIBLE_DEVICES vide), pas de mesure GPU en cours. Scellé : néant (relevé, pas une mesure de perf). Mesuré : 1485 passed, 8 failed, 6 errors, 575 skipped.

## Corrigés par moi (0c48d75b) : 2
- `test_lanceur_source.py::test_paquet_par_defaut` / `::test_arbre_opt_in_propre_puis_sale` — régression : préfixe "à sec :" → "commande :" (fusion commande-unique, ba6a5187). Test migré, 4/4 verts.

## Corrigés par moi (pièce 5) : test_menus règle affinée
- `test_menus.py::test_b_disque_et_menu_dans_les_deux_sens` et `::test_l_inventaire_tsv_suit_le_disque` — 8 « sur disque, absent du menu ». 6 = témoins bf16/ablation sans fiche verdict (aucun TSV servi, absents de l'inventaire Katy) → exclus par règle encodée (`temoins_sans_fiche`). **2 restent rouges, réels** : `gemma-4-12B-it-bf16`, `gemma-4-31B-it-nvfp4-vision` — servis (~/TSV/acvram-chemins.tsv) mais absents des menus. À qui pousse la fiche (Katy/Océane), pas corrigé ici.

## Non touchés, hors mon domaine (6 failed + 6 errors)
- `test_mm_api.py::test_texte_pur_jetons_inchanges[vision]` (FAIL)
- `test_mm_api.py::test_image_sur_alias_vision_acceptee` (ERROR)
- `test_mm_api.py::test_deux_images_deux_fragments_deux_sha` (ERROR)
- `test_mm_api.py::test_n_jetons_image_au_dela_de_max_model_len` (ERROR)
- `test_mm_api.py::test_file_hors_dossier_refuse` (ERROR)
- `test_mm_api.py::test_http_non_local_refuse` (ERROR)
- `test_mm_api.py::test_schema_inconnu_et_data_non_image_refuses` (ERROR)
- `test_mm_conversion.py::test_la_ligne_de_regime_nomme_la_vision_seulement_modele_charge` (FAIL)
- `test_cadrage_perplexite.py::test_la_liste_des_variables_lues_ne_derive_pas` (FAIL) — à nommer : instrument ou défaut
- `test_regime_noyaux.py::test_toute_variable_de_chemin_est_dans_la_table` (FAIL) — à nommer : instrument ou défaut

Verdict : 0.6.34 ne part pas avec les deux menus réels (gemma bf16/nvfp4-vision) et les 3 lignes ci-dessus non nommées.
