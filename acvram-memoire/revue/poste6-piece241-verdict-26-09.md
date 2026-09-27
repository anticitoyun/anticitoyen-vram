# 241 — la CI produit les noyaux précompilés que le chargeur 240 relit : garde à sec VERTE (contre-lecture de poste2 : verdict manquant)

instrument : pytest sous `outils/carte.sh` (prise poste6-240-241-249, à sec : `CUDA_VISIBLE_DEVICES=""`), fichier suivi `tests/test_noyaux_precompiles_ci_241.py`
commit : origin/poste6-241 772c8bf61 (worktree travail/poste6-241 ; = 5d84c763e noyaux précompilés par la CI + fusion poste3-236 + release.yml)
régime : aucun (à sec, aucune carte, aucun noyau chargé) ; carte tenue par la prise, llama-server 4219 seul
scellé : 7 tests verts, sinon FAUX — casse si le job CI attend un chemin (`build/noyaux`) ou une disposition (`<src_sha16>/acvram_kernels.so`
  + `empreinte.json`) que `compiler_precompile`/`_ecrire_empreinte` ne produit pas, si le manifeste Flathub n'expose pas
  `ACVRAM_KERNELS_PRECOMPILES` sur le répertoire installé, ou si les architectures forcées ne donnent pas les `-gencode` attendus
mesuré : 04:10:32 → 04:10:33 (26/09), `7 passed in 0.08s`, rc 0 — test_le_job_produit_le_dossier_que_le_manifeste_embarque,
  test_ce_que_la_ci_ecrit_est_ce_que_le_chargeur_relit, test_archs_depuis_texte ×2 (paramétré), test_archs_illisibles_levent,
  test_les_archs_forcees_donnent_les_gencode_famille_sans_carte
verdict : TENU — la CI (job `noyaux-precompiles` de release.yml, conteneur CUDA sans carte, ex-job flatpak `if: false`), le manifeste
  Flathub et le chargeur de la 240 nomment le même chemin et la même empreinte. Ce que ce verdict ne dit PAS : que le `.so` produit
  par la CI charge et sert sur une carte réelle (240 le prouve pour un `.so` local ; le `.so` de la CI se prouve à la première release
  qui l'embarque, par `acvram doctor` sur le paquet installé — reste de la 236/241, à mettre à la sortie de la 0.7.0).
durée : prévu ≤ 1 min / tenu 1 s dans une prise de 3 min (240 + 241 + 249), attente 651 s
