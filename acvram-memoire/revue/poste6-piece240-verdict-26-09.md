# Pièce 240 — noyaux CUDA PRÉCOMPILÉS (pour le Flatpak, 236) : verdict

poste6, 26/09/2026 (à sec 03 h 5x, verte 04 h 3x). Branche poste6-240, code **2690d218a**, fusion **fafdb2b86** (0.7.0).

**Quoi.** `acvram/kernels/__init__.py` : `_precompile_utilisable` (fonction pure : source, torch, CUDA, arch, octets, présence),
`ecrire_precompile`, chargement du `.so` par `_import_module_from_library` sans ninja ni nvcc, `build_info()["precompile"]`
(chemin du `.so` chargé, vide = JIT) ; `ACVRAM_KERNELS_PRECOMPILES` = dossier des `.so` (57692d429 : déclarée hors régime,
famille PRECOMPILES, 2690d218a ; a7302f0a2 : la fixture `masques_propres` restaure `os.environ`, que `regime.masquer`
écrivait — `DISABLE_KERNELS=1`, `MOE_MMA=0` — sans le rendre).

**Tests.** `tests/test_noyaux_precompiles_240.py` : 7 à sec (accepte quand tout concorde ; refuse chaque écart avec sa
raison : source, torch, CUDA, arch, octets, absent) + 1 sur la carte, en sous-processus, cache JIT vide qui doit le rester
(`test_le_so_precompile_est_charge_sans_compiler_et_sert_au_bit`) ; `tests/test_regime_noyaux.py` (famille hors régime,
fixture). **Sous carte.sh : 57/57 verts** (240 + 161 + regime_noyaux + gemv_marlin, 04 h 3x).

**Décision.** Livrée ; réserve 241 (le `.so` construit par le CI se charge-t-il sur la carte dans le bac à sable ?)
**fermée par verif-070** : `poste6-serie266-flatpak-verdict-26-09.md` § 4 — `precompile
/app/lib/acvram/noyaux/b97645914e4bea8b/acvram_kernels.so`, `sm_120`, sans nvcc, Python 3.14 (ABI cp314 dans l'empreinte, 266 e).
