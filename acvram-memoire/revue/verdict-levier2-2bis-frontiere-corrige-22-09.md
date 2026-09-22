# Levier 2 (2 bis) : frontiere-pas rejoué après correctif 0c17a2bf — TENU, nettement — 22/09 (poste2)

* instrument : `outils/gpu/mesure/frontiere-pas.py` (worktree poste2-w-21-09, main 32eb7352), 4 processus A0/B0/B1/A1, Coder nvfp4, b=12, 300 pas ; `build_info` vérifié avant prise (`ninja: no work to do`, aucune recompilation)
* commit : 32eb7352 (correctif poste1 inclus : tampon `[2, n]` contigu, la copie épinglée visait avant une vue non contiguë `[:, :12]` d'un tampon `[2,16]` → copie synchrone)
* régime : A = `rapatriement=flux` (défaut) ; B = `rapatriement=epingle`
* scellé (poste1) : `trou_gpu` → 10-30 µs (contre 173 µs de référence) ; réfuté si > 100 µs ; prédit aussi `suite_prep` ≤ 200 µs, `attente_evt` ≈ 6 500 µs
* mesuré : A `trou_gpu` 166,3/164,2 µs (moy 165,3), `attente_evt` 5,6/5,1 µs, `suite_prep` 149,8/149,5 µs ; B `trou_gpu` **24,2/24,8 µs (moy 24,5)**, `attente_evt` **6631,3/6622,1 µs (moy 6626,7)**, `suite_prep` 155,3/156,5 µs (moy 155,9). `consommer` reste effondré (≈15 µs, comme au 1er essai)
* verdict : **TENU, nettement** — `trou_gpu` 165,3 → 24,5 µs (dans la fourchette prédite 10-30, réfutation à 100 µs largement évitée) ; `attente_evt` 6626,7 µs, à 2 % de la prédiction 6 500 ; `suite_prep` 155,9 µs, sous le seuil 200. Le 1er essai (7f967671, avant correctif, `verdict-levier2-2-frontiere-22-09.md`) était RÉFUTÉ pour cause identifiée : la latence n'avait pas disparu, elle s'était déplacée dans `suite_prep` (151 → 6860 µs, copie synchrone sur vue non contiguë) — le correctif la reloge dans `attente_evt`, où le recouvrement du pipeline peut la masquer.
* durée : 4 passes, ≈ 3 min

## Suite
ABBA débit (A graphe / B graphe+épinglé, prédit +2,0-2,4 %) rejoué sur ce commit.
