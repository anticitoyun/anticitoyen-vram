# C9 M-HÔTE — RÉFUTÉ, arrêt (a) confirmé — 22/09 (poste2)

* instrument : `outils/gpu/mesure/c9-m-hote.py` (4 experts réels 119B, `nvfp4_matmul_cpu` OpenMP, MLP complet gate/up/down), sous `carte.sh` (aucune allocation GPU, charge CPU voisine comptée), OMP_NUM_THREADS=16
* commit : main à jour, worktree poste2-w-21-09
* régime : Mistral-Small-4-119B-2603-NVFP4, couche 0, 4 experts, charge étrangère load1=1,37/32 nproc
* scellé (poste1, `poste1-c9-conception-21-09` § 3.1) : prédit 0,80 ± 0,15 ms les 4 experts (0,20 ± 0,04 ms/expert) ; arrêt (a) si > 1,2 ms les 4 (< 47 Go/s, l'hôte ne bat plus llama.cpp)
* mesuré : contrôle d'exactitude `2,98e-03` (sous le seuil ± 2⁻⁶ = 1,56e-2, TENU — le calcul est correct) ; **0,855 ms/expert (p90 0,872), 3,417 ms les 4, 16,6 Go/s**
* verdict : **RÉFUTÉ — ARRÊT C9 (a) confirmé** — 3,417 ms très au-dessus du seuil d'arrêt 1,2 ms (×2,85), 16,6 Go/s bien sous 47 Go/s. Confirme l'alarme d'poste1 du 21/09 (« noyau hôte nvfp4 scalaire, 2,2 Go/s par fil → M-HÔTE probablement > 1,2 ms → arrêt (a) ») : l'hôte ne peut pas porter le calcul d'experts pour ce chantier sans un noyau AVX-512 (plusieurs jours de travail, non engagé).
* durée : < 1 min de carte (charge CPU pure)

## Suite
C9 arrêté sur ce chantier (conception déjà « FAITE » avec cette issue nommée d'avance). M-TRACE ensuite pour vérifier S3 (mixte).
