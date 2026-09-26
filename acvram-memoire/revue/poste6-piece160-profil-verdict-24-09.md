# Verdict — pièce 160 étape 1 : profil par familles du préfill de lot (poste6, 24/09) — les lancements NE dominent PAS

* **instrument** : `scratchpad/poste6-p160-24-09/profil-familles.py` (torch.profiler CPU+CUDA, annotations par monkeypatch,
  3 passes profilées après 3 de chauffe, mur médian de 5 passes synchronisées) ; `prise-profil.sh` ; sortie `profil.json`
* **commit** : ef98869f (branche poste6 = main 7525b14e + instrument) ; `acvram.__file__` = worktree
* **régime** : -lgc 2700 (SM 2 992 → verrou posé, mem 13 801), ACVRAM_ECO=off, cpu-safe 100, défaut `Qwen3.8-27B-nvfp4`,
  ACVRAM_GDN_PREFILL_LOT=1 (état 150 bis), carte 0 seule (llama-server 5,6 Gio sur la 3080 Ti, bus 02, pendant la prise)
* **scellé** : dossier `poste6-piece160-dossier-gdn-varlen-24-09.md` § 4 (prédit : la boucle GDN = premier poste, préfill
  borné par l'hôte) · **mesuré** : oui · **verdict** : prédiction RÉFUTÉE sur le poste dominant, tenue sur les lancements
* **durée** : carte obtenue après 1 231 s d'attente, tenue 34 s (19:11:03 → 19:11:37) ; prévu ≤ 10 min, tenu

## Mesure (un ForwardBatch de préfill, hors service)

| composition | jetons | mur/passe | Σ noyaux | carte occupée | noyaux/passe | dont gdn-cœur | carte gdn-cœur | aten gdn-cœur |
|---|---|---|---|---|---|---|---|---|
| C1 8 × 78 | 624 | **275,9 ms** | 215,3 | 78 % | 13 544 | **11 897 (88 %)** | **45,6 ms (17 %)** | 53 732 |
| C2 mêlées 33-200 | 775 | 337,2 | 286,7 | 85 % | 13 444 | 11 904 | 49,5 | 53 760 |

Classement des noyaux (C1, par passe) : GEMM bf16 cutlass 163,1 ms (368 lancements) + `nvfp4_dequant_kernel` 38,3 ms
(496) = **201 ms, 73 % du mur** ; fla `ChunkGatedDeltaRuleFunction` 23,9 (384 appels, 62 µs chacun) ; copies 11,1 (2 307) ;
élémentaires 9,4 (1 920) ; clones 7,0 (1 552). C2 : GEMM 231, dequant 39, fla 23,9.

## Lecture

* **Les ~14 600 lancements sont là (13 544, dont 88 % dans la boucle par séquence), mais ils ne dominent pas** : la carte est
  occupée 78-85 % du mur ; l'hôte n'est devant que 50-60 ms par passe. La boucle GDN coûte 46-50 ms de carte à 4 µs le
  noyau — cadence de lancement — donc ≈ 0,10 s de mur par passe (36 %), pas la majorité.
* **Le poste dominant est le chemin nvfp4 du préfill : dépaquetage nvfp4 → bf16 de TOUT le modèle à chaque passe (38 ms,
  ≈ 70 Go de trafic) puis GEMM bf16 à 207 TFLOPS effectifs (163 ms)** — même mécanisme que la 147 sur PROJ_MARLIN, ici
  sur le défaut. C'est le levier du préfill du défaut, pas la récurrence.
* Instrument : `device_time_total` des annotations imbriquées double-compte (gdn-cœur 88,9 contre 45,6 par somme des
  noyaux) → seules les sommes de noyaux sous annotation sont retenues ; `cpu_time_total` des annotations rend 0 sous kineto.

## Prédiction révisée pour l'étape 2 (varlen, conv par séquence), écrite avant tout code

* Gain = la boucle (≈ 0,10 s de mur/passe) ramenée à ≈ 0,01-0,02 (48 appels fla à T = 624 + élémentaires sur Σ t lignes) :
  **−0,06 à −0,09 s/passe → défaut 0,34 → 0,25-0,28 s/lot** ; le seuil scellé ≤ 0,26 s/lot reste, atteint seulement
  dans le haut de la fourchette. Issue nommée : tenu à −0,08 ou plus ; sinon « gain réel mais sous le seuil », sans
  reformulation du seuil. Au bit : inchangé (§ 3 du dossier).
* Levier plus gros, hors pièce : GEMM sur poids nvfp4 sans dépaquetage complet par passe (fp4 tensor cores = activations
  quantifiées, hors bit ; W4A16 façon Marlin = ± 1 ulp, 147 L2) — prédit −0,15 à −0,20 s/passe sur le défaut. À chef.
