# Reste à faire — 21/09 17 h (pause)

Objectif : plus rapide et moins de joules que llama.cpp, vLLM, TensorRT-LLM. **Vitesse contre vLLM : tenue sur les trois cellules** — b=1 380,8 (vLLM 290,6), prefill 22 707 (vLLM 21 054), b=12 1 625,5 (vLLM 1 596,1, 22/09). **Contre TensorRT-LLM (22/09, même checkpoint NVFP4, deux bras au plafond 400 W) : b=1 tenu (390 contre 46 t/s ; 0,44 contre 7,2 J), b=12 NON tenu (1 638 contre 1 998 t/s, +22 % ; 0,196 contre 0,156 J/jeton).** Reste : énergie à 4 moteurs (banc), cellule A/V vLLM rejouée, +8-12 % au bit (poste1), qualité W4A4 de TRT-LLM (poste3), puis choix d'un opt-in hors bit pour la parité.

## A — porte immédiate (utilisateur)
1. `sudo dpkg -i acvram_0.6.34_amd64.deb` (racine du dépôt, sha256 dans poste3.md) — feu vert donné, non installé.
2. Vibe série 2 : répondue (qrvibe01.md), verdicts 2.1-2.5 en fin de qr.md — 2.4 (ordre des leviers) et 2.5 (protocole énergie) intégrés ci-dessous.
3. **OUI utilisateur (21/09 17 h) : C9 119B et bf16 30B (60 Go)** → pièces 19-20 ci-dessous.

## B — objectif b=12 (poste1, poste2, poste4)
4. ~~ABBA sampler~~ FAIT : H1 (ordre/dérive) confirmée ; levier 1 (échantillonnage dans le graphe) TENU et DÉFAUT (13de908e) ; protocole A B B A A B B A avec horloges = règle.
5. Frontière : décomposée (frontiere-pas.py) ; levier 2 (rapatriement épinglé, double tampon contigu) TENU 2 passes (+6,84 % / +3,22 %, B 1 641 / 1 607 t/s) — verdict poste4 puis défaut ; levier 3 = modules B (prêt, compilé) et C (fusions d'épilogue MoE, −0,20 à −0,30 ms) après la table nsys.
6. Cellule b=12 finale ≥ 1 596 ou écart nommé comme résultat.

## C — 30B-VL (défaut de conversion, pas du moteur)
7. ~~Reconversion~~ FAITE (3b785f52) : P3 (4) TTFT 0,114 s / J 41,1 tenus ; P3 (3) RÉFUTÉ 26,1 % — **calibration AWQ jamais collectée** (experts_sans_stats 18 432/18 432, cli.py:654 « 0 tenseurs », setStorage en échec) → **7 bis (poste1, première pièce à la remise)** : quant/collect.py doit voir les experts groupés du hub ≥ 5 (même scission que 3b785f52 côté collecte, ou crochets sur le module fusionné) + test « 0 expert sans stats sur un MoE hub ≥ 5 » ; puis reconversion et P3 (3) rejoué (poste2).
8. Fiche alias 30B-VL mise à jour ; --decoder parc (pièce 6) après.

## D — 31B 4sur6 (qualité nvfp4)
9. Reconversion gemma-4-31B `--echelle=4sur6` en mode service (≈ 75 min, tuée à 24/60 par DUREE_MAX 1800) ; weight_format contrôlé.
10. ~~Scellé E~~ RÉFUTÉ (B 4sur6 1,39 > 1,2 ; A max/6 recalibré 0,87) ; complément PPL relative A/B (≤ 20 min, poste2) avant retrait de l'option 4sur6.

## E — énergie et TensorRT-LLM (pièces 10-11)
11. TRT-LLM : run minimal TENU (8d7706f6, temoin-3B 187,6 t/s indicatif) ; cellules b=1/b=12/prefill sur le checkpoint hub NVFP4 (liste de contrôle 3.4) — poste3, après 30B-VL de poste2.
12. J/jeton 4 moteurs (banc-4moteurs.py) selon verdict 2.5 : J net = ∫(P − P_repos), ≥ 6 fenêtres ≥ 20 s alternées, rejet charge > 5 % / sd > 10 % / throttle actif, horloge médiane par fenêtre écart ≤ 3 %, en-tête TSV ; colonne J/jeton du README à remettre.

## F — modularisation (4 bis, à sec)
13. model.py : 2-3 FAITS (5aca3424, 2 912 l.) ; 4-5 (moe.py, couches.py) reportés après b=12 — 95 monkeypatchs de tests sur `engine.model` à repointer par script + test-garde, 2 commits, suite complète au trou.
14. ~~GUI module 3~~ FAIT (c365094d, 4 bis GUI clos : lanceurs 80/78 l.).
15. Ensuite : convert.py (2 398), loader.py (1 749), layers.py (1 449), kernels/__init__.py (1 401).

## G — livraison
16. Après dpkg : rejeu 19/19 installé, --decoder parc, feu vert parc (pièces 4, 6).
17. Poste THP/EPP (pièce 12, dernier : change la signature).
18. Republier GitHub (`outils/publier-github.sh --pousser`) à chaque livraison ; release du prochain moteur.

## H — ouverts par le OUI du 21/09 (après B-D, jamais avant l'objectif b=12)
19. C9 — S1 reborné (poste1 9530e314) : 26-31 j/s prédits (barre llama.cpp 24,2), vaut la peine ssi zéro-copie ≥ 17 Go/s (M-UVA ≤ 5 min, poste2) ET h(119B) ≥ 0,72 ; J/jeton ≈ 4-5 (bus) — vitesse seulement. M-HÔTE : arrêt (a) confirmé (16,6 Go/s) ; M-TRACE h_pin 0,733 → S1 cache PCIe à reborner (poste1, après remise) ; noyau AVX-512 = pièce 22. Conception FAITE ; ALARME poste1 21 h 2x : noyau hôte nvfp4 scalaire (2,2 Go/s par fil) → M-HÔTE probablement > 1,2 ms → arrêt (a) ; S2/S3 exigeraient un noyau hôte AVX-512 (plusieurs jours) ; tranché à la mesure c9-m-hote à vide (≤ 3 min). Conception FAITE (poste1-c9-conception-21-09 : S3 mixte ≈ 43 j/s à h=0,41, arrêt écrit) ; mesures M-HÔTE (≤ 5 min) et M-TRACE (≤ 10 min) par poste2 après le scellé E ; puis 3 commits du premier run (spec, table de noms + import nvfp4 compressed-tensors au bit, charge tout-hôte). Porte ajoutée 22/09 12 h 1x (verdict croisé duck.ai Q(6), qr.md) : E = 128, k = 4 → en uniforme h = C/E, 0,72 exige ≈ 92 experts résidents/couche, impossible en 32 Gio ; donc AVANT tout code de cache, histogramme de routage par (couche, expert) tiré de la passe de calibration du 119B (commit 3), h_static(C_abordable) = Σ top-C p_(i) ; S1 fermée si h_static < 0,72 ; C_abordable chiffré par poste1 avec la prédiction.
20. bf16 30B (60 Go) : alias `Qwen3-VL-30B-A3B-awq-dequant-bf16` servi par acvram (référence P3 (3), déjà sur disque 62,1 Go) — fiche alias, cellule b=1 en étagé, sert de témoin qualité aux conversions nvfp4/4sur6.

## Sans pièce
poste4 : verdict A/B quand une cellule tombe. chef : réarmer `/loop 20m` à la relance.

## I — ouverts par Vibe série 3 (21/09 22 h, verdicts en fin de qr.md)
21. Verrou : classe PARTAGÉ « carte visible sans calcul » (second fichier flock, LOCK_SH ; EXCLUSIF = LOCK_EX sur les deux ; `.qui/<pid>` ; promesse vérifiée par compute-apps) — les conversions longues y passent (poste3, outils/carte.sh + test).
22. Noyau hôte NVFP4 AVX-512 (table vpermb, 68 instr/64 quartets, micro-banc perf stat, test au bit) — seulement si c9-m-hote à vide réfute (> 1,2 ms).
23. Four Over Six : profil réel de la conversion (horodatages du journal) puis vectorisation de la recherche amax/6 vs amax/4 sur [n_blocs, 16, 2] au bit (poste2 profil, poste1 vectorisation).
24. Cellules TRT-LLM : liste de contrôle du verdict 3.4 (graphes actifs des deux côtés, exclusions listées, KV FP8/int8 vérifié dans la doc 1.3, glouton, même contexte/émission, en-tête de publication).
25. Calibration AWQ experts : instruments FAITS (poste1 7c658b08 : awq-stabilite-experts.py, `convert --repli-experts mediane_couche`, invites-experts-sans-stats.py) ; mesures poste2 : (a) stabilité ≈ 20 min → (c) sur l alias actuel → reconversion mediane_couche (≈ 25 min) → (c) + P3 (3), prédit 12,6 → 6-9 %.

## J — ouverts par Vibe série 4 (22/09 07 h 5x, verdicts en fin de qr.md)
26. Prédicteur Four Over Six hors ligne par bloc (MSE amax/6 contre amax/4 sur les quartets réels E2M1/E4M3, part de blocs gagnants, gain moyen ; < 10 min) — doit prédire la perte du scellé E sur le 31B avant toute conversion (poste1, 09 h 15 en régime mesuré).
27. Corpus de calibration guidé par routage : instrument de synthèse d'invites sous contrainte de score de routage + mesure de couverture (jetons par expert) — remplace le corpus générique qui n'atteint aucun expert froid (poste1 instrument, poste2 mesure ; après 25(a) qui donne le seuil). Verdict croisé duck.ai Q(5) (22/09 11 h 4x, qr.md) : critère « score max ≥ 0,27 » invalide → comptage réel top-8 par (couche, expert), min C ≥ 512 ; les experts froids d'un modèle vision calibré sur texte sont probablement des experts de jetons visuels → diagnostic positions image/texte d'abord (si ≥ 80 % image-only, voie texte fermée), puis corpus multimodal + boucle génération→routage→filtre par score de nouveauté S(x).
28. Énergie : deux périmètres publiés (carte seule NVML ; hôte + carte RAPL + NVML), note « experts sur processeur » par moteur, client isolé par soustraction (banc, après les cellules TRT-LLM b=1 réparées). Verdict croisé duck.ai Q(4) (22/09, qr.md) : au plafond 400 W, J/jeton = P_moy/débit — l'écart avec TRT-LLM est l'écart de débit ; ajouter la colonne (J/jeton)/MHz_moy aux cellules ; cellule séparée CUDA_DISABLE_PERF_BOOST=1 (jamais défaut).
29. Écart TRT-LLM b=12 (+22 %) : décomposition par poste1 (08 h 15) → leviers au bit ou choix de format (KV FP8 opt-in).
30. Cellule TRT-LLM à KV égalisé : acvram KV fp8_e4m3 (opt-in) contre TRT-LLM KV fp8, nommée « kv-fp8 », séparée de la cellule KV natif (poste3, après le b=1 ; jugée aussi par KL). Se juge par le débit et la KL, pas par les joules (Q(4) : même octet que int8).
31. Tests supplémentaires du levier 1 (duck.ai Q3.6) : slot EOS réadmis dans le même pas ; exécution avec et sans CUDA_LAUNCH_BLOCKING=1 (poste1, heure creuse).
32. [seuil relevé à 512 obs/expert par ef8835f7] Collecte AWQ : critère d'arrêt = min d'observations par expert ≥ 256 (512 si hétérogène), distribution publiée au manifeste (pièces 25(a)/27).
33. [FERMÉE 22/09 08 h 5x — RÉFUTÉ b08a3d34 : aucun T ≤ 1 ulp ne gagne, crochet à retirer avec la 35] **Mode opt-in « rapide ± 1 ulp » (décision utilisateur 22/09 08 h 4x, REGLES § 1)** : projections étroites int8 en split-K — conception poste1 (11 h 15) → code opt-in ACVRAM_ETROITES_SPLITK → poste2 : PPL + SE et KL contre le chemin exact, cellule b=12 « ± 1 ulp » A B B A A B (prédit 1 970-2 070 t/s, parité TRT-LLM) ; jamais défaut, jamais agrégée avec la cellule exacte.
34. ~~PDL Marlin~~ RÉFUTÉ avant code (trace nsys du service : trou entre noyaux 0,19 µs) ; banc-marlin-plancher joué (5e2bd345 : couche 96-99 %, « 8 couches » = artefact du banc).
35. Marlin `down` (K = 768 court, 1,26 To/s = 81 %, occupation/latence) : ptxas -v + banc `down` seul à L2 froid, levier au bit seulement si l'ordre de réduction intra-ligne est conservé ; plafond 0,26 ms (+ gate·up 0,12) = 5,6 % (poste1, après la conception split-K). Élargie 22/09 09 h 4x (verdict croisé duck.ai Q(3), qr.md) : aux étroites qkv/o aussi — `ptxas -v` (registres/thread, spills, programmes résidents par SM) au défaut, puis balayage borné BLOCK_M ∈ {16, 32} × num_warps ∈ {2, 4, 8} × num_stages ∈ {2, 3}, exact au bit (ordre de réduction K inchangé, test au bit dans le commit) ; prédit ≥ 10 % sur o (10,14 µs) à occupation doublée, réfuté sinon ; le crochet split-K se retire dans le même commit.
36. `acvram serve` : peupler `CompletionChoice.logprobs` (top-K par position, et `echo` avec logprobs du prompt = teacher forcing) — compatibilité API OpenAI + KL par API pour tous les moteurs ; test au bit contre `_sample_lent` (les logprobs du graphe existent depuis le levier 1) ; poste1 (moteur : logits par position au prefill) + poste3 (app.py). Après split-K.
37. Diagnostic de l'eval PPL sur gemma-4-31B (max6 53 202 après BOS ae519990, 4sur6 7 302, attendu 8-16) : contrôler `Tokenizer.bos_id` non-None, le gabarit appliqué contre celui de `serve`, le chemin prefill 2048 (fenêtre glissante locale/globale de Gemma 4) hors service ; un test qui rend « faux » = PPL ctx512 d'un texte du corpus d'entraînement ≤ 30. poste1, 11 h 15, AVANT la 26 (toute qualité 31B et le mode ± 1 ulp en dépendent). Verdict croisé duck.ai (qr.md, 09 h 3x) : ordre de diagnostic (a) BOS dupliqué (le tokeniseur pose-t-il déjà le BOS ? A a empiré après ae519990), (b) alignement logits[:, :-1]/ids[:, 1:], (c) softcap final de Gemma omis/doublé hors service, (d) masque local 1024/global et position_ids à 512/1024/2048 ; test discriminant : teacher-forcing court sans fenêtrage, NLL jeton par jeton contre Transformers bf16, première position divergente = cause.
38. [FAITE 22/09 12 h 0x — 538781f1 poste3] Replis silencieux signalés par le relecteur externe (22/09 11 h) et confirmés : `acvram/server/chat.py:118` (`except Exception: pass` → ChatML au moment du rendu, sans trace : l'invite change de format sans que le client le sache) ; `acvram/quant/gguf.py:581-582` (`rope_freqs.weight` illisible → `partial_rotary_factor_full = 0.25` sans trace). Correctif : avertissement une fois (logger) + valeur dans la ligne de régime (`gabarit=jinja|chatml-repli`, `rotary=mesuré|repli0.25`) + test qui échoue si le repli redevient muet ; à sec, ≤ 15 min, poste3. Le troisième point (FEUILLE-DE-ROUTE en-tête contredisant l'erratum) corrigé par la chef dans le dépôt, à republier au prochain feu GitHub.
