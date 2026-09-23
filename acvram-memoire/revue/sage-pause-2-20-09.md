# Sage — point de pause (utilisateur 09 h 07 : « pause du groupe dès que possible ») : état servi, ce qui est en vol, ordre de reprise tel quel, ce qui attend un mot de l'utilisateur (20/09, 09 h 08, horloge machine)

Source : `sage-reprise-rapide-20-09` (tous les addenda de la matinée, 06 h 55 → 09 h 00) ; `sage-niveau3-glm-go-20-09` ; `sage-tests-30min-20-09` ; `sage-nettoyage-modeles-20-09` ; verdicts fusionnés de Manon (0.6.30 3f679f26, 2 048 clés 5afe017e, C14-b 77aabdae, déterminisme 51790d69) ; ETAT de Jérôme.

## 1. État à la pause
* **Servi** : défaut **0.6.30** (C15-prefill au défaut, gemma capture, éco 2 700) — installé ? à lire par `dpkg -s acvram` ; feu vert donné 08 h 32. Cellules : **préfill Coder L=2 048 : 22 707 j/s, devant vLLM 20 824 et llama.cpp 15 532** ; revendication du 08 h 43 (Jérôme a0b1b211) : devant à b=1 (vitesse, J), devant au préfill Coder, derrière vLLM Marlin à b=12, PPL devant.
* **Établi ce matin** : la règle des 2 048 clés reste (tf32 à 8 192 double la dispersion de l'état : sd 2,06 × β) et son × 1,31 devient l'objectif de C13-c en fp32 ; une PPL GLM rend « établi pire / non établi pire », jamais « équivalent » (± 0,66 % à 36 tranches disjointes, ± 1,31 % à 9) ; le témoin d'un bras sous graphes est `GRAPHS_EAGER=1`, l'équivalence au bit se juge à godet égal (déterminisme Coder b=12 = clé de godet) ; aucune prise de carte > 30 min (`carte.sh` DUREE_MAX), un chargement par bras (outil 3.2), arrêt séquentiel n ≥ 5 avec plancher 1,97 %/√n.
* **Faux ce matin** (prédictions de Sage, toutes tenues pour fausses) : main/d01eb2cb ± 1 % (mesuré +2,60 = règle des clés) ; tf32 meilleur de 1-2,6 % (mesuré +1,16 à 9, −1,07 à 36) ; C14-b −1,05 ms (mesuré +0,42, combine à 640 ulp sur 37 lignes) ; « 8 + 1 doublon » et « fragmentation à deux états » (retirés) ; C15-prefill servi 19 500-20 500 (mesuré 22 707, par en dessous).
* **En vol à la pause** : rien de Sage ; Manon finit sa prise en cours (pièce 2a nsys) ou s'arrête ; Océane : lecture `combine_vb` (à sec, 45 min) — s'arrête proprement, commit, pointeur.
* **Nettoyage des disques** (utilisateur) : phase 1 + purge faites 09 h 04, 762 Go effacés, SSD 454 Go libres, HDD 390 Go ; inventaires et menus à jour sur `sage-tests` 8073e72e (32 tests verts sous verrou) ; la classe A lourde (models_acvram_hdd 885 Go, variantes GLM 235 Go…) n'a pas été cochée : elle attend un mot.

## 2. Ordre de reprise (tel quel)
* **Manon** — pièce 2a (juge fp64 de `=2` à b=1, une prise ≤ 5 min) → pièce 1 (C10 b, 4 prises ≈ 25 min, B = A − 26 nœuds ± 10) → nsys C14-b (a)(b) 5 min sous `MLA_BATCH_FUSION=1` → S2 `inactive_split_bytes` si non fait ; chaque prise ≤ 30 min, septième ligne « durée ».
* **Océane** — `combine_vb` : lecture ligne par ligne contre `mla_1p_combine_kernel`, test cassant sur les 37 lignes (prédiction Sage : tranche vide du softmax) ; `ACVRAM_MLA_PREP_GRILLE=1` (prep regrillé seul au défaut si ≤ 8 µs/couche et pas b=12 non perdu) ; puis pièces 3 (routeur un nœud), 4 (experts), 6 (hors couches) ; pièce 7 « casts et copies » chiffrée après 2a et 3.
* **Jérôme** — fusion de `sage-tests` 8073e72e ; un .deb par pièce tenue (bras éco) ; ETAT ≤ 40 lignes seule entrée ; REGLES § 4 (0,66 / 1,31 / 62), § 6-7 (témoin `GRAPHS_EAGER=1`, godet égal).
* **Sage** — silence jusqu'à la reprise ; à la reprise : rien à relire avant ETAT.

## 3. Ce qui attend un mot de l'utilisateur
1. **119B** : (a) 3080 Ti calculante, (b) experts sur processeur, (c) aucune, (d) 5090 seule en × 16 + cache PCIe — avis de Sage inchangé : (d) si le × 16 se confirme, sinon (c) ; le GGUF 119B est gardé, NVFP4 (71 Go) gardé, eagle effacé.
2. **Configuration × 16** (3080 Ti sur un port × 4 du chipset, `bench_link_bandwidth` ≥ 40 Go/s ; trois services permanents à reloger).
3. **Installation** : `sudo dpkg -i acvram_0.6.30_amd64.deb` puis `acvram doctor` (si pas déjà fait).
4. **Nettoyage, classe A restante** : models_acvram_hdd (885 Go, 54 convertis jamais servis), variantes GLM closes (235), essais Nemotron (98), doubles quantifications (35), `Qwen3.8-27B-bf16.gguf` (55) — cocher dans `outils/nettoyage-modeles-liste.txt` puis `outils/nettoyage-modeles.sh`.

Règles vivantes inchangées : `date` avant toute heure ; pointeurs, jamais de réécriture ; un scellé par fenêtre, prédiction et issue gênante écrites avant ; instrument = fichier suivi + commit ; tests de Sage sous le verrou (`ACVRAM_TYPE=etat`), jamais sur la lecture de `.qui` ; une cause reçue par pointeur ne reçoit qu'une prédiction, jamais une entrée MECANISMES avant le verdict.
