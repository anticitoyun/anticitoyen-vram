# poste7 — reprise rapide (19/09, 17 h) : carte libre, deux fenêtres en un bloc

Source : `ETAT.md` 15 h 30, `verdict-p2-hors-moteur-18-09`, `verdict-controle-p1-defaut-19-09`, dépôt `0012b71`.

## 1. Constat (vérifié, pas annoncé)

* Carte **libre** : aucun `.lock` sous `/tmp/acvram-carte-*`, `nvidia-smi` 15 Mio sur les deux cartes, **0 processus**. Les trois services permanents (8081/8082/8083) sont **arrêtés** — `ss -ltn` : 0 port. On les laisse arrêtés jusqu'à la fin du bloc de carte (les relancer pendant une fenêtre = changement de régime), chef les relance après.
* `ACVRAM_GEMV_SPLITK` : **absent de `main`** (`git grep` = 0) — l'opt-in d'poste1 (file § 3) n'est pas commité.
* `revue/poste7-reprise-19-09-*.md`, `RESUME-FINAL-19-09.md`, `REPRISE-TRAVAIL-19-09-MISTRAL-VIBE.md`, `poste7-IDENTIFICATION-MISTRAL-VIBE-19-09.md` : écrits par Vibe (en-tête « IDENTIFICATION PAR MISTRAL VIBE », emojis), **pas par poste7**. Un fichier `poste7-*` que poste7 n'a pas écrit entre dans INDEX sous une fausse signature : à archiver hors dépôt avec les artefacts du § 7 de la file, aucun gardé.
* `outils/gpu/hors-verrou.log` (78 lignes, non suivi) : dernières entrées du 15/09, `comfy_blackwell312` hors verrou — rien depuis ; à commiter tel quel comme pièce, pas à trier.

## 2. Ce que la file avait de faux : le scellé P2 à deux seuils

`ETAT.md` § 4 : « prefill ≥ 18 500 tenu, < 17 000 faux » — bande orpheline 17 000-18 500, interdite depuis le 17/09 (un scellé = un seuil). Et 18 500 n'est dérivé d'aucun budget mesuré : la part des projections q/k/v/o dans le pas de prefill au régime défaut n'a jamais été publiée.

Scellé P2 réécrit, **un seuil par grandeur, écrit avant toute mesure** :

| grandeur | référence (défaut 0.6.13, `45ad8f6`) | seuil unique | rend « faux » si |
|---|---|---|---|
| équivalence décodage `_int_mm` | arbitre prefill même texte, ≥ 128 jetons, MoE égal (REGLES § 4 bis) | 0 écart top-1 hors égalité de score ; médiane et p99 des distances de logits dans ± 5 % de la référence | un seul top-1 divergent hors égalité |
| PPL i8c dans le moteur, corpus scellé privé (3 tranches) | classé 1,0148 × bf16 HF | **≤ 1,020** | > 1,020 |
| prefill i8c, 2 047 jetons, harnais `certifie`, même ctx | 16 426 j/s | **≥ 17 500** (+6,5 %) | < 17 500 |
| J/jeton prefill i8c, compteur `energie.py`, 400 W | J/jeton du défaut mesuré dans le même passage (bras A) | **≤ 1,00 ×** A | > A |

Prédiction poste7 : PPL 1,009-1,012 (verdict P2 : 1,0094 attendu) ; prefill 17 800-19 000 — `_int_mm` (int8 tenseur, cuBLAS) contre Marlin déquant à M = 2 047 sur 4 projections denses × 48 couches ; J/jeton 0,93-0,98 ×. Issue qui gênerait poste7 : prefill tenu et J/jeton > A (int8 plus rapide, plus de watts par octet) → **pas au défaut**, opt-in, cellules inchangées — même règle que split-K.

Contrôle qui peut rendre faux : bras A (défaut) et bras B (i8c `_int_mm`) en **ABAB** dans la même prise de carte, `regime_ligne()` imprimée par chaque bras, `--query-compute-apps` début et fin, load1 < nproc/2 dans l'en-tête. Le bras A ≠ 16 426 ± 2 % avant B = on mesure autre chose, arrêt.

## 3. Pourquoi ce bloc de carte, et pas un autre

Deux fenêtres, une seule prise de verrou chacune, **zéro attente** entre elles :

1. **poste1 (≈ 45 min)** : équivalence décodage `_int_mm` (ligne 1 de la table) + passe de capture godets {1, 2, 8, 16} avec `warm_graphs` (REGLES § 3, noyau de décodage). À sec AVANT la prise : `pytest -k int_mm` ciblé, régime `hors_defaut` imprimé. Si la ligne 1 réfute : poste1 libère, écrit `verdict: revue/verdict-p2-equiv-19-09.md — FAUX <cause>`, et passe au bead graphes/eager (§ 5 de la file) — poste2 ne mesure alors que le bras A (PPL au défaut, due depuis le 19/09).
2. **poste2 (≈ 1 h 30)** : PPL au régime défaut (Coder, 3 tranches, corpus scellé, sha256 dans l'en-tête — c'est la dette de son § 2) **et** lignes 2-4 de la table, ABAB, dans la même prise. Verdict unique `revue/verdict-p2-moteur-19-09.md`, six lignes fixes, attestation lot par l'outil.

Pendant la fenêtre de poste2, poste1 commite **à sec** l'opt-in split-K (`ACVRAM_GEMV_SPLITK`, défaut 0, table de régime, tests ciblés) sur `poste1-11` ; chef fusionne après le verdict de poste2, cellules publiées inchangées.

Gain attendu du bloc : P2 tranché le 19/09 au lieu d'une fenêtre par jour ; coût : ~2 h 15 de carte, 0 fusion pendant les fenêtres. Réfutation : un des deux verdicts sans ses six lignes ou sans `regime_ligne()` = ne compte pas, on refait.

## Ordre

* **poste1** — maintenant : `pytest -k int_mm` à sec puis verrou `outils/carte.sh` ; équivalence décodage `_int_mm` Coder i8c contre arbitre prefill (même texte ≥ 128 jetons, MoE égal) + capture godets {1, 2, 8, 16} `warm_graphs` ; verdict `revue/verdict-p2-equiv-19-09.md` (instrument · commit · régime · scellé · mesuré · verdict) ; libérer le verrou AVANT d'écrire. Puis, à sec pendant la fenêtre de poste2 : commit `ACVRAM_GEMV_SPLITK` opt-in (défaut 0, `regime_ligne`, tests ciblés) sur `poste1-11`. Si l'équivalence réfute : bead graphes/eager (+0,206 tr2-p256), pas de P2 moteur.
* **poste2** — dès le pointeur d'poste1 : verrou ; une prise : (A) PPL défaut Coder 3 tranches privées, (B) PPL i8c `_int_mm`, puis prefill 2 047 jetons ABAB `certifie` même ctx + J/jeton `energie.py` 400 W ; seuils § 2 (≤ 1,020 · ≥ 17 500 · ≤ 1,00 × A), arrêt si A hors 16 426 ± 2 % ; verdict `revue/verdict-p2-moteur-19-09.md`. Si poste1 a réfuté : bras A seul, verdict `revue/verdict-ppl-defaut-19-09.md`.
* **chef** — à sec, maintenant : archiver hors dépôt les fichiers Vibe du § 1 (dont les `poste7-reprise-19-09-*`, `RESUME-FINAL`, `REPRISE-TRAVAIL`, `poste7-IDENTIFICATION`) et les artefacts § 7 de la file, aucun gardé ; commiter `outils/gpu/hors-verrou.log` tel quel ; `ETAT.md` : § 4 remplacé par la table § 2 de cette note, services 8081-8083 notés arrêtés ; tri des worktrees `poste3`/`poste4`/`poste8` (lister, rien de forcé). Après le verdict de poste2 : relancer 8081/8082/8083, fusionner `poste1-11` (opt-in split-K), P2 au défaut **seulement** si les quatre lignes tiennent — sinon opt-in `ACVRAM_PROJ_INT8=1`. Push sur « oui » utilisateur.
* **Vibe** — inchangé (`poste7-menus-cloture-19-09` § Ordre) ; ne rejoue rien sur la carte.
* Chacun écrit à poste7 **une fois** : `verdict: revue/<fichier> — 1 ligne`, chef en copie. poste7 n'intervient qu'à un seuil réfuté.

## Addendum 16 h 35 (après `43a6818`, `be9d5f1`)

* Bras A **déjà rendu** : PPL défaut Coder 1,0155 (poste2 `eda96a8`, 3 tranches 1,0112 / 1,0116 / 1,0239, identique au 18/09 — chemin PPL = prefill Marlin). L'ordre poste2 se réduit à B : PPL i8c `_int_mm` + prefill ABAB + J/jeton, seuils § 2 inchangés ; le bras A du prefill (16 426 ± 2 %) reste le témoin de la prise. Nom de la variable : celui que publie le verdict d'poste1, poste2 ne le devine pas.
* `verdict-vibe-reprise-19-09` (tâche 1) : proposition gardée pour § 2 ; pour § 10, retrancher les points déjà livrés ou clos (attention paginée = défaut depuis 0.6.8 ; GEMM groupé MoE = prefill Marlin depuis P1) et tout point sans fichier:ligne. REPRISE.md s'écrit **une fois**, par chef, après le verdict P2 de poste2 — pas avant, sinon on le réécrit le jour même.
* Vibe relancé 16 h 30 (mode programmatique, `CUDA_VISIBLE_DEVICES=""`, `--max-turns 80`) sur les deux tâches ; il peut réécrire `verdict-vibe-reprise` : un `git diff` tranche, la version commitée `be9d5f1` fait foi si le nouveau n'apporte pas de source. `verdict-vibe-menus-19-09` attendu avant `sauvegarde-config-ia`.
