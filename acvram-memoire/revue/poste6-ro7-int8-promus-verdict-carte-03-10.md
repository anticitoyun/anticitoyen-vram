# ro7, carte : les int8 promus servis par le GEMM int8 (re-quantifiés par canal au chargement) donnent **+15,8 % de préfill à M = 4 096 (+31,6 % à 512)** pour une PPL wiki 2048 × 1,0119 (≤ 1,020) — les deux seuils du chef sont tenus, l'option B (promotion par canal à la conversion) est proposée ; réserve nommée : sous l'opt-in la sortie dépend du chemin selon M (A8 au-delà de 16 lignes), témoin reprise à 0,31

instrument : `scratchpad/poste6-bf16/carte-ro7.sh` (garde de chaîne `poste6-ro7-chaine`, 5 bras `service`) — `prise-ro7.py` (R1 chrono nu `gemm_i8c_cublas` sur poids canal symétrique contre `linear_prefill` bf16, cinq formes ; R2 débit du préfill moteur, 6 passes, médiane, un processus par bras : le mode est posé au chargement ; R3 ligne de régime ; R6 pic mémoire), chaîne S1 A1 sous l'opt-in (`prise-s1-morceaux-kv31b.sh`, REQUETES=2), `acvram eval` wiki-gptq fenêtre 2048 pas 2048, 131 008 jetons, 64 fenêtres, défaut puis opt-in ; journal `scratchpad/poste6-bf16/carte-ro7.log` (non suivi ; le script s'est arrêté sur mon analyse R5 — clé JSON mal lue —, refaite à sec sur les fichiers)
commit : poste6-gemma-anneau 827f06bf9, arbre propre sous `acvram/` et `outils/`
régime : RTX 5090 seule, aucun autre poste (journal du verrou, 10 lignes toutes à moi) ; Devstral-24B `srcawq-nvfp4` (55 int8 promus g128 affines), NOMINAL ; `prefill_int8=repli-bf16×55` (défaut) contre `prefill_int8=cublas` (`ACVRAM_INT8_PROMUS=canal`) ; torch 2.14.0+cu130
scellé : `poste6-ro7-int8-promus-scelle-03-10.md` (R1-R6) ; seuils du chef (05 h 0x) : R2 ≥ +5 % à 4 096 ET R5 ≤ × 1,020 → proposer B
mesuré : 03/10 05:11:07 → 05:14:28 (3 min 21 s)
verdict : **seuils tenus, B proposée.** R1 : le GEMM int8 cuBLASLt vaut × 0,45-0,63 du bf16 déquantifié dès M ≥ 1 024 sur les cinq formes (k / v à 512 : × 1,24, seule exception). R2 : préfill **2 347 → 3 089 j/s à 512 (+31,6 %), 2 955 → 3 624 à 1 024 (+22,6 %), 3 319 → 3 943 à 2 048 (+18,8 %), 3 632 → 4 204 à 4 096 (+15,8 %)** — bien au-delà de mes +6 à +10 % : le repli payait aussi la déquantification à chaque préfill et le trafic bf16, pas seulement le GEMM. R5 : PPL **5,6644 → 5,7320 (× 1,0119)**, 64 fenêtres sur 64 plus mauvaises (Δ log-ppl par fenêtre + 0,0069, IC95 ± 0,0009) : écart petit, systématique, sous le seuil. **Réserve** (R4) : sous l'opt-in, la même requête rejouée sur cache de préfixe s'écarte de **0,31** (contre 0,027 sous le défaut) : au-delà de 16 lignes le chemin est W8A8 (A8 par jeton), en dessous W8A16 (GEMV) — la sortie dépend du CHEMIN selon M, bien plus que par la réduction cuBLAS qui a occupé les pièces précédentes. Opt-in gardé, défaut inchangé.
durée : 3 min 21 s de carte (prévu ≈ 6 min)

## Prédit / mesuré

| # | prédit | mesuré | |
|---|---|---|---|
| C0 garde de chaîne | tenue | 05:11:07 → 05:14:28, 5 bras, aucun autre poste | tenu |
| R1 int8 / bf16 au noyau | 0,50-0,70 sur les larges à M ≥ 2 048 ; 0,7-1,1 à 512 et sur k / v | **gate / up 0,52-0,51, down 0,45-0,49, q 0,45-0,48, o 0,49-0,50** (M ≥ 1 024) ; 512 : 0,54-0,63 sauf k / v × 1,24 ; k / v ≥ 1 024 : 0,80-0,54 | tenu (meilleur que prédit sur q, o, down) |
| **R2 débit moteur, canal contre défaut** | 4 096 : +6 à +10 % · 2 048 : +5 à +9 · 1 024 : +3 à +7 · 512 : +1 à +5 | **4 096 : +15,8 %** (974,3 contre 1 127,9 ms) · 2 048 : +18,8 % · 1 024 : +22,6 % · 512 : +31,6 % (165,8 contre 218,2 ms) ; étendues disjointes, processus séparés (bruit inter-processus vu ≤ 0,5 %) | **≥ 5 % : tenu**, prédiction trop basse |
| R3 ligne de régime | `cublas` sous l'opt-in, `repli-bf16×55` au défaut | `prefill_int8=cublas` (prise et serveur), `repli-bf16×55` | tenu |
| R4 seul tenant canal contre défaut | ids différents attendus (hors bit) | ids DIFFÉRENTS dès la position 0, Δ 0,28 sur 8 valeurs ; **témoin reprise sous canal : Δ 0,31 sur 315 valeurs** (défaut : 0,027) | dit ; réserve ci-dessous |
| **R5 PPL wiki 2048** | × 1,005-1,020 | **× 1,0119** (5,6644 → 5,7320 ; nll 1,7342 → 1,7461) ; par fenêtre + 0,0069 ± 0,0009, 64 / 64 pires | **≤ 1,020 : tenu** |
| R6 pic mémoire au préfill | −0,5 à −1,5 Gio | 17,31 → 17,54 Gio (+0,23) : la déquantification par tranches ne pesait pas, les tampons A8 / int32 s'ajoutent | FAUX (sens inverse, petit) |

## Ce que la fenêtre apprend

1. **Le repli bf16 des promus coûtait plus que son GEMM** : 19,8 % des opérations prédisaient +6 à +10 % ; mesuré +15,8 %
   à 4 096 et +31,6 % à 512. Déquantification (`int8_dequant` + copie bf16) et trafic mémoire des matrices bf16 à chaque
   préfill n'étaient pas dans mon compte — à 512 lignes ils dominent.
2. **La qualité tient le seuil, mais l'écart est systématique** (64 fenêtres sur 64, IC95 serré) : double quantification
   (g128 affine → canal symétrique) + A8 par jeton sur les couches que la conversion avait justement jugées sensibles.
   L'option B (promotion directement par canal à la conversion) supprime la double quantification ; l'A8 reste.
3. **Réserve R4** : `gemm_i8c_cublas` refuse M ≤ 16 (`kernels/__init__.py:1156`), donc un préfill court (reprise sur cache
   de préfixe, décodage spéculatif, petits morceaux) passe en W8A16 et un long en W8A8 : deux arithmétiques pour le même
   poids selon M, écart 0,31 au témoin reprise. Pour un modèle servi en option B, le même effet existera — à mesurer avant
   le défaut, et à fermer soit en servant A8 aussi au GEMV (noyau), soit en acceptant l'écart dit sur la ligne de régime.
   Les variantes `-i8c` déjà servies ont ce comportement aujourd'hui, non dit.
4. R6 à l'envers : la mémoire n'est pas le gain, le temps l'est.

## Décision appliquée et ce qui reste au chef

* `ACVRAM_INT8_PROMUS=canal` reste opt-in (827f06bf9), défaut inchangé ; la ligne de régime dit le chemin pris.
* **Option B proposée** (seuils du chef tenus) : promouvoir par canal symétrique à la conversion (comme
  `--attn-qkvo-int8-canal` pour q / k / v / o, étendu aux rôles promus) — reconversion par modèle, parc entier concerné
  (47-370 poids par modèle) ; prédiction pour son scellé : PPL entre × 1,005 et × 1,012 (une quantification de moins),
  débit identique à R2.
* Avant tout défaut (A ou B) : la réserve R4 (W8A8 contre W8A16 selon M) et la garde de qualité complète (KL, pas
  seulement PPL wiki) ; le ticket ro7 reste ouvert sur « gemm_i8c_cublas rend None » : ce n'est pas un bogue de forme,
  c'est la quantification g128 affine des promus — le ticket peut être requalifié « promotion par canal à la conversion ».
