# ro7 option B, carte : la promotion int8 par canal à la conversion donne le débit de A (+15,1 % à M = 4 096) et une PPL × 1,0114, mais **B ≈ A (× 0,9995 : l'écart n'était pas la double quantification), KL@64 médiane 0,049 nat (seuil 0,015) et témoin reprise × 3,4 celui de l'actuel** — B écartée du parc selon le scellé, A reste opt-in, pas de lot des 9 autres

instrument : `scratchpad/poste6-bf16/carte-ro7b.sh` — (1) conversion `acvram convert … --no-awq --snr-floor 25 --max-promotions 0.15 --mixed-precision auto --int8-canal` sous `carte.sh` (mesure), horodatée ; (2) chaînes sous garde `poste6-ro7b-chaine` : `prise-ro7.py moteur` (B5), `acvram eval` wiki-gptq 2048 / 2048 / 131 072 (B3), `kl-fenetres.py` (B6 : KL@64 tronquée au top-64 de l'actuel, masse hors top-64 de B bornée par son 64ᵉ logprob — sous-estime la KL vraie), `temoin-reprise.py` (B7 : séquence de chauffe de 7 865 jetons générée deux fois, 32 jetons gloutons, cache de préfixe — PAS l'invite S1 de la chaîne serveur : instrument différent de celui du scellé, dit). Journaux `carte-ro7b.log`, `carte-ro7b-2.log` (non suivis)
commit : poste6-gemma-anneau 07faa2989 (conversion, chaîne 1) et 05cae524b (chaîne 2 : instrument KL corrigé, B5 rejoué)
régime : RTX 5090, carte 0 ; converti B `/mnt/AI_GENERATOR/models_acvram/devstral-24b-srcawq-int8canal-nvfp4` (16 G) contre l'actuel `devstral-24b-srcawq-nvfp4` ; `prefill_int8=cublas` (B) contre `repli-bf16×55` (actuel) ; torch 2.14.0+cu130. **Entorses dites par chef** : copie rsync (NVMe `/home` → SSD SATA, pas mon disque de sortie) de 06:39 à ≈ 06:41 pendant la conversion (B8 seule concernée) et ≈ 2,5 Go de 06:43:18 à ≈ 06:44 pendant mon premier B5 — **B5 rejoué** à 06:46 hors copie, ce sont ces chiffres qui comptent
scellé : `poste6-ro7-b-conversion-scelle-03-10.md` (B1-B8) ; seuils du chef : PPL × 1,005-1,012, B < A de 0,002, KL médiane ≤ 0,015 / 95ᵉ ≤ 0,05, témoin reprise ≤ 0,06
mesuré : conversion 06:39:31 → 06:43:18 ; chaîne 1 06:43:18 → 06:44:33 (B5 marqué, B3, B4) ; chaîne 2 06:46:05 → 06:50:20 (B5, B6, B7)
verdict : **B écartée du parc, A reste opt-in** — par la décision écrite au scellé (« B écartée si B6 fausse »). Ce qui tient : la conversion fait exactement ce qui est demandé (B1 : 55 int8 tous par canal, mêmes promus, aucun autre format changé ; B2 : 16 G ; B8 : 227 s), le débit est celui de A (B5 : **+15,1 % à 4 096**, +31,4 % à 512), la PPL tient le seuil de justesse (B3 : **× 1,0114**, 63 fenêtres sur 64 pires). Ce qui casse : **B4** — B / A = × 0,9995 : enlever la double quantification ne rend rien, l'écart est dans le canal lui-même (SNR des promus 53,6 → 38,0 dB médian, 28,5 min) et dans l'A8 ; **B6** — KL@64(actuel ‖ B) médiane **0,0485 nat**, 95ᵉ 0,077, max 0,087, sur 64 fenêtres (seuil 0,015 / 0,05) : la distribution bouge trois fois plus que ce que j'avais fixé, à toutes les positions, ce que la PPL (× 1,0114 sur les seuls jetons du texte) ne montrait pas ; **B7** — témoin reprise 0,48 sous B contre 0,141 pour l'actuel avec le même instrument (× 3,4) : les deux arithmétiques W8A8 / W8A16 selon M sont là. **Pas de lot des 9 autres modèles.**
durée : 3 min 47 s de conversion + 1 min 15 s + 4 min 15 s de chaînes

## Prédit / mesuré

| # | prédit | mesuré | |
|---|---|---|---|
| B1 bilan de conversion | 55 int8 tous canal, mêmes promus | 55 int8, canal 55, `promoted_from` gardé 55, mêmes tenseurs, 0 autre format changé ; SNR des int8 : médiane **38,0 dB**, min 28,5 (actuel g128 : 53,6) | tenu ; SNR sous mon « 45-50 dB » |
| B2 taille | ≈ 16 Gio | 16 G | tenu |
| B3 PPL B / actuel | × 1,005-1,012 | **× 1,0114** (5,6644 → 5,7290), par fenêtre + 0,0066 ± 0,0009, 63 / 64 pires | tenu (à 0,0006 du bord) |
| **B4 B contre A** | B < A d'au moins 0,002 | **× 0,9995** (5,7320 → 5,7290) : B < A de 0,0005 | **FAUX** : la double quantification de A ne coûtait rien |
| B5 débit B / actuel (rejoué hors copie) | +15 à +17 % à 4 096 ; +30 à +33 % à 512 | 4 096 : 4 179 j/s (980,2 ms) contre 3 632 : **+15,1 %** · 2 048 : +18,0 % · 1 024 : +21,8 % · 512 : +31,4 % ; A : 4 204 (× 0,994) | tenu |
| **B6 KL@64(actuel ‖ B)** | médiane ≤ 0,015, 95ᵉ ≤ 0,05 | **médiane 0,0485 ; 95ᵉ 0,0766 ; max 0,0874 ; min 0,0181** (64 fenêtres, tronquée : la vraie est ≥) | **FAUX** (× 3,2 et × 1,5) |
| **B7 témoin reprise** | sous B 0,2-0,4 ; critère ≤ 0,06 pour un défaut | B : **0,480** sur 311 valeurs (32 ids égaux) ; actuel, même instrument : 0,141 sur 47 valeurs (divergence au jeton 4) — instrument en processus sur séquence de chauffe, pas l'invite S1 (0,027 au serveur) : le critère 0,06 ne s'applique pas tel quel, le rapport **× 3,4** si | **FAUX** (rapport) ; critère à réécrire pour cet instrument |
| B8 temps de conversion | 10-25 min | **227 s** (copie rsync concurrente 06:39-06:41, autre disque) | faux dans le bon sens |

## Ce que la fenêtre apprend

1. **Le prix n'est pas la double quantification, c'est le canal (et l'A8).** Passer les promus de g128 affine à canal symétrique
   leur retire 15 dB de SNR médian (53,6 → 38,0) : une échelle par ligne de 5 120 ou 32 768 coefficients ne suit pas des
   poids que le routeur avait promus justement parce qu'ils se quantifient mal. La PPL ne bouge que de 1,1 % parce que le
   texte évalué pardonne ; la KL@64 à toutes les positions dit 0,048 nat médian.
2. **Mon seuil KL (0,015) n'a pas de témoin** : je l'avais fixé sans mesurer ce que vaut, dans ce même instrument, un
   changement jugé acceptable (par exemple l'actuel contre lui-même sous le drapeau bf16 exact, ou l'actuel contre le HF
   bf16 : la KL de la quantification nvfp4 elle-même). Le résultat « FAUX » tient par le scellé, mais la lecture « trois
   fois trop » n'aura de sens qu'avec ces deux témoins — à mesurer avant de rouvrir B. Mesurés ici, ils auraient coûté
   deux relevés d'une minute : c'est ma faute de scellé.
3. **R4 est réelle et grande** : avec le même instrument, le témoin reprise passe de 0,141 à 0,480 (× 3,4) quand les promus
   prennent le W8A8 — le préfill court (M ≤ 16, GEMV W8A16) et le long (W8A8) ne calculent pas la même chose. Toute
   voie int8-canal (A, B, les `-i8c` servis) porte ce défaut tant que `gemm_i8c_cublas` refuse M ≤ 16.
4. **Le débit, lui, est acquis** (+15 % à 4 096, +31 % à 512, identique entre A et B) : c'est le prix qualité qui reste à
   payer autrement — un GEMM int8 à échelles de groupe (le poids g128 affine servi tel quel, point zéro et échelle par
   groupe dans le noyau, hors de cette pièce) garderait les 53,6 dB.

## Décision appliquée et ce qui reste au chef

* B écartée (scellé : B6 fausse) ; `--int8-canal` reste une option de conversion mesurée et documentée (opt-in, aucun
  modèle du parc converti avec), `ACVRAM_INT8_PROMUS=canal` (A) reste opt-in ; **aucun lot de reconversion préparé**.
  Le converti B (16 G, `/mnt/AI_GENERATOR/models_acvram/devstral-24b-srcawq-int8canal-nvfp4`) peut être supprimé ou gardé
  comme témoin — à toi.
* Si tu veux rouvrir : d'abord les deux témoins KL@64 (actuel ‖ actuel-exacte, HF bf16 ‖ actuel) pour donner un sens au
  seuil ; puis la voie « GEMM int8 à échelles de groupe » (noyau) plutôt que le canal, et la fermeture de R4.
* ro7 : requalifié par toi ; ce verdict en est la fin côté conversion.
