# Verdict — latent fp8 (f4607c2, bras D) : 0 ms de gain et PPL identique au bit à bf16 → le cache fp8 n'est pas lu par le chemin batché ; deux familles d'arithmétique MLA (boucle 8,0175 / batché 8,0003) que personne n'avait comparées

- **instrument** : identique à `verdict-mla1/2/3` + **7 contrôles de sensibilité** tranche 0 (`campagne-mla-controle{,2,3}-16-09.sh`) ; sorties `scratchpad/mla5-16-09/`
- **commit** : arbre mesuré **travail/poste3-qa @ f4607c2** (poste4 ; test fp8 corrigé, noyau inchangé depuis 3a1d2fd) ; tests **25/25** ; bras **C** = défaut, **D** = `ACVRAM_MLA_LATENT_FP8=1`
- **régime** : `-k48`, prefill W4A16, décodage MMA=1 MIN_T=5, MLA_BATCH=2, une passe, prep ; graphes actifs (et eager en contrôle)
- **scellé** : chef : PPL D/C dans 1 ± 0,004 (réfuté → bf16 gardé), mla_* −0,2 à −0,5 ms. Moi : mla_* 1,9-2,2, PPL D/C 1,001-1,004, top-1 ≥ 99 %, pas bit-identique
- **mesuré** : D pas **17,29** (C 17,08), `mla_1p_kernel<fp8>` **2,11 = 2,11** bf16, 1 835 lancements ; ties C/D **bit-identique 12/12** ; PPL D/C **1,000000** × 3 tranches ; contrôles : C eager 8,000319 = C graphes ; D eager 8,000319 ; **`MLA_BATCH=0` 8,017474** (graphes et eager) ; `MOE_DECODE_MMA=0` 7,888561
- **verdict** : bras D **non adoptable, et pas pour la raison scellée** : ni gain (0,0 ms à ctx ≈ 300 — la lecture du latent n'est pas ce qui borne `mla_1p` ici), ni coût de qualité mesurable — parce que **le contenu du cache latent ne change rien à la sortie du chemin batché** (E4M3 par ligne ≡ bf16 au 10⁻¹² sur 3 × 22 664 jetons : impossible si les codes étaient lus). Anomalie d'intégration, poste4 (§ 1). Mes prédictions : mla_* tenue (2,11), PPL et « pas bit-identique » réfutées — par l'anomalie, pas par le fp8

## 1. Ce que les sept contrôles établissent (tranche 0, 22 664 jetons, même arbre)
```
chemin                         graphes   eager      lecture
batché (BATCH≥1, 1p, prep)     8,000319  8,000319   une seule arithmétique, insensible aux graphes
batché + latent fp8            8,000319  8,000319   INSENSIBLE au format du cache → le cache écrit n'est pas ce qui est lu
boucle par séquence (BATCH=0)  8,017474  8,017474   une autre arithmétique : +0,21 % de PPL
MoE W4A16 (MMA=0)              7,888561     —       l'instrument voit bien le décodage : W4A4 = +1,42 % au décodage GLM (chiffre pour poste7 § 8.3)
```
- **Les « bit-identiques » des verdicts mla1-3 comparaient des chemins batchés entre eux** (BATCH 1 ↔ 2, une passe, prep) : vrais, mais ils ne disaient rien de la boucle d'origine. Boucle et batché diffèrent de 0,21 % — au-dessus du ± 0,004 du scellé, dans le sens « batché meilleur », et personne ne sait lequel est la référence : à trancher par un troisième chemin (prefill teacher-forcé `acvram eval` du même texte en décodage, ou b=1 sans statics).
- **fp8 inerte** : le test unitaire prouve le noyau seul (codes × échelle en float64, 1e-4) ; en intégration, `new_static` alloue bien uint8, `mla_ecrit_latent(fp8=1)` et `mla_1p<FP8=true>` sont bien lancés (nsys) — et la PPL ne bouge pas d'un bit. Pistes pour poste4, non vérifiées : (i) le kernel `<FP8>` lit `cache8` mais l'écriture fp8 dans `mla_ecrit_latent_kernel` (`acvram_kernels.cu:4265-4307`) n'écrit pas là où le lecteur lit (foulée `W+16` des deux côtés ?) ; (ii) `static_load` quantifie le prefill mais l'attention batchée lit une autre source pour les positions déjà présentes ; (iii) le chemin batché lit le latent AVANT écriture (le pas courant seulement) — ce qui expliquerait aussi le 0,21 % contre la boucle. Le test qui tranche : après 100 pas, relire `st["cache"]` (uint8) et le comparer au latent recalculé ; puis mettre le cache à zéro et vérifier que la PPL bouge.

## 2. Conséquence
Commits 1-2 (batch, une passe, prep) restent tenus en temps ; leur équivalence est acquise **entre eux**, pas avec la boucle : la question « lequel des deux a raison » précède la fusion dans main, et le fp8 attend la réponse.
