# Scellé — pièce 141 : PDL sur les GEMV int8 (poste6, 24/09, AVANT toute mesure) — feu de chef sur le dossier 61903975

## Code (branche poste6, commit dans le verdict)
* `acvram_kernels.cu` : `lancer_pdl()` (cudaLaunchKernelEx + attribut programmatique, repli <<<>>> compté et nommé),
  `pdl_attendre()` (`griddepcontrol.wait`, sm_90+, no-op sans arête) ; `int8_gemv_kernel` : premier tour de poids/échelles/zéros
  émis AVANT l'attente, x/res/norme après ; `pdl_mode == 1` (ACVRAM_PDL_CASSANT, test) repousse l'attente après la lecture de x ;
  lanceurs `int8_gemv` et `int8_gemv_norme` par `lancer_pdl` ; `pdl_jouet_ecrire` (producteur jouet), `pdl_etat`.
* `regime.py` : `Variable("PDL", "")` ; `kernels.pdl_texte()` → `pdl=off|inerte|on(n)|repli(raison:n)` sur la ligne moteur.
* Défaut inchangé : sans ACVRAM_PDL=1, même lancement <<<>>> qu'avant, même arithmétique (le préchargement du premier tour
  ne change que l'ORDRE des chargements, pas celui des sommes).

## Étape 1 — paire jouet sous graphe (`scratchpad/poste6-p141-24-09/jouet.py`, prise-jouet.sh)
200 paires (producteur jouet 3 µs puis GEMV QKV 5 120 × 2 048, N = 1) capturées, 20 rejeux, médiane par paire ; bras off / on
en deux processus ; aussi spin 10 µs. **Prédit** : on < off d'au moins **0,5 µs par paire** (montée du GEMV recouverte) ;
`pdl_etat` : lancements = 200 + chauffe, replis = 0. **FAUX** si |on − off| < 0,1 µs (arête ignorée par le pilote sous graphe →
« repli » à nommer, étape 3 annulée) ; alarme : replis > 0 → la capture a refusé l'attribut, dit tel quel.

## Étape 2 — équivalence au bit + bras cassant (`tests/test_pdl_gemv_int8.py`, même commit)
3 formes × 6 lots, eager et sous graphe, variante à norme : PDL = référence à `torch.equal` ; cassant ≠ référence sur ≥ 1 forme
à N = 1. Le test est rouge si le cassant rend la référence (contrôle inopérant) ou si un repli est compté.

## Étape 3 — cellule b=1 (`prise-b1.sh`, deux prises de 5 lots : A B B A A puis B B A A B)
`certifie-b12.py <bras> … 1`, CERT_PUR=1, Coder i8c, `-lgc 2700`, A = PDL off, B = ACVRAM_PDL=1 ; preuve : `pdl=on(n)` dans
`engine_regime` des lots B, `pdl=off` des lots A ; médianes de 5. Seuils (dossier § 3, inchangés) : **TENU si Δpas ≤ −0,10 ms/pas**
(≥ +3 % de t/s, pas b=1 ≈ 3,2 ms), **FAUX si Δpas > −0,05 ms** ; entre : partiel. Prédit −0,14 à −0,24 ms/pas, J/jeton
−4 à −7 % (même puissance, moins de temps). Ce qui me gênerait : étape 1 tenue et étape 3 fausse → les prédécesseurs réels
(MoE down, paged_attention) n'ont pas de queue à recouvrir sans `cudaTriggerProgrammaticLaunchCompletion` chez eux.
B (persistant) seulement si A tient et laisse ≥ 0,05 ms (ordre).

## Addendum 07 h 0x — étape 1 mesurée (prise 07:04, 7 s de carte, scratchpad/poste6-p141-24-09/jouet-*.json)
Paire jouet : off 8,349 µs / on 8,430 (spin 3) ; off 14,491 / on 14,537 (spin 10) — |on − off| < 0,1 µs → **FAUX au sens du
scellé**. Mais le témoin « GEMV seul, 200 lancements sous graphe » du même processus : **off 5,143 / on 4,798 µs (−0,35 µs, −7 %)**,
`pdl_etat` lancements 411, replis 0 : **l'arête programmatique EST honorée sous graphe** — la prémisse de la clause « FAUX ⇒
arête ignorée ⇒ étape 3 annulée » est contredite par son propre témoin. Pourquoi la paire ne gagne rien : les poids (10,5 Mo)
restent CHAUDS en L2 d'une paire à l'autre (rien à précharger depuis la DRAM) et le producteur à 1 bloc n'a pas de queue à
recouvrir ; le jouet ne reproduit pas le pas réel (poids froids, 1 280 blocs devant). Le seuil ne se rouvre pas : étape 1 = FAUX ;
la décision de jouer ou non l'étape 3 revient à chef (le scellé l'annulait sur une prémisse fausse). Étape 2 : le bras a cassé
sur (2 048 × 4 096, N = 8) — variante norme hors budget de mémoire partagée (65 664 o > 48 Kio, lanceur), défaut de l'AIDE du test,
pas du noyau ; corrigé (norme seulement si N·K·2 ≤ 32 Kio), rejoué.

## Addendum 07 h 1x — étape 1 rejouée avec déclenchement précoce dans le producteur jouet (3c6e7296), étape 2 verte
`griddepcontrol.launch_dependents` au début du producteur : paire off 8,357 → **on 6,518 µs (−1,84 µs, −22 %)** à spin 3 ;
14,470 → 12,657 (−1,81) à spin 10 ; GEMV seul 5,14 → 4,80 ; lancements 411, replis 0. Étape 1 TENUE sur le jouet corrigé (≥ 0,5 µs).
Étape 2 : `tests/test_pdl_gemv_int8.py` **3 passed** (au bit eager + graphe + norme sur 3 formes × 6 lots ; bras cassant ≠ référence).
Mécanisme établi : sans déclenchement précoce chez le PRIMAIRE, le dépendant ne monte qu'à la sortie des blocs du primaire →
gain = latence de lancement seule (−0,35 µs/GEMV) ; avec, la montée ET le préchargement recouvrent tout le primaire (−1,8 µs).
Conséquence pour l'étape 3 telle que scellée (A seul, prédécesseurs inchangés) : prédit ≈ −0,35 µs × 96 = **−0,03 ms/pas → FAUX
probable** ; le −0,14 à −0,24 prédit suppose A' = `pdl_declencher()` au début des prédécesseurs (attention paginée Triton :
`gdc_launch_dependents` existe en Triton 3.8 ; fin du MoE b=1) — code hors du feu actuel, décision à chef avant toute carte.

## Addendum A' (07 h 4x, AVANT la cellule) — déclenchement précoce chez les primaires, ordre de chef (option b)
* Code (branche poste6) : `pdl_declencher()` = `griddepcontrol.launch_dependents` au DÉBUT de `moe_reduce_kernel` (prédécesseur du
  GEMV q/k/v à norme fusionnée, moe.py:1436), de `rmsnorm_bf16_kernel` et `rmsnorm_bf16_warp_kernel` (norme finale, prédécesseur
  de la tête int8, model.py:190 → layers.py:597-599), et `gdc_launch_dependents()` (Triton 3.8) au début de `_partiel_reduit_kernel`
  et `_reduce_kernel` de `attn_paginee.py` (prédécesseur de o_proj à b=1 : PAGED_ATTN=triton, d=128, n_rep=8). Drapeau hôte
  (`pdl_drapeau_primaire()`, 0 hors ACVRAM_PDL=1) côté CUDA ; `PDL: tl.constexpr = False` côté Triton (le noyau du défaut est
  compilé sans la branche). Compteur `déclencheurs` (CUDA + Triton) sur la ligne moteur : `pdl=on(n, déclencheurs=k)`.
* **Invariant écrit et testé** (`tests/test_pdl_gemv_int8.py`) : tout noyau lancé avec l'attribut programmatique appelle
  `griddepcontrol.wait` avant de lire une sortie de son prédécesseur — (1) seuls `int8_gemv` / `int8_gemv_norme` empruntent
  l'attribut : lancements programmatiques == appels GEMV du bras (compte exact) ; (2) bras cassant (attente déplacée après la
  lecture de x) ≠ référence sur la variante nue ET la variante norme, derrière un producteur qui déclenche tôt ; (3) les quatre
  primaires rendent la même sortie au bit off/on. Un déclencheur précoce est sûr pour tout dépendant qui attend ; il serait
  faux pour un dépendant qui n'attend pas — d'où (1).
* Prédiction pour la cellule A' (inchangée par rapport au dossier § 3) : **−0,14 à −0,24 ms/pas**, seuil TENU ≤ −0,10, FAUX > −0,05.
  Le jouet a montré −1,8 µs par paire avec déclenchement précoce ; les trois prédécesseurs réels (moe_reduce ~2 µs, attention
  paginée ~10 µs, norme ~3 µs) ont une durée ≥ montée du GEMV : le recouvrement attendu par GEMV est ≈ 1,5-2 µs.
* Étape 3 = cellule A (PDL off) contre A' (ACVRAM_PDL=1 avec déclencheurs) : prise-b1.sh inchangé ; preuve `pdl=on(n, déclencheurs=k)`
  avec k ≈ 3 × 48 + 1 par pas dans `engine_regime` des lots B.

## Addendum 07 h 5x — prise 1 (07:42-07:47) NULLE pour deux raisons, instrument de rechange AVANT la remesure
(1) chef : de 07:31 à 07:50 un sous-agent du chef a chargé le processeur (pip torch CPU, pytest) — tout lot de cette fenêtre est
écarté ; (2) de toute façon 0 lot n'a rendu de chiffre : `certifie-b12.py … 1` en CERT_PUR=1 lève « lot variable pendant la fenêtre
pure : 0 != 1 » — à 312 t/s la séquence unique épuise le contexte (2 048) avant les 20 s de fenêtre ; les lots B disaient
`pdl=inerte` parce que la ligne de régime est imprimée AVANT la capture des graphes (aucun GEMV lancé encore), pas parce que
l'attribut manquait. **Instrument de rechange = celui des cellules b=1 publiées** (README ³ : `acvram serve` + `banc-llamacpp-16-09.py
decode`, BANC_SLOTS=1×5, 1 024 jetons, fenêtre ≥ 10 s, énergie nvml nette) : `prise-b1-http.sh`, ABBA par relance du serveur
(A : ACVRAM_PDL=0 ; B : ACVRAM_PDL=1), `-lgc 2700` posé pour la prise, preuve `pdl=on(n, déclencheurs=k)` dans le journal du
serveur après la capture. Seuils inchangés : Δpas = 1000/t/s(B) − 1000/t/s(A) ≤ −0,10 ms TENU, > −0,05 FAUX ; J/jeton net.
Le pas b=1 de référence est ≈ 3,2 ms (312,3 t/s) : −0,10 ms = +3,2 % de t/s.
