# Dossier à sec — pièce 160 : préfill GDN du lot en longueurs variables (poste6, 24/09)

* **instrument** : lecture de code (fla 0.5.2 du `.venv`, `acvram/engine/gdn.py`, `couches.py`) ; comptage aten à sec
  (`scratchpad/poste6-p160-24-09/compte-ops-gdn.py`, CUDA_VISIBLE_DEVICES vide, bouchon à la place de la règle delta)
* **commit** : 8cc64606 (poste6 = main 7525b14e) · **régime** : aucun (pas de carte) · **scellé** : § 4 ci-dessous
* **mesuré** : rien · **verdict** : dossier · **durée** : 40 min à sec

## 1. fla 0.5.2 accepte-t-il les longueurs variables ? OUI, sur les deux noyaux
* Règle delta : `fla/ops/gated_delta_rule/chunk.py:397` (`def chunk_gated_delta_rule`), paramètre `cu_seqlens` :410,
  docstring :434-436 « initial_state `[N, HV, K, V]` for N input sequences », :463-465 (cu_seqlens `[N+1]`, B = 1,
  T = Σ t). Seule restriction sur l'état : `cp_context` (:468-470, :539-541), que nous n'utilisons pas.
  Découpe par séquence : `ops/utils/index.py:156-164` (`prepare_chunk_indices` : chunks de 64 comptés PAR séquence) ;
  `ops/common/chunk_h.py:67-70` (IS_VARLEN : bos/eos et NT de la séquence, même boucle). Un état absent = zéros
  (`USE_INITIAL_STATE` charge h0, sinon `zeros`) : passer des zéros aux séquences neuves ne change pas un bit.
* Convolution : `fla/modules/conv/causal_conv1d.py:17-27` — `initial_state [N, D, W]`, `output_final_state`,
  `activation='silu'`, `cu_seqlens`, backend triton par défaut. Autre noyau que `F.conv1d` (cuDNN) : PAS au bit
  par construction.

## 2. Où part le temps du préfill par lot (à sec, 150 bis en main)
* Défaut 0,34 s/lot après 150 bis, pour 8 × 78 jetons, 48 couches GDN. Sur la carte, la boucle `forward_lot` →
  `_coeur` (gdn.py:188-196, :198-254) fait par séquence et par couche : ~28 lancements hors fla (comptés à sec :
  pad, conv, silu, 3 copies contiguës q/k/v, sigmoid, exp, neg, softplus, add, mul, 2 repeat_interleave, clone de
  l'état conv, g/beta contigus, norme gated 9 ops) + 8-10 noyaux fla (cumsum, l2norm, intra ×3, fwd_h, fwd_o)
  ≈ 38 par (séquence, couche) → 8 × 48 × 38 ≈ **14 600 lancements par passe**, tous à formes petites (t = 78).
* À 15-25 µs l'opération eager (Python + dispatcher, valeur implicite de la 150 bis : 1 680 lancements de projections
  → −74 ms), la boucle vaut 0,22-0,36 s par passe : c'est le premier poste du 0,34 s, devant les 16 attentions et 64 MLP
  dont le calcul GPU pour 624 jetons tient en ≈ 60-80 ms (34 TFLOP + 15 Go de poids). Le préfill du défaut est
  borné par l'hôte, pas par la carte. **À vérifier par un profil par familles** (§ 5, 1 prise ≤ 10 min).

## 3. Au bit contre la boucle ?
* Règle delta varlen : **au bit prédit** — mêmes noyaux, même BT = 64, mêmes chunks par séquence, seuls les offsets
  changent ; zéros ≡ état absent. Opérations par ligne (sigmoid, softplus, norme gated…) sur Σ t lignes : au bit
  (aucune réduction ne traverse une ligne). Convolution : au bit SEULEMENT si elle reste `F.conv1d` — un seul appel
  sur le flux concaténé [état_i ou 3 zéros | x_i] avec les colonnes de garde jetées ; même op, autre L, cuDNN peut
  choisir un autre algorithme → le test tranche ; repli : conv par séquence (8 lancements gardés). `causal_conv1d`
  de fla : hors bit, écarté pour le défaut.
* Chemin torch (`_refs`, transformers) sans cu_seqlens : la boucle reste pour le CPU et les tests sans carte ; le test
  au bit se joue sur la carte (pytest sous carte.sh, ≤ 2 min).

## 4. Prédiction et seuils (écrits avant toute mesure)
* Lancements par passe : 14 600 → ≈ 48 × 40 + 2 ≈ 1 900 (−87 %). **Préfill défaut 0,34 → 0,15-0,24 s/lot** (−0,10
  à −0,19) ; débit banc chat b=8 +2 à +5 % (le décodage domine). **Seuil : ≤ 0,26 s/lot** (−0,08, celui de la 150 bis).
* Au bit : B = boucle sur 100 % des logits (3 compositions de la 150 bis, dont états repris) ; sinon ÉCHEC nommé.
* Issues nommées : (a) −0,10 à −0,19 : tenu ; (b) −0,05 à −0,10 : la boucle n'est qu'une moitié, le reste est
  l'ordonnanceur/attention → profil § 5 dit où ; (c) < −0,05 : le coût n'est pas dans les lancements GDN (le profil
  montre la carte occupée) ; (d) conv hors bit → repli conv par séquence, −0,01 s de moins.
## 5. Ordre proposé (aucun code avant le feu)
1. Profil par familles sur 1 préfill 8 × 78 (défaut, LOT=1) : temps mur, Σ GPU, ops par famille (gdn-cœur / proj /
   attention / MLP / autre) — instrument à écrire, 1 prise ≤ 10 min. 2. Code varlen opt-in `ACVRAM_GDN_PREFILL_LOT=2`
   + test au bit sur carte. 3. ABBA banc chat 150 bis (kl-lot.py de poste5) : vitesse et KL.
