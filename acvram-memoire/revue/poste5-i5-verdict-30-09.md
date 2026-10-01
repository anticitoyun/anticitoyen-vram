# I5 — récurrence GDN fp32 réglée AU BIT : plafond, noyau `gdn_tuiles` au PTX identique à fla, scellé (poste5, 30/09)
* instrument : `scratchpad/poste5-i5-30-09/compare_ptx.py` (sortie `compare_ptx.txt`), `compile_fla.py`, à sec (Triton 3.8.0 → sm_120) ; banc de demain `outils/gpu/mesure/banc-gdn-recurrence.py`
* commit : poste5-leap (voir git log)
* régime : à sec, CUDA_VISIBLE_DEVICES="", aucune prise ; aucun pytest (campagne de poste2) : le test est écrit, pas lancé
* scellé : § 3, écrit AVANT la prise de demain matin
* mesuré : rien sur carte ; à sec, suites d'instructions flottantes de 12 compilations contre fla
* verdict : plafond au bit ≤ 0,25 ms/pas à b=12 (≈ 1 %) ; noyau `gdn_tuiles` J tuiles/programme, PTX = fla × J (VRAI) ; opt-in `ACVRAM_GDN_TUILES`
* durée : 0 (à sec)
## 1. Plafond (ce qu'un noyau au bit peut gagner)
Au bit, les octets ne bougent pas : 3,15 Mo lus par couche et séquence. Noyau servi (Qwen3.8) : b=8 23,1 µs/couche
(scratchpad/poste1-p202-25-09/familles.txt:14, 1,109 ms/48), b=12 30,2-31,6 (profils 17-18/09, 1,451-1,516 ms/48).
Plancher de lecture à 1,2-1,45 To/s (REGLES § 9) : b=8 17,4-21,0, b=12 26,0-31,4 µs. NInfer (état fp32 aussi,
recurrent.cu:56-57) : 18,7 µs à b=8 = 1,35 To/s. **Gain au bit maximal : b=8 ≤ 0,27 ms/pas, b=12 ≤ 0,25, b=1 ≤ 0,05.**
ERRATUM de poste5-leap-verdict-30-09 § 3, I5 : « ≥ 0,45 ms/pas à b=12 » est hors d'atteinte au bit (plafond 0,25) ;
LeapQuant n'est donc pas rendu inutile par I5, il reste le seul levier au-delà de ~1 %.
## 2. Ce que fixe fla, et ce qui casse le bit (lu dans le TTGIR et le PTX, compilés à sec)
Instrument vérifié d'abord : le PTX compilé hors carte est identique à celui du cache Triton servi (FLTHVEKL, 9ba36e8f).
Programme d'un warp, tuile [128, 8] (`#blocked1` sizePerThread [1, 1], threadsPerWarp [4, 8]) : 32 lignes de K dans
le fil puis papillons 16, 8 ; 95 registres → 21 programmes/SM, 2,6 vagues à b=12 ; 45 LDG par tuile.
LLVM contracte 64 voies FMA par tuile (dont 6 fma.f32x2) : cette décision dépend de la FORME du code, pas des opérations.
* K1 (q, k, portes hoistés, sans boucle T) : même TTGIR, mais 32 FMA contre 64 à J=1 → PAS au bit.
* v4 (deux tuiles, prologue partagé) : J=1 = fla, mais 12 voies FMA pour la 2e tuile → PAS au bit.
* **retenu (`acvram/engine/gdn_tuiles.py`)** : corps de fla recopié, boucle T gardée, répété J fois : suite ordonnée des
  instructions flottantes = fla × J pour F1 on/off × HV 48/32 × J 1, 2, 4 (12/12 VRAI). Coût : q, k, portes refaits par
  tuile et tuiles sérialisées dans le programme (SASS : LDG puis STG, tuile après tuile) ; J=4 : 128 registres.
## 3. Prédiction et seuils (scellés avant la prise ; banc : 48 états distincts par graphe, L2 froid, fla avant/après)
* au bit : `tests/test_gdn_tuiles_au_bit.py` — à sec (PTX = fla × J) et sur carte (`torch.equal` sortie + état, 3 pas,
  b ∈ {1, 2, 8, 12, 16}) ; contrôle de sensibilité : fla en BV=16 doit DIFFÉRER de BV=8, sinon la comparaison est aveugle.
* prédit, µs/couche, Qwen3.8 : fla b=12 30-31,5, b=8 22,5-23,5 ; tuiles2 b=12 28-31 (gain 0-2,5), tuiles4 28-33 ;
  b=1 tuiles plus lent de 1-4 µs (768 → 192 programmes) → jamais sous b=8 ; plancher lecture b=12 26-31,4 ;
  plancher copie en place 26-32 si les écritures restent dans le L2 le temps du noyau, 52-63 sinon.
* **GO (défaut par godet ≥ 8)** : au bit partout ET gain ≥ 2 µs/couche à b=12 (≥ 0,1 ms/pas) ET ≥ 2 σ, fla encadrant.
Issues : **T1** GO. **T2** au bit, gain < 2 µs (le plus probable) : la voie Triton est close ; alors, si
plancher_r ≤ fla − 4 µs à b=12, transcription CUDA à arrondis explicites (inline PTX des 223 instructions de fla,
4 warps/CTA, lignes de 512 o, prologue partagé), qui sert aussi de base au noyau INT8 de LeapQuant ; sinon I5 clos, fla
est au plancher. **T3** PTX identique mais bits différents sur carte → l'instrument à sec est troué, tout est rouvert.
**T4** tuiles plus lentes → abandon. La copie en place à 52-63 µs dirait que les écritures se paient DANS le noyau :
M2 du verdict leap (2,0-2,4 ms à b=12) deviendrait l'estimation basse de LeapQuant, pas la haute.
