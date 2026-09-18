# poste7 — Profil lu : prefill = trois scellés tenus, P0 part à sec, P1 attend le oui ; hôte b=1 réfuté (1,5-5 %, pas 11 %) — pas de chantier M ; le 0,4 ms était l'attention qui grandit avec le contexte + un `nvidia-smi` dans la boucle de mesure, à retirer de l'instrument avant tout re-tampon b=1 (18/09)

Entrée : poste3 c611bc7 `verdict-profil-prefill-b1-18-09`. (a) 197,0 ms GPU (9 844 j/s) : `gemm_groupe` 82,4 (41,8 %), `nvfp4_dequant` 50,9 (25,8 %), int8_dequant 11,2, flash 11,8, cutlass 19,4, colle MoE 8,3, elementwise 6,7 — experts 67,6 % (≥ 55 tenu), préparation 31,5 % (≥ 25 tenu), hors-experts 63,7 ms (≥ 60 tenu) ; B0 n'a rien absorbé (= 17/09 à 0,3 ms). (b) hôte b=1 = mur − GPU : 0,047 ms (ctx 600), 0,164 (ctx 1 300) = 1,5-5 % ; le 0,4 ms pur→rondes = attention paginée (+0,21 ms de ctx 300 à 2 000) + `nvidia-smi` appelé par `certifie` dans la boucle (27 ms / 200 pas = 0,12 ms/pas) + prefill amorti (0,05).

## 1. Prefill : les décisions écrites s'appliquent telles quelles

* (i) `nvfp4_dequant` 50,9 ≥ 40 → **P1 déclenchée**, en suspens sans carte jusqu'au oui (`poste7-p1-porte-marlin-18-09`). Le profil confirme la soustraction de poste4 : 197,0 − 50,9 − (82,4 − 54,3) = 118 ms GPU ⇒ ~15 700 j/s en situ à X = 137, cohérent.
* (ii) int8_dequant + colle + elementwise = 26,2 ≥ 25 → **P0 part maintenant, à sec, poste4** (P1 attend, elle est libre) : GEMM int8 sans déquantification par appel sur q/k/v/o, activations A8 par jeton, régime `ACVRAM_PREFILL_INT8=a8` dans `regime_ligne()` ; colle MoE fusionnée. Scellé inchangé : ≥ **11 000 j/s** (< 11 000 faux) ; porte PPL privé ≤ 1,020 (prédiction ≤ 1,017). Marge de déclenchement 1,2 ms : je le note, la règle s'applique au chiffre, pas à la marge.

## 2. b=1 : ma prémisse est réfutée, et le poste est GPU

« ≥ 60 % des 0,4 ms dans un poste hôte » : faux — l'hôte pèse 1,5-5 %. Réfuté sur moi (carnet) : j'ai attribué à l'hôte un écart pur→rondes sans regarder ce que les rondes changent au GPU (le contexte croît : +0,21 ms d'attention paginée, poste E) ni ce que l'instrument ajoute (0,12 ms de `nvidia-smi`). **Pas de chantier M.** b=1 reste derrière llama.cpp (≈ 3,36 vs 2,93 ms) pour des raisons GPU : attention paginée à contexte long et colle (glue 0,43 + norm/act 0,34 + elementwise 0,41 au profil du 17/09). C'est le poste **après** P0/P1, pas maintenant ; il s'ouvrira sur le top 5 par poste à ctx 1 300 de ce profil, pas sur une nouvelle conjecture.

## 3. Instrument : `nvidia-smi` dans la boucle de `certifie` — REGLES § 4, « instrument corrélé à la variable »

0,12 ms/pas = 3,5 % à b=1 (3,48 → ≈ 3,36 ms, 287 → ≈ 298 t/s), 0,4 % à b=12 (négligeable). Ordre poste1, à sec, 30 min, un commit : `certifie` lit la puissance par NVML en processus (`pynvml`, déjà l'instrument d'`energie.py`), plus jamais par sous-processus dans la fenêtre ; garde : `certifie` refuse la cellule si un `nvidia-smi` a été lancé dans la fenêtre (compte des appels, comme la troncature). Contrôle qui peut rendre faux (poste3, 2 × 20 s, ABAB) : Coder b=1 ancien/nouveau instrument, écart attendu **0,10-0,14 ms/pas** ; < 0,06 → la cause du 0,12 n'était pas le smi, on cherche avant d'éditer un chiffre.
Cellules : **étiqueter, pas effacer** (REGLES § 4). poste3 dit dans le verdict quels bras du comparatif ont tourné sous ce `certifie` (acvram seul ? llama.cpp/EXL3 aussi ?) : le biais se corrige uniformément ou pas du tout. Coder b=1 acvram re-tamponné après le contrôle (une ligne, note datée) ; b=12 inchangé.

## Ordre

* poste4 : P0 à sec (§ 1 ii), un commit par fusion, test d'équivalence dans chaque commit ; P1 reste gelé.
* poste1 : § 3 instrument (NVML en processus + garde), à sec.
* poste3 : liste des bras mesurés sous `certifie`+smi (5 min, à sec) ; contrôle ABAB § 3 quand poste1 livre (carte 5 min) ; puis re-tampon Coder b=1.
* chef : ETAT — (a) tenus, P0 lancé, P1 en suspens ; M retiré ; ligne b=1 « surestimée de 0,12 ms/pas par l'instrument, correction en cours » ; b=1 GPU = poste après P0/P1.

## 4. Harnais différents (poste3, 18/09) : deux biais de sens opposé — le comparatif se remesure au harnais des concurrents, pas au nôtre corrigé

Concurrents b=1 : `banc-llamacpp-16-09.py` / `banc-4moteurs.py` (HTTP + SSE + `energie.py`, aucun `nvidia-smi` dans la fenêtre, `:445-450`). acvram b=1 (287,1) : `certifie` rondes, en processus — porte +0,135 ms/pas de smi (−3,9 %) **et ne porte pas le coût HTTP/SSE** que les concurrents portent. Corriger le smi seul rendrait une cellule biaisée dans l'autre sens. Règle (REGLES § 3, à écrire) : **une cellule du comparatif se mesure avec le harnais des concurrents** ; `certifie` reste l'instrument des cellules moteur (verdicts, INDEX), pas du comparatif.
Ordre poste3, après l'ABAB § 3 (mécanisme smi confirmé) : acvram Coder **b=1 et b=12 sous `banc-4moteurs.py`** contre `acvram serve` (même chaîne HTTP/SSE, `energie.py`, 2 × 20 s chacun, régime en tête). Prédictions écrites : b=1 **275-295 t/s** (3,36 ms + 0,1-0,3 ms de HTTP par jeton : les deux biais se compensent en partie), b=12 **1 150-1 200 bridé** (HTTP amorti sur 12 flux). Issue qui me gênerait : b=1 < 270 — alors le chemin HTTP d'`acvram serve` coûte plus que celui des concurrents et c'est un poste moteur (§ 2, b=1 GPU + serveur). Le comparatif reprend ces deux chiffres avec note datée ; 287,1 et 1 198 restent lisibles, étiquetés « certifie, en processus ». Rien n'est édité avant la mesure.
