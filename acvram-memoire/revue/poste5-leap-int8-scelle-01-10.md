# État GDN INT8 par fenêtre (LeapQuant, opt-in `ACVRAM_ETAT_GDN=int8`) : livré à sec, scellé AVANT la carte (poste5, 01/10)
* instrument : à sec `tests/test_gdn_etat_int8.py` (référence torch, processeur) ; carte demain : mêmes tests (noyaux), `outils/gpu/mesure/banc-gdn-recurrence.py --bras int8`, `outils/gpu/mesure/garde-etat-gdn.py`
* commit : poste5-leap (voir git log)
* régime : à sec, CUDA_VISIBLE_DEVICES="" ; aucune prise
* scellé : § 3 ci-dessous ; garde qualité reprise de revue/poste5-leap-verdict-30-09.md § 3 (chef 30/09)
* mesuré : à sec seulement (référence, K = V = 32) : algèbre 2,6e-7 ; fenêtre P=16 2,1e-3 relatif sans dérive sur 256 pas ; requantification à chaque pas 8,3e-3 (4 × pire)
* verdict : livré à sec (référence + noyaux compilés sm_120 + branchement moteur) ; rien de revendiqué avant la carte
* durée : 0 (à sec)
## 1. Ce qui est livré (fichier:ligne dans le commit)
* `acvram/engine/gdn_etat_int8.py` : référence (`pas_reference`, `geler`, `etat_fp32`, témoin `pas_fp32`) et noyaux Triton
  `_pas_kernel` (grille N·HV × V/32, 79-96 registres) et `_bord_kernel` (persistant ≤ nb de SM, 255 registres, 0 débordement).
  INT8 symétrique par colonne de valeur, lissage des lignes de clé, R = 4 compensateurs fp16 (itération de puissance, 2
  itérations à CHAUD depuis la fenêtre précédente : 20 produits matrice-vecteur par gel au lieu de 68 à froid), tampon fp16
  (k par tête de clé, w et g par tête de valeur), P = 16 (`ACVRAM_ETAT_GDN_FENETRE`). Octets stockés : 1,20 Mo/couche/séq,
  54,9 Mio/séq sur Qwen3.8 (fp32 : 144) → −1,07 Gio à b=12.
* `gdn.py` : créneaux INT8 (`new_static`), `static_load` (gel d'un état fp32 de préfill), `static_export` (fp32 reconstruit
  + copie compressée : l'aller-retour d'un changement de lot est EXACT, pas une requantification), `_decode_int8`.
  `lot_etats.nouveau_static` prend des dtypes. Variables déclarées (regime.VARIABLES, cli VARIABLES_LUES) ; ligne de
  régime « etat=int8(P=16,R=4) ». Sans carte : refus nommé (noyaux Triton).
## 2. Tolérances nommées (écrites avant toute exécution, tests/test_gdn_etat_int8.py)
TOL_ALGEBRE 1e-5 (fenêtre sans quantification contre la récurrence exacte) · TOL_INT8 3e-2 par pas sur 256 pas et sans
dérive (derniers 64 ≤ 2 × pas 16-80) · contrôle qui peut rendre faux : la requantification par pas (P = 1) doit faire PIRE
· carte : TOL_NOYAU 1e-3 (noyaux contre référence, sortie et état, compteurs n égaux), couche entière int8 contre fp32 ≤
TOL_INT8 sur 40 pas, export → chargement → pas suivant AU BIT.
## 3. Prédiction et seuils (banc et tests : poste5 ; garde et service : poste2, REGLES § 3 « l'auteur ne couronne pas »)
* banc (µs/couche, Qwen3.8, même banc que I5, fla b=12 = 49,25) : int8 b=12 médiane (pas sans gel) 9-15, moyenne amortie
  12-24 (les 48 états gèlent ensemble 1 rejeu sur 16) ; b=8 8-18 (fla 32,5) ; **b=1 5-9 contre fla 4,37 : PLUS LENT**.
  **GO banc : moyenne b=12 ≤ 30 µs/couche** (gain ≥ 19 µs, ≥ 0,9 ms/pas) ET tests carte verts.
* garde (b=12, court 2 048 + 256 notés, long 16 384 + 256 notés, `--concat 2`, tranches-128 scellées) : prédit Δppl géo
  0 à +0,3 %, KL moyenne 1e-4 à 1e-3, max ≤ 0,05. **TENU si |Δppl| ≤ 2 SE (ou négatif) ET KL max ≤ 0,74, court ET long.**
* service (ABBA ≥ 2 paires, certifie-b12, même horloge) : gain b=12 1,0-2,4 ms/pas (référence M2 2,0-2,4 moins le calcul
  du noyau), soit +4 à +9 % t/s ; b=1 −0,5 à −1,7 %.
Issues : **L1** tout tenu et gain b=12 > 2 σ → proposition de défaut à chef, b=1 compris dans la balance. **L2** gain
sous le bruit à b ≤ 12 → abandonné (chef 30/09). **L3** garde FAUX court ou long → abandon ; P = 8 ou R = 8 seulement
s'ils sont scellés avant. **L4** médiane ≤ 15 mais moyenne > 30 → le gel coûte trop, retravaillé avant le service.
**L5** b=1 plus lent de > 1 % au service → reste opt-in (un format par serveur, pas par godet). **L6 (celle qui me
gêne)** court tenu, long FAUX : la fenêtre ne borne pas l'accumulation sur nos poids nvfp4 — l'argument de l'article
ne vaut pas ici.
## 4. Prise de demain (poste5, ≤ 10 min) puis poste2
pytest tests/test_gdn_etat_int8.py + banc `--bras int8 --lots 1,8,12,16` ; si GO banc : garde (poste2, 4 processus), ABBA.
