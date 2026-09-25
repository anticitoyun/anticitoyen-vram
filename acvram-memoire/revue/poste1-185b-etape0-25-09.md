# 185 b étape 0 — verdict (poste1, 25/09) : BN 32/16 à tranches inchangées AU BIT, mais −0,058 ms/pas seulement (prédit −0,4) → pas de code moteur

* instrument : `scratchpad/poste1-p185b-25-09/banc-bn-tranches.py` (lance `_etroit_reduit_kernel` servi à BN 32/16 avec les
  tranches de BN 64 ; chrono `banc-etroites-latence.chrono_froid`, poids `banc-etroites-occupation.poids_int8`), b = 8,
  L2 froid (copies ≥ 256 Mo en rotation, graphe), 60 rejeux, comparaison octet pour octet au servi (`gemm_etroit(compact=True)`)
* commit : a352b523 (poste1-185b = origin/main 7e5a4bd2 + scellé et banc)
* régime : carte 0 seule, -lgc 2700, verrou poste1-p185b-banc, 4 warps / 3 étages (défaut), compute-apps début = fin (4242 seul)
* scellé : `scratchpad/poste1-p185b-25-09/scelle.md` (a352b523, avant la mesure)
* mesuré : 4 formes × 3 largeurs ; au bit 8/8 variantes (écart max 0,0)
* verdict : au bit TENU ; gain FAUX sur l'ampleur (0,058 contre 0,4 prédit, sous le seuil de 0,15 pour passer au code) ;
  H ni validée (o/out −0,73 µs < 2) ni réfutée (BN 32 < BN 64 sur o/out)
* durée : prévu ≤ 5 min ; tenu < 3 min (fin 09:10:03)

| forme (N × K) | prog BN 64 | BN 64 servi µs | BN 32 µs | BN 16 µs | max/moy par SM 64 → 32 | coût par octet implicite BN 32 |
|---|---|---|---|---|---|---|
| o/out 5120 × 6144 | 400 | 26,01 | **25,28** | 35,51 | 1,275 → 1,062 | +17 % |
| down 5120 × 17408 | 400 | 66,28 | 66,09 | 96,04 | 1,275 → 1,062 | +20 % |
| qkv attn 14336 × 5120 | 448 | **50,77** | 62,44 | 84,14 | 1,138 → 1,138 | +23 % |
| gate‖up 34816 × 5120 | 544 | 145,58 | 144,48 | 184,67 | 1,25 → 1,094 | +13 % |

Lecture : le modèle max/moy prédit bien le SENS. À BN 32, là où le rapport ne bouge pas (qkv attn), on paie +23 %. Là
où il baisse de 17 %, on gagne 0 à 3 %. Une tuile de 32 colonnes coûte 13 à 23 % de plus par octet (moins d'octets en vol
par programme, comme BN 128 dans l'autre sens à la 57) : cette perte absorbe tout le rééquilibrage.

Fait acquis : **la sortie ne dépend pas de BN** à tranches égales (BN 16/32/64 au bit, 4 formes). La réduction interne de
`tl.dot` ne suit pas la largeur de tuile, et la découpe en N est donc libre au bit. Seules les tranches K fixent l'arithmétique.

Suite (étape 1, au bit, chaque levier seul) : (a) diagnostic direct de H : même K, N choisi pour 340 / 400 / 408 / 510
programmes, µs par octet contre max/moy (≤ 2 min, aucun code moteur) ; (b) épilogue : réduction du dernier arrivé
déroulée (même ordre de somme) ; (c) segments GDN : grille compacte des 864 programmes actifs (160 vides aujourd'hui),
les plus longs d'abord.
