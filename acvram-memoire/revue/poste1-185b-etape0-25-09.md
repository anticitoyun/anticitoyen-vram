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

## Étape 1 — à sec (gel du 25/09 : purge GitLab, aucun code neuf) ; à prendre après la purge, dans cet ordre
**(a) Test direct de H, sans code moteur (banc, ≤ 2 min).** Même K = 6144 (ng = 48), `decouper_k` servi, N choisi :
| N | tuiles × tranches = prog | gpt | max prog/SM | octets int8 max par SM | octets totaux |
|---|---|---|---|---|---|
| 2560 | 40 × 8 = 320 | 6 | 2 | 98 304 | 15,7 Mo |
| 5120 (servi) | 80 × 5 = 400 | 10 | 3 | 245 760 | 31,5 Mo |
| 5440 | 85 × 4 = 340 | 12 | 2 | 196 608 | 33,4 Mo |
| 6400 | 100 × 4 = 400 | 12 | 3 | 294 912 | 39,3 Mo |
Couple décisif : 5440 contre 6400, même gpt. Par les octets seuls, t(5440)/t(6400) = 0,85 ; si H vaut (charge du SM le
plus chargé), 0,67. **H validée si le rapport est ≤ 0,75, réfutée s'il est ≥ 0,82.** Contrôle : 5120 contre 5440
(servi contre 2 prog/SM exacts), où H prédit 5440 plus RAPIDE malgré 6 % d'octets en plus.
**(b) Épilogue déroulé (au bit).** La réduction du dernier arrivé lit ses `tranches` partiels dans une boucle
dynamique (`gemm_etroit.py`, `_etroit_reduit_kernel`, `for t in range(0, tranches)`, lectures `.cg`), probablement
une latence L2 par itération. Variante : `tranches` en constexpr, lectures émises d'un bloc, somme dans le même ordre
0..T−1. Prédit : −0,3 à −0,5 µs par tranche au-delà de la première, soit ≈ −0,15 ms/pas (o/out 64 × 5 tranches,
segments 48 × 4). Sondes à mesurer d'abord : noyau sans épilogue (store seul) et noyau sans lecture (acc nul).
**(c) Segments GDN (au bit).** Grille 256 × 4 = 1 024 programmes, dont 160 vides : les tuiles qkv ont 3 tranches de
14 groupes, les tuiles gate 4 tranches de 10. Grille compacte des 864 actifs, programmes de 14 groupes d'abord
(ordonnancement du plus long au plus court) : même arithmétique par programme, même réduction. Prédit −3 à −5 µs
par appel × 48 ≈ −0,15 à −0,25 ms/pas.
Chaque levier sera seul dans son commit, avec son test au bit et son ABBA servi (ordre de chef).

**Addendum 25/09 (croisement avec poste3, `revue/poste3-piece185-etroit-qkvo-debit-25-09.md`, poste3-185 62e6b418)** : son
banc de débit confirme que BN 32 à tranches de BN 64 ne retrouve pas le débit de BN 64 (gdn_qkv_gate −10 à −13 %,
q/o/gdn_out neutres, k/v +2 à +3 % à M = 8 seulement). **Levier BN clos.** L'étape 1 garde (a) comme diagnostic, puis (b) et (c).
