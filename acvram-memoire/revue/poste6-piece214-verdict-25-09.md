# Verdict — pièce 214 : le GEMV des experts du Coder-30B à b=1 (62 % du pic, 203 poste 1) — le déficit est de la STRUCTURE (5 µs par noyau hors transfert), pas des chargements ; les regrouper (P5, au bit) ne rend rien à la forme servie ; le tensor à T=1 est 1,5 × plus lent (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p214-25-09/banc-gemv-214.py` (piles synthétiques 128 experts aux formes du Coder, gate·up K 2 048 × N 768,
  down 768 × 2 048, disposition Marlin par `preparer_pile` ; graphe de 16 appels à jeux d'experts tournants, médiane de 40 rejeux ;
  un processus par `ACVRAM_GEMV_SPLITK`), `temoin-bit.py` (8 sorties à graine fixe, sha256), `appliquer-p5.py` (patch/retrait),
  `chaine.sh` / `chaine-p5.sh`. Résultats `banc-S{1,0,2,4,8}.json`, `p5-banc-S{1,0}.json`, `p5-temoin.log`.
* **commit** : noyau d'avant 581a4745d (= origin/main 1b56f68e9 + scellé) ; P5 a45867695 ; extension du worktree recompilée sous verrou.
* **régime** : carte 0, -lgc 2700 (horloges 2 572-2 992 SM), llama-server 4242 (5,6 Go, tiers) présent début = fin, aucun autre processus.
* **scellé** : `scratchpad/poste6-p214-25-09/scelle.md` (581a4745d, avant les prises) — P1-P5.
* **mesuré** : gate·up G = 8 **12,91 µs** (S = 4 auto), down **8,10** (S = 2) ; L2 tiède 7,34 / 5,25 (0,57 / 0,65 de la froide) ; S forcé gate·up
  1 / 2 / 8 : 17,97 / 13,97 / 14,26 ; down S = 1 **7,34 < 8,10 servi** ; t(G ≥ 8) à S = 4 : **4,1 µs + 1,10 µs/expert (1,60 To/s)** ; down S = 1 :
  2,9 + 0,55 (1,62 To/s) ; chaîne GEMV b=1 21,6 µs contre chaîne tensor T = 1 **32,4** (Marlin gate seul 9,7 µs pour 7 Mo = 0,73 To/s) ;
  P5 : gate·up 12,97 (**+0,5 %**), down 8,12, S = 1 15,89 (−12 %), **AU BIT** (temoin-bit : 0 valeur différente sur 8 sorties, sha256 égal).
* **verdict** : P1 TENUE (4,1 + 1,10 : coût fixe dominant, marginal à 90 % du pic) ; P2 TENUE (0,57) mais mal lue : la chaîne d'allers-
  retours n'est pas celle des chargements ; **P3** S = 8 FAUSSE (14,3 > S = 4 : seconde vague) ; **P4 FAUSSE** (tensor T = 1 : 32,4 ≫ 21,6, levier
  mort) ; **P5 RÉFUTÉE à la forme servie** (+0,5 %, seuil −0,15 ms/pas non atteint) → **patch retiré, aucun code servi ne change**.
* **durée** : prévu 5 × ≤ 60 s + 1 200 s ; tenu banc 173 s (dont 160 s de compilation) + P5 164 s (dont 150 de compilation) ; files 465 s et 1 709 s (« carte obtenue apres » dans prise.log / prise-p5.log).

## 1. Où sont les 5 µs (par couche, G = 8 : gate·up 12,9 = 7,9 de transfert au pic + 5,0 ; down 8,1 = 3,95 + 4,1)
* Depuis la L2 (tiède), gate·up prend encore **7,3 µs** pour 14 Mo que la L2 sert en < 3 : ≥ 4,3 µs ne sont ni HBM ni L2 — rampe de
  lancement d'une vague unique de 384 blocs × 256 fils, remplissage de x en mémoire partagée puis barrière (:2225), épilogue split-K
  (dépôt, `__threadfence`, atomique, relecture volatile par le dernier bloc, :2300-2325), queue de la dernière colonne.
* Les chargements « deux tuiles en vol » (:2244) SONT sérialisés par nvcc (SASS 581a4745d : LDG 0x0cd0, PRMT sur le registre
  chargé, LDG 0x12d0, 0x1b30, 0x24d0) — et ça ne compte pas : les 384 blocs partent ensemble, la mémoire est saturée en agrégat
  pendant la phase de transfert ; P5 (barrière de registres, `MB_EN_VOL4`) ne gagne qu'à S = 1 (96 blocs, 4,5 warps/SM : −12 %),
  rien à S = 4. 64 registres après P5 (63 avant), même occupation.
* Points G ≤ 4 du balayage : 16 jeux × G × 1,77 Mo ≤ 113 Mo ≈ L2, donc tièdes — l'ajustement est fait sur G ≥ 8 seulement (défaut du
  banc, nommé ; sans effet sur les conclusions).
* Marlin MoE (tensor) à T = 1 : 9,7 µs par GEMM de 7 Mo — le pipeline cp.async ne vaut rien sans lignes à traiter ; à T = 8 (64 paires
  distinctes, synthétique) 37,4 µs = 1,51 To/s. Le 98 % de la 203 (30,5 distincts, 81 Mo) vient de l'amortissement d'un coût fixe
  d'environ 3 µs sur 4 × plus d'octets, pas d'une autre mécanique par octet.

## 2. Ce qui reste comme leviers (chiffrés, non joués)
1. **Structure entre les deux noyaux et entre les couches** : PDL (lancement dépendant programmatique) sous graphe, pour recouvrir
   rampe + remplissage de x du noyau suivant avec la queue du précédent — c'est le poste 2 de la 203 (0,43 ms/pas de trous à b=1),
   pas ce noyau ; gain plausible 1-2 µs par frontière × 2 par couche = **0,1-0,2 ms/pas**, au bit. À banc-er (194 § a).
2. **Down à S = 1 au lieu de S = 2** : −0,76 µs/couche = −0,037 ms/pas (1,2 % du pas b=1) — hors bit (ordre des sommes), pas au seuil.
3. **Un seul noyau persistant gate·up → down par paire** (compteurs par paire, 680 blocs résidents) : −1 lancement, −1 rampe, −1 épilogue
   ≈ 3-4 µs/couche = 0,15-0,2 ms/pas, au bit possible ; ≈ 2 jours ; à ne pas ouvrir avant le levier 1.
4. x lu en registres depuis la L2 au lieu du remplissage partagé + barrière : ≤ 1 µs/couche (0,05 ms/pas), au bit — sous le seuil seul.

## Suite (à chef)
Pièce close sans code servi : le GEMV b=1 n'a pas 0,3 ms à rendre par ses chargements ; les 9 µs/couche hors transfert sont de la
structure de lancement, à traiter avec les trous (PDL, 203 poste 2). Instruments et témoin au bit réutilisables pour tout noyau GEMV.
