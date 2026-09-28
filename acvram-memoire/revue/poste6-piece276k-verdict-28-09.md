# Pièce 276 k — préfill en file dans le pas : FAUX (TTFT moyen +6 %, mur +12 % contre J) ; un forward par séquence coûte 2,9 ms de plus que sa part du forward packé — verdict

poste6, 28/09/2026 07 h 1x. Scellé : `poste6-piece276k-a-sec-28-09.md` (avant le code, inchangé). Code : origin/poste6-276i c86dbc6ca
(mesuré 3d48a2475, même code moteur). Sous carte.sh (ACVRAM_NOM=poste6-276k, 07:07:19 → 07:10:26, 187 s), arbre importé contrôlé,
charge hôte par cellule, échelle **M G J K K J G M** : M = main 2bede2bca, G = eager (TOUR_GRAPHE=0 PREFILL_FILE=0), J = graphe
(276 j), K = graphe + `ACVRAM_PREFILL_FILE=1`. Qwen3-VL-2B, une image 448×448 ; par bras : jetons b = 1 k0-k4 en premier (serveur
neuf), b = 12 × 7 tours, solo × 5, équivalences b = 4/12 ; pour J et K un second serveur neuf `--no-prefix-cache` pour l'identité
« premier jeton à b = 12 = b = 1 » (12 invites). Traces (sha256) : M1 d80ae1a3 G2 b8df2fdc J3 aed1aa1d K4 8cf45890 K5 bf215d42
J6 4695948e G7 00fb23af M8 8c04c86a ; identités J3 09d68619 K4 70400645 K5 efecc824 J6 42948dd3 ; analyse ffaa7b52.

## Essais 1 et 2 (avortés, à moi)
* Essai 1 (06:56-07:03, 67c323daf) : K déjà FAUX (TTFT moyen 186,8/186,2 contre J 174,4/176,0 ; mur 264,6/263,0 contre 233,7/236,4).
  Lecture : dans chaque forward, `batch.tokens.to(device)` et `block_tables[i].to(q.device)` (à chaque couche) copient depuis de la
  mémoire PAGINÉE — une telle copie synchronise le flux avant de partir : le forward k + 1 attendait la fin du forward k. Correctif :
  lot sur la carte par mémoire épinglée non bloquante (`_lot_sans_attente`).
* Essai 2 (07:06, fd6ccfa85) : correctif INERTE — `getattr(self.model, "device", cpu)` : `Model` n'a pas de `.device`, le chemin
  épinglé ne s'appliquait jamais (ni la copie épinglée des jetons de l'essai 1). Arrêté à M1, corrigé (device des embeddings), test.
* Essai 3 (07:07, 3d48a2475) : le correctif s'applique ; **résultat inchangé**. Ce n'était donc pas (seulement) cela.

## 1. Identité : TENUE, mais le témoin ne rend pas faux
* Jetons b = 1 : K4, K5 = M1 **5/5** (et G, J, M8 5/5).
* Premier jeton à b = 12 = premier jeton à b = 1 (serveur `--no-prefix-cache`) : **K4 12/12, K5 12/12** — mais aussi **J3 12/12,
  J6 12/12** : le forward packé rend ici les mêmes premiers jetons que les forwards seuls (la dépendance en M des GEMM, 276 i, n'a
  pas fait basculer un argmax sur ces 12 invites). Le critère tient pour K ; il ne distingue pas K de J sur ce corpus.
* b = 4/12 : publiés, inapplicables.

## 2. Tableau (b = 12, 84 requêtes par bras ; ms — côté client ; « pas » de K faux : les sorties émises dans le pas ne sont plus
   attribuables par la trace, 79-80 « incomplets »)
| bras | TTFT moyen | p50 | p95 | mur max (méd) | solo |
|---|---|---|---|---|---|
| M1 main | 176,3 | 178 | 266 | 290,8 (264,2) | 32,3 |
| G2 eager | 177,1 | 193 | 264 | 287,4 (263,8) | 32,4 |
| J3 graphe | 174,4 | 189 | 235 | 238,5 (235,4) | 31,5 |
| **K4 file** | **185,0** | 190 | 266 | **272,4 (259,8)** | 28,0 |
| **K5 file** | **185,2** | 186 | 265 | **274,2 (264,5)** | 29,4 |
| J6 graphe | 178,0 | 206 | 237 | 239,4 (235,5) | 29,4 |
| G7 eager | 179,1 | 183 | 263 | 293,3 (264,6) | 32,3 |
| M8 main | 177,6 | 178 | 262 | 285,1 (263,3) | 32,2 |

## 3. Scellé, point par point
* **TTFT moyen K ≤ 0,90 J** : 185,1/176,2 = **1,05 : FAUX** (prédit 145-155). p50 ≤ 0,90 J : 188/197 = 0,95 : FAUX. p95 +13 %.
* **Mur K ≤ 1,03 J** : 262,2/235,5 = **1,11 : FAUX** — le seuil « ≥ 1,10 = le recouvrement a échoué » est atteint : la pièce ne vaut
  rien, comme annoncé au scellé (issue n° 1, celle qui me gênait, en pire : le TTFT moyen ne baisse même pas).
* Solo : −2 à −3 ms (n = 1 : chemin groupé inchangé ; le solo reflète la tour graphe, comme en 276 j).
* Si l'hypothèse était fausse (K = J ± 3 %) : non — K est PIRE que J, de 5 % (moyenne) et 11 % (mur).

## 4. Où vont les millisecondes (traces d'identité, sans tour concurrente : 12 invites de ≈ 370 jetons)
* Packé (J) : 11 séquences en 3 pas = **10,0 ms/séquence** ; en file (K) : 11 séquences en 2 pas de 4 et 7 forwards = **12,8-13,1
  ms/séquence** (+2,9 ms, +29 %). Une requête SEULE (cellule b = 1, chemin normal, synchronisation à chaque forward) : **10,6-11,0
  ms**. Un forward en file coûte donc 2 ms de plus qu'un forward seul de même forme, et 2,9 de plus que sa part du packé.
* Sur 12 séquences : +35 ms par tour, exactement la perte de mur (+27 ms) ; l'attente des voisines dans le groupe (≈ −25 ms sur la
  moyenne, scellé) est mangée par ce surcoût → moyenne +9 ms.
* Causes candidates, NON départagées (3, la plus probable d'abord) : (1) allocations épinglées par forward (`pin_memory()` sur 6
  tenseurs de tailles variables, `cudaHostAlloc` quand l'allocateur hôte n'a pas de bloc : ~ms chacune) ; (2) une synchronisation
  restante dans le forward (le lancement de k + 1 n'est pas devant la carte) ; (3) efficacité GEMM à M ≈ 370-650 (mais la requête
  seule, même M, coûte 10,7 : cette cause ne fait pas les 2 ms).
* nsys (prise 07:10-07:11, `-t cuda`, bras J et K) : la trace s'arrête à 5,6 s, avant la première requête (chargement, captures
  des godets : 150 k noyaux) — la phase servie n'y est pas ; non poursuivi (2 rapports de 17 Mo sur disque, `nsys-276k-{J,K}.nsys-rep`,
  sha non calculés). Un profil en process (événements CUDA autour de chaque forward) est l'instrument juste, pas fait.

## 5. Décision
**FAUX** : `ACVRAM_PREFILL_FILE` reste **opt-in (défaut 0)**, documenté comme plus lent sur ce modèle ; `ACVRAM_TOUR_GRAPHE`
reste opt-in (décision chef, 276 j). Ce que la pièce établit : à b = 12 images sur Qwen3-VL-2B, le préfill packé est déjà au
plus serré (10 ms/séquence, linéaire) ; défaire le groupe pour servir le premier jeton dans l'ordre coûte plus (2,9 ms/séquence)
qu'il ne rend (≈ 2 ms/requête de moyenne). Deux suites possibles, à chef : (a) **sous-groupes plafonnés** (n ≤ 3 par forward :
le 7-groupe devient 3 + 3 + 1) — prédiction : moyenne −8 %, mur +4 % (le surcoût par forward tombe de 12 à 4 forwards) ; (b)
trouver les 2 ms du forward en file (profil en process) — si c'est (1), un tampon épinglé réutilisé les efface et la file
redevient ≈ packé + 0,9 ms/séquence, ce qui rendrait ≈ −15 ms de moyenne pour +11 de mur : encore en dessous de l'ordre
« sans perdre le temps du lot ». Mon avis : (a) seulement si le TTFT p50 vaut 4 % de mur ; sinon clore et garder J opt-in.
