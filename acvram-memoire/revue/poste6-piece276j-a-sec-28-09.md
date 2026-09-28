# Pièce 276 j — tour de vision à lancements réduits (graphe CUDA à forme fixe) : scellé à sec, AVANT le code

poste6, 28/09/2026. Ordre chef : ramener les 12 tours (≈ 105 ms en série à chaud, 276 h) à ≤ 40 ms ; critère d'identité
écrit avant de coder ; prédiction scellée (12 tours, mur, TTFT moyen à b = 12) ; échelle à arbre contrôlé ; jetons sur serveur
neuf (REGLES § 4, 276 i).

## Choix : graphe CUDA de la tour eager, pas torch.compile
Un graphe rejoue LES MÊMES noyaux (mêmes formes, même ordre) : au bit par construction, comme les graphes du moteur ;
inductor réécrit les noyaux (norme, GELU, softmax) et l'identité au bit n'est pas acquise. Le graphe se capture à la première
image d'une forme (pixel_values, image_grid_thw) et se rejoue ensuite : copie de l'entrée dans le tampon statique, `replay()`,
clones des sorties (traits + niveaux deepstack). Ce qui dépend de la grille seulement (indices/poids d'interpolation,
position_ids, cu_seqlens — transformers 5.17 les accepte précalculés par kwargs) est calculé UNE fois hors capture par les
mêmes fonctions HF, sur le même device : mêmes valeurs. Sous capture, aucune synchronisation : grille et cu_seqlens sur CPU
(seuls `.tolist()` les lisent, chemin sdpa). Capture en mode `thread_local` (le pas moteur continue sur l'autre fil), sous
le verrou lecteur de la 276 h (la capture du moteur, rédactrice, est exclue). Rejeu sérialisé (un verrou : tampons statiques,
réserve mémoire partagée) ; une capture qui échoue → forme marquée eager, journal, jamais un refus de requête.
Une tour dont les annexes ne sont pas une petite grille (Gemma 4 : `image_position_ids`, 2 520 valeurs) reste eager.

## Critère d'identité (décide du défaut)
* **Au bit** : graphe = eager sur 12 images 448×448 distinctes, deux passages (le même graphe rejoué sur 12 entrées
  différentes), traits ET niveaux deepstack : 12/12 et 12/12, sinon **opt-in** (`ACVRAM_TOUR_GRAPHE` défaut 0).
* **Jetons b = 1** (24 jetons, temperature 0, 5 invites) J = M **5/5**, cellule PREMIÈRE de chaque bras (serveur neuf, 276 i).
  Un écart sur cette cellule = opt-in, quelle que soit la vitesse.
* b = 4/12 : publiés, inapplicables (composition du lot, 276 c-h).

## Prédiction (écrite avant la mesure) et seuils
Ordre de grandeur : tour Qwen3-VL-2B = 24 blocs × 784 patches, h 1 024, i 4 096 ≈ 530 GFLOP/image ; 5090 à 2 700 MHz
bf16 : 2,5 ms au pic, 4-6 ms réalistes à M = 784. Les ≈ 8,7 ms/image de la série eager = noyaux + lancements ; le graphe
retire les lancements, pas les noyaux.
* **12 tours à chaud (identité, série)** : 105 → **55-75 ms** (−30 à −50 %). Seuils : ≤ 40 ms = **TENU** (cible chef, vrai
  seulement si les noyaux font ≤ 3,3 ms/image) ; ≤ 75 ms = **PARTIEL** (lancements retirés, borne = la carte) ; ≥ 95 ms =
  **FAUX** (le graphe n'apporte rien : la tour était bornée par la carte, pas par les lancements). Le temps noyaux par image
  (événements CUDA autour d'un rejeu) est publié : c'est lui qui dit si ≤ 40 est atteignable par ce levier.
* **Mur b = 12 (médiane des 7 tours)** : J ≤ **0,90 × G** (276 h : G ≈ 262 ms → ≤ 236) = TENU ; contre M (≈ 281) ≤ 0,85.
* **TTFT moyen b = 12** : J ≤ **0,90 × G** (G ≈ 176 → ≤ 158) = TENU.
* Si l'hypothèse est fausse (les tours ne sont pas sur le chemin critique, les préfills dominent) : **J = G ± 3 %** sur mur et
  TTFT alors que l'identité montre les 12 tours plus courtes.
* Solo (b = 1, 5 requêtes) : ± 1 ms (le graphe ne coûte rien à une image seule ; capture payée avant, cellule jetons).

## Issues nommées, dont celle qui me gêne
1. La capture (chauffe 2 passes + capture, ≈ 100-300 ms) est payée par la PREMIÈRE requête d'une forme : sur le bras J elle
   tombe dans la cellule jetons (première), pas dans les cellules de temps. En service réel, une image d'une forme nouvelle
   paie ce prix une fois — dit dans le régime.
2. **Ce qui me gênerait** : 12 tours ≤ 75 mais mur/TTFT = G ± 3 % — le levier réel est ailleurs (préfills groupés, 276 h § 3).
3. Le rejeu est sérialisé : `ACVRAM_TOUR_FLUX=2` (opt-in) ne recouvre plus deux tours de même forme ; documenté, hors défaut.
4. Mémoire : réserve privée du graphe (activations 784 × 4 096 bf16, quelques dizaines de Mo) par forme — bornée par le
   nombre de formes vues ; au-delà de `ACVRAM_TOUR_GRAPHE_MAX` formes (défaut 8), eager (jamais une capture sans fin).
5. `torch.cuda.graph.__enter__` synchronise le device et vide le cache de l'allocateur : un hoquet unique du pas en cours.

## Échelle (arbre contrôlé, serveur neuf par bras, charge hôte par cellule) : M G J J G M
M = main 2bede2bca (travail/main-276j), G = cet arbre `ACVRAM_TOUR_GRAPHE=0` (eager, régime 276 g), J = cet arbre graphe.
Par bras, dans l'ordre : jetons b = 1 k0-k4 (serveur neuf), b = 12 × 7 tours, b = 1 × 5, équivalences b = 4/12.
Identité (au bit + 12 tours + temps noyaux) en tête, sous carte.sh, arbre J.
