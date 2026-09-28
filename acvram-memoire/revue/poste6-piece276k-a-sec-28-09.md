# Pièce 276 k — préfill EN FILE dans le pas (premier jeton par séquence, coût fixe du pas payé une fois) : scellé à sec, AVANT le code

poste6, 28/09/2026 07 h 0x. Ordre chef (après la 276 j) : neutraliser le regroupement des préfills quand les tours sont courtes, pour
que le gain de la tour se voie sur le TTFT moyen et le p50 sans perdre le temps du lot ; même protocole.

## Ce que la 276 j a mesuré (traces J3/G2, ajustement sur 45/72 pas de préfill)
Durée d'un pas de préfill = **6,8-8,6 ms + 18-20 µs/jeton** (σ 6 ms) : le coût par jeton est LINÉAIRE (aucune économie à grouper 7
requêtes de 650 jetons : 83 ms = 7 × 12), la seule économie du groupe est le coût fixe du pas (≈ 8 ms). Avec la tour graphe (6,5 ms
par image), les requêtes sortent de préparation plus vite que le préfill ne les consomme (≈ 13 ms de calcul chacune) : une file se
forme et `_admit` prend TOUT ce qui attend → pas [1, 1, 3, 7] : les 7 dernières reçoivent leur premier jeton ensemble à la fin d'un pas
de 83 ms — la 4e attend les 6 autres. C'est ce qui rend le TTFT moyen égal (175 = 175) et le p50 pire (+17 ms) alors que la
préparation a gagné 47 ms par requête. Le fixe de 8 ms ne se paie pas par forward mais par PAS (admission, décodage, registre) : dans
un pas, plusieurs forwards en file gardent la carte occupée si l'hôte lance le suivant avant de rapatrier le jeton du précédent
(le recouvrement du pipeline de décodage, `_plain_decode_pipeline`).

## Mécanisme (`ACVRAM_PREFILL_FILE`, opt-in tant que non mesuré ; défaut si le scellé tient)
Dans le chemin groupé de `_step` (`new` de n ≥ 2 séquences, sans budget ni frontière), au lieu d'un forward packé : pour chaque
séquence dans l'ordre d'admission (FIFO) — `_build_batch([seq])`, forward, `_sample_only` (device), copie asynchrone des jetons vers
une mémoire épinglée + événement, PUIS lancement du forward suivant, PUIS attente de l'événement, `_consommer`, `_register_complete_blocks`,
et **émission immédiate** des sorties vers le serveur (`Engine.emettre`, posé par `AsyncEngine` : `_deliver` par
`call_soon_threadsafe`) — les sorties émises dans le pas ne sont pas rendues une seconde fois à la fin. Le pas de décodage qui suit
est inchangé. n = 1 : chemin groupé tel quel (même forward que main à b = 1).

## Critère d'identité (décide du défaut)
* **Premier jeton à b = 12 = premier jeton à b = 1 de la même invite, 12/12** (arbre K, serveur neuf pour chaque cellule ou
  `--no-prefix-cache`) : un préfill par séquence rend le premier jeton indépendant de la composition du lot (GEMM à M fixe, 276 i).
  Sur l'arbre J (groupé) ce contrôle rend « faux » (composition) : c'est le témoin que le critère peut échouer.
* Jetons b = 1 (24 jetons, 5 invites, cellule première) K = M **5/5**.
* Un écart → opt-in quelle que soit la vitesse.

## Prédiction et seuils (b = 12, Qwen3-VL-2B, une image 448×448, 7 tours ; J = 276 j graphe, 235 ms de mur, TTFT moyen 175, p50 193)
* **TTFT moyen K ≤ 0,90 J** (≤ 158 ms) = TENU ; prédit 145-155 : dans le groupe de 7, l'attente moyenne des voisines
  (≈ (n − 1)/2 × 12 ms ≈ 36 ms sur 7 requêtes) disparaît → ≈ −25 ms sur la moyenne du tour.
* **p50 K ≤ 0,90 J** (≤ 174) = TENU ; prédit 160-170.
* **Mur K ≤ 1,03 J** (≤ 242) = TENU (le travail carte est le même ; le fixe du pas est payé une fois par groupe) ; **mur ≥ 1,10 J**
  = le recouvrement hôte/carte a échoué (l'hôte attend chaque jeton avant de lancer le suivant : + ≈ 5-8 ms × forwards) = FAUX,
  la pièce ne vaut rien même si le TTFT moyen baisse.
* Contre G (eager) : TTFT moyen ≤ 0,90 G aussi (c'est la porte de la 276 j déplacée ici).
* Si l'hypothèse est fausse : TTFT moyen K = J ± 3 % — le premier jeton n'est pas retenu par le groupe mais par autre chose (fenêtre
  d'admission 5-20 ms, gabarit, sortie HTTP).
* Solo : ± 1 ms (n = 1 : chemin inchangé).

## Issues nommées, dont celle qui me gêne
1. **Ce qui me gênerait** : TTFT moyen −10 % mais mur +10 % (forwards en file mal recouverts) — le gain se paie sur la fin du lot,
   ce que chef refuse ; je le mesurerais et le dirais : opt-in.
2. La fenêtre d'admission (5 ms, plafond 20) reste : elle groupe AVANT le pas ; avec la file, un groupe ne coûte plus à ses membres
   que l'ordre — la fenêtre ne devrait plus gêner ; si elle gêne, ce sera lu dans « file » (submit → admission).
3. Séquences à `logprobs` : `_tops_si_demande` rapatrie en synchrone (chemin lent, inchangé) — hors mesure.
4. Le pas de décodage suivant voit toutes les séquences du groupe préfillées : identique au groupé.
5. Les sorties émises dans le pas passent par `_deliver` depuis le fil moteur : même chemin que la fin de pas, aucune serrure nouvelle.

## Échelle (arbre contrôlé, serveur neuf par bras, charge hôte par cellule) : M G J K K J G M
M = main 2bede2bca (travail/main-276j), G = cet arbre TOUR_GRAPHE=0 PREFILL_FILE=0, J = TOUR_GRAPHE=1 PREFILL_FILE=0 (276 j), K =
TOUR_GRAPHE=1 PREFILL_FILE=1. Par bras : jetons b = 1 k0-k4 (serveur neuf), b = 12 × 7 tours, solo × 5, identité « premier jeton b = 12
= b = 1 » (12 invites, b = 12 puis b = 1 chacune, sur un SECOND serveur neuf du même bras pour K et J), équivalences b = 4/12.
