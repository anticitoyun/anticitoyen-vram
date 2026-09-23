# Prédiction SmoothQuant + A4 (balayage α) et isolation down_proj — avant mesure

Manon, 13/09/2026. Suite du verdict A4 nu
(`revue/verdict-a4-fakequant-llama2-7b.md` : PPL 5,7550, +2,58 %, seuil
dépassé). Hypothèse de Jérôme : un lissage SmoothQuant (calibration
statique, comme les poids) pourrait combler l'écart avec la littérature.

## Ce qui est livré, à sec

* `acvram/quant/smoothquant.py` : `echelle_smoothquant` (s = max|X|^α /
  max|W|^(1-α) par canal d'entrée) et `replier_smoothquant` (déquantifie,
  applique s aux colonnes, requantifie NVFP4 bloc 16). Identité algébrique
  du lissage vérifiée exactement en pleine précision
  (`tests/test_smoothquant.py`, 8 tests) ; erreur après repliement+requant
  <50 % (contrôle de non-régression grossier, pas une mesure fine).
* `outils/hooks_activations_a4.py` : `installer_hooks_smoothquant` (replie
  le poids EN MÉMOIRE, jamais sur disque, hooke l'activation avec la même
  échelle) et `installer_hooks_genres` (restreint le fake-quant à un
  sous-ensemble de genres, pour isoler `down_proj`). Vérifiés sur le
  modèle jouet, CPU : couverture, non-inertie, alphas distincts donnent
  des sorties distinctes, genres absents des statistiques signalés (pas
  d'échelle inventée).
* `outils/smoothquant-a4-sweep.py` : campagne à 4 régimes, calibration
  512 séquences sur `Llama-2-7b-hf` (même mécanisme que l'AWQ du
  convertisseur), gardé derrière `--pour-de-vrai`.

## Régime nommé

Identique à `prediction-a4-fakequant-llama2-7b.md` : Llama-2-7B,
wiki-gptq.txt, ctx 2048, GPTQ. Étalon A16 = 5,6102. **Seuil scellé
inchangé : PPL ≤ 5,6663 (+1 %).**

## Prédiction chiffrée par régime

* **α=0,50** (symétrique) : **échec probable, marge modeste**. PPL
  prédite **5,68-5,75** — un peu mieux que le A4 nu (5,7550) mais sans
  franchir le seuil, car un partage égal ne cible pas spécifiquement les
  canaux difficiles.
* **α=0,65** (déjà la valeur par défaut publiée dans le papier SmoothQuant
  original pour des modèles de cette taille) : **le plus prometteur des
  trois, réussite possible mais pas certaine**. PPL prédite **5,63-5,70**
  — je place le bord du seuil dans cette fourchette, sans savoir de quel
  côté elle tombera.
* **α=0,80** (pousse la difficulté vers le poids) : **probablement pire
  qu'α=0,65**, car les poids NVFP4 sont déjà à leur limite de résolution
  (4,5 bits) — leur donner encore plus de dynamique à absorber devrait
  coûter plus qu'il ne fait gagner aux activations. PPL prédite **5,70-5,80**.
* **downproj-seul** (A4 sur down_proj uniquement, reste en A16) : **doit
  récupérer la MAJORITÉ du coût du A4 nu**, si l'hypothèse Bridging Gap
  (outliers concentrés dans down_proj,
  `piste-hadamard-downproj-refutee.md`) est le bon moteur de la
  dégradation. Prédiction : écart entre **+0,3 % et +1,2 %** — nettement
  moins que le +2,58 % du A4 complet, mais pas forcément sous le seuil.

**Issue qui me gênerait** (règle 4) : que `downproj-seul` reproduise la
QUASI-TOTALITÉ de l'écart du A4 complet (+2,3 % ou plus) — cela voudrait
dire que down_proj domine totalement, ce qui est plausible d'après
Bridging Gap, mais rendrait alors étrange que SmoothQuant (qui traite les
7 genres également) n'améliore pas nettement les choses pour au moins un
α, puisque down_proj a justement le motif d'outliers que SmoothQuant est
censé traiter.

## Ce qui reste, sur carte

`python outils/smoothquant-a4-sweep.py --pour-de-vrai`, carte libre
(Jérôme). ~1 h par régime × 4 = ~4 h, ou sous-ensemble via `--regimes`.
