# poste7 — avis extérieur « moteur C++/CUDA » : un point utile (résidence dynamique des experts, hors VRAM seulement), deux déjà couverts, trois sans objet pour nous (16/09)

Grille : ce que l'avis décrit est un moteur à écrire ; le nôtre existe et son goulot mesuré du jour est le MLA (×4 au duel, `poste7-duel-verdict`). Un point n'entre en file que s'il porte un chiffre mesurable sur un modèle qu'on sert.

| # | point de l'avis | statut chez nous | décision |
|---|---|---|---|
| 1 | double buffering RAM→VRAM, pas de retransfert par jeton | **couvert** : poids résidents en VRAM (NOMINAL 0/47 exilées) ; l'exil par expert (arène épinglée, `loader.py:301-315`) ne sert que hors VRAM, mesuré ×9 | rien tant que le duel se joue en VRAM ; le double buffering est le chantier (b) de `poste7-objectif` s'il rouvre |
| 2 | Expert Residency Manager (fréquents en VRAM, rares en RAM, éviction) | **absent** : placement figé au manifeste | **seul chantier retenu**, après MLA et seulement pour les modèles hors VRAM (GLM-4.5-Air, Kimi) ; gain attendu : exil ×9 → ×2-3 si ≥ 80 % des activations tombent sur ≤ 50 % des experts (à mesurer d'abord, à sec, sur les traces de routage d'une PPL : 20 min) ; **réfuté** si la fréquence est plate (< 60 % sur 50 %) → le point ne vaut rien sur nos MoE |
| 3 | graphes CUDA sur la partie statique, routage/KV/échantillonnage dynamiques | **couvert autrement, et mieux mesuré** : nos graphes capturent le pas entier par godet (b, longueur) avec routage groupé *dans* le graphe (`_tuiles` à grille fixe, bead runner), échantillonnage hors graphe (`sampler.py:79`) ; le rejeu est 98 % du pas | rien ; sortir le routage du graphe nous **coûterait** des lancements (c'est ce que le bead runner du 14/09 a supprimé) |
| 4 | quantification par composant + plusieurs versions du même expert selon la charge | **couvert** pour la première moitié (plancher SNR, promotions int8, routeur fp32 conditionnel `prediction-routeur-fp32`) ; multi-version : sans objet en VRAM (une version suffit), en RAM elle double le stockage pour un gain non chiffré | rien |
| 5 | référence lente par opérateur avec tolérance | **couvert** (`noyau == référence Python`, REGLES §7) ; la généralisation utile est celle du § 4 bis d'aujourd'hui : *l'arithmétique du noyau*, pas une tolérance | rien de neuf |
| 6 | KV paginé, quantifié, réutilisation de préfixes | paginé : **oui** (tables de blocs `paged_attn`) ; préfixes : **oui** (cache LRU, runner) ; **quantifié : non** — le latent MLA est bf16 `[t, 576]`, vLLM le tient en fp8 (moitié d'octets sur le seul tenseur que le noyau MLA lit) | **à faire dans le noyau MLA à une passe, pas avant** : cache latent en fp8 E4M3 par ligne = ÷2 des octets lus ; gain attendu ≤ 0,5 ms sur les ≤ 6 ms scellés (le noyau à une passe est déjà borné compute/latence, pas bande passante) ; **réfuté** si PPL 3 tranches sort de ± 0,004 → bf16 gardé. Ordre : après le scellé ≤ 6 ms, comme deuxième commit |

**Sources** : la doc TensorRT-LLM KV reuse et le blog CUDA Graphs décrivent ce qu'on a ; llama.cpp « améliorer le planificateur d'un moteur existant » est exactement la ligne tenue depuis le 14/09 (pas de réécriture).

## Ordre

* **chef** : `ETAT.md` +2 lignes : « résidence dynamique des experts (hors VRAM) — après MLA, mesure de fréquence à sec d'abord » ; « cache latent fp8 — 2ᵉ commit du noyau MLA ». Rien d'autre ne change ; l'avis est intégré par cette note (INDEX).
* **poste4** : rien de plus maintenant ; fp8 latent = commit 2 du chantier MLA, scellés ci-dessus.
* **poste1** (à son retour) : fréquence d'activation des experts sur les traces de routage d'une PPL GLM (à sec, 20 min) — seuil 80 % / 50 %.
