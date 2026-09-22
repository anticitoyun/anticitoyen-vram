# Veille : architectures hybrides et récurrence

Relevé du 8 septembre 2026. Champ : implémentations du delta rule et leurs
divergences entre moteurs, précision et quantification de l'état récurrent,
portes et décroissance, noyaux par blocs, sensibilité des couches récurrentes.

Chaque fiche dit **ce qui est mesuré**, **ce qui se transpose à notre matériel**,
**ce qui ne s'y transpose pas**, et **la mesure qui trancherait chez nous**.

---

## 1. DeltaLog — différer l'écriture de l'état récurrent (2608.15533)

*Deferred Materialization of Recurrent States for Linear Attention Decoding.*

**Mesuré.** Le réécriture de l'état récurrent en mémoire domine le trafic au
décodage. Les auteurs la diffèrent et la fusionnent par intervalles (8 pas pour
GDN, 4 pour KDA), avec vérification par compteurs matériels sur H200. Noyaux
Triton autonomes, plus une intégration dans un prototype vLLM en mode graphe.
Modèles : Qwen3.6-35B-A3B pour GDN, Kimi-Linear-48B pour KDA.

**Ne se transpose pas — et les auteurs l'écrivent eux-mêmes.** « Les charges à
**petits lots** ou à petits états récurrents offrent moins d'occasion pour cette
optimisation. » Notre cas de service est le lot de taille 1. Le gain annoncé
vient du régime grand lot, où le trafic d'état se cumule sur les séquences.
Ils notent aussi que le gain bout-en-bout est borné par la part du temps que
l'optimisation touche — perceptron, normalisations, ordonnancement et rejeu de
graphe restent hors de portée.

**Ce qui se transpose quand même, et c'est précieux.** Leur table de précision
décrit **exactement notre configuration** : état dense en FP32, métadonnées de
décroissance en FP32, activations en BF16, et « GDN et KDA gardent le journal de
mise à jour côté valeur en FP32 parce que leurs corrections de règle delta sont
plus sensibles aux annulations ». C'est une validation externe d'un choix que
nous avions fait sans référence — et qui explique pourquoi le chemin processeur
n'existe pas chez nous : ces noyaux sont écrits en Triton.

**La mesure qui trancherait chez nous.** Avant toute implémentation : quelle
part du temps de décodage part réellement dans l'écriture de l'état récurrent,
à lot 1 ? Si elle est marginale — ce que la limite des auteurs laisse attendre —
la piste est close pour notre usage. Un relevé de compteurs sur une couche GDN
suffit.

---

## 2. La divergence entre moteurs est un phénomène documenté (2605.19537, 2608.04714)

*The Silent Hyperparameter* et *What We Observe as LLM Behavior Can Be a
Side-effect of Inference Backend.*

**Mesuré.** Cinq moteurs — vLLM, SGLang, llama.cpp, LMDeploy, Ollama — comparés
à `transformers` comme référence, sur cinq modèles et quatre jeux d'épreuves.
L'écart entre le meilleur et le pire moteur atteint **16,3 points** sur GSM8K
pour DeepSeek R1 7B (78,1 % chez `transformers`, **61,7 % chez Ollama**), et
persiste en échantillonnage stochastique — donc il ne vient pas du décodage
glouton mais d'une différence au niveau des logits eux-mêmes.

**Se transpose entièrement, et valide la journée du 8 septembre.** Nous avons
mesuré un facteur 22 entre acvram et llama.cpp sur le même modèle, cherché
quatorze causes, et fini par trouver une convention perdue au transport. La
littérature dit que ce genre d'écart est **attendu** entre moteurs, qu'il est
rarement un bug isolé, et qu'il ne se voit pas sans référence externe.

**La leçon opérationnelle, que ces articles rendent générale.** Un chiffre de
qualité mesuré sur un seul moteur ne décrit pas le modèle : il décrit le couple
modèle-moteur. Toute comparaison de format publiée sans référence externe est
donc suspecte — y compris les nôtres avant aujourd'hui.

**La mesure qui trancherait chez nous.** Elle est déjà en place : l'étalon
llama.cpp sur `wiki.test.raw` au cadrage `n_ctx/2`. Ce qu'il faut y ajouter,
c'est de le **rejouer à chaque changement de moteur**, pas seulement de format.

---

## 3. Ce que la littérature ne dit pas, et que nous avons mesuré

**La sensibilité comparée des couches récurrentes.** Les travaux sur la
sensibilité par couche (2503.06518, 2601.11663) et sur la propagation d'erreur
de quantification (2504.09629) portent **tous sur des transformeurs à
propagation avant**. Aucun ne compare une couche récurrente à un perceptron de
même largeur. Notre mesure du 8 septembre — **une couche GDN est 2,3 fois plus
sensible qu'un perceptron de même largeur à une perturbation de poids, et ce
rapport est constant avec la profondeur** — ne semble pas avoir d'équivalent
publié.

**Une contradiction à instruire.** 2504.09629 affirme que les erreurs de
quantification « croissent approximativement **exponentiellement** avec la
profondeur ». Notre mesure donne **√N** : de 1 à 8 couches, l'écart passe de
1,87e−03 à 5,50e−03, soit ×2,93 pour ×8 de profondeur quand √8 vaut 2,83 — et
le témoin dense fait exactement pareil.

Les deux ne parlent probablement pas du même objet : ces travaux quantifient
**séquentiellement**, chaque couche absorbant l'erreur de la précédente, ce qui
peut composer multiplicativement ; notre essai perturbe chaque couche
**indépendamment** et empile avec résiduel, ce qui borne la croissance. Mais ce
n'est pas vérifié.

**La mesure qui trancherait.** Refaire notre empilement en propageant l'erreur
comme le fait une quantification séquentielle, et voir si la loi passe de √N à
exponentielle. Si oui, les deux résultats se réconcilient et le nôtre décrit un
régime différent — celui d'une quantification indépendante par tenseur, qui est
justement le nôtre. Si non, l'un des deux décrit mal son propre objet.

---

## 4. Pistes écartées pour notre usage

**Améliorations d'architecture** : Gated DeltaNet-2 (2605.22791, portes d'effacement
et d'écriture découplées), Kaczmarz Linear Attention (2605.08587, meilleure
perplexité que GDN à taille égale), Erase-then-Delta (2606.26560). Toutes
demandent un **réentraînement**. Nous servons des modèles existants : hors de
portée, et sans intérêt tant que nous n'entraînons pas.

---

## 5. Veille elargie hors arXiv — ce qu'elle donne et ce qu'elle ne donne pas

**MOSAIC n'est pas adapte a ce domaine.** L'agregateur installe pour la veille
multi-sources rend du hors-sujet sur nos requetes (batteries, psychiatrie,
pollen pour une recherche sur l'attention lineaire) et ne trouve pas les
identifiants arXiv — `similar 2504.09629` echoue. Il n'a pas non plus de
commande `gaps`. Il agrege OpenAlex et Europe PMC, qui couvrent mal les
preprints d'apprentissage automatique. **A ne pas utiliser pour ce champ** ;
les recherches ciblees par moteur restent superieures.

**Le chiffre qui ferme definitivement DeltaLog pour nous.** Leur gain
bout-en-bout, dans leur regime le plus favorable — lots de 64 a 256 sur H200 —
vaut **1,08 a 1,20 fois**. Ils invoquent explicitement la loi d'Amdahl : le
gain est borne par la part du temps passee dans la mise a jour de l'etat. A lot
1, cette part est plus petite encore. Huit a vingt pour cent dans le meilleur
cas d'autrui n'est pas un argument pour reecrire notre chemin de decodage.

**La validation d'implementations numeriques est une tradition etablie, et nous
l'avons reinventee aujourd'hui sans le savoir.** Le genie logiciel numerique
pratique depuis longtemps ce que nous avons bricole en une journee :

- **une reference en precision superieure** — la bibliotheque MPFR sert d'etalon
  a arrondi correct, et les implementations industrielles se valident en
  comparant leurs sorties aux siennes. C'est exactement ce que nous avons fait
  en mesurant en float64 contre le service en bfloat16 ;
- **une verification a deux couches** — d'abord une comparaison bit-a-bit
  contre un modele de reference, puis une validation applicative contre une
  reference FP32 avec erreur quadratique et metriques de stabilite. C'est
  precisement notre couple *contribution par couche* / *perplexite contre
  etalon*, decouvert par tatonnement le 8 septembre ;
- **des outils dedies** (Encapsulated Error, ACM Algorithm 1029) pour evaluer
  la precision atteinte plutot que de l'estimer.

**Ce qu'il faut en retenir pour le projet** : cette litterature a un vocabulaire
et des methodes pour ce que nous appelons « comparer a une reference
exterieure ». Avant de rebatir un banc de validation, il vaut la peine d'aller
y chercher les protocoles — c'est du temps gagne sur des problemes deja resolus
ailleurs.

**Et une resolution probable de la contradiction de la section 3.** L'analyse
numerique classique connait les deux regimes que nous opposions : une somme
d'erreurs **independantes** croit en racine de n (marche aleatoire), une borne
**deterministe** de pire cas croit en n. Notre mesure en racine de N decrit donc
le premier regime, et la croissance « exponentielle » annoncee par 2504.09629
decrit vraisemblablement une quantification sequentielle ou chaque couche
amplifie l'erreur recue — un troisieme regime, distinct des deux precedents.
**A verifier plutot qu'a supposer**, mais l'hypothese est maintenant nommee et
elle est classique.

## Sources

- [DeltaLog: Deferred Materialization of Recurrent States for Linear Attention Decoding](https://arxiv.org/html/2608.15533)
- [The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility](https://arxiv.org/html/2605.19537)
- [What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend](https://arxiv.org/html/2608.04714)
- [Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](https://arxiv.org/html/2504.09629v1)
- [Towards Superior Quantization Accuracy: A Layer-sensitive Approach](https://arxiv.org/html/2503.06518v1)
- [Activation Sensitivity as a Unifying Principle for Post-Training Quantization](https://arxiv.org/html/2601.11663v1)
- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/pdf/2605.22791)
- [Kaczmarz Linear Attention](https://arxiv.org/abs/2605.08587)
- [Erase-then-Delta Attention](https://arxiv.org/pdf/2606.26560)
- [KVBuffer: IO-aware Serving for Linear Attention](https://arxiv.org/pdf/2605.19049)
- [When Good Enough Is Optimal: Matrix Inversion Approximation for Quantized Gated DeltaNet](https://arxiv.org/pdf/2606.06034)
- [Enhancing linear attention with residual learning (OpenReview)](https://openreview.net/forum?id=dy6tnQMeyI)
- [A precision- and range-independent tool for testing floating-point arithmetic](https://dl.acm.org/doi/10.1145/382043.382404)
- [Algorithm 1029: Encapsulated Error, a Direct Approach to Evaluate Floating-Point Accuracy](https://dl.acm.org/doi/fullHtml/10.1145/3549205)
