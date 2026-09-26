# Avis extérieur — déterminisme sous partage dynamique (question 1)

Source : GPT-5.6 via duck.ai (aucun compte), 10/09, avec recherche et sources citées
(arXiv FlashInfer, docs flashinfer.ai, vllm.ai, GitHub FlashAttention,
thinkingmachines.ai). Perplexity écarté : il exige un compte.

**Séparation affirmé / vérifié / à vérifier** — rien de ce qui suit n'est vérifié dans
le code des projets cités ; tout est à traiter comme piste.

## AFFIRMÉ, et cela confirme directement notre analyse

- **FlashInfer a délibérément évité l'agrégation atomique à la Stream-K parce qu'il
  voulait des sorties déterministes** (attribué à l'article). Son `plan()` côté CPU
  construit les tuiles, les assigne aux CTA par un coût équilibré **déterministe**, et
  produit une **carte de réduction** que le noyau de contraction consomme dans cet
  ordre. C'est de la planification dynamique, **pas du vol de travail côté GPU**.
- **`FP32` réduit l'erreur mais n'est pas la source du déterminisme.** La réponse le
  dit explicitement et refuse l'option (b) comme solution. `FP64` n'est utilisé nulle
  part.
- **vLLM `paged_attention_v2` a exactement notre architecture** : grille indexée
  `(tête, séquence, partition KV)`, chaque partition écrit sa sortie, son maximum et sa
  somme d'exponentielles, puis un noyau de réduction combine **par index de partition**.
  Pas de CTA accumulant par atomiques.
- **FlashAttention dense avant** : un CTA possède sa tuile de requête et parcourt K/V
  dans un ordre fixe — déterministe. Le drapeau `deterministic` ne concerne que la
  passe arrière.

## LE POINT QUI CONTREDIT CE QUE NOUS CROYONS — et il est neuf

**Le découpage adaptatif casse l'invariance au lot.** Trois affirmations concordantes :

- FlashInfer expose `fixed_split_size` : avec cette option les frontières de partition
  sont **fixées en pages** et la fusion est déterministe **et invariante au lot** ; sans
  elle, l'ordre de réduction dépend de la forme du lot.
- vLLM est « déterministe à configuration de noyau fixe, mais accepte des écarts de bits
  faibles quand la configuration change » — et le nombre de partitions **change avec la
  longueur de séquence, la forme du lot et les contraintes de graphe CUDA**.
- Le travail de déterminisme au-dessus de vLLM (thinkingmachines) utilise des splits KV
  **de taille fixe** et note que la stratégie adaptative par défaut **n'est pas
  invariante au lot**.

**Conséquence pour nous, à vérifier d'urgence : notre tranche adaptative — le gain de
+88/+104 % — est précisément une stratégie de découpage variable.** Si `C` dépend de la
longueur ou de la forme du lot, alors **deux requêtes identiques placées dans des lots
différents ne rendront pas la même sortie**. Ce n'est pas un défaut de justesse (les
deux sorties sont également valides) mais une perte d'**invariance au lot**, qui est une
propriété que les serveurs d'inférence sérieux annoncent et que nous n'avons jamais
regardée. Le circuit a arbitré la qualité de la tranche adaptative contre l'attention
dense ; il n'a pas arbitré son invariance au lot.

## CE QUE ÇA VALIDE DE NOTRE COTÉ

Notre architecture `partial` + `reduce` avec réduction **par index de tranche** est la
même que celle de vLLM, et elle est du bon côté : déterministe à configuration fixe. Ma
conclusion sur Stream-K — « la variante à compteur est exclue par le déterminisme, pas
par la légalité » — est exactement le choix qu'a fait FlashInfer, et pour la même
raison.

## À VÉRIFIER AVANT DE S'APPUYER DESSUS

1. lire l'article FlashInfer pour confirmer la phrase sur l'évitement de Stream-K ;
2. vérifier dans notre code si `C` dépend du lot ou seulement du contexte de chaque
   séquence — c'est ce qui décide si nous avons ou non l'invariance au lot ;
3. la réponse mentionne une « accumulation turnstile par sémaphores » décrite comme
   travail futur dans l'article : à retrouver, c'est une troisième voie que nous n'avons
   pas envisagée.

---

# VÉRIFIÉ DANS NOTRE CODE — la non-invariance au lot existe, et elle est bornée par les godets

Le point le plus actionnable de l'avis extérieur est vérifié, et le résultat est plus
nuancé que l'alerte : **nous sommes exactement dans le statut de vLLM par défaut.**

`acvram/engine/model.py:87` :

    n = bucket_blocks(max(t.shape[0] for t in self.block_tables))

**La table de blocs est dimensionnée sur le MAXIMUM du lot**, et `C = ceil(N × 16 /
PA_CHUNK)` en découle. Donc une requête courte placée dans un lot contenant une longue
reçoit un `C` plus grand, l'ordre de réduction change, et **la sortie change au dernier
bit**. La non-invariance au lot est donc réelle et démontrée par le code, pas seulement
affirmée par une IA.

**Mais `bucket_blocks` arrondit aux puissances de deux** (`kvcache.py:36`), depuis 8
blocs de 16 positions :

    godet     positions    C a PA_CHUNK=512
        8           128           1
       16           256           1
       32           512           1
       64          1024           2
      128          2048           4
      256          4096           8
      512          8192          16

**La non-invariance est donc par godet, pas continue** : deux lots dont la plus longue
séquence tombe dans le même godet donnent la même sortie au bit près. Et à
`PA_CHUNK = 512`, les trois premiers godets donnent tous `C = 1`, donc aucun découpage
et aucune sensibilité. C'est le même mécanisme que le `fixed_split_size` de FlashInfer,
obtenu par un autre chemin.

**Ce que les auteurs avaient déjà vu, et ce qu'ils n'avaient pas vu.** Les deux
commentaires sont explicites : les godets existent pour que *« le chemin eager et le
graphe CUDA rendent des sorties identiques au bit près »*. Le déterminisme **entre
chemins d'exécution** était donc pensé. L'invariance **au lot** ne l'était pas — elle
n'est nulle part nommée, et personne ne l'a mesurée.

**Ce qui reste à décider, et ce n'est pas à moi** : est-ce un défaut ? Non par rapport
à l'état de l'art — vLLM a exactement ce comportement, et la littérature du déterminisme
le corrige par des splits de taille fixe. C'est une **propriété non documentée** de
notre moteur, et le geste juste est de l'écrire plutôt que de la corriger : un serveur
qui annonce l'invariance au lot doit poser `PA_CHUNK ≥ taille du plus grand godet servi`,
ce qui la garantit au prix du découpage.

**Et cela touche la tranche adaptative de plein fouet** : elle fait varier `PA_CHUNK`
selon la longueur, donc elle **ajoute** une source de variation là où le godet en
limitait une. Le gain de +88/+104 % reste mesuré et entier ; ce qui n'a jamais été
regardé, c'est ce qu'il coûte en invariance. À arbitrer avant de la mettre sur `main`,
au même titre que la barrière de qualité.
