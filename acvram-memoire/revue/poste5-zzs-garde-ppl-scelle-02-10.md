# zzs — garde de qualité avant de basculer le défaut : PPL de décodage à 8 192 + 512, SCELLÉE à sec (poste5, 02/10)

Ordre de chef, après `poste5-zzs-verdict-02-10.md` : la troncature causale entre en OPT-IN (`ACVRAM_MLA_CAUSAL=1`,
défaut 0), parce qu'elle change la sortie (E1 au bit faux, 2 réponses gloutonnes sur 6 divergentes). Le défaut ne bascule
qu'après cette garde (REGLES § 3 : un changement d'arithmétique du cœur MLA se juge à 8 192 + 512). **La mesure revient à
poste2** (REGLES § 3 : celle qui a écrit la méthode ne produit pas la mesure qui la couronne). Rien n'est mesuré ici.

## Protocole

* **Instrument** : `outils/gpu/mesure/ppl-decode-kv.py`, inchangé. Préfixe de 8 192 jetons en préfill, puis 512 jetons
  notés en forçage (`--prefixe 8192 --notes 512`). Cache de préfixe au défaut servi : sur un hybride, il coupe le préfill
  à un multiple de 256, ce qui fait passer le chemin à passé > 0, et c'est ce que la troncature change.
* **Modèle** : Kimi-Linear-35B-kda-nvfp4, b=1, carte 0. Le contexte de 8 704 tient en régime résident : le plan à sec
  calibré (`zzs-plan-a-sec.py`) donne 12 416 résident avec 0,5 à 1 Gio de marge. La ligne de régime de chaque bras le prouve
  (graphes=on, 0 couche exilée), sinon le bras est jeté.
* **Corpus** : les 9 tranches DISJOINTES de `scratchpad/corpus-prive/tranches-9/` (REGLES § 4 : au moins 9 tranches),
  identifiées par sha256 : 0a0baf4f5e9215ee, 616e09210ee55e68, 749e37bbac23a78d, f53f4d02d6b3185b, f723cc0ab55850df,
  150b236f446e595f, f426c5395d4f2b9e, b902f1470d8ea81a, 0b3fc50de6516d0e. Elles sont référencées par empreinte seulement.
* **Bras** : A = défaut (aucune `ACVRAM_MLA_CAUSAL`), B = `ACVRAM_MLA_CAUSAL=1`. Même arbre (HEAD figé, `ATTENDU`),
  un processus neuf par bras, ordre **A1 B1 B2 A2**. La variable est lue à l'import, ce qui interdit de basculer dans le
  même processus. Le régime de B doit porter `mla_causal=1(opt-in)`, celui de A non, sinon le bras est jeté (REGLES § 3 :
  prouver que la configuration a pris).
* **Durée prédite** : par bras, chargement d'environ 50 s, puis 9 × (préfill 8 k ≈ 1,0 à 1,4 s + 512 pas ≈ 1,6 s), soit
  ≈ 1,5 min. Avec les 4 bras, ≈ 6-8 min dans une prise de ≤ 15 min.

## Contrôles qui peuvent rendre faux (avant tout verdict)

1. **Témoin** : PPL de A1 = PPL de A2 à 10⁻⁴ sur chaque tranche, et de même B1 = B2. Sinon l'instrument n'est pas
   déterministe ici, et AUCUN verdict n'est rendu.
2. **Mêmes jetons notés** : le nombre de positions notées et les ids sont identiques entre A et B (la ligne `RESULTAT` de
   chaque tranche les donne).

## Prédiction (scellée)

Sur carte, la sortie du cœur diffère d'environ 1 ulp bf16 (E1(a) : max 3,9e-3 à 8 192, 2,0e-3 à 3 000 + 1 000), et le
module tient E2 (dans 2 × l'erreur d'arrondi du chemin complet contre fp64). C'est l'ordre de grandeur d'un réordonnancement
de somme, comme le C14 (≤ 8,3 ulp fp32 : géo −0,18 %, tranches −0,9 / −0,5 / +0,9 %). Prédit :

* **Δ géo des 9 tranches (B/A − 1) : dans ± 0,15 %** ;
* **|Δ| par tranche ≤ 1 %**, de signe quelconque. C'est la sensibilité d'une PPL de 512 jetons à une perturbation
  d'arrondi, pas un biais.

## Seuils de décision (fixés maintenant)

On calcule le Δ géo, son SE sur les 9 tranches (écart-type des log-rapports / √9) et le max |Δ| par tranche.

* **Bascule du défaut permise** (décision du chef) si **Δ géo ≤ +0,3 % ET Δ géo + 2 SE ≤ +0,6 % ET max |Δ| par tranche
  ≤ 2 %**. Ce sont la règle du C14 à 8 k (`verdict-tf32-8k-19-09`) et la règle de REGLES § 4 sur la géo du lot à 2 SE.
* **Reste en opt-in** si l'une des trois conditions tombe.

## Issues nommées

1. **Tenu** (l'issue prédite) : Δ géo dans ± 0,15 % et tranches ≤ 1 %. Le défaut peut basculer, sur décision du chef,
   avec la ligne DEFAUTS_PAR_VERSION de la version qui le porte.
2. **Indécidable** : Δ géo ≤ +0,3 % mais Δ géo + 2 SE > +0,6 %. Opt-in maintenu ; un second jeu de 9 tranches ou un modèle
   MLA de plus (GLM) peut trancher.
3. **La gênante : une tranche à |Δ| > 2 %**, comme le tf32 à 8 k (+6,2 % sur une tranche, géo −0,22 %). La troncature
   déplace alors la numérique du préfill long au-delà de l'arrondi, malgré E2 au niveau du module. FAUX pour la bascule,
   opt-in définitif, et l'on cherchera pourquoi un écart d'1 ulp au module coûte autant au bout de 8 192 jetons.
4. **B meilleur que A au-delà de −0,3 % en géo** : improbable pour un réordonnancement. Le premier suspect est
   l'instrument (régimes, jetons notés) ; aucune victoire ne se publie avant ce contrôle.
5. **Témoin A1 ≠ A2** : aucun verdict ; on cherche la source de non-déterminisme (cache de préfixe, autotune) avant de
   rejouer.

## Complément (hors garde, si la fenêtre le permet)

Les 2 invites divergentes du verdict (8 192 k=1 au pas 4, 12 288 k=102 au pas 0) rejouées sous `ACVRAM_SAMPLER_LENT=1`
avec `logprobs` : l'écart entre les deux premiers candidats au pas de divergence dit s'il s'agit d'une quasi-égalité
(écart < 1e-2 nat, l'issue attendue d'un ulp) ou non. Cela ne décide pas la bascule ; c'est une lecture.
