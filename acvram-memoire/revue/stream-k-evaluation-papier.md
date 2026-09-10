# Stream-K pour `paged_attn_partial` — évaluation sur le papier

Trois questions posées, plus une quatrième qui les précède et que la source n'a pas
vue. Rien ici n'est mesuré ; les chiffres viennent de comptes exacts ou de latences
documentées, et sont marqués comme tels.

## 0. La question préalable : Stream-K supprime les blocs vides, pas l'absence de travail

Le travail disponible au décodage d'**une** séquence est `HQ × ceil(ctx / tuile)` :

    ctx   128, tuile 512 :   40 tuiles pour 170 SM   MOINS DE TRAVAIL QUE DE SM
    ctx   128, tuile  64 :   80 tuiles              MOINS DE TRAVAIL QUE DE SM
    ctx  1024, tuile 512 :   80 tuiles              MOINS DE TRAVAIL QUE DE SM
    ctx  1024, tuile  64 :  640 tuiles              3,8 par SM
    ctx  4096, tuile  64 : 2560 tuiles             15,1 par SM

**À ctx 128, il n'existe que 80 unités de travail pour 170 SM, quel que soit le
mapping.** Aucun ordonnancement ne crée du travail : Stream-K ne remplirait pas la
carte dans ce régime, il ne ferait qu'éviter d'y laisser des blocs *vides*.

Et cela corrige une emphase que j'ai mise moi-même : j'ai présenté ctx 128 comme « le
régime décisif où `C` ne peut rien ». C'est vrai, mais **le gain moteur y est de 2,0 %**
(28,687 → 28,134 ms) contre **13,0 % à ctx 1024**, parce que l'attention n'y pèse que
2,1 % du pas. Le régime qui rapporte est le contexte long, pas le contexte frais. Ce
qui remplirait la carte à contexte court n'est pas le mapping mais **le lot** (plusieurs
séquences) ou la spéculation (`q_len > 1`) — deux dimensions déjà présentes dans la
grille `(BQ, HQ, C)` et aujourd'hui à 1.

## 1. Le coût des atomiques — et pourquoi ils ne sont pas nécessaires

Un compteur global sérialise ses accès en L2. À ~40 cycles par accès sérialisé et
2,1 GHz (ordre de grandeur, **non mesuré chez nous**) :

     170 atomiques    3,2 us
     680 atomiques   13,0 us
    2040 atomiques   38,9 us

Le noyau entier vaut 10 à 15 µs. **Un compteur atomique global est donc disqualifié par
le calcul à notre échelle** — il coûterait autant que le travail qu'il ordonnance.

Mais la formulation d'origine de Stream-K **n'a pas besoin de compteur** : le partage
est calculé par arithmétique — le bloc `i` traite les tuiles `[i·T/N, (i+1)·T/N)` — donc
**déterministe et sans atomique**. Les atomiques n'y servent qu'à combiner les tuiles à
cheval sur deux blocs, et nous avons déjà un second noyau pour ça : `paged_attn_reduce`.
**Notre architecture partial + reduce est donc déjà la bonne forme** ; ce qui changerait
est le mapping bloc → travail, pas la structure.

## 2. Les graphes CUDA : compatible, et même plus qu'aujourd'hui — sauf sur un point

La question était bloquante, la réponse est en trois parties.

- **Légalité** : un graphe capture des *lancements* — grille, blocs, arguments. Le corps
  du noyau, atomiques comprises, n'est pas concerné. Rien n'interdit les atomiques dans
  une région capturée.
- **Mieux qu'aujourd'hui** : notre grille est `(1, 40, C)` avec `C` fonction du
  contexte, donc **la grille change avec le contexte** — d'où les godets de longueur et
  `warm_graphs()`. Stream-K lance un nombre **constant** de blocs : un seul graphe
  suffirait au lieu d'un par godet. C'est un argument *pour*, que la source n'a pas vu.
- **Le point qui coince, et ce n'est pas la légalité** : si le partage dépend de l'ordre
  d'arrivée des blocs (variante à compteur), **le résultat numérique varie d'un rejeu à
  l'autre** — l'attention est un softmax en ligne avec correction de maximum, l'ordre
  d'accumulation change le dernier bit. Cela casserait `serie-determinisme.sh` et la
  reproductibilité de toutes nos mesures, et rendrait la barrière de qualité écrite ce
  matin inapplicable. **La variante déterministe sans compteur n'a pas ce défaut** — et
  c'est une raison de plus de la préférer, indépendante du coût.

Réponse nette : oui, compatible, **à condition d'exclure la variante à compteur** — et
elle était déjà exclue par son coût.

## 3. La mémoire de travail : elle cesserait de croître avec le contexte

Aujourd'hui `part[BQ, HQ, C, D]`, donc **linéaire en `C`**, donc linéaire en contexte à
tuile fixe. J'ai chiffré ce trafic plus haut : 4,0 Mo par pas à `chunk 512`, **128,8 Mo
à `chunk 16`** — plus que les 92,3 Mo de cache KV lus.

Avec Stream-K, le nombre de partiels par (séquence, tête) est borné par le nombre de
blocs, non par le contexte : `S ≤ ceil(170/40) + 1 = 6`.

    ctx 1024, tuile 512 :  C =  2   ->  S <= 6    equivalent
    ctx 1024, tuile  64 :  C = 16   ->  S <= 6    borne par la CARTE
    ctx 4096, tuile  64 :  C = 64   ->  S <= 6    borne par la CARTE

**La mémoire de travail deviendrait bornée par la carte et non par le contexte.** C'est
le bénéfice le plus solide des trois, parce qu'il est un compte et non une hypothèse de
performance.

## 4. Ce que je ne sais pas, et ce qu'il faudrait mesurer avant d'écrire une ligne

- Le plancher de travail par tuile ne disparaît pas : le balayage a montré que le
  travail cesse de décroître sous 64 positions par tuile. **Stream-K évite les blocs
  vides, il ne rend pas les petites tuiles rentables.**
- Le gain attendu n'est pas chiffrable sur le papier : il vaut au plus ce que vaut
  l'écart entre l'occupation actuelle et 100 %, et `PA_WARPS = 16` vient déjà d'en
  prendre une part (−72,9 % sur l'attention). **Les deux ne s'additionnent pas.**
- Ordre que je proposerais : finir `PA_WARPS` (barrière de qualité, puis 32 après le
  plafonnement par instanciation), **puis** mesurer le lot et la spéculation — deux
  dimensions déjà présentes dans la grille et à 1 aujourd'hui, qui remplissent la carte
  sans réécrire le noyau. Stream-K vient après, et son vrai argument n'est pas
  l'occupation : c'est **un seul graphe au lieu d'un par godet** et **une mémoire de
  travail bornée**.

## 5. Sur les deux autres pistes

- **Split-V sur le vocabulaire** : nous n'avons jamais mesuré ce que coûte la projection
  finale. Chez `Qwen2.5-Coder-14B`, `lm_head` fait 152 064 × 5 120 en bf16, soit
  **1,45 Gio lus par jeton** — 5,6 % des 26,06 Gio du pas. Ce n'est donc pas un poste
  négligeable, et il est mesurable sans rien réécrire : c'est le seul des trois qui
  commence par une mesure plutôt que par un noyau.
- **Ordonnancement page-centric** : il traite un déséquilibre **entre séquences**, qui
  n'existe pas à `BQ = 1`. Sans objet tant que le lot est à une séquence — à reprendre
  le jour où le lot augmente, c'est-à-dire précisément après la mesure du point 4.
