# L'ordre du sac à dos est mal orienté — établi, à budget égal, par un signe

```
                    promus     depense      PPL     vs etalon
bras A, ordre actuel  198/225  5,9859 Gio  5,4918   +1,435 %
bras B, ordre inverse 149/225  5,9979 Gio  5,4482   +0,630 %
ecart                                     -0,0436   (-0,794 %)
```

Prédiction écrite **avant** la mesure : `5,4144 <= PPL(B) < 5,4918`. Obtenu
**5,4482**, dans la fourchette. Les trois issues étaient nommées d'avance —
un B au-dessus de A réfutait l'inversion, un B égal à A retirait le sujet.

Étalon extérieur 5,4141 ; dispersion de l'instrument **0,000000** à six
décimales ; les deux bras reconvertis par le même binaire, chacun vérifiant
`budget.ordre_glouton` à son propre manifeste.

## Le détail qui rend le résultat brutal

**Le bras inverse promeut 49 tenseurs DE MOINS et fait mieux.**

```
A   0,0774 PPL de retard sur le plafond   avec 198 tenseurs promus
B   0,0338 PPL de retard                  avec 149 tenseurs promus
```

**56,3 % du retard au plafond comblé en renversant un signe**, en dépensant le
même budget sur moins de tenseurs. Le glouton ne se contente donc pas d'être
sous-optimal : sur la tête de son classement, il est **activement
contre-productif** — il consomme le budget sur des tenseurs qui ne rendent
presque rien, et n'a plus de place pour ceux qui rendent.

## Ce que ce résultat n'est pas

**L'ordre inverse n'est pas un candidat.** C'était un instrument, et il a fait
son travail : il mesure ce que l'ordre vaut, il ne propose pas de le remplacer
par son contraire.

**Et ce n'est pas une BORNE — j'avais écrit le mot faux.** Un ordre inverse est
un ordre parmi 225 !, choisi parce qu'il est **facile à nommer**, pas parce
qu'il majore quoi que ce soit. Correction de Jérôme, et elle décide de la
suite : l'inverse établit qu'**au moins 56,3 % du retard au plafond étaient
récupérables par le seul ordre** — un **plancher** sur le gain accessible, pas
un plafond. Rien n'interdit qu'un meilleur ordre en récupère 80 %.

Le mot importait : avec « borne », le jour où une clé rend 0,0300 nous aurions
conclu « presque tout récupéré » sans avoir la moindre idée de ce qui restait.

## La clé qui a un argument, et le test qui la distingue

Une clé fondée existe : l'**erreur de sortie évitée par octet** au lieu du gain
de décibels par octet. Elle n'est **pas** une transformation monotone de la
première, parce qu'elle applique `10^(-snr/20)` aux **deux** SNR avant la
soustraction :

```
A   SNR 20 -> 30 dB   erreur 0,1000 -> 0,0316   gain 0,0684
B   SNR 40 -> 50 dB   erreur 0,0100 -> 0,0032   gain 0,0068
```

Même écart de dix décibels, **dix fois moins de gain en erreur**. La clé en dB
les classe ex æquo ; la clé en erreur place A dix fois devant B. Argument de
Jérôme, **vérifié sur nos données** :

```
correlation entre SNR de base et deplacement de rang   -0,6645
SNR de base moyen des tenseurs qui MONTENT             20,83 dB
SNR de base moyen de ceux qui DESCENDENT               29,66 dB
desaccord de tete  top-10 90 %   top-25 60 %   top-100 3 %
```

Le réordonnancement est concentré **en tête** — là où le glouton puise en
premier — et il privilégie les tenseurs les plus mal quantifiés, ceux où un bit
rapporte le plus. **C'est ce qu'on demande à un sac à dos.**

Une exception dans les données, signalée plutôt que lissée : l'un des six
tenseurs qui descendent le plus a un SNR de base de 20,84 dB, ce qui contredit
le mécanisme pour lui. La corrélation de −0,66 est une tendance, pas une loi.

## Ce qui est en place

`ACVRAM_ORDRE_SAC` accepte `snr` (défaut inchangé), `erreur`, et `inverse`.
Le défaut **n'a pas bougé** : un A/B jugeait sur `snr`, et déplacer le défaut
sous lui l'aurait invalidé. Un mode inconnu **lève une erreur** au lieu de
retomber en silence sur le défaut — sinon une campagne lancée avec une faute de
frappe mesurerait le défaut en croyant mesurer autre chose. Le mode employé est
inscrit au manifeste.

Deux épreuves CPU : la clé `erreur` doit **réordonner** et non rehabiller — si
elle était monotone en la clé `snr`, la substituer ne changerait rien — et un
mode inconnu doit lever. Suite complète : **473 passés**.

## Ce qui reste

La mesure du bras `erreur` à budget égal, contre A et contre B. C'est la seule
qui dira si la clé fondée récupère les 0,0436 PPL que l'inversion a révélés, ou
seulement une partie. Et **ni le SNR ni son transformé ne sont la perplexité** :
trancher entre plusieurs clés fondées demanderait un ΔL par tenseur mesuré sur
une perte de calibration.
