# La part réelle de `mla_*` sous graphes à slots=12 — et pourquoi 32,8 ms > 18,8 ms

Laure, sur carte, 13/09/2026. Réponse à la contradiction de Jérôme : un noyau
ne peut pas devenir 2× plus court parce qu'il est capturé — donc l'écart entre
les 32,8 ms de noyaux MLA (eager) et les 18,8 ms de pas entier « sous graphes »
(campagne du 11/09) devait avoir une cause. Il y en a deux, empilées.

## Hypothèse 2 confirmée — et bien pire que « le graphe ne capture pas »

`GraphRunner.run()` (graphs.py:234-244) :

    b_reel = batch.batch_size
    b = bucket_batch(b_reel)          # godet puissance de 2
    ...
    if self.hybrid_layers:
        if (... or b > self.max_slots):
            return None                # eager, silencieux

`max_slots = ACVRAM_HYBRID_SLOTS` (12 dans toutes nos campagnes « slots=12 »).
`bucket_batch(12) = 16` (arrondi au godet supérieur). **16 > 12 est vrai** :
pour un modèle hybride (GDN + MLA, GLM-4.7 en fait partie —
`self.hybrid_layers` non vide), **tout lot concurrent de 9 à 15 séquences
retombe en eager, systématiquement, sans aucun message** — contrairement à la
désactivation globale des graphes qui, elle, s'annonce en console. Vérifié
directement : `bucket_batch(n)` pour n=9..16 vaut 16 dans tous les cas ; seuls
les lots ≤ 8 (godets 1,2,4,8) passent sous `max_slots=12`.

**Ce que ça veut dire pour les campagnes antérieures.** `sync-A1.json` etc.
(11/09) appelaient `engine.generate(UNE_SEULE_invite, params)` par tour :
`b_reel` valait **1** pendant tout le décodage, jamais 12. Le réglage
`ACVRAM_HYBRID_SLOTS=12` de cette campagne ne dimensionnait que les tampons
VRAM (c'est exactement ce que Prédiction 4 a vérifié) — **il n'a jamais fait
passer le graphe par un lot concurrent de 12.** Les 18,8 ms « sous graphes,
slots=12 » du 11/09 sont donc un pas à **b_reel=1** (66 couches × 1 séquence =
66 appels `decode_static`), pas un pas à 12 séquences concurrentes. Comparer
ce chiffre aux 32,8 ms mesurés aujourd'hui à b_reel=12 (792 appels) compare
deux régimes différents par construction — l'écart n'a besoin d'aucune autre
explication.

**Bead créé : `anticitoyen-vram-x0s`.** Ce n'est pas seulement le blocage de
`anticitoyen-vram-6wa` — c'est une régression silencieuse pour **tout**
déploiement hybride dont `ACVRAM_HYBRID_SLOTS` n'est pas une puissance de 2 :
le générateur perd le bénéfice des graphes CUDA (mesuré ailleurs : -11,9 % de
débit, `_eligible()` docstring) pour tout lot dans `(godet_inférieur, N]`,
sans qu'aucune ligne ne le signale.

## Mesure corrigée — `ACVRAM_HYBRID_SLOTS=16` (godet), 12 séquences réelles

Contournement pour mesurer quand même : plafonner au godet (16) au lieu de la
valeur nominale (12) — `Engine(max_batch_size=12)` garde 12 séquences réelles
admises, seul le plafond de comparaison de `GraphRunner` change.

    voie du pas : 'graphe' (confirmé, contre 'eager' avant le contournement)
    pas_total (3 pas consécutifs, sans profiler, synchronize explicite) :
        124,56 / 121,44 / 121,34 ms
    bind   : ~6,5 ms      fill : ~0,26 ms      replay (host, SANS sync) : ~7,5 ms

**`replay` sans `ACVRAM_CHRONO_SYNC` ne vaut ici que ~7,5 ms alors que
bind+fill+replay ≈ 14,3 ms << pas_total ≈ 121 ms.** C'est piste 1 (11/09) qui
se reproduit à l'identique : sans synchronisation, le chrono host autour de
`.replay()` ne mesure que le lancement asynchrone, pas l'exécution. Je n'ai
**pas** relancé avec `ACVRAM_CHRONO_SYNC=1` (budget de carte) ; à la place,
j'ai pris la mesure indépendante du profileur — CUPTI ne dépend pas du chrono
host et n'est pas sujet à ce biais.

## Le chiffre qui répond à la question de Jérôme

`torch.profiler`, même pas (bucket 16, 12 séquences réelles, voie='graphe') :

    total CUDA busy (tous noyaux, self)         : 111,98 ms
    mla_scores_kernel + mla_reduce_kernel (self) :  44,29 ms   (1056 + 1056 lancements)
    part de mla_* dans le CUDA busy total        :  39,55 %
    part de mla_* dans le pas mur (profiler actif) : 34,23 %
    pas mur SANS profiler (témoin, 3 pas)         : ~121-125 ms → CUDA busy ~92 % du pas

**La part réelle de `mla_*` sous graphes, à slots=12 correctement capturés,
est ~34 à 40 % du pas — pas 98 %.** Les 98 % du 11/09 mesuraient un régime à
1 séquence (66 lancements/pas) ; ici, à 12 séquences réelles capturées dans un
godet de 16 (1056 lancements/pas, 16× plus de travail MLA), le poste ne
domine plus le pas de la même façon : d'autres noyaux (GEMV NVFP4 groupés,
RMSNorm, copies/cat/index_copy — cf. TOP 15 du profil) prennent une part
comparable une fois 16 séquences traitées par couche.

## Conséquence pour `anticitoyen-vram-6wa`

Le bead reste valide (1584→2112 lancements par pas selon le godet, occupation
de `mla_reduce_kernel` toujours ~20 blocs/2040 par lancement, boucle Python
toujours présente) mais **la borne de gain doit être recalculée sur ~35-40 %
du pas, pas 49,5 %.** Avec le même rapport que `PA_WARPS` (-40 à -48,5 % du
poste pour un doublement d'occupation analogue) :

    borne basse :  ~35 % x 15 % ≈  -5 %  sur pas_total
    borne haute :  ~40 % x 45 % ≈ -18 %  sur pas_total

(contre -8/-22 % annoncé le 13/09 avant cette mesure, sur la base erronée du
49,5 % hérité de la campagne à b_reel=1). `bd update` fait sur les deux beads.

## Ce qui reste sans réponse

Pourquoi `bind` (6,5 ms) et `fill` (0,26 ms) ne rendent-ils pas compte du
reste ? Parce que la quasi-totalité du travail (mla_*, GEMV groupés, etc.) est
DANS le graphe, donc dans le temps de `.replay()` — dont le chrono host est
justement invalide ici sans sync. Une mesure `ACVRAM_CHRONO_SYNC=1` propre
donnerait un `replay` proche de 110-115 ms au lieu de 7,5 ms ; non faite,
budget de carte épuisé pour ce tour. À faire si le protocole A/B du bead 6wa
a besoin d'un chiffre plus serré que le profileur.
