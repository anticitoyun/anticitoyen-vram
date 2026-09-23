# PREDICTION ECRITE AVANT le classement des 36 486 par nom de noyau

Laurine, 10/09/2026, **avant** que le classement de Laure soit lu. Par lecture
du code seule, plus un ancrage mesure.

## L'ancrage : ce que coute une couche DENSE, mesure par moi aujourd'hui

Qwen3-4B, 36 couches, b <= 8, graphes coupes : **564 noyaux par pas**, soit
**15,7 par couche**. Decomposition relevee au profil, ramenee a la couche :

    nvfp4_gemv   3,9    rmsnorm  2,0    int8_gemv  1,7
    elementwise  1,3    rope     1,0    kv_write   1,0    (reste ~4,8)

Une couche dense coute donc **~16 noyaux, INDEPENDAMMENT du lot** — c'est le
fait important : le dense ne boucle pas sur les sequences.

## Ce que je predis pour GLM-4.7 (MLA + MoE, 66 couches, b = 12)

    bloc                                noyaux/couche/pas   d'ou
    attention MLA, appel ext                    24          2 x 12, etabli
    attention MLA, le RESTE de decode_static   216          18 x 12  <- voir plus bas
    normes (entree, post-attn)                   3          par lot
    routage (_route, topk, cast eid)             6          par lot
    plomberie MoE (tok, seq x2, views)           5          par lot
    experts (gateup groupe, down, ponderation)   4          par lot
    expert partage                               3          par lot
    residuels et divers                          3          par lot
                                              ------
    TOTAL PREDIT                              ~264

**Le poste qui domine n'est pas l'appel au noyau MLA, c'est tout ce qui
l'entoure.** `decode_static` (mla.py:268) est ecrit « un jeton, une sequence » :
la boucle par sequence n'enveloppe pas seulement `ext.mla_decode`, elle
enveloppe **tout le corps de la methode**. J'y compte ~18 lancements par
sequence — deux projections, une rope, quatre `cat`, une norme, deux `einsum`,
deux conversions de `k_b`/`v_b`, un `index_copy_`, un `contiguous`, la sortie
`o_proj`, l'increment de `len` — soit **216 par couche a douze sequences,
contre 24 pour le noyau lui-meme**. Si ma lecture est bonne, le chantier MLA
par lot vaut donc **neuf fois** ce que mon compte de 1 584 lancements disait.

## LES DEUX ISSUES, ecrites avant de voir le chiffre

    predit ~264 contre 553 mesures  ->  IL MANQUE UN FACTEUR ~2,1.
        La lecture ne voit pas tout : il reste une boucle ou une repetition que
        le code ne montre pas a la lecture. Le classement dira laquelle, et le
        chantier est « TROUVER ».

    predit ~264 et mesure ~264-300  ->  la structure explique tout, il n'y a pas
        de boucle cachee. Le chantier devient « FUSIONNER » : sortir la
        plomberie de la boucle, batcher decode_static, fusionner les normes.

**Je prédis la PREMIERE**, et j'en donne la raison : mon estimation de 18
lancements par sequence dans `decode_static` est une borne BASSE — je n'ai
compte que les operations que la lecture nomme, sans les conversions de dtype
implicites, sans les `contiguous` que PyTorch insere, et sans le `_proj_entree`
que je n'ai pas ouvert. Il est plus probable que je sous-estime le poste MLA que
d'y voir un poste inconnu.

**Conséquence de cette asymetrie, et c'est elle qui compte** : si l'ecart se
comble par un `decode_static` plus lourd que prevu, l'issue « trouver » et
l'issue « fusionner » designent **le meme chantier** — la boucle par sequence de
MLA. Si l'ecart vient d'ailleurs (MoE, normes), ce sont deux chantiers
differents. **C'est le classement par nom de noyau qui separe les deux**, et
c'est pour ca qu'il doit passer avant toute autre mesure.

## Le second calcul, conditionnel

Laure mesure **629 lancements par pas** sur un temoin dense a douze sequences.
Il me manque son nombre de couches. Avec mon ancrage a 15,7 par couche, 629
implique **~40 couches** — coherent avec un 3B. Alors :

    dense      629 / 40 couches  =  ~16 noyaux/couche
    GLM-4.7  36 486 / 66 couches =  ~553 noyaux/couche
    rapport                          ~35

**Un facteur ~35 par couche entre un dense et un MLA+MoE, a concurrence
egale.** Et comme le dense ne boucle pas sur les sequences (mesure : 564 a b=1
comme a b=8), **tout ce facteur est du a MLA et au MoE**. La soustraction nomme
le chantier sans qu'aucune lecture de code soit necessaire — mais elle ne dit
pas lequel des deux, et ma prediction dit MLA a neuf contre un.
