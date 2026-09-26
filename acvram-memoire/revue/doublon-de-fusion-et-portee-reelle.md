# Le doublon de fusion : réel, corrigé, et beaucoup plus petit que son titre

## Ce qui était vrai

`MLP.fuse()` et `Attention.fuse()` construisent une pile et **ne libèrent pas**
les projections d'origine — à raison : `forward` retombe sur elles au-delà de
`SEUIL_FUSION`. Les deux **chemins** sont nécessaires. Les deux **copies** ne
l'étaient pas.

Quatre empileurs, deux comportements :

```
stack_plain_linears    (bf16)      repointe les originaux en VUES   ✓
stack_nvfp4_linears                repointe les originaux en VUES   ✓
stack_int8_linears                 vues pour le BIAIS seulement     ✗
stack_int4_awq_linears             vues pour le BIAIS seulement     ✗
```

Et le commentaire de `stack_int8_linears` affirmait « les originaux deviennent
des vues, **comme pour les poids** » — au-dessus du seul code qui repointait le
biais. La disposition le permettait depuis toujours : `cat` sur l'axe 0 rend un
tenseur contigu dont chaque tranche `[d:d+n]` l'est aussi, et deux des quatre
empileurs le prouvaient avec l'argument écrit dans leur docstring.

**Corrigé.** Épreuve CPU `tests/test_fusion_sans_doublon.py`, quatre cas, qui
vérifie le partage de `untyped_storage().data_ptr()` et la contiguïté des vues.
Elle **discrimine** : lancée contre l'acvram installé, qui porte l'ancien code,
elle échoue sur int8 et int4_awq et passe sur bf16 — le témoin qui était déjà
juste. Un contrôle qui ne peut pas rendre faux ne contrôle rien.

## Ce qui était faux dans mon titre

J'avais annoncé « 461 Mo, 6,4 % de VRAM pour +2,6 % de débit, et la VRAM
déclenche l'exil qui coûte 61 à 70 % ». Les deux premiers chiffres sont exacts.
La conséquence ne l'est pas.

**Arithmétique sur les 18 modèles du parc qui portent de l'int8 ou de
l'int4_awq**, tailles réelles des dossiers contre capacités réelles :

```
                      5090 (31,36 Gio)   3080 Ti (11,63 Gio)
bascules de placement          0                  0
modeles deja etages            4                 10
```

**Aucun modèle ne bascule, sur aucune des deux cartes, même à la borne
supérieure.** Ceux qui sont touchés étaient déjà étagés ; le doublon aggrave
leur étagement de 0,02 à 0,44 Gio (estimation réelle) au lieu de créer un exil.
La chaîne causale que j'avais tendue — doublon → dépassement → exil → −70 % —
**ne se referme sur aucun modèle du parc.**

## Et ma borne valait dix fois le réel

Le manifeste ne dit pas si `_scaler_commun` aurait accepté la paire. Sur le seul
modèle mesuré :

```
borne du manifeste        4,29 Gio    64 groupes empilables
reel mesure               0,43 Gio     5 groupes empiles
rapport                   x 10,0       7,8 % des groupes
```

Les deux comptes concordent — 5/64 groupes valent 7,8 %, et 0,43/4,29 vaut
10,0 % des octets — ce qui confirme que **59 des 64 empilements ont été
refusés**. Transporté au parc, le gain réel cumulé vaut ~2,13 Gio contre 27,37
à la borne. Ce taux de transport de 7,8 % est **à vérifier** : il dépend des
`act_scale` de chaque modèle.

## Le vrai défaut est ailleurs, et il porte sur le débit

`_scaler_commun` exige `torch.equal(s.scale, tete.scale)`. `gate` et `up` lisent
la **même entrée**, donc leurs échelles d'activation devraient être identiques ;
elles ne le sont pas, et **59 empilements sur 64 sont refusés pour cette
raison**. Ces 59 groupes lancent deux ou trois GEMV par pas là où un seul
suffirait.

Le gain de la fusion a été mesuré à **+2,6 %** — mais sur un modèle bf16, où
aucun `act_scale` n'existe et où **tous** les empilements aboutissent. Sur un
int8 calibré, la même fusion n'atteint que 7,8 % des groupes. **Le +2,6 % est un
chiffre juste dans un régime et transporté dans un autre**, exactement la faute
que ce carnet nomme.

À vérifier avant toute conclusion : d'où le convertisseur tire l'`act_scale` de
chaque tenseur. Si elle est calculée depuis les statistiques de l'**entrée**,
`gate` et `up` doivent recevoir la même et le refus est un défaut à corriger —
gain potentiel sur le débit de tous les modèles int8 calibrés. Si elle dépend
aussi du **poids** (comme une échelle AWQ), le refus est légitime et c'est la
portée annoncée de la fusion qu'il faut corriger.

## Ce que ça coûte de le dire ainsi

Rien, et c'est le point : le correctif reste bon puisqu'il est gratuit, mais il
ne vaut pas ce que son premier titre promettait. Publier « 6,4 % de VRAM
reprise » sans « et cela ne change le placement d'aucun modèle » aurait été
juste dans le chiffre et faux dans la décision.
