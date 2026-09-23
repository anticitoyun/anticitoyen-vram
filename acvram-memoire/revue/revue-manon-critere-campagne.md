# Revue adverse de `valider-campagne.py`

Manon, 9 septembre 2026. Question directrice : **quelle campagne défectueuse
passerait ?** — et sa symétrique, que la relecture a rendue plus urgente :
**quelle campagne valide serait refusée ?**

## 1. La garde anti-spéculation est un no-op CERTAIN, pas conditionnel

Ligne 70 : `prop = (l.get("proposed") or l.get("proposes") or "0")`.

**Aucune de ces deux colonnes n'existe dans le TSV.** Relevé sur l'en-tête écrit
par le banc :

    modele moteur alias ctx t_s ttft_ms W j_kJ jetons chargement_s etat apercu
    J J_net W_repos j_kJ_net plafond_W horloge_min horloge_max temp_max bridages
    dispersion_pct t_s_min t_s_max t_s_passages ttft_passages jetons_moteur
    jetons_flux jetons_source binaire ctx_servi empreintes textes_identiques

`proposed` ABSENTE · `proposes` ABSENTE. Le `or "0"` transforme donc l'absence
en zéro, et **la garde approuve toujours** — y compris une campagne où les deux
moteurs spéculent à plein.

Ce n'est pas « si la colonne manque » : elle manque, aujourd'hui, dans toutes
les campagnes. `proposed_tokens` vient de `/metrics`, pas du banc.

**Remède** : soit publier la colonne (le banc ne relève pas encore
`proposed_tokens`), soit vérifier la spéculation autrement — par le journal du
serveur, qui imprime `speculation : <mode>, k=<n>`. Tant que la colonne n'existe
pas, **la condition doit ÉCHOUER en son absence**, jamais l'ignorer.

## 2. Une seconde condition a la même forme : `etat`

Ligne 41 : `if (l.get("etat") or "ok") != "ok"`.

Colonne absente ou vide → `or "ok"` → **la condition passe**. C'est exactement
le défaut du point 1, sur l'état de la mesure elle-même. Ici la colonne existe,
donc le trou est théorique — mais la forme est la même et elle se propagera.

**Les autres sont correctes** et méritent d'être citées comme modèle :
`jetons_source`, `textes_identiques`, `ctx`, `binaire` utilisent `or ""` puis
comparent à une valeur non vide — une colonne absente y produit `""`, qui
échoue. C'est la bonne forme : **le défaut par défaut doit être le refus.**

## 3. Le critère d'étendue REFUSERA TOUTE CAMPAGNE, même parfaite

C'est le défaut le plus coûteux, et il va dans le sens inverse des autres.

`PASSAGES_MIN = 4` porte le commentaire « cinq lancés, le premier jeté : quatre
utiles ». **Mais `t_s_passages` publie les CINQ passages, premier compris** —
c'est sa raison d'être : min et max ne disent pas *lequel* s'écarte, il fallait
l'ordre pour appliquer la table de décision.

Conséquence, avec les chiffres mesurés le 8/09 :

    passages publies   112,9 · 150,5 · 150,2 · 151,2 · 152,3
    etendue / median   (152,3 - 112,9) / 151,05 = 26 %
    seuil du critere   1 %

**Toute campagne saine échoue.** Un critère qui refuse tout est aussi inutile
qu'un critère qui accepte tout — et il finira désactivé, comme la garde `vmstat`
qui lisait la ligne 1.

**Remède** : écarter le premier passage avant de calculer l'étendue —
`passages[1:]` — ce qui est justement ce que la campagne fait pour publier son
débit. Sur les mêmes chiffres : 1,2 % sur les quatre utiles. Le seuil de 1 %
reste alors serré mais discutable, au lieu d'être impossible.

## 4. Le critère vérifie `ctx`, c'est-à-dire l'intention — pas `ctx_servi`

Ligne 62 : `if (l.get("ctx") or "") != CTX_ATTENDU`.

**`ctx` est ce que le banc a demandé. `ctx_servi` est ce que le processus a
reçu**, lu dans `/proc/<pid>/cmdline`. La colonne existe précisément parce que
le 9/09 le banc croyait 8192 pendant que le serveur tournait à **32 768** —
`acvram-serveur` écrasait la variable d'environnement par un argument absent.

Vérifier `ctx` reconduit donc exactement le défaut qui a motivé l'ajout de
`ctx_servi`. **Il faut vérifier les deux, et surtout leur égalité** : si `ctx`
et `ctx_servi` diffèrent, la plomberie a menti et le chiffre ne vaut rien.

## 5. Le seuil de 1 % n'est pas de moi

Je n'ai jamais posé de seuil d'étendue. J'ai posé une règle de comparaison :
**un écart ne conclut que s'il dépasse la dispersion**, mesurée sans le premier
passage. C'est une relation entre deux grandeurs de la même campagne, pas une
constante absolue.

Un seuil fixe est une constante inventée ; la règle, elle, s'adapte au modèle
mesuré — 0,03 % de dispersion sur `nemo-12b` contre 0,53 % sur les Huihui.
**Ce que le critère devrait exiger, c'est que la dispersion soit petite devant
l'écart qu'on veut publier** — donc il ne peut pas être écrit avant de savoir
quel écart on mesure, ou alors il faut assumer le 1 % comme un choix
d'ingénierie non mesuré et l'écrire comme tel.

## 6. L'égalité stricte des comptes de jetons peut échouer légitimement

Lignes 83-85 : deux moteurs qui s'arrêtent sur un jeton de fin à des longueurs
différentes produisent des comptes différents sans qu'aucun ne soit fautif. À
`max_tokens = 200` et température nulle, l'égalité est probable mais non
garantie. **Condition à assouplir ou à justifier**, sinon elle rejettera une
campagne saine — et c'est le défaut du point 3 une seconde fois.

## Ce que je n'ai pas vérifié

Le script n'a pas été **exécuté** sur un TSV réel : la campagne `bounebtni`
mesure et je ne lance rien. Les points 1, 2 et 4 sont établis par lecture des
colonnes, le point 3 par calcul sur des chiffres mesurés le 8/09 — aucun ne
demande d'exécution, mais aucun ne l'a reçue.
