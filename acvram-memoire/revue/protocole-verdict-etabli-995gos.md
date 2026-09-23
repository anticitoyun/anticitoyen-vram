# Protocole — trancher `ETABLI.md:2476-2481` (995 Go/s effectifs)

Demandé par Jérôme le 10/09/2026, en réponse à l'audit Go/s. Écrit par Laure,
**à exécuter dès qu'une carte se libère** — après Laurine et Manon.

## 1. Le modèle exact — identifié, pas déduit du compte de lancements

`Qwen2.5-Coder-14B-bf16-pur` (Qwen2ForCausalLM, `tie_word_embeddings: false`,
`vocab_size` 152 064, `hidden_size` 5 120, 48 couches).

**Preuve indépendante des 549 lancements** (qui ne sont qu'un indice de
« dense », pas une identification) :

    poids totaux sur disque (7 safetensors)         27,5114 Gio
    − table d'entrée embed_tokens (152064×5120×2)    1,4503 Gio   (indexée par
                                                                   gather, pas
                                                                   lue en entier
                                                                   au décodage)
    = poids réellement lus par pas                  26,0611 Gio

contre **26,09 Gio** dans `ETABLI.md:2476` — écart 0,03 Gio (0,1 %), imputable
aux en-têtes safetensors. C'est la même campagne que celle nommée en toutes
lettres à `ETABLI.md:2278` (`Qwen2.5-Coder-14B-bf16-pur`, cache KV apparié
q8_0) ; aucun changement de modèle n'intervient entre cette ligne et 2453.

## 2. Instrument — nsys pour le temps, comptage pour l'attribution

Deux défauts déjà traversés sur CE modèle commandent le choix :

* **`ncu` surestime les noyaux courts de ~45 %** ([[ncu-surestime-les-noyaux-courts]])
  — sérialisation pour la collecte de compteurs. Utilisable pour le **nom** et
  le **compte** des noyaux (exacts), jamais pour asseoir une part sur le
  **temps** seul.
* **`nsys` trace un graphe CUDA comme un bloc par défaut** — `6 099` noyaux
  comptés pour `108 800` réels lors de la trace du 9/09. Remède déjà identifié :
  `--cuda-graph-trace=node`.

Protocole retenu, dans l'ordre :

1. **Warm-up** : `ACVRAM_CHAUFFE_S=180` avant toute capture (palier thermique
   posé le 9/09).
2. **Trace `nsys`**, graphes actifs, `--cuda-graph-trace=node`, bornée par
   NVTX aux 200 pas de décodage seuls (pas le chargement, pas le prefill —
   défaut n°3 du 9/09 : `nsys stats` sur toute l'exécution avait donné
   103,7 % d'occupation, numérateur plus large que son dénominateur).
   `--capture-range=nvtx` seul avait échoué (« No reports were generated ») ;
   vérifier la présence du rapport avant de dépouiller.
3. `nsys stats --report cuda_gpu_kern_sum` **sur le fichier de sortie
   fraîchement produit** — vérifier l'horodatage du CSV contre celui de la
   trace avant lecture (défaut n°4 du 9/09 : un dépouillement a lu des CSV de
   la trace précédente et rendu 97,4 % de temps mort fantôme).
4. **Grouper les noyaux par NOM, puis classer chaque nom en deux catégories
   seulement : « lit du poids » (qkv/o_proj/gate_up/down, `lm_head`, tout GEMV
   nvfp4/int8) ou « ne lit pas de poids » (attention `partial`+`reduce`,
   écriture KV, RoPE/normes fusionnées, échantillonnage, mémoire).** Ce n'est
   plus `paged_attn_partial` contre le reste (§3 quater : établi que son
   plancher seul, 0,687-1,369 %, ne peut pas suffire) — **c'est la somme de
   toute la catégorie non-poids qui donne `p`.** L'écriture KV, le RoPE fusionné
   et l'échantillonnage sont les suspects principaux restants (453 des 549
   lancements ne sont toujours pas répartis, cf §3 quater).
5. **Asseoir `p` sur le compte de lancements par nom, par pas**, croisé avec le
   temps `nsys` (fiable, pas de replay contrairement à `ncu`) — pas sur un
   pourcentage `ncu` brut. Si `ncu` est utilisé en complément pour des
   compteurs (dram__, coalescence), diviser tout temps de noyau court par
   ~1,45 avant de l'additionner à quoi que ce soit.
6. Publier `Σ(temps nsys des noyaux non-poids) / t_mural du pas` à côté de `p`
   (demande d'Océane) : sans ce rapport la part n'est pas publiable ; avec lui
   c'est un rapport de deux quantités biaisées dans le même sens, donc
   utilisable.

**Deux contrôles miroir à vérifier AVANT de faire confiance au profil (Océane)**,
sans quoi un désaccord entre son banc et `nsys` se lirait comme un résultat au
lieu d'un défaut de montage :

* **Le total de temps GPU sommé par `nsys` doit retrouver ~28,140 ms par pas**
  sur ce modèle (`ETABLI.md:2460`). Un écart de plus de quelques pourcents veut
  dire que la trace ne couvre pas exactement les mêmes 200 pas de décodage que
  la mesure d'origine — corriger le montage, pas lire l'écart comme la part.
* **L'écart entre le pas complet (plage NVTX) et l'union d'intervalles GPU
  doit retrouver ~0,608 ms** (le temps mort déjà mesuré). Si cet écart diffère
  nettement, c'est le même signal d'alarme que le précédent.

Ces deux contrôles jouent le même rôle qu'`ACVRAM_PA_ARM=9` dans
`outils/part-attention-trois-bras.py` (banc d'Océane, commit `3d66541`,
branche `oceane`) : une vérification qui peut arrêter la mesure avant qu'elle
ne produise un chiffre, plutôt qu'un chiffre qu'on croit sur parole.

**Avant de lancer quoi que ce soit, trois vérifications sur le binaire chargé
(Jérôme/Océane) — vérifiées par moi à l'instant, PAS encore présentes dans mon
arbre** :

* `~/.cache/acvram/kernels` est **partagé entre les quatre worktrees**
  (`acvram/kernels/__init__.py:252`, confirmé). Le correctif à deux lignes
  annoncé (`ACVRAM_KERNEL_CACHE`) **n'est pas encore dans `origin/oceane`**
  (vérifié par `git show origin/oceane:acvram/kernels/__init__.py`, dernier
  commit visible `3d66541`) — je repointerai vers un répertoire privé au
  moment de lancer, en relisant l'état du dépôt à ce moment-là plutôt que de
  supposer que le correctif est arrivé entre-temps.
* **Publier le sha256 du `.so` chargé** dans le relevé — un `.so` peut être
  cohérent avec le `.cu` du mauvais arbre (Manon, ce matin).
* **Vérifier `acvram.__file__`** avant de mesurer quoi que ce soit — le
  worktree sans venv important `acvram` du dépôt principal est la faute qui a
  coûté un essai à Océane.

**Fait qui déplace le sens d'une des deux issues, pas le protocole** : Manon a
établi que `paged_attn_partial` est **sérialisé sur la longueur de tranche**
(`chunk=64` : 25,47 µs contre 78,72 en production, ×3,1) — son coût est un
choix de découpage, pas une propriété physique fixe. Si `p > 5,24 %`,
l'issue ne se lit plus comme « rien à faire », mais comme **« l'attention est
une cible dont le remède est déjà identifié »**.

**Ordre d'exécution sur la carte, imposé par Jérôme** : Manon finit ses douze
mesures, PUIS Océane compile ses trois branches en `nice -19` (~9 min, elle
signale la fin) — sa compilation ne peut pas cohabiter avec ce profil. **Ne pas
démarrer sur la seule libération de la carte : attendre le signal d'Océane.**

## 3. Le seuil — dérivé du chiffre contesté lui-même (Océane), pas choisi

`p` = part du temps de pas prise par des noyaux qui ne lisent pas de poids
(attention paginée comprise). La bande passante réellement exigée des GEMM
seules est `26,09 Gio / (28,14 ms × (1 − p))`. Elle franchit la borne mesurée
de la carte (1050 Go/s, [[borne-memoire-5090-1050-go-s]]) dès que :

    p > 1 − 995/1050 = 5,24 %

Au-delà, le calcul d'`ETABLI.md` ne s'affaiblit pas, **il se contredit** : il
exige des GEMM un débit que la 5090 n'atteint pas. Les deux issues :

* **`p ≤ 5,24 %`** → la conclusion **tient**, lever la mention SUSPECTE.
* **`p > 5,24 %`** → elle **tombe**, question rouverte pour la famille dense.

**Ne pas transporter 42,9-49,5 %** (Océane vient d'être corrigée sur ce point) :
ce chiffre vient d'un MoE à routage creux, ce modèle est un dense où le MLP
entier est lu à chaque jeton. `p` se mesure sur CE modèle, pas comparé à une
part empruntée.

**La réserve `ncu` se clôt par la marge, pas par une correction** : attribuer
au biais toute son amplitude (÷1,45, soit −45 % du temps mesuré) ferait passer
un `p` observé de 42,9 % à 29,6 % — encore six fois le seuil. Aucune correction
du biais ne peut sauver le calcul si `p` observé est dans cet ordre de
grandeur. Publier quand même `Σ(temps profileur) / t_mural non profilé` à côté
de `p` : sans ce rapport aucune part n'est publiable ; avec lui, c'est un
rapport de deux quantités biaisées dans le même sens, donc utilisable.

## 3 bis. Minorant arithmétique — fait avant la carte, sans profileur

**CORRIGÉ après vérification directe de la source — le format supposé
(bf16, ou q8_0 uniforme) était faux.** `acvram_kernels.cu:797-808` :
`paged_attn_partial_kernel` prend `kc`/`vc` en `signed char [NB,16,HKV,D]`
et `ks`/`vs` en `__half [NB,16,HKV]` — **int8 avec une échelle fp16 par
vecteur `D`** (une échelle par position, par tête, par K ou V), pas du bf16.
D = 128 (head_dim, confirmé §1). Par position et par tête KV :

    (D + 2 octets d'échelle) × 2 (K et V) = (128 + 2) × 2 = 260 octets

Les octets de cache KV se comptent, ils ne se profilent pas :

    octets_KV(n_ctx) = n_ctx × n_kv_head × 260 × n_couches
                     = n_ctx × 8 × 260 × 48
                     = n_ctx × 99 840 octets  (97,5 KiB/jeton de contexte)

    plancher de temps = octets_KV / 1050 Go/s
    p_minorant = plancher / 28,14 ms

**Phrase à écrire avant de calculer, pas après** : un `p_minorant` faible ne
réhabilite rien — l'attention paginée est un noyau **court**, dominé par la
latence de lancement, pas par les octets qu'il lit. Un minorant faible dit
seulement que la voie arithmétique ne suffit pas à trancher ; il ne dit pas
que la conclusion tient.

**Calcul, avec `ctx 1024` lu à `ETABLI.md:2279`** (même campagne, spéculation
éteinte, 0 couche exilée — aucune autre longueur de contexte n'est déclarée
entre cette ligne et 2453) :

    octets_KV(1024) = 1024 × 99 840 = 102 236 160 octets = 97,5 Mio = 0,1022 Go
    plancher = 0,1022 / 1050e9  = 0,0974 ms
    p_minorant = 0,0974 / 28,14 = 0,346 %

**0,346 % est ~15× sous le seuil de 5,24 %** (Océane, corrigeant mon calcul
initial en bf16 qui donnait 0,681 % — **valeur remplacée, ne pas la
réutiliser**). Conforme à l'attente : sur un contexte court, la voie
arithmétique pure ne clôt rien. **Le dossier reste ouvert et dépend de la
mesure au profileur (§2)**.

## 3 ter — RETIRÉ. Le minorant par lancements et son addition confondaient deux budgets

**Retiré le 10/09/2026 par Océane, avant toute mesure.** Tout ce qui suit dans
cette section et la suivante (§3 quater, conservées ci-dessous pour la trace)
additionnait un coût de frontière de lancement au plancher d'octets KV, les
deux rapportés à `p` sur les **28,14 ms de temps GPU**. C'est une erreur de
catégorie : les **28,14 ms** sont l'union des intervalles d'EXÉCUTION des
noyaux (`ETABLI.md:2460`) — le temps où un noyau tourne réellement. Le coût de
frontière (rampe, dépendances, attente entre noyaux) est précisément ce qui
est **hors** de cette union : c'est le **temps mort**, `0,608 ms`, **déjà
mesuré et déjà publié** (`ETABLI.md:2461`), soit **1,1 µs de moyenne sur les
549 lancements** (`0,608 / 549,3`) — une valeur mesurée sur CETTE campagne,
plus précise que la plage 1-3 µs empruntée à une autre.

**Conséquence : `p` dans le budget de 28,14 ms n'a qu'un plancher arithmétique
valide, celui du §3 bis — 0,346 %.** Le temps de frontière n'entre pas dans ce
budget ; il vit dans le pas complet (28,748 ms), un dénominateur différent.
**Les deux instruments à venir doivent donc publier leur part dans les DEUX
dénominateurs** (temps GPU 28,14 ms et temps mural 28,748 ms), ou en
millisecondes brutes qui n'ont pas ce problème — 5,24 % de 28,14 ms vaut
5,13 % du mural, l'écart est petit mais réel.

**Ce qui reste utile de tout ce travail, correctement requalifié :**

* Le compte de **96 lancements d'attention sur 549** (§3 quater, vérifié dans
  `acvram_kernels.cu`) reste un **contrôle de comptage** valable pour le profil
  du §2 : si `nsys stats` ne trouve pas 96 noyaux d'attention par pas à
  `ctx 1024`, c'est le profil qu'il faut corriger, pas le nombre.
* Les **453 lancements non répartis** restent la vraie question — mais leur
  répartition poids/non-poids **se lit sur les NOMS de noyaux dans
  `nsys stats`, pas sur un temps** (Océane) : c'est un compte, non biaisé par
  la surestimation `ncu`, publiable même avant que les temps le soient.
* `p` se mesure désormais directement et uniquement par **la somme des durées
  `nsys` (fiables, sans replay) des noyaux non-lecteurs de poids, divisée par
  28,14 ms** — aucune arithmétique de rampe n'est plus nécessaire une fois le
  profil pris ; l'arithmétique de cette section ne servait qu'à sonder AVANT
  la carte, et elle a fait son travail (aucun des deux plancher n'a clos le
  dossier), pas à remplacer la mesure.

**Ancien contenu conservé ci-dessous sans être corrigé dans le texte, pour que
l'erreur reste visible plutôt qu'effacée :**

## 3 ter. Second minorant, par le compte de lancements — trois réserves, deux tombent en partie

**Réserve à écrire avant de chercher** : ce minorant ne clôt le dossier que si
les trois conditions ci-dessous tiennent ENSEMBLE ; si l'une tombe, il faut le
dire plutôt que forcer une conclusion.

1. **Les graphes CUDA sont actifs** sur cette campagne — lu, pas supposé :
   `ETABLI.md:2455` (« 200 pas, graphes actifs, noyaux… ») ET `:2340` pour la
   mesure d'énergie de la même chaîne. Cette réserve NE tombe PAS, mais elle
   change le régime applicable : sous rejeu de graphe, le coût par nœud n'est
   pas un lancement CPU dispatché, c'est une frontière GPU amortie — le
   chiffre à utiliser est celui mesuré DANS ce régime, pas un coût de
   dispatch à froid.

2. **« 3 à 6 µs de rampe » ne se retrouve dans aucune mesure protocolée.**
   Cherché dans `ETABLI.md` et dans `docs/FEUILLE-DE-ROUTE.md` (la campagne du
   3/09 elle-même, celle qui fait passer Qwen3-Coder-30B de 88 à **166-172**
   t/s, `docs/FEUILLE-DE-ROUTE.md:677-786`) : cette campagne dit « environ
   vingt-cinq lancements par couche, chacun payant **quelques microsecondes**
   de rampe » (`:682`) — de la prose, pas un chiffre mesuré isolément ; le gain
   vient d'un paquet de sept optimisations distinctes (conflits de banque,
   fusion RoPE, écriture KV, N-par-lecture, quatre lancements de moins par
   couche…), pas d'une seule constante de rampe.
   **Le seul chiffre mesuré et protocolé pour un coût par frontière de noyau
   est ailleurs, et il a une histoire** : `ETABLI.md:1742` mesurait
   `832 × 2,15 µs`, **puis `ETABLI.md:1773-1792` retire ce chiffre** — 2,15 µs
   venait d'une exécution sérialisée par `ncu` (collecte de compteurs entre
   noyaux), pas d'un rejeu réel. Le retrait recalcule sur la valeur mesurée
   d'un rejeu de graphe libre : **1 à 3 µs par frontière**, retenu à 2 µs dans
   son propre calcul de clôture.
   **Cette réserve tombe telle que formulée** : 3-6 µs n'est pas la mesure
   disponible ; 1-3 µs (déjà elle-même une correction d'un chiffre `ncu`
   biaisé) l'est.

3. **549 par PAS, confirmé** : `ETABLI.md:2459` l'écrit explicitement
   (« noyaux dans la plage 549,3 par pas (544 attendus : cohérent) »), sous le
   même régime de graphes actifs que le compte. Cette réserve tient.

**Recalcul avec la plage réellement mesurée (1-3 µs, retenue à 2 µs) :**

    549 x 1 us = 0,549 ms -> 1,95 % de 28,14 ms
    549 x 2 us = 1,098 ms -> 3,90 %
    549 x 3 us = 1,647 ms -> 5,85 %

**La plage vérifiée encadre le seuil de 5,24 % au lieu de le dépasser aux deux
bornes.** À la valeur retenue par le calcul qui a produit ce chiffre (2 µs),
le minorant par les lancements reste SOUS le seuil (3,90 %), comme le minorant
par les octets. Seule l'extrémité haute de la plage vérifiée (3 µs, jamais
observée directement sur ce modèle) le dépasserait.

## 3 quater. L'addition des deux minorants — réserve avant de vérifier, puis un recouvrement trouvé

**Réserve d'abord** : l'addition n'est valide que si (a) les deux postes sont
disjoints dans le temps ET (b) chaque terme compte le bon dénombrement, pas le
total du pas.

**(a) tient pour `paged_attn_partial` lui-même** : le temps d'un noyau se
décompose en rampe de lancement PUIS octets/calcul, séquentiellement — ce sont
deux tranches du même intervalle, pas deux comptages du même temps. Pas de
recouvrement à ce niveau.

**(b) ne tient PAS tel que formulé, et c'est plus grave que le recouvrement
cherché.** `549` (`ETABLI.md:2459`) est le compte de **tous** les lancements du
pas, GEMM/GEMV lisant du poids compris (qkv, o_proj, gate_up, down — au moins
quatre par couche pour ce dense, donc une bonne partie des 549). Or `p` est
défini (§3, Océane) comme la part du temps prise par des noyaux qui **ne
lisent PAS de poids**. Le coût de frontière des lancements GEMM appartient au
bucket `(1-p)` — c'est déjà implicitement compris dans le temps que le calcul
« bande passante exigée des GEMM = 26,09 Gio / (28,14 ms × (1-p)) » attribue à
la lecture de poids — pas au bucket `p`.

**Utiliser 549 tel quel double-compte une partie du côté qui ne devrait pas y
être.** Le bon dénombrement pour ce terme est le sous-compte des lancements de
noyaux NON lecteurs de poids (attention + son noyau de combinaison, écriture
KV, RoPE/normes fusionnées, échantillonnage — précisément la famille « tout le
reste » déjà prévue à séparer de `paged_attn_partial` au §2), pas les 549
entiers.

**Conséquence** : la somme `0,68 % + [1,95-5,85] % = [2,63-6,53] %`, retenue à
4,58 %, est une SURESTIMATION du vrai minorant additif — donc le budget restant
qu'elle implique (0,66 % à la valeur retenue) est plus pessimiste qu'il ne
devrait l'être, pas une contrainte fausse dans l'autre sens.

**Le sous-compte demandé existe déjà pour l'attention, vérifié dans la source
(Océane, confirmé par moi à `acvram_kernels.cu:793,918-1924`)** :
`PA_CHUNK = 512`, grille `C = ceil(N×16/512)` ; à `ctx 1024`, `N=64`, `C=2`.
Le noyau `reduce` n'est lancé **QUE si `C > 1`** (`if (C > 1) paged_attn_reduce_kernel<<<...>>>`,
ligne 1922) — donc **2 lancements d'attention par couche** (`partial` + `reduce`)
à cette longueur de contexte, soit **96 sur les 549,3 du pas (17,5 %)**, un
compte lu dans le code, pas déduit.

**Plancher combiné pour l'attention seule** (octets KV §3 bis + sa propre
frontière, sur 96 lancements et non 549) :

    96 x 1 us = 0,096 ms -> 0,341 %      + 0,346 % (octets) = 0,687 %
    96 x 2 us = 0,192 ms -> 0,682 %      + 0,346 %           = 1,028 %
    96 x 3 us = 0,288 ms -> 1,023 %      + 0,346 %           = 1,369 %

**Loin du seuil de 5,24 % dans les trois cas — l'attention seule ne clôt pas
le dossier.** Ce qui reste non compté : les 453 lancements restants (549 − 96)
mélangent noyaux lecteurs de poids (qkv, o_proj, gate_up, down, lm_head — au
moins 5 par couche pour ce dense) et noyaux non-lecteurs (écriture KV,
RoPE/normes fusionnées, échantillonnage). **Ce partage-là n'est pas déductible
du code sans compter les lancements réels par nom** — c'est exactement ce que
le profil du §2 doit rendre, kernel par kernel, avant de compléter la
contrainte additive au-delà de l'attention.

**Verdict de ce deuxième minorant : il ne clôt pas le dossier.** Contrairement
à la prédiction initiale (3-6 µs, non retrouvée dans une mesure), la valeur
correctement sourcée ne dépasse pas le seuil de façon univoque — elle
l'encadre. **Le maillon manquant reste le même : une mesure directe de la part
de `paged_attn_partial` sur CE modèle, au profileur (§2), pas un calcul par
comptage.** Ce que ce deuxième minorant apporte quand même : il exclut que le
dossier se referme "par le compte seul" avant la carte, et il resserre la
plage de ce qu'il faudra confirmer au profileur (quelque part entre ~2 % et
~6 % rien que pour le poste lancement, avant même le calcul propre à
`paged_attn_partial`).

## Ce qui n'est PAS dans ce protocole

Aucune nouvelle campagne de comparaison acvram/llama.cpp, aucune touche au
code — lecture de profil seule. Pas de GPU sollicité avant que Laurine et
Manon aient libéré la carte.
