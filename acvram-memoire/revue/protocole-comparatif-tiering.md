# Protocole — comparatif sur le terrain du tiering (manche 2)

Rédigé par chef le 9/09. **Non lancé** : soumis à poste1, poste2, poste4.

## LA QUESTION DE LA MANCHE — a ecrire avant de lancer (poste1)

**Cette manche demande : « que rapporte le tiering ? »** — pas « qui est le
plus rapide ». Le fait mesure est : *l'un tourne la ou l'autre doit deborder
sur le processeur.*

Cette ligne decide du statut du desaccord de quantification (38 contre 40 Go
sous le meme nom) : pour CETTE question, un ecart de 5 % sur les poids est
negligeable devant un moteur qui lit depuis la RAM hote. **Si un jour la
question devient « qui est le plus rapide », le desaccord redevient
DISQUALIFIANT** : il faudrait apparier en qualite d'abord (perplexite des deux
cotes) et aucun debit ne se publierait sans elle.

Les deux manches sont legitimes ; elles ne se publient pas de la meme facon.
Sans cette ligne, le lecteur choisirait la lecture qui l'arrange.

## Pourquoi cette manche

La manche 1 a mesuré le **terrain le moins favorable** : un dense de 27,5 Gio
qui tient entièrement sur la 5090, où notre pari architectural n'intervient
pas. Résultat validé : −2,3 % en débit, **−8,5 % en énergie**.

Cette manche mesure le terrain **où acvram est censé gagner** : un modèle qui
ne tient pas sur 32 Gio. Chez nous, le planificateur exile ; chez llama.cpp,
`--n-gpu-layers` partiel décharge sur le processeur. **C'est la comparaison qui
décide si notre conception vaut ce qu'elle coûte.**

## Modèles disponibles des deux côtés

    Llama-3.3-70B-Instruct-heretic.i1-Q4_K_M          acvram 38G   gguf 40G
    Hermes-4-70B-heretic.i1-Q4_K_M                    acvram 38G   gguf 40G
    DeepSeek-R1-Distill-Llama-70B-heretic.i1-Q4_K_M   acvram 38G   gguf 40G

## LE PIÈGE À LEVER AVANT DE MESURER

**Les tailles diffèrent : 38 Go contre 40 Go.** Le nom porte `i1-Q4_K_M` des
deux côtés, mais notre conversion est partie de ce GGUF et a pu **requantifier
dans un autre format**. Si nos poids sont en NVFP4 et les leurs en Q4_K_M, la
comparaison oppose deux quantifications, pas deux moteurs — et l'écart de
2 Go le suggère.

**À vérifier AVANT tout lancement, dans le manifeste acvram :**

    formats reellement presents (bf16 / nvfp4 / int4_awq / q3n ?)
    octets par poids de chaque cote
    et si les formats different -> la manche nue est INVALIDE en l'etat

**Issue si les formats diffèrent** : soit convertir en Q4_K_M équivalent des
deux côtés, soit déclarer la manche comme « chacun dans son meilleur format »
— ce qui est une autre question, légitime mais distincte.

## Conditions (identiques à la manche 1, sauf mention)

    ctx 1024 des deux cotes, verifie par ctx_servi ET par le journal
    SPECULATIF=none  ·  ACVRAM_PLAN_FIGE=1  ·  CUDA_VISIBLE_DEVICES=0
    LLAMACPP_BIN=~/llama.cpp/build/bin/llama-server  (e34f042)
    5 passages, le premier jete  ·  200 jetons de chaque cote
    critere d'acceptation a 10 conditions

**Nouveau et propre à cette manche** : le placement doit être **publié des deux
côtés**, pas seulement contrôlé.

    acvram    : N couches exilees, N MLP en RAM hote, embed/lm_head
    llama.cpp : n_gpu_layers effectif, et ce qui reste sur le processeur

**Sans ces deux lignes le résultat n'est pas interprétable** : un moteur qui
décharge 10 couches et un qui en décharge 40 ne font pas le même travail.

## Ce que la manche peut rendre

    acvram nettement devant  -> le tiering vaut ce qu'il coute, c'est notre creneau
    ecart faible             -> notre avantage architectural n'est pas realise
    acvram derriere          -> la conception ne tient pas sur son propre terrain

**Écrit avant la mesure, et aucune ligne ne se renégocie après lecture.**

## Contrainte machine

poste2 mesure le trafic mémoire sur la 5090 (chantier énergie). **Cette manche
ne se lance pas avant qu'elle ait rendu la carte.** Un 70B exilé épingle
beaucoup de RAM hôte : borner à `MemoryMax=24G MemorySwapMax=0` et vérifier
que le PC reste utilisable — contrainte dure de l'utilisateur.

---

# LE PIEGE EST CONFIRME, ET IL EST DOUBLE (chef, 9/09)

Manifeste de `Llama-3.3-70B-Instruct-heretic.i1-Q4_K_M` cote acvram :

    couches 80   poids 37,65 Gio
    formats     : nvfp4 59 couches + int4_awq 21 couches
    mlp_storage : cuda:0 59 couches + cuda:1 21 couches
    embed_device: cpu        lm_head_device: cuda:0

## 1. Le nom n'est pas le contenu

Le dossier porte `i1-Q4_K_M` — le nom du GGUF source — mais **le contenu est
nvfp4 + int4_awq**, deux formats qui ne sont ni l'un ni l'autre du Q4_K_M.
L'ecart de 38 contre 40 Go n'etait pas un arrondi : **ce sont deux
quantifications differentes sous une meme etiquette**.

**Consequence** : la manche nue opposerait deux quantifications, pas deux
moteurs. Et deux formats **melanges** d'un cote contre un format homogene de
l'autre.

## 2. Le plan utilise DEUX cartes

`mlp_storage` place 21 couches sur `cuda:1` — la 3080 Ti. **Le plan de ce
modele est un attelage**, pas une carte. C'est precisement ce que nous avons
passe la journee a eliminer sur Qwen.

**A trancher avant toute mesure** : la manche 2 compare-t-elle
* acvram sur **deux cartes** contre llama.cpp sur deux cartes (`--split-mode`) ?
* ou acvram sur **une carte avec exil vers l'hote** contre llama.cpp sur une
  carte avec dechargement CPU ?

**Ce sont deux questions differentes**, et la seconde est celle que le
protocole annonce (« que rapporte le tiering »). Avec `CUDA_VISIBLE_DEVICES=0`
et `ACVRAM_PLAN_FIGE=1`, le plan fige demanderait `cuda:1` qui n'existe pas —
**il faut savoir ce que fait le chargeur dans ce cas avant de lancer**.

## 3. `embed_device: cpu` — le meme defaut qu'on a corrige sur Qwen

Corrige a la main sur Qwen le 9/09 au matin. **Present ici aussi**, et il le
sera sur tout modele converti avant le correctif de fond. La table de
plongements cote hote coute deux traversees PCIe par jeton (latence, pas
volume) — voir la note de poste2.

**Le correctif de fond n'est toujours pas ecrit** : `tiering.py:204` a
`embed_device: str = "cpu"` par defaut, voisin de `lm_head_device: "cuda:0"`.
Deux defauts opposes sur des lignes consecutives.
