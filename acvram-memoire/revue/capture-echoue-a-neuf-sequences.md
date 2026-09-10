# La capture de graphe échoue à neuf séquences — et l'échec tue le moteur

Mesuré le 10/09, sous verrou de carte. Tout ce qui suit est **mesuré**, sauf la
dernière section, marquée.

## 1. Ce qui était à vérifier, et qui est RÉFUTÉ

Mon hypothèse : le plateau de 1,287 ms par séquence viendrait du repli de
`model.py:346` — deux boucles par séquence et un `gather` qui déquantifie tout le
cache — pris quand `paged_attention` rend `None`.

**Les portes sont ouvertes, sur les deux modèles :**

    Qwen3-4B-srcgguf-nvfp4      36 couches  head_dim 128  cache int8  k_scale present
    Qwen2.5-Coder-14B-bf16-pur  48 couches  head_dim 128  cache int8  k_scale present

Et sur le témoin, l'espion des appels le confirme de bout en bout : **144 appels au
noyau paginé, dont 0 retour `None`**. Le repli n'est pas pris. **Mon hypothèse tombe**,
et le plateau garde sa cause.

## 2. Ce que la vérification a trouvé à la place

    modele                       BQ    capture
    Qwen3-4B-srcgguf-nvfp4        2    OK
    Qwen3-4B-srcgguf-nvfp4        4    OK
    Qwen3-4B-srcgguf-nvfp4        6    OK
    Qwen3-4B-srcgguf-nvfp4        8    OK
    Qwen3-4B-srcgguf-nvfp4        9    ECHEC   cudaErrorStreamCaptureInvalidated
    Qwen3-4B-srcgguf-nvfp4       10    ECHEC
    Qwen3-4B-srcgguf-nvfp4       12    ECHEC
    Qwen2.5-Coder-14B-bf16-pur   12    OK      1 capture, 3 rejeux

**Seuil net entre 8 et 9, et propre à ce modèle** : ce n'est ni le moteur en général
— le 14B capture à 12 — ni le modèle en général, qui capture jusqu'à 8.

**Le calcul, lui, est juste** : à `BQ = 9` avec `enable_cuda_graphs=False`, le
décodage se déroule sans erreur. C'est **la capture** qui échoue, pas le noyau.
L'erreur d'origine reste masquée même sous `CUDA_LAUNCH_BLOCKING=1`.

## 3. Et l'échec ne se replie pas : il tue le moteur

`graphs.py:198` :

    except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
        # capture impossible faute de VRAM : le decodage continue en eager
        # plutot que de tuer le serveur
        if "out of memory" not in str(e).lower():
            raise

**La garde ne couvre que l'OOM.** Une capture qui échoue pour toute autre raison —
comme celle-ci — relève l'exception. Le commentaire annonce « plutôt que de tuer le
serveur » ; le filtre par le texte du message ne tient cette promesse que pour un
seul cas. **`acvram serve` sur ce converti à neuf séquences ou plus est donc une
panne, pas une dégradation.**

Le correctif est d'une ligne et il est déjà écrit dans l'intention du commentaire :
se replier en eager pour **toute** exception de capture, en journalisant la cause.
Le seul risque à couvrir est qu'un échec masque un vrai défaut — d'où la
journalisation, pas le silence.

## 4. Conséquence sur le plateau de 1,287 ms — à établir, non à conclure

Ce plateau a été mesuré sur ce modèle à quatre concurrences dont douze, avec
« graphes actifs, 1 rejeu par pas, 1 capture ». **Or à douze séquences, la capture
échoue ici et le processus s'arrête.** Trois lectures possibles, aucune tranchée :

- le modèle mesuré n'est pas ce chemin exact (un homonyme) ;
- les graphes étaient désactivés dans ce montage, et « graphes actifs » a été relevé
  à une concurrence plus basse puis transporté ;
- son montage diffère du mien d'une manière qui compte — et alors la différence est
  le renseignement.

**Rien de ceci ne met le plateau en doute** : 1,287 ms par séquence reste mesuré. Ce
qui est en doute est le **régime** dans lequel il a été obtenu, et c'est justement ce
qu'il faut attacher à un chiffre pour qu'il serve.

## 5. Ce que je n'ai pas établi

La cause de l'échec de capture. Le seuil à 9 coïncide avec `num_key_value_heads = 8`
de ce modèle, ce qui **suggère** une borne franchie quelque part quand `BQ` dépasse le
nombre de têtes KV — mais c'est une coïncidence de nombres, pas une mesure, et j'ai
transporté trois chiffres hors de leur régime aujourd'hui pour ne pas en ajouter un
quatrième. Le moyen de trancher est de retrouver l'erreur d'origine, que la capture
masque : la piste est de rejouer le même lot **hors capture** avec un contrôle d'accès
mémoire, non de deviner la borne.
