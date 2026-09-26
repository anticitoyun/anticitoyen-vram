# Repli au chargement — proposition soumise à relecture

Non écrit dans l'arbre : le témoin MoE tourne. Ceci est la forme exacte à
appliquer à `acvram/engine/loader.py`, à partir de la ligne 285.

## La contrainte que le code impose

Le corps de `for lp in plan.layers:` fait environ 460 lignes, avec une branche
par architecture. Deux choses en découlent :

* on ne peut pas **dupliquer** le corps pour le rejouer après un repli ;
* on ne peut pas envelopper chaque tenseur — il y a une quarantaine de `.to()`
  dispersés, et il en manquerait un.

La reprise se fait donc par un `while` à index non incrémenté : le corps n'est
indenté que d'un cran, jamais recopié.

## La forme

```python
    bascule = False          # irréversible une fois posé
    il = 0
    while il < len(plan.layers):
        lp = plan.layers[il]
        if bascule:
            lp.mlp_storage = "cpu"
        n_couches = len(layers)          # pour défaire une couche partielle
        try:
            # ... corps existant, inchangé, indenté d'un cran ...
            pass
        except torch.OutOfMemoryError:   # jamais `as e` : le traceback
                                         # retiendrait les tenseurs de la couche
            if bascule:
                raise                    # déjà en hôte et ça échoue encore :
                                         # ce n'est plus un problème de place
            alloc_avant = torch.cuda.memory_allocated()
            res_avant = torch.cuda.memory_reserved()

            # étape 2 — faire tomber les références de la couche en cours
            del layers[n_couches:]
            caches.pop(lp.index, None)
            gc.collect()                 # les modules PyTorch font des cycles
            torch.cuda.empty_cache()     # étape 3, après seulement

            alloc_apres = torch.cuda.memory_allocated()
            res_apres = torch.cuda.memory_reserved()
            if alloc_apres >= alloc_avant and res_apres >= res_avant:
                raise RuntimeError(
                    f"repli au chargement inopérant, couche {lp.index} : "
                    f"alloué {alloc_avant} -> {alloc_apres}, "
                    f"réservé {res_avant} -> {res_apres}. Rien n'a été libéré — "
                    f"référence oubliée, cycle non collecté, ou traceback retenu. "
                    f"Le chargement échouait de toute façon ; ce message dit où.")
            if alloc_apres >= alloc_avant:
                print(f"[acvram] couche {lp.index} : OOM au premier tenseur, "
                      f"rien à libérer pour elle — normal", flush=True)

            bascule = True
            print(f"[acvram] VRAM pleine à la couche {lp.index} : le reste du "
                  f"modèle passe en RAM hôte (réservé {res_avant / 2**30:.2f} -> "
                  f"{res_apres / 2**30:.2f} Gio)", flush=True)
            continue                     # refait CETTE couche, vers l'hôte
        il += 1
```

## Ce que chaque ligne évite

| ligne | ce qu'elle empêche |
|---|---|
| `if bascule: lp.mlp_storage = "cpu"` avant le `try` | trente OOM au lieu d'un : sans drapeau, chaque couche suivante retenterait la VRAM |
| `except torch.OutOfMemoryError` sans `as` | le `__traceback__` référence les frames, donc les locales, donc les tenseurs déjà placés — `empty_cache()` ne rendrait rien, et `del e` n'est automatique qu'à la **sortie** du bloc, alors que la libération se fait dedans |
| `if bascule: raise` | boucle infinie si l'hôte lui-même échoue |
| `del layers[n_couches:]` et `caches.pop` | c'est **l'étape qui libère réellement** ; `empty_cache()` seul ne rend que les blocs non référencés |
| `gc.collect()` avant `empty_cache()` | les modules PyTorch forment des cycles ; `del` seul ne libère pas immédiatement |
| deux compteurs | `memory_allocated` mesure ce que les **références** retiennent, `memory_reserved` ce que l'**allocateur** garde : ensemble ils nomment l'étape fautive au lieu de dire « ça n'a pas marché » |
| échec dur seulement si **aucun** ne bouge | l'OOM au premier tenseur d'une couche est le cas dominant — rien n'a été placé, donc `alloc` ne baisse pas, et un contrôle sur `alloc` seul se déclencherait le plus souvent quand tout va bien |
| `continue` sans incrémenter | refait la couche **entière** vers l'hôte : aucun tenseur à moitié placé |

## Limites connues, écrites ici parce qu'elles mordent ici

* **Le repli exile les dernières couches**, puisqu'il bascule au point de
  rupture. C'est aussi ce que fait le planificateur aujourd'hui (il exile 18 à
  47), donc pas de perte. Mais s'il choisissait un jour selon un autre critère
  — fréquence de routage, taille des experts, profondeur — le repli l'ignorerait
  **en silence**.
* **Les deux compteurs ne valent que si rien d'autre n'alloue en parallèle** sur
  la même carte. La règle des trois accords avant toute mesure est ce qui le
  garantit : une discipline d'organisation devenue précondition technique.
* **Le contrôle transforme un repli en échec dur.** C'est voulu : si le repli
  est inopérant, le chargement allait échouer de toute façon, mais avec un OOM
  opaque au lieu d'un message qui nomme la cause.

## À vérifier avant de pousser

* qu'aucun `except` plus large n'englobe cette boucle en amont — un
  `except Exception` au-dessus avalerait ce qu'on prend soin de laisser passer ;
* que `gc` est importé dans le module ;
* que `plan.layers` est bien indexable (liste) et non un générateur.
