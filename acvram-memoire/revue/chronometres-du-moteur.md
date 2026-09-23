# Les 26 chronomètres du moteur, et lesquels mesurent un travail

**10/09/2026.** Trouvé par `97` en profilant `bind/fill/replay`, circonscrit par
`0a` en suivant le chemin d'appel, complété sur le prefill.

## Le défaut

```python
# graphs.py:317
entry["graph"].replay()          # aucun synchronize hors ACVRAM_TRACE_PTRS
entry["out"].clone()             # lancement asynchrone lui aussi
t3 = time.perf_counter()         # mesure le LANCEMENT, pas l'execution
```

**CUDA rend la main au CPU avant que le noyau tourne.** Un `time.perf_counter()`
autour d'une opération CUDA ne mesure donc rien de réel — **sauf si la fenêtre
contient une synchronisation ou un transfert vers l'hôte** (`.tolist()`,
`.item()`, `.cpu()`, `cuda.synchronize()`).

Traces mesurées par `97` :

```
len=680   chemin graphe entier  0,9 ms      emit  146,4 ms
len=687   chemin graphe entier  0,9 ms      emit  148,3 ms
```

**Le GPU réel est facturé au premier point de synchronisation en aval, et porte
son nom.** `emit` n'est pas 148 ms d'échantillonnage : c'est du travail GPU mal
étiqueté.

## Le tri — il se fait LIGNE PAR LIGNE, pas fichier par fichier

| site | synchronisation dans la fenêtre | verdict |
|---|---|---|
| `runner.py:678-684` `decode_seconds` | `_plain_decode` → `_emit` → `.tolist()` | **juste** |
| `runner.py:628-631` `prefill_seconds` | lecture : non — **mesure du 11/09 : OUI**, +0,09 % sous sync | **juste** (réhabilitée) |
| `runner.py:643-664` `prefill_seconds` | idem — mesure du 11/09 | **juste** (réhabilitée) |
| `graphs.py` `bind`/`fill`/`replay` | **non** — rien entre les bornes | **faux** |
| `runner.py:711/713/720` `t0..t2` | **non** — batch, `graphs.run` | **faux**, jamais publié |
| `runner.py:724` `t3` | oui, après `_emit` | juste |
| `graphs.py:241/297/317` | **non** | **faux** |
| `bench.py:388-390` « mur » | oui — `generate()` consommé par un `sum(1 for _ in …)` | juste, mais englobe le chargement |
| `app.py` (7 sites) | sans objet — latence HTTP, CPU pur | légitime |

**`decode_seconds` et `prefill_seconds` sont dans le même fichier, publiées par
la même classe, écrites de la même façon. La différence est dans l'ORDRE : quatre
lignes de position séparent une statistique juste d'une statistique fausse.**

## Ce qui tient, ce qui tombe

```
TIENNENT   tout chiffre bati sur decode_seconds : les +88 % / +104 % de la
           tranche adaptative, les debits du duel, les pas/s
TIENNENT   TTFT, jetons/s de prefill, la reference 6 420 j/s — REHABILITES le
           11/09 : prefill_seconds ne bouge pas sous sync (+0,09 %), donc une
           synchronisation existait deja dans la fenetre. La lecture disait
           « faux », la mesure dit « juste ». La mesure tranche.
FAUSSE     la decomposition bind / fill / replay
```

**`prefill_seconds` sous-estime — elle rend moins que le vrai, jamais plus.
Donc tout GAIN de prefill mesuré avec elle est SURESTIMÉ.**

## Le contrôle, et il peut rendre « faux »

```python
if os.environ.get("ACVRAM_CHRONO_SYNC"):
    torch.cuda.synchronize()
```

**Sous variable d'environnement, jamais en dur** : un `synchronize` par rejeu
sérialise le pipeline — il ne révèle pas seulement le coût, il en crée.

```
attendu     replay monte, emit s effondre D AUTANT (somme constante)
REFUTATION  replay monte SANS que emit baisse
            -> on a AJOUTE du cout, l hypothese tombe

attendu     prefill_seconds MONTE
REFUTATION  prefill_seconds ne bouge pas
            -> il y avait une synchronisation que le grep ne voit pas
```

## Ce que l'épisode enseigne

**Le même geste avait déjà été corrigé — dans le banc, le 9/09, parce qu'il
ajoutait ~9 µs par appel.** On l'a retiré de l'instrument et jamais du moteur.
`git grep perf_counter` aurait rendu `graphs.py:317` ce jour-là.

**Quand on corrige un défaut de méthode, chercher le même geste PARTOUT, pas
seulement là où il a fait mal.**

**Et l'inverse est une faute symétrique** (`0a`) : *retirer plus que nécessaire
est aussi une faute de mesure*. Sans suivre le chemin jusqu'à `.tolist()`, les
deux gains principaux de la journée auraient été retirés par excès de prudence.
**La prudence n'est pas une méthode ; suivre le chemin en est une.**


## Trois choses que la table dit et que la liste ne disait pas

**1. Le jumeau a lui-même un jumeau.** Le prefill découpé (`runner.py:643-664`)
répète la même faute vingt lignes plus bas, sur la même statistique. **Ce ne sont
pas deux erreurs : c'est une erreur dupliquée avec le code.**

**2. Quatre des treize sites ne servent qu'aux messages de diagnostic**
(`[pas-lent]`, `[graphe-lent]`, sous `ACVRAM_TRACE_STEPS`). Ils sont faux, et
**leur fausseté est plus pernicieuse que celle des statistiques** : ce sont
précisément les nombres qu'on lit quand on cherche pourquoi c'est lent. **Un
instrument de diagnostic faux envoie chercher au mauvais endroit** — c'est ce qui
s'est passé ce soir avec « avant < 1 ms, emit 120-148 ms ».

**3. `decode_seconds` est juste par accident de position, pas par conception.**
Sa validité dépend d'un `.tolist()` situé dans `_emit`, qu'une refonte pourrait
déplacer sans que personne fasse le lien. **Le remède honnête est donc un
`synchronize()` explicite plutôt qu'une dépendance tacite à un transfert** : il
dit ce qu'il fait.

**Le remède, partout, tient en un mot : fermer la fenêtre APRÈS le retour vers
l'hôte.**

## Mesure du 11/09 — le contrôle `ACVRAM_CHRONO_SYNC` (Laure, 4 bras × 15 tours)

GLM-4.7-42B MLA nvfp4, `slots=12`, `ctx=2048`, 5090.

```
bras  sync  replay_ms  pas_ms   prefill_s  tok/s
A1    non    0,017     18,829   0,0732     54,65
A2    non    0,016     18,819   0,0730     54,76
B1    oui   18,482     18,791   0,0731     54,69
B2    oui   18,590     18,879   0,0732     54,60
```

* **`replay` : 0,016 → 18,5 ms (×1137).** Sans sync on mesurait le lancement.
  **Le rejeu du graphe est 98 % du pas** ; bind, fill et emit sont négligeables.
* **`pas_total` : stable à 0,1 %.** Le total encaissait déjà le GPU.
* **`prefill_seconds` : +0,09 %.** Une synchronisation existait dans la fenêtre
  que la lecture n'avait pas vue. **Condamnation d'hier annulée.**
* **Le `synchronize` ne coûte rien de mesurable** (pas_total identique) : le
  garde peut rester éteint par défaut sans perdre d'information, ou allumé
  pour les diagnostics sans fausser le total.

**Ce que ça enseigne, une fois de plus : la lecture avait raison sur `graphs.py`
et tort sur `prefill_seconds`, et rien dans la lecture ne distinguait les deux
cas.** Les dimensions absentes se découvrent par la mesure.
