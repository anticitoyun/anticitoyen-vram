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
| `runner.py:628-631` `prefill_seconds` | **non** — `_emit` est ligne 635 | **faux** |
| `runner.py:643-664` `prefill_seconds` | **non** — ferme après `self.model()` | **faux** |
| `graphs.py` `bind`/`fill`/`replay` | **non** — rien entre les bornes | **faux** |
| `app.py` (7 sites) | sans objet — latence HTTP, CPU pur | légitime |

**`decode_seconds` et `prefill_seconds` sont dans le même fichier, publiées par
la même classe, écrites de la même façon. La différence est dans l'ORDRE : quatre
lignes de position séparent une statistique juste d'une statistique fausse.**

## Ce qui tient, ce qui tombe

```
TIENNENT   tout chiffre bati sur decode_seconds : les +88 % / +104 % de la
           tranche adaptative, les debits du duel, les pas/s
A REPRENDRE  TTFT, jetons/s de prefill, et la reference 6 420 j/s du chantier
           GEMM groupee NVFP4
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
