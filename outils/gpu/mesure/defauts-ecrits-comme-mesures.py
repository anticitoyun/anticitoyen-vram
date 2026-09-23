"""Quels champs du manifeste portent un DEFAUT là où rien n'a été mesuré ?

Question de 1c le 10/09 : `mamba_state_size` a été trouvé parce qu'il gênait.
Les autres ne gênent personne pour l'instant. Un défaut de dataclass écrit dans
le manifeste est indiscernable d'une valeur lue — et tout inventaire bâti sur
les manifestes le compte comme mesuré.

Le critère est double, et il faut les DEUX :
  1. la valeur du manifeste égale le défaut de `ModelSpec`
  2. la clé est ABSENTE du `config.json` source (donc rien ne l'a fournie)

Un champ qui vaut son défaut ET que la source déclare n'est pas en cause : la
source dit la même chose que le défaut. C'est la seconde condition qui
distingue « coïncidence » de « invention ».

Ne mesure rien, ne touche pas la carte : lit 126 manifestes et leurs configs.
"""
import dataclasses
import glob
import json
import os
import sys
from collections import Counter, defaultdict
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from acvram.engine.config import ModelSpec  # noqa: E402

BASE = os.environ.get(
    "ACVRAM_PARC", _RACINE)

DEFAUTS = {}
for ch in dataclasses.fields(ModelSpec):
    if ch.default is not dataclasses.MISSING:
        DEFAUTS[ch.name] = ch.default
    elif ch.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            DEFAUTS[ch.name] = ch.default_factory()      # type: ignore[misc]
        except Exception:
            pass

# Les alias reellement lus par load_model_spec : une cle du manifeste peut
# venir d'un AUTRE nom dans la source. Sans cette table, `mamba_state_size`
# serait declare invente sur un modele qui ecrit `ssm_state_size`.
ALIAS = {
    "mamba_state_size": ("mamba_state_size", "ssm_state_size"),
    "num_experts": ("num_experts", "num_local_experts", "n_routed_experts"),
    "num_experts_per_tok": ("num_experts_per_tok", "top_k"),
    "num_layers": ("num_hidden_layers", "n_layer"),
    "hidden_size": ("hidden_size", "n_embd"),
    "num_attention_heads": ("num_attention_heads", "n_head"),
    "rms_norm_eps": ("rms_norm_eps", "norm_epsilon"),
}

mans = sorted(glob.glob(os.path.join(BASE, "*", "acvram_manifest.json")))
if not mans:
    sys.exit(f"aucun manifeste sous {BASE}")

# COMPTE DE TRAITEMENT, pas de fourniture : un balayage rend un resultat sur ce
# qu'il a pu traiter, jamais sur ce qu'on lui a donne. L'ecart s'imprime.
donnes = len(mans)
lus = apparies = 0
sans_config = []
illisibles = []
invente = Counter()
concerne = Counter()
exemples = defaultdict(list)

for m in mans:
    d = os.path.dirname(m)
    try:
        man = json.load(open(m))
    except Exception as e:
        illisibles.append((os.path.basename(d), type(e).__name__))
        continue
    lus += 1
    cfg_p = os.path.join(d, "config.json")
    if not os.path.exists(cfg_p):
        sans_config.append(os.path.basename(d))
        continue
    try:
        cfg = json.load(open(cfg_p))
    except Exception as e:
        illisibles.append((os.path.basename(d) + "/config", type(e).__name__))
        continue
    for cle in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(cle), dict):
            cfg = {**cfg[cle], **{k: v for k, v in cfg.items() if k not in cfg[cle]}}
            break
    apparies += 1
    mo = man.get("model") or {}
    for champ, defaut in DEFAUTS.items():
        if champ not in mo:
            continue
        concerne[champ] += 1
        if mo[champ] != defaut:
            continue
        noms = ALIAS.get(champ, (champ,))
        if any(cfg.get(n) is not None for n in noms):
            continue          # la source le declare : coincidence, pas invention
        invente[champ] += 1
        if len(exemples[champ]) < 2:
            exemples[champ].append(os.path.basename(d))

print(f"manifestes DONNES     : {donnes}")
print(f"  LUS                 : {lus}")
print(f"  APPARIES a un config: {apparies}")
print(f"  sans config.json    : {len(sans_config)} {sans_config[:3]}")
print(f"  illisibles          : {len(illisibles)} {illisibles[:3]}")
print(f"  ECART donnes-apparies : {donnes - apparies}")
print()
print(f"{'champ':34s}{'defaut':>14s}{'invente':>9s}{'/present':>9s}")
print("-" * 68)
for champ, n in invente.most_common():
    print(f"{champ:34s}{str(DEFAUTS[champ])[:14]:>14s}{n:9d}{concerne[champ]:9d}"
          f"   ex. {exemples[champ][0][:26]}")
print()
print(f"champs a defaut ECRIT SANS SOURCE : {len(invente)} sur "
      f"{len(DEFAUTS)} champs a defaut")
print(f"total d ecritures inventees : {sum(invente.values())}")
