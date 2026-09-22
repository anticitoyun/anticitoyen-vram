"""Prix du doublon de fusion sur le parc, par arithmetique de manifestes.

BORNE SUPERIEURE : le manifeste ne dit pas si _scaler_commun aurait accepte la
paire (il compare des tenseurs act_scale par torch.equal). Sur
Llama-2-7b-int8, 5 fusions sur 32 ont reellement abouti — la borne y vaut donc
6,4 fois le reel. Le chiffre ci-dessous dit « au plus », jamais « exactement ».
"""
import json, glob, os, re
from collections import defaultdict
import sys as _s, pathlib as _p  # noqa: E401
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)

_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

RACINES = [MODELES,
           _RACINE]
# Seuls int8 et int4_awq dupliquaient : bf16 et nvfp4 repointaient deja en vues.
TOUCHES = {"int8", "int4_awq"}
GROUPES = (("gate_proj", "up_proj"), ("q_proj", "k_proj", "v_proj"))
CAPACITE = {"5090": 31.36, "3080Ti": 11.63}

def octets(fmt, forme, g=128):
    out, inn = forme[0], forme[1]
    ng = max(1, inn // g)
    if fmt == "int8":
        return out * inn + out * ng * 2 + out * ng
    if fmt == "int4_awq":
        return out * (inn // 2) + out * ng * 2 + out * (ng // 2)
    return 0

lignes = []
for r in RACINES:
    for m in sorted(glob.glob(os.path.join(r, "*", "acvram_manifest.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        t = d.get("tensors", {})
        g = (d.get("options") or {}).get("group_size", 128)
        par_couche = defaultdict(dict)
        for nom, e in t.items():
            if e.get("format") not in TOUCHES:
                continue
            mm = re.match(r"(.*layers\.\d+)\.(?:self_attn|mlp)\.(\w+)\.weight$", nom)
            if mm:
                par_couche[mm.group(1)][mm.group(2)] = e
        gagne = 0
        n_paires = 0
        for couche, projs in par_couche.items():
            for grp in GROUPES:
                if all(p in projs for p in grp):
                    formes = [projs[p]["shape"] for p in grp]
                    if len({f[1] for f in formes}) != 1:
                        continue
                    gagne += sum(octets(projs[p]["format"],
                                        projs[p]["shape"],
                                        projs[p].get("group_size", g))
                                 for p in grp)
                    n_paires += 1
        if gagne:
            total = sum(os.path.getsize(f) for f in
                        glob.glob(os.path.join(os.path.dirname(m), "*.safetensors")))
            lignes.append((gagne, n_paires, total, os.path.basename(os.path.dirname(m))))

lignes.sort(reverse=True)
print(f"{'modele':52s} {'dossier':>9s} {'au plus':>9s} {'%':>6s} {'groupes':>8s}")
for gagne, n, total, nom in lignes:
    print(f"{nom[:52]:52s} {total/2**30:8.2f}G {gagne/2**30:8.2f}G "
          f"{100*gagne/total:5.1f}% {n:8d}")
print(f"\n{len(lignes)} modeles concernes (int8 ou int4_awq)")
if lignes:
    print(f"gain au plus, cumule : {sum(l[0] for l in lignes)/2**30:.2f} Gio")
    print("\nBASCULE DE PLACEMENT sur une 5090 de 31,36 Gio "
          "(dossier + doublon contre capacite) :")
    for gagne, n, total, nom in lignes:
        avec = (total + gagne) / 2**30
        sans = total / 2**30
        if sans <= CAPACITE["5090"] < avec:
            print(f"  BASCULE  {nom[:46]:46s} {sans:6.2f}G tient, "
                  f"{avec:6.2f}G ne tient pas")
    print("  (aucune ligne ci-dessus = le doublon ne fait basculer aucun modele)")
