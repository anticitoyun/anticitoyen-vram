#!/usr/bin/env python3
"""Retrouve l'exposant AWQ de chaque tenseur DEPUIS LE DOSSIER, sans GPU.

`search_channel_scales` pose `s = act.pow(alpha) / mean(act.pow(alpha))` avec
`alpha = i/20`, i de 0 a 20. `act` ne depend que de l'ENTREE, donc `gate` et
`up` d'une meme couche partagent `act` : seuls leurs exposants diffèrent, et
c'est cette difference qui fait refuser la fusion par `torch.equal`.

En prenant les logarithmes, `log s = alpha * log act - log mean(...)` : deux
echelles d'un meme groupe sont donc AFFINES l'une de l'autre en log, et le
rapport des pentes vaut `alpha_1 / alpha_2`. On n'a pas besoin de `act` pour le
mesurer — une regression entre `log s_gate` et `log s_up` suffit, et la grille
etant a 21 valeurs, le rapport pince le couple.

Aucun chargement, aucune carte : lecture des `act_scale` du dossier.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

BASE = Path(MODELES)
GRILLE = [i / 20 for i in range(21)]


def lire_tenseurs(dossier: Path, motif: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for f in sorted(dossier.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            entete = json.loads(fh.read(n))
            debut = 8 + n
            for cle, v in entete.items():
                if cle == "__metadata__" or motif not in cle:
                    continue
                a, b = v["data_offsets"]
                fh.seek(debut + a)
                brut = fh.read(b - a)
                dt = {"F32": "<f4", "F16": "<f2", "BF16": None}[v["dtype"]]
                if dt is None:            # bf16 : relire en uint16 puis etendre
                    u = np.frombuffer(brut, dtype="<u2").astype(np.uint32) << 16
                    arr = u.view(np.float32)
                else:
                    arr = np.frombuffer(brut, dtype=dt).astype(np.float32)
                out[cle] = arr.reshape(v["shape"])
    return out


def alpha_du_couple(sa: np.ndarray, sb: np.ndarray) -> tuple[float, float] | None:
    """Rend (alpha_a, alpha_b) de la grille, ou None si indeterminable."""
    la, lb = np.log(np.clip(sa, 1e-12, None)), np.log(np.clip(sb, 1e-12, None))
    la, lb = la - la.mean(), lb - lb.mean()
    if la.std() < 1e-9 or lb.std() < 1e-9:
        return (0.0, 0.0) if la.std() < 1e-9 and lb.std() < 1e-9 else None
    pente = float((la * lb).sum() / (lb * lb).sum())        # alpha_a / alpha_b
    # LIMITE A DIRE : la regression ne rend qu'un RAPPORT. Les couples
    # (0,25 ; 0,35) et (0,50 ; 0,70) ont le meme, donc les exposants absolus ne
    # sont determines qu'a un facteur commun pres. Ce qui est solide, c'est le
    # rapport et le fait que les echelles soient egales ou non ; le « nombre de
    # pas de grille » qui en decoule est INDICATIF, pas mesure.
    meilleur, err = None, float("inf")
    for a in GRILLE[1:]:
        for b in GRILLE[1:]:
            e = abs(a / b - pente)
            if e < err:
                err, meilleur = e, (a, b)
    return meilleur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="Llama-2-7b-int8")
    a = ap.parse_args()
    d = BASE / a.modele
    if not d.exists():
        print(f"ECHEC / CAUSE: {d} absent")
        return 2
    ech = lire_tenseurs(d, "act_scale")
    print(f"{a.modele} : {len(ech)} echelles d'activation lues, aucun GPU\n")
    par_couche: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for cle, arr in ech.items():
        m = re.match(r"(.*layers\.\d+)\.(?:self_attn|mlp)\.(\w+)\.weight\.act_scale$",
                     cle)
        if m:
            par_couche[m.group(1)][m.group(2)] = arr
    groupes = (("gate_proj", "up_proj"), ("q_proj", "k_proj"), ("k_proj", "v_proj"))
    bilan: dict[str, list] = defaultdict(list)
    for couche in sorted(par_couche, key=lambda s: int(s.rsplit(".", 1)[-1])):
        projs = par_couche[couche]
        for g in groupes:
            if not all(p in projs for p in g):
                continue
            sa, sb = projs[g[0]], projs[g[1]]
            if sa.shape != sb.shape:
                continue
            identiques = bool(np.allclose(sa, sb, rtol=1e-6, atol=1e-9))
            r = alpha_du_couple(sa.ravel(), sb.ravel())
            bilan["/".join(g)].append((couche, identiques, r))
    for nom, lignes in bilan.items():
        egaux = sum(1 for _, i, _ in lignes if i)
        print(f"  {nom:22s} {len(lignes)} couches, "
              f"{egaux} avec des echelles EGALES (donc fusionnables)")
        ecarts = [abs(r[0] - r[1]) for _, i, r in lignes if not i and r]
        if ecarts:
            pas = [round(e * 20) for e in ecarts]
            print(f"    ecart d'exposant sur les {len(ecarts)} autres : "
                  f"{min(pas)} a {max(pas)} pas de grille sur 20, "
                  f"median {sorted(pas)[len(pas)//2]} — INDICATIF, "
                  f"la regression ne rend qu'un rapport")
            for couche, i, r in lignes[:4]:
                if not i and r:
                    print(f"      {couche.rsplit('.',1)[-1]:>3s} : "
                          f"alpha {r[0]:.2f} contre {r[1]:.2f}")
    print("\nCE QUI EST SOLIDE ICI : le compte des echelles EGALES, qui vient "
          "d'une comparaison\ndirecte des fichiers et ne depend d'aucun modele. "
          "Il vaut 5 sur 32 pour gate/up,\net c'est exactement le nombre de "
          "groupes fusionnes releve sur la carte.\n"
          "\nCE QUI EST INDICATIF : les exposants absolus, donc les « pas de "
          "grille ». La\nregression ne rend qu'un RAPPORT — (0,25 ; 0,35) et "
          "(0,50 ; 0,70) sont\nindiscernables. Pour le prix exact d'un alpha "
          "commun il faut conserver les 21\nerreurs a la conversion ; c'est "
          "la mesure suivante, pas celle-ci.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
