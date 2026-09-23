#!/usr/bin/env python3
"""Inventaire acvram pondere par OCTETS (pas par nombre de tenseurs) + doublons."""
import json, os, collections, hashlib
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

A = MODELES
ETIQ = ["q4_k_m","q4_k_xl","q4_k_l","q5_k_m","q5_k_s","q6_k","q8_0","q5_k","q4_k","q3_k_s",
        "iq3m","exl3","awq","bf16","nvfp4","gguf","bpw","q4_k_s","ud-q5_k_m","i1"]

def octets(d):
    t = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t

def sig(d):
    """empreinte bon marche : tailles exactes des safetensors, triees"""
    ts = sorted(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                if f.endswith(".safetensors"))
    return hashlib.sha256(str(ts).encode()).hexdigest()[:16] if ts else ""

print("nom\tfmt_octets\trepartition_octets\tGio\tetiq_nom\tconforme\tsignature")
for nom in sorted(os.listdir(A)):
    d = os.path.join(A, nom)
    if not os.path.isdir(d): continue
    mf = os.path.join(d, "acvram_manifest.json")
    tot = octets(d)
    if not os.path.isfile(mf):
        print(f"{nom}\tSANS-MANIFESTE\t\t{tot/2**30:.2f}\t\tNON\t{sig(d)}"); continue
    try: m = json.load(open(mf))
    except Exception: print(f"{nom}\tILLISIBLE\t\t{tot/2**30:.2f}\t\tNON\t{sig(d)}"); continue
    par = collections.Counter()
    for t in (m.get("tensors") or {}).values():
        if not isinstance(t, dict): continue
        f = t.get("format") or t.get("quant") or t.get("dtype") or "?"
        n = t.get("nbytes") or t.get("bytes") or t.get("size")
        if n is None:
            sh = t.get("shape") or []
            n = 1
            for x in sh: n *= x
            # Un format par blocs ne coute pas que ses bits de poids : NVFP4
            # ajoute une echelle e4m3 par bloc de 16, soit 4,5 bits par poids
            # et non 4. Compter 0,5 sous-estimait le nvfp4 de 12,5 % et faisait
            # passer des modeles nvfp4 pour des modeles a dominante int8 ou
            # bf16. Verifie le 9/09/2026 contre la taille des safetensors :
            # 8,781 Gio calcules contre 8,785 mesures, soit 0,05 % d ecart,
            # la ou l ancien compte donnait 7,967.
            n *= {"bf16": 2, "fp16": 2, "int8": 1.0625,
                  "nvfp4": 0.5625, "int4_awq": 0.5625}.get(str(f), 2)
        par[str(f)] += int(n)
    if not par: print(f"{nom}\t?\t\t{tot/2**30:.2f}\t\t?\t{sig(d)}"); continue
    T = sum(par.values())
    dom = par.most_common(1)[0][0]
    mix = "+".join(f"{k}:{100*v/T:.0f}%" for k, v in par.most_common(4))
    bas = nom.lower()
    porte = [e for e in ETIQ if e in bas]
    # conforme si le nom ne porte AUCUNE etiquette, ou porte exactement le format dominant
    conf = "OUI" if (not porte or porte == [dom]) else "NON"
    print(f"{nom}\t{dom}\t{mix}\t{tot/2**30:.2f}\t{','.join(porte)}\t{conf}\t{sig(d)}")
