"""L'echelle de sortie varie-t-elle d'un tenseur a l'autre ? Sans carte.

`out_ref_norm` = ||diag(act) @ w.T||_F, donc ||y_ref||^2 = sum_i act[i]^2 ||w[:,i]||^2.
Ni `act` ni les poids ne sont au manifeste, et AUCUN manifeste existant ne
porte le champ : la simulation exacte de `absolu` est IMPOSSIBLE sur les
donnees deja collectees.

SUBSTITUT ASSUME : sous act ~ constant, ||y_ref|| est proportionnel a ||w||_F.
On lit donc les normes de Frobenius des poids SOURCE, sur processeur. C'est une
APPROXIMATION — elle repond a la question qui decide (l'echelle varie-t-elle de
plusieurs ordres de grandeur ?) et pas au classement exact.
"""
import glob, json, os, struct, sys
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
import numpy as np
SRC=f"{MODELES}/Llama-2-7b-hf"
CIBLE=set()
m=json.load(open(f"{MODELES}/Llama-2-7b-quota-6g00/acvram_manifest.json"))
for k,v in m["tensors"].items():
    if v["format"] in ("int8","nvfp4"): CIBLE.add(k)
normes={}
lus=0
for f in sorted(glob.glob(os.path.join(SRC,"*.safetensors"))):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n)); deb=8+n
        for k,v in h.items():
            if k=="__metadata__" or k not in CIBLE: continue
            a,b=v["data_offsets"]; fh.seek(deb+a); brut=fh.read(b-a)
            if v["dtype"]=="BF16":
                u=np.frombuffer(brut,dtype="<u2").astype(np.uint32)<<16
                arr=u.view(np.float32)
            elif v["dtype"]=="F16":
                arr=np.frombuffer(brut,dtype="<f2").astype(np.float32)
            else:
                arr=np.frombuffer(brut,dtype="<f4")
            normes[k]=float(np.linalg.norm(arr)); lus+=1
    fd=os.open(f,os.O_RDONLY)
    try: os.posix_fadvise(fd,0,os.fstat(fd).st_size,os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)
print(f"{lus} tenseurs lus sur {len(CIBLE)} cibles")
vals=np.array(list(normes.values()))
print(f"||w||_F : min {vals.min():.4g}  median {np.median(vals):.4g}  max {vals.max():.4g}")
print(f"          rapport max/min = {vals.max()/vals.min():.1f}x")
print(f"          ecart-type / moyenne = {vals.std()/vals.mean():.3f}")
q=np.percentile(vals,[5,25,50,75,95])
print(f"          quantiles 5/25/50/75/95 : " + "  ".join(f"{x:.4g}" for x in q))
# Sortie relative a la racine du depot, pas a un chemin de poste : le fichier
# pointait vers le dossier acvram-memoire SANS DISTANT (le doublon du 10/09).
_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SORTIE = os.environ.get("ACVRAM_NORMES_SORTIE",
                         os.path.join(_RACINE, "acvram-memoire", "corpus", "normes-poids-source.json"))
json.dump(normes, open(_SORTIE, "w"))
print("\npar type de projection :")
import os
import re, collections
par=collections.defaultdict(list)
for k,v in normes.items():
    mm=re.search(r"\.(\w+_proj)\.weight$",k)
    if mm: par[mm.group(1)].append(v)
for t,vs in sorted(par.items(), key=lambda kv:-np.median(kv[1])):
    a=np.array(vs); print(f"  {t:12s} n={len(a):3d}  median {np.median(a):8.3f}  min {a.min():8.3f}  max {a.max():8.3f}")
