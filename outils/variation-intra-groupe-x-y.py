"""Les 76 ont-ils une variation INTRA-GROUPE plus forte que les 27 ?

Si oui, le mecanisme de l hypothese A tient : l echelle int8 (une par 128) ne
peut pas suivre ce que l echelle nvfp4 (une par 16) rattrape. Lu sur les poids
SOURCE, sans carte, AVANT toute conversion.

Mesure : pour chaque tenseur, rapport entre l amplitude max du groupe de 128 et
l amplitude max des sous-blocs de 16 qu il contient. Un rapport eleve dit qu un
seul aberrant domine son groupe de 128 alors que les blocs de 16 le confinent.
"""
import glob, json, os, struct
import numpy as np
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402
BASE=MODELES
SRC=os.path.join(BASE,"Llama-2-7b-hf")
groupes=json.load(open(sorties() / "groupes-x-y.json"))
X,Y=set(groupes["X_76"]),set(groupes["Y_27"])
CIBLE=X|Y
def lire(f,h,k,deb,fh):
    v=h[k]; a,b=v["data_offsets"]; fh.seek(deb+a); brut=fh.read(b-a)
    if v["dtype"]=="BF16":
        u=np.frombuffer(brut,dtype="<u2").astype(np.uint32)<<16; arr=u.view(np.float32)
    elif v["dtype"]=="F16": arr=np.frombuffer(brut,dtype="<f2").astype(np.float32)
    else: arr=np.frombuffer(brut,dtype="<f4")
    return arr.reshape(v["shape"])
res={}
for f in sorted(glob.glob(os.path.join(SRC,"*.safetensors"))):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n)); deb=8+n
        for k in list(h):
            if k=="__metadata__" or k not in CIBLE: continue
            w=np.abs(lire(f,h,k,deb,fh))
            out,inn=w.shape
            n128=inn//128
            if n128==0: continue
            w=w[:,:n128*128]
            g128=w.reshape(out,n128,128).max(axis=2)            # amplitude par groupe de 128
            g16=w.reshape(out,n128,8,16).max(axis=3)            # amplitude par bloc de 16
            # rapport moyen : combien l aberrant du groupe depasse la moyenne des blocs
            rap=(g128/np.maximum(g16.mean(axis=2),1e-12)).mean()
            res[k]=float(rap)
    fd=os.open(f,os.O_RDONLY)
    try: os.posix_fadvise(fd,0,os.fstat(fd).st_size,os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)
x=np.array([res[k] for k in X if k in res]); y=np.array([res[k] for k in Y if k in res])
print(f"lus : {len(x)}/{len(X)} du groupe X (les 76), {len(y)}/{len(Y)} du groupe Y (les 27)")
print()
print(f"{'groupe':30s}{'n':>4s}{'median':>10s}{'moyenne':>10s}{'min':>8s}{'max':>8s}")
print(f"{'X = promus par A seul (76)':30s}{len(x):4d}{np.median(x):10.4f}{x.mean():10.4f}{x.min():8.4f}{x.max():8.4f}")
print(f"{'Y = promus par B seul (27)':30s}{len(y):4d}{np.median(y):10.4f}{y.mean():10.4f}{y.min():8.4f}{y.max():8.4f}")
print()
d=(np.median(x)-np.median(y))/np.median(y)*100
print(f"ecart des medianes : {d:+.2f} %")
print(f"PREDICTION DU MECANISME A : X doit avoir une variation intra-groupe PLUS FORTE.")
print(f"  => mecanisme {'COMPATIBLE' if np.median(x)>np.median(y) else 'INFIRME'} avec ces donnees")
