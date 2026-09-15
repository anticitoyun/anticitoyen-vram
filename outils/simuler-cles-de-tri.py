import json, math
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402
BASE=Path(MODELES)
DOS=[(0.0,"Llama-2-7b-nvfp4"),(4.50,"Llama-2-7b-quota-4g50"),(5.00,"Llama-2-7b-quota-5g00"),
     (5.50,"Llama-2-7b-quota-5g50"),(6.00,"Llama-2-7b-quota-6g00"),(6.55,"Llama-2-7b-quota-plafond")]
GIO=1024**3; BUDGET=6.00
S=json.load(open(sorties() / "normes-poids-source.json"))
def octets(v):
    out,inn=v["shape"][0],v["shape"][1]; g=v.get("group_size") or 128; ng=max(1,inn//g)
    if v["format"]=="int8": return out*inn+out*ng*2+out*ng
    if v["format"]=="nvfp4": return out*(inn//2)+out*(inn//16)
    return int(out*inn*v.get("bpw",16.0)/8)
err=lambda s:10.0**(-s/20.0)
et={b:json.load(open(BASE/n/"acvram_manifest.json")) for b,n in DOS}
bs=sorted(et); tens={b:et[b]["tensors"] for b in bs}
plancher=et[6.00]["budget"]["plancher_gib"]
cand=[]
for k in sorted(tens[bs[0]]):
    base=None
    for b in bs:
        v=tens[b].get(k)
        if v is None: break
        if v["format"]=="nvfp4": base=v
        elif v["format"]=="int8" and base:
            if k in S:
                cand.append({"nom":k,"sb":base["out_snr_db"],"sp":v["out_snr_db"],
                             "cout":max(octets(v)-octets(base),1),"ech":S[k]})
            break
reste_total=BUDGET*GIO-plancher*GIO
cles={
 "snr":     lambda c:-((c["sp"]-c["sb"])/c["cout"]),
 "inverse": lambda c:+((c["sp"]-c["sb"])/c["cout"]),
 "erreur":  lambda c:-((err(c["sb"])-err(c["sp"]))/c["cout"]),
 "absolu~": lambda c:-(c["ech"]*(err(c["sb"])-err(c["sp"]))/c["cout"]),
 "base_croissant": lambda c: c["sb"],
}
res={}
for nom,cle in cles.items():
    reste,pr=reste_total,[]
    for c in sorted(cand,key=cle):
        if c["cout"]<=reste: pr.append(c["nom"]); reste-=c["cout"]
    res[nom]=pr
    print(f"  {nom:16s} {len(pr):3d} promus, {reste/2**20:7.1f} Mio inemployes")
def sp(x,y):
    n=len(x)
    def rg(v):
        o=sorted(range(n),key=lambda i:v[i]); r=[0.0]*n
        for i,j in enumerate(o): r[j]=i+1
        return r
    a,b=rg(x),rg(y); ma,mb=sum(a)/n,sum(b)/n
    num=sum((p-ma)*(q-mb) for p,q in zip(a,b))
    da=sum((p-ma)**2 for p in a)**.5; db=sum((q-mb)**2 for q in b)**.5
    return num/(da*db)
re_=[c["nom"] for c in sorted(cand,key=cles["erreur"])]
ra=[c["nom"] for c in sorted(cand,key=cles["absolu~"])]
print(f"\n  absolu~ contre erreur : top-10 {len(set(ra[:10])&set(re_[:10]))}/10, "
      f"top-100 {len(set(ra[:100])&set(re_[:100]))}/100, "
      f"ensembles {len(set(res['absolu~'])&set(res['erreur']))}/{max(len(res['absolu~']),len(res['erreur']))}")
r_e={n:i for i,n in enumerate(re_)}; r_a={n:i for i,n in enumerate(ra)}
dec=[r_e[c["nom"]]-r_a[c["nom"]] for c in cand]
ech=[c["ech"] for c in cand]
print(f"  correlation entre l'echelle ||w||_F et le deplacement de rang : {sp(ech,dec):+.4f}")
print(f"  (deplacement > 0 = monte dans le classement absolu~)")
