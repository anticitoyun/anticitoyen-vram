import json
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
BASE=Path(MODELES)
DOS=[(0.0,"Llama-2-7b-nvfp4"),(4.50,"Llama-2-7b-quota-4g50"),(5.00,"Llama-2-7b-quota-5g00"),
     (5.50,"Llama-2-7b-quota-5g50"),(6.00,"Llama-2-7b-quota-6g00"),(6.55,"Llama-2-7b-quota-plafond")]
GIO=1024**3; BUDGET=6.00
def octets(v):
    out,inn=v["shape"][0],v["shape"][1]; g=v.get("group_size") or 128; ng=max(1,inn//g)
    if v["format"]=="int8": return out*inn+out*ng*2+out*ng
    if v["format"]=="nvfp4": return out*(inn//2)+out*(inn//16)
    return int(out*inn*v.get("bpw",16.0)/8)
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
            n=v["shape"][0]*v["shape"][1]
            cand.append({"nom":k,"sb":base["out_snr_db"],"sp":v["out_snr_db"],
                         "cout":max(octets(v)-octets(base),1),"params":n}); break
reste=BUDGET*GIO-plancher*GIO
cles={"snr":lambda c:-((c["sp"]-c["sb"])/c["cout"]),
      "inverse":lambda c:+((c["sp"]-c["sb"])/c["cout"]),
      "erreur":lambda c:-((10**(-c["sb"]/20)-10**(-c["sp"]/20))/c["cout"])}
print(f"{'cle':10s}{'promus':>8s}{'params promus':>16s}{'octets depenses':>17s}{'params/octet':>14s}{'PPL':>9s}")
ppl={"snr":5.4918,"inverse":5.4482,"erreur":5.4927}
for nom,cle in cles.items():
    r,pr,pa,oc=reste,[],0,0
    for c in sorted(cand,key=cle):
        if c["cout"]<=r: pr.append(c); pa+=c["params"]; oc+=c["cout"]; r-=c["cout"]
    print(f"{nom:10s}{len(pr):8d}{pa:16,}{oc:17,}{pa/oc:14.3f}{ppl[nom]:9.4f}".replace(",", " "))
print()
# les 49 que A promeut et B non
a=set(c["nom"] for c in sorted(cand,key=cles["snr"])[:0] )
def ens(cle):
    r,s=reste,set()
    for c in sorted(cand,key=cle):
        if c["cout"]<=r: s.add(c["nom"]); r-=c["cout"]
    return s
A,B=ens(cles["snr"]),ens(cles["inverse"])
seulA,seulB=A-B,B-A
pA=sum(c["params"] for c in cand if c["nom"] in seulA)
pB=sum(c["params"] for c in cand if c["nom"] in seulB)
oA=sum(c["cout"] for c in cand if c["nom"] in seulA)
oB=sum(c["cout"] for c in cand if c["nom"] in seulB)
print(f"promus par A seulement : {len(seulA):3d} tenseurs, {pA:,} params, {oA/2**20:7.1f} Mio".replace(",", " "))
print(f"promus par B seulement : {len(seulB):3d} tenseurs, {pB:,} params, {oB/2**20:7.1f} Mio".replace(",", " "))
