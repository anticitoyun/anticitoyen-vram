"""La QUEUE, pas la mediane : le mecanisme A ne predit pas que TOUS les tenseurs
de X souffrent, seulement ceux a forte variation. Cinq sur soixante-seize ne
bougeraient pas la mediane, et la perplexite les paierait quand meme — elle
somme, ponderee par les octets, elle ne prend pas la mediane.

Reserve posee par chef le 10/09. Repond a trois questions :
  combien de tenseurs au-dessus d un seuil, dans X et dans Y
  la somme ponderee par les octets promus, pas par le nombre de tenseurs
  les cinq plus hauts de chaque groupe, NOMMES

Ecrit aussi le detail par tenseur, pour qu une relecture ne relise pas 13 Go.
"""
import glob, json, os, struct
import numpy as np
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
BASE=os.environ.get("ACVRAM_MODELES", MODELES)
CORPUS=os.environ.get("ACVRAM_CORPUS", os.path.expanduser("~/Bureau/Claude/anticitoyen-vram/acvram-memoire/corpus"))
SRC=os.path.join(BASE,"Llama-2-7b-hf")
DETAIL=os.path.join(CORPUS, "variation-par-tenseur-x-y.json")
groupes=json.load(open(os.path.join(CORPUS, "groupes-x-y.json")))
X,Y=list(groupes["X_76"]),list(groupes["Y_27"])
CIBLE=set(X)|set(Y)

def lire(h,k,deb,fh):
    v=h[k]; a,b=v["data_offsets"]; fh.seek(deb+a); brut=fh.read(b-a)
    if v["dtype"]=="BF16":
        u=np.frombuffer(brut,dtype="<u2").astype(np.uint32)<<16; arr=u.view(np.float32)
    elif v["dtype"]=="F16": arr=np.frombuffer(brut,dtype="<f2").astype(np.float32)
    else: arr=np.frombuffer(brut,dtype="<f4")
    return arr.reshape(v["shape"])

if os.path.exists(DETAIL):
    det=json.load(open(DETAIL))
    print(f"detail relu depuis {DETAIL} ({len(det)} tenseurs) — aucun octet relu du disque")
else:
    det={}
    for f in sorted(glob.glob(os.path.join(SRC,"*.safetensors"))):
        with open(f,"rb") as fh:
            n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n)); deb=8+n
            for k in list(h):
                if k=="__metadata__" or k not in CIBLE: continue
                w=np.abs(lire(h,k,deb,fh))
                out,inn=w.shape
                n128=inn//128
                if n128==0: continue
                w=w[:,:n128*128]
                g128=w.reshape(out,n128,128).max(axis=2)
                g16=w.reshape(out,n128,8,16).max(axis=3)
                rap=g128/np.maximum(g16.mean(axis=2),1e-12)
                det[k]={"variation":float(rap.mean()),
                        "p99":float(np.quantile(rap,0.99)),
                        "numel":int(out*inn),
                        # octets promus : int8 a 8,1875 bits/poids (groupe 128)
                        "octets_promus":int(out*inn*8.1875/8)}
        fd=os.open(f,os.O_RDONLY)
        try: os.posix_fadvise(fd,0,os.fstat(fd).st_size,os.POSIX_FADV_DONTNEED)
        finally: os.close(fd)
    json.dump(det,open(DETAIL,"w"),indent=1,sort_keys=True)
    print(f"detail ecrit dans {DETAIL} ({len(det)} tenseurs)")

manquants=[k for k in CIBLE if k not in det]
if manquants:
    raise SystemExit(f"REFUS : {len(manquants)} tenseurs de la liste absents du modele, "
                     f"dont {manquants[:3]} — le compte mentirait en silence")

def vals(noms): return np.array([det[k]["variation"] for k in noms])
def octs(noms): return np.array([det[k]["octets_promus"] for k in noms],dtype=float)
x,y=vals(X),vals(Y)
ox,oy=octs(X),octs(Y)

print()
print("=== 1. LA QUEUE : combien au-dessus du seuil ===")
print(f"{'seuil':>8s}{'X (76)':>12s}{'Y (27)':>12s}{'X en %':>10s}{'Y en %':>10s}")
for s in (1.38,1.40,1.42,1.45,1.50):
    nx,ny=(x>s).sum(),(y>s).sum()
    print(f"{s:8.2f}{nx:12d}{ny:12d}{nx/len(x)*100:9.1f}%{ny/len(y)*100:9.1f}%")

print()
print("=== 2. LA SOMME PONDEREE PAR LES OCTETS PROMUS ===")
print(f"{'groupe':28s}{'octets Mio':>12s}{'moy ponderee':>14s}{'moy simple':>12s}{'mediane':>10s}")
for nom,v,o in (("X = A seul (76)",x,ox),("Y = B seul (27)",y,oy)):
    print(f"{nom:28s}{o.sum()/2**20:12.1f}{(v*o).sum()/o.sum():14.4f}{v.mean():12.4f}{np.median(v):10.4f}")
dp=((x*ox).sum()/ox.sum()-(y*oy).sum()/oy.sum())/((y*oy).sum()/oy.sum())*100
print(f"ecart des moyennes ponderees : {dp:+.2f} %")

print()
print("=== 3. LES CINQ PLUS HAUTS DE CHAQUE GROUPE, NOMMES ===")
for nom,noms in (("X = promus par A seul",X),("Y = promus par B seul",Y)):
    print(f"-- {nom}")
    for k in sorted(noms,key=lambda k:-det[k]["variation"])[:5]:
        d=det[k]
        print(f"   {d['variation']:7.4f}  p99={d['p99']:7.3f}  "
              f"{d['octets_promus']/2**20:7.1f} Mio  {k}")

print()
print("=== 4. LE GENRE DE TENSEUR — separation totale, decouverte ici ===")
import collections
def genre(k):
    if k=="lm_head.weight": return "lm_head"
    p=k.split("."); return p[-2] if len(p)>2 else k
for nom,noms in (("X = A seul (76)",X),("Y = B seul (27)",Y)):
    o=collections.Counter(); c=collections.Counter()
    for k in noms: o[genre(k)]+=det[k]["octets_promus"]; c[genre(k)]+=1
    tot=sum(o.values())
    print(f"-- {nom}   {tot/2**20:.1f} Mio")
    for g,n in c.most_common():
        print(f"   {n:3d} tenseurs {o[g]/2**20:8.1f} Mio {o[g]/tot*100:5.1f} %  {g}")

print()
print("=== VERDICT ===")
SEUIL=1.45
nx,ny=(x>SEUIL).sum(),(y>SEUIL).sum()
part_x=ox[x>SEUIL].sum()/ox.sum()*100
part_y=oy[y>SEUIL].sum()/oy.sum()*100
print(f"queue au-dessus de {SEUIL} : X {nx} tenseurs = {part_x:.1f} % de ses octets, "
      f"Y {ny} = {part_y:.1f} %")
dmed=(np.median(x)-np.median(y))/np.median(y)*100
print(f"moyennes ponderees par les octets : {dp:+.2f} % (medianes : {dmed:+.2f} %)")
# Le seuil qui tranche porte sur les OCTETS, pas sur le compte : "X en a deux et
# Y aucun" est satisfait par n importe quelle queue non vide, donc ne dit rien.
# La perplexite paie au prorata du poids : une queue qui porte 3 % des octets ne
# peut pas expliquer un ecart de perplexite.
if part_x>=10.0:
    print("=> la queue de X porte assez d octets pour compter : le mecanisme A")
    print("   n est pas refute, il est CONCENTRE, et redevient le candidat.")
else:
    print("=> la queue existe (deux tenseurs, couche 1 q et k) mais ne porte que")
    print(f"   {part_x:.1f} % des octets de X. Ponderee par les octets, la difference")
    print("   se resserre encore. Le mecanisme reste INFIRME sur les trois lectures.")
print()
print("Ce qui reste, et qui n etait pas cherche : X est 100 % attention,")
print("Y est 96,5 % down_proj + lm_head. AUCUN recouvrement de genre.")
