"""Notre chaine et l'etalon lisent-ils le meme texte, et le decoupent-ils
pareil ? Compter n'est pas comparer : on compare les SUITES d'identifiants."""
from transformers import AutoTokenizer
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402
M=f"{MODELES}/Llama-2-7b-fp16pur"
C=sorties()
rapide=AutoTokenizer.from_pretrained(M, use_fast=True)
lent  =AutoTokenizer.from_pretrained(M, use_fast=False)
for nom in ("wiki-gptq.txt","wiki.test.raw"):
    t=(C/nom).read_text(encoding="utf-8", errors="replace")
    ir=rapide(t).input_ids; il=lent(t).input_ids
    n=min(len(ir),len(il),344064)
    ident = ir[:n]==il[:n]
    seg_r=len(ir)//2048; seg_l=len(il)//2048
    print(f"{nom:16s} rapide {len(ir):>7} jetons ({seg_r} segments)  "
          f"lent {len(il):>7} ({seg_l})  "
          f"suites identiques sur {n} : {ident}")
    if not ident:
        d=next(i for i in range(n) if ir[i]!=il[i])
        print(f"   premiere divergence a l'indice {d} : rapide {ir[d]} vs lent {il[d]}")
    print(f"   positions notables a 168 segments : {168*2047}")
