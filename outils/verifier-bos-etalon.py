from transformers import AutoTokenizer
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402
M=f"{MODELES}/Llama-2-7b-fp16pur"
t=(sorties() / "wiki-gptq.txt").read_text(encoding="utf-8",errors="replace")
for uf in (True, False):
    tok=AutoTokenizer.from_pretrained(M, use_fast=uf)
    gptq = tok(t).input_ids                      # protocole GPTQ : __call__
    notre = tok.encode(t)                        # notre chaine : .encode
    print(f"use_fast={uf}")
    print(f"  bos_token_id={tok.bos_token_id}  eos={tok.eos_token_id}")
    print(f"  GPTQ  ({len(gptq)} jetons) commence par {gptq[:3]}  BOS={gptq[0]==tok.bos_token_id}")
    print(f"  notre ({len(notre)} jetons) commence par {notre[:3]}  BOS={notre[0]==tok.bos_token_id}")
    print(f"  suites identiques : {gptq==notre}")
