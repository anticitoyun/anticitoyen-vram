"""Second vidage de l'option A : ce que Rust ne peut pas recalculer au bit de son côté.

* Tables RoPE fp32 (`RotaryEmbedding.tables32`, layers.py:757) : calculées par torch sur la carte ; un cos/sin
  recalculé ailleurs peut différer au dernier bit.
* Noyaux Triton RÉELLEMENT compilés par le décodage servi : hash, nom, signature, constexpr, attributs, warps,
  mémoire partagée, et le cubin lui-même — le choix de la variante se fait sur ces clés, jamais par deviner.

Usage (sous outils/carte.sh) :
  PYTHONPATH=<racine> python moteurs/acvram_rust/outils/vidage_tables.py <modèle> <invites.json> <dossier-sortie>
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.server.chat import load_tokenizer, render_chat

MODELE, INVITES, SORTIE = sys.argv[1:4]
MAX_LEN = 2304
os.makedirs(os.path.join(SORTIE, "triton"), exist_ok=True)

tok = load_tokenizer(MODELE)
loaded = load_model(MODELE, dtype=torch.bfloat16, max_model_len=MAX_LEN, max_concurrent_seqs=1)
eng = Engine(loaded, tok, max_batch_size=1, max_model_len=MAX_LEN, enable_prefix_cache=True,
             speculator=None, enable_cuda_graphs=True)
eng.demarrer_service(strict=False, warm_max_len=int(os.environ.get("ACVRAM_WARM_GRAPHS", "2048")))
modele = loaded.model
# Une génération par invite : les godets de contexte rencontrés par la porte sont compilés et capturés.
for inv in json.load(open(INVITES, encoding="utf-8")):
    ids = tok.encode(render_chat(tok, inv["messages"], True))
    for _ in eng.generate(list(ids), SamplingParams(temperature=0.0, max_tokens=128)):
        pass

ropes = {id(l.self_attn.rope) for l in modele.layers}
assert len(ropes) == 1, f"{len(ropes)} objets RoPE distincts : non porté"
rope = modele.layers[0].self_attn.rope
cos32, sin32 = rope.tables32(MAX_LEN, torch.device("cuda:0"))
save_file({"cos32": cos32[:MAX_LEN].contiguous().cpu(), "sin32": sin32[:MAX_LEN].contiguous().cpu()},
          os.path.join(SORTIE, "rope.safetensors"))

from acvram.kernels import attn_paginee  # noqa: E402

inventaire = []
for nom in dir(attn_paginee):
    fn = getattr(attn_paginee, nom)
    caches = getattr(fn, "device_caches", None)
    if not isinstance(caches, dict):
        continue
    for dev, tup in caches.items():
        for cle, ck in tup[0].items():
            md = ck.metadata
            src = getattr(ck, "src", None)
            chemin = os.path.join(SORTIE, "triton", f"{ck.hash}.cubin")
            with open(chemin, "wb") as fh:
                fh.write(ck.asm["cubin"])
            inventaire.append({
                "fonction": nom, "nom": ck.name, "hash": ck.hash, "cle": str(cle),
                "signature": {str(k): str(v) for k, v in getattr(src, "signature", {}).items()},
                "constexprs": {str(k): (v if isinstance(v, (int, float, bool, str)) else str(v))
                               for k, v in getattr(src, "constexprs", {}).items()},
                "attrs": str(getattr(src, "attrs", None)),
                "src_attributs": sorted(vars(src).keys()) if src is not None else None,
                "num_warps": md.num_warps, "num_stages": md.num_stages, "shared": md.shared,
                "global_scratch_size": getattr(md, "global_scratch_size", None),
                "profile_scratch_size": getattr(md, "profile_scratch_size", None),
            })
json.dump({"rope": {"head_dim": rope.head_dim, "base": rope.base, "lignes": MAX_LEN,
                    "d": int(cos32.shape[-1])},
           "triton": inventaire}, open(os.path.join(SORTIE, "tables.json"), "w"), indent=1)
eng.fermer()
print("ok", len(inventaire), "noyaux Triton")
