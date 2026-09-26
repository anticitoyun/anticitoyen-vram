#!/usr/bin/env python3
"""Pièce 100 B : KL Coder(bf16 ‖ alias) au décodage, à b=1 (5 invites, seuil 0,74) et porte appariée à lot mêlé
(REGLES § 1, pièce 98) — un alias par processus, le témoin (qkvo-i8c) joué dans la même prise par la chaîne.

Instrument de composition.py (p98) / kl-b.py (p81) : teacher forcing 8 pas gloutons bf16 sur les 5 dumps HF de
`kl-coder-texte-22-09/dumps`, eager, la séquence 0 toujours en ligne 0. Compositions par invite (L = longueur de l'invite) :
  C1 seule (b=1) · C3 préfixes de la même invite (L−7, L−14, L−21, b=4) · C5, C6 contenu ÉTRANGER (README × 6, deux
  segments distincts, b=4) · C7 contenu étranger, b=12. Chaque composition est jouée DEUX fois : écart de rejeu en ulp
  bf16 (doit être 0), chauffe jetée avant la première invite.
Sortie JSON sans chemin absolu (cliquet de la CI). Usage : kl-lot-mele-p100.py <alias> <sortie.json>
"""
from __future__ import annotations

import json
import os
import sys

import torch

# Pièce 211 : racine dérivée de __file__, pas du cwd (garde a86fa1dd refusait depuis un
# worktree si le cwd n'était pas la racine, constat poste5 25/09).
_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _RACINE)
sys.path.insert(0, os.path.join(_RACINE, "outils"))
from racine_modeles import racine_modeles  # noqa: E402
from acvram import regime  # noqa: E402
from acvram.engine.loader import load_model  # noqa: E402
from acvram.engine.runner import Engine  # noqa: E402
from acvram.engine.sampler import SamplingParams  # noqa: E402
from acvram.quant.equivalence import ulp_bf16  # noqa: E402
from acvram.server.chat import load_tokenizer  # noqa: E402

import acvram  # noqa: E402

ALIAS, SORTIE, N = sys.argv[1], sys.argv[2], 8
D = os.path.expanduser(os.environ.get("P100_DUMPS", "~/Bureau/Claude/anticitoyen-vram/scratchpad/kl-coder-texte-22-09/dumps"))
assert os.path.realpath(acvram.__file__).startswith(os.path.realpath(os.getcwd()) + os.sep), acvram.__file__
chemin = os.path.join(racine_modeles(), ALIAS)
tok = load_tokenizer(chemin)
loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=2048, max_concurrent_seqs=16)
print("REGIME", regime.regime_ligne(), flush=True)
attn = [m for m in loaded.model.modules() if hasattr(m, "qkv_proj")]
piles_qkv = sum(1 for m in attn if m.qkv_proj is not None)
print(f"PILES_QKV {piles_qkv}/{len(attn)}", flush=True)
etranger = tok.encode(open(os.path.join(os.getcwd(), "README.md"), encoding="utf-8").read() * 6)
assert len(etranger) > 6000


def lancer(prompts, cibles):
    B = len(prompts)
    eng = Engine(loaded, tok, max_batch_size=16, max_model_len=max(len(p) for p in prompts) + N + 32, enable_cuda_graphs=False)
    eng.pipeline_actif = False
    for j, p in enumerate(prompts):
        eng.add_request(list(p), SamplingParams(temperature=0.0, max_tokens=N + 1), request_id=f"s{j}")
    eng._admit()
    seqs = list(eng.running)
    assert len(seqs) == B and seqs[0].request_id == "s0"
    sortie = []
    with torch.inference_mode():
        logits = loaded.model(eng._build_batch(seqs, prefill=True))
        for s in seqs:
            s.prefill_len = len(s.prompt_ids)
        for k in range(N):
            lg = (logits.view(B, -1, logits.shape[-1])[:, -1].float() if logits.dim() == 3
                  else logits.view(B, -1)[:, -logits.shape[-1]:].float() if logits.dim() == 2 and logits.shape[0] != B
                  else logits.float())
            sortie.append(lg[0].cpu())
            for s in seqs:
                s.output_ids.append(int(cibles[k]))
                if not eng._grow(s):
                    raise RuntimeError("blocs épuisés")
            if k < N - 1:
                logits = loaded.model(eng._build_batch(seqs, prefill=False))
    return torch.stack(sortie)


def ecart(x, ref):
    return [round(float((x[k].double() - ref[k].double()).abs().max()) / ulp_bf16(float(ref[k].abs().max())), 2) for k in range(N)]


def kl(x, hf):
    out = []
    for k in range(N):
        lph = torch.log_softmax(hf[k], -1)
        out.append(round(float((lph.exp() * (lph - torch.log_softmax(x[k].double(), -1))).sum()), 4))
    return out


res = {"alias": ALIAS, "dumps": "kl-coder-texte-22-09/dumps", "n_pas": N, "piles_qkv": f"{piles_qkv}/{len(attn)}",
       "regime": regime.regime_ligne(), "invites": []}
_st = torch.load(os.path.join(D, "lot-hf", "decode-pas-hf-invite0.txt.pt"), weights_only=False)
_ids = list(_st["ids"])
lancer([_ids], list(_st["cibles"]))
lancer([_ids] + [_ids[:len(_ids) - 7 * j] for j in (1, 2, 3)], list(_st["cibles"]))
print("CHAUFFE faite", flush=True)
for i in range(5):
    dump = os.path.join(D, "lot-hf", f"decode-pas-hf-invite{i}.txt.pt") if i < 4 else os.path.join(D, f"decode-pas-hf-invite{i}.txt.pt")
    st = torch.load(dump, weights_only=False)
    ids, cibles, lg_hf = list(st["ids"]), list(st["cibles"]), st["logits"].double()
    L = len(ids)
    assert len(etranger) >= 12 * L, (len(etranger), L)
    a, b_ = etranger[:3 * L], etranger[3 * L:6 * L]
    comp = {"C1": [ids],
            "C3": [ids, ids[:L - 7], ids[:L - 14], ids[:L - 21]],
            "C5": [ids, a[:L - 7], a[L:2 * L - 14], a[2 * L:3 * L - 21]],
            "C6": [ids, b_[:L - 7], b_[L:2 * L - 14], b_[2 * L:3 * L - 21]],
            "C7": [ids] + [etranger[j * L:(j + 1) * L - 7 * (j % 3 + 1)] for j in range(11)]}
    out = {c: lancer(p, cibles) for c, p in comp.items()}
    rejeu = {c: max(ecart(lancer(p, cibles), out[c])) for c, p in comp.items()}
    rep = {"invite": i, "L": L,
           "kl_par_pas_C1": kl(out["C1"], lg_hf),
           "kl_max": {c: max(kl(x, lg_hf)) for c, x in out.items()},
           "kl_moy": {c: round(sum(kl(x, lg_hf)) / N, 4) for c, x in out.items()},
           "argmax_egaux_hf": {c: sum(int(x[k].argmax() == lg_hf[k].argmax()) for k in range(N)) for c, x in out.items()},
           "rejeu_ulp_max": rejeu,
           "ecart_ulp_C1_vs": {c: max(ecart(out[c], out["C1"])) for c in ("C3", "C5", "C6", "C7")}}
    res["invites"].append(rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
res["kl_max_b1"] = max(r["kl_max"]["C1"] for r in res["invites"])
res["tenu_b1_5_sur_5"] = sum(1 for r in res["invites"] if r["kl_max"]["C1"] <= 0.74)
res["max_rejeu_ulp"] = max(max(r["rejeu_ulp_max"].values()) for r in res["invites"])
print("RESULTAT " + json.dumps({k: res[k] for k in ("alias", "piles_qkv", "kl_max_b1", "tenu_b1_5_sur_5", "max_rejeu_ulp")},
                               ensure_ascii=False), flush=True)
json.dump(res, open(SORTIE, "w"), ensure_ascii=False, indent=1)
