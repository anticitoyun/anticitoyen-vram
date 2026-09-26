#!/usr/bin/env python3
"""Pièce 100 B, confirmation : KL Coder(bf16 ‖ alias) sur des compositions NEUVES (autres invites, autres graines,
autres tailles de lot), lecture Δ (le lot n'abîme pas le candidat plus que le témoin) fixée avant la prise.
Même mécanique que kl-lot-mele-p100.py (teacher forcing 8 pas, séquence 0 en ligne 0, rejeu ×2) ; compositions :
  C1 seule (b=1) · E1 préfixes L−5, L−11, L−17 (b=4) · E2 étranger REGLES.md (b=4) · E3 étranger REGLES.md, autres
  segments (b=8) · E4 étranger mêlé README+REGLES (b=12). Décalage des segments : 1 000 jetons × numéro d'invite.
Usage : kl-lot-mele-p100-confirmation.py <alias> <liste_dumps.txt> <sortie.json>
  liste_dumps.txt : une ligne par invite « nom<TAB>chemin du .pt HF » (chemins absolus admis en entrée, jamais en sortie).
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

ALIAS, LISTE, SORTIE, N = sys.argv[1], sys.argv[2], sys.argv[3], 8
assert os.path.realpath(acvram.__file__).startswith(os.path.realpath(os.getcwd()) + os.sep), acvram.__file__
DUMPS = [(n, os.path.expanduser(c)) for n, c in (l.rstrip("\n").split("\t") for l in open(LISTE) if l.strip() and not l.startswith("#"))]
chemin = os.path.join(racine_modeles(), ALIAS)
tok = load_tokenizer(chemin)
loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=2048, max_concurrent_seqs=16)
print("REGIME", regime.regime_ligne(), flush=True)
attn = [m for m in loaded.model.modules() if hasattr(m, "qkv_proj")]
piles_qkv = sum(1 for m in attn if m.qkv_proj is not None)
print(f"PILES_QKV {piles_qkv}/{len(attn)}", flush=True)
JEU = os.environ.get("P100_JEU", "E")            # E : prise 2 (REGLES+MECANISMES, README) ; F : prise 3 (corpus et lots jamais vus)
if JEU == "E":
    regles = tok.encode("".join(open(os.path.join(os.getcwd(), "acvram-memoire", f), encoding="utf-8").read() for f in ("REGLES.md", "MECANISMES.md")))
    readme = tok.encode(open(os.path.join(os.getcwd(), "README.md"), encoding="utf-8").read() * 6)
    assert len(regles) > 30000 and len(readme) > 6000, (len(regles), len(readme))
    ETRANGER = ["acvram-memoire/REGLES.md + MECANISMES.md", "README.md x6"]
elif JEU == "F":
    F_FICHIERS = ("REPRISE.md", "CLAUDE.md", "acvram-memoire/ANNUAIRE.md", "acvram-memoire/revue/organisation-22-09.md")
    corpus_f = tok.encode("".join(open(os.path.join(os.getcwd(), f), encoding="utf-8").read() for f in F_FICHIERS))
    assert len(corpus_f) > 14000, len(corpus_f)
    ETRANGER = list(F_FICHIERS)
else:                                            # G (107 bis) : prose française (Germinal, Gutenberg #5711), lots 7 / 3 / 9 / 12
    G_FICHIER = os.environ.get("P100_CORPUS_G", "scratchpad/poste6-p107-23-09/prose-fr.txt")
    corpus_g = tok.encode(open(os.path.join(os.getcwd(), G_FICHIER), encoding="utf-8").read())
    assert len(corpus_g) > 60000, len(corpus_g)
    ETRANGER = [G_FICHIER + " (Germinal, Gutenberg #5711)"]


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


def segments(src, base, n, longueurs):
    """n segments consécutifs de `src` à partir de `base`, chacun de la longueur demandée."""
    out, pos = [], base
    for lg in longueurs[:n]:
        out.append(src[pos:pos + lg])
        assert len(out[-1]) == lg, (pos, lg, len(src))
        pos += lg
    return out


res = {"alias": ALIAS, "n_pas": N, "piles_qkv": f"{piles_qkv}/{len(attn)}", "regime": regime.regime_ligne(),
       "etranger": ETRANGER, "jeu": JEU, "invites": []}
nom0, ch0 = DUMPS[0]
_st = torch.load(ch0, weights_only=False)
_ids = list(_st["ids"])
lancer([_ids], list(_st["cibles"]))
lancer([_ids] + [_ids[:len(_ids) - 4 * j] for j in (1, 2, 3, 4)], list(_st["cibles"]))
print("CHAUFFE faite", flush=True)
for i, (nom, ch) in enumerate(DUMPS):
    st = torch.load(ch, weights_only=False)
    ids, cibles, lg_hf = list(st["ids"]), list(st["cibles"]), st["logits"].double()
    L = len(ids)
    if JEU == "E":
        base = 1000 * i
        e2 = segments(regles, base, 3, [L - 5, L - 11, L - 17])
        e3 = segments(regles, base + 5000, 7, [L - 3, L - 6, L - 9, L - 12, L - 15, L - 18, L - 21])
        e4 = segments(readme, (base // 2) % 2000, 6, [L - 4, L - 8, L - 12, L - 16, L - 20, L - 24]) + \
            segments(regles, base + 12000, 5, [L - 2, L - 7, L - 13, L - 19, L - 23])
        comp = {"C1": [ids],
                "E1": [ids, ids[:L - 5], ids[:L - 11], ids[:L - 17]],
                "E2": [ids] + e2,
                "E3": [ids] + e3,
                "E4": [ids] + e4}
        tailles = {"C1": 1, "E1": 4, "E2": 4, "E3": 8, "E4": 12}
    elif JEU == "G":
        base = 3000 * i
        g2 = segments(corpus_g, base, 2, [L - 6, L - 13])
        g3 = segments(corpus_g, base + 1000, 8, [L - 2, L - 5, L - 8, L - 11, L - 14, L - 17, L - 20, L - 23])
        g4 = segments(corpus_g, base + 2000, 11, [L - 1, L - 4, L - 7, L - 9, L - 12, L - 15, L - 18, L - 21, L - 24, L - 26, L - 28])
        comp = {"C1": [ids],
                "G1": [ids, ids[:L - 3], ids[:L - 8], ids[:L - 12], ids[:L - 16], ids[:L - 20], ids[:L - 25]],
                "G2": [ids] + g2,
                "G3": [ids] + g3,
                "G4": [ids] + g4}
        tailles = {"C1": 1, "G1": 7, "G2": 3, "G3": 9, "G4": 12}
    else:
        base = 500 * i
        f2 = segments(corpus_f, base, 5, [L - 3, L - 7, L - 10, L - 14, L - 18])
        f3 = segments(corpus_f, base + 6000, 9, [L - 2, L - 4, L - 6, L - 8, L - 10, L - 12, L - 14, L - 16, L - 18])
        f4 = segments(corpus_f, base + 9000, 11, [L - 1, L - 3, L - 5, L - 7, L - 9, L - 11, L - 13, L - 15, L - 17, L - 19, L - 21])
        comp = {"C1": [ids],
                "F1": [ids, ids[:L - 4], ids[:L - 9], ids[:L - 15], ids[:L - 22]],
                "F2": [ids] + f2,
                "F3": [ids] + f3,
                "F4": [ids] + f4}
        tailles = {"C1": 1, "F1": 5, "F2": 6, "F3": 10, "F4": 12}
    assert all(len(p) == tailles[c] for c, p in comp.items())
    LOTS = [c for c in comp if c != "C1"]
    out = {c: lancer(p, cibles) for c, p in comp.items()}
    rejeu = {c: max(ecart(lancer(p, cibles), out[c])) for c, p in comp.items()}
    rep = {"invite": nom, "L": L,
           "kl_par_pas_C1": kl(out["C1"], lg_hf),
           "kl_max": {c: max(kl(x, lg_hf)) for c, x in out.items()},
           "kl_moy": {c: round(sum(kl(x, lg_hf)) / N, 4) for c, x in out.items()},
           "argmax_egaux_hf": {c: sum(int(x[k].argmax() == lg_hf[k].argmax()) for k in range(N)) for c, x in out.items()},
           "rejeu_ulp_max": rejeu,
           "ecart_ulp_C1_vs": {c: max(ecart(out[c], out["C1"])) for c in LOTS}}
    res["invites"].append(rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
res["kl_max_b1"] = max(r["kl_max"]["C1"] for r in res["invites"])
res["tenu_b1"] = f"{sum(1 for r in res['invites'] if r['kl_max']['C1'] <= 0.74)}/{len(res['invites'])}"
res["max_rejeu_ulp"] = max(max(r["rejeu_ulp_max"].values()) for r in res["invites"])
print("RESULTAT " + json.dumps({k: res[k] for k in ("alias", "piles_qkv", "kl_max_b1", "tenu_b1", "max_rejeu_ulp")},
                               ensure_ascii=False), flush=True)
json.dump(res, open(SORTIE, "w"), ensure_ascii=False, indent=1)
