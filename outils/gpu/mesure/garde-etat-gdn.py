"""Garde qualité de l'état GDN INT8 (LeapQuant, ``ACVRAM_ETAT_GDN=int8``) contre l'état fp32, au chemin SERVI : graphes
CUDA, créneaux statiques (le chemin INT8 n'existe que là, gdn.py `decode_static_batch`), 12 requêtes en lot, teacher
forcing. Seuils scellés AVANT la mesure dans revue/poste5-leap-int8-scelle-01-10.md.

Un processus par bras (la variable est lue à l'import ; le bras le PROUVE : refus si `gdn._ETAT_GDN` ou la ligne de
régime ne disent pas le bras demandé, ou si les créneaux GDN n'ont pas la forme attendue) :
    ACVRAM_ETAT_GDN=fp32 garde-etat-gdn.py mesurer --bras fp32 --modele M --prefixe 16384 --notes 256 --concat 2 \\
        --sortie long-fp32.json
    ACVRAM_ETAT_GDN=int8 garde-etat-gdn.py mesurer --bras int8 --modele M --prefixe 16384 --notes 256 --concat 2 \\
        --reference long-fp32.json --sortie long-int8.json
    garde-etat-gdn.py comparer long-fp32.json long-int8.json
Textes : scratchpad/corpus-prive/tranches-128 (scellé), `--concat c` colle c tranches consécutives par requête (16 k
jetons et plus à c = 2), requêtes i = 0..11 → tranches [c·i, c·i + c). Le bras fp32 garde, par pas et par requête, la
NLL de la cible et ses 256 log-probs les plus fortes ; le bras int8 en tire KL(fp32 ‖ int8) sur ce support, plus la
masse hors support regroupée (borne basse de la vraie KL : traitement des données). Sortie sans chemin absolu."""
import argparse, json, math, os, statistics, sys
from pathlib import Path

_REPO = os.environ.get("ACVRAM_ARBRE", str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, _REPO)
TOPK, LOT = 256, 12


def textes(dossier, n, concat):
    return ["\n\n".join(open(os.path.join(dossier, f"tranche{concat * i + j}.txt"), encoding="utf-8").read()
                        for j in range(concat)) for i in range(n)]


def mesurer(a):
    import torch
    import acvram
    from acvram.engine import gdn as G
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer
    if G._ETAT_GDN != a.bras or (a.bras == "int8") != ("etat=int8" in acvram.regime_ligne()):
        sys.exit(f"REFUS : bras {a.bras} mais gdn._ETAT_GDN={G._ETAT_GDN}, régime « {acvram.regime_ligne()} »")
    ref = json.load(open(a.reference)) if a.reference else None
    tok = load_tokenizer(a.modele)
    ids = []
    for t in textes(a.corpus, LOT, a.concat):
        x = tok.encode(t)
        if len(x) < a.prefixe + a.notes + 1:
            sys.exit(f"REFUS : texte de {len(x)} jetons < {a.prefixe + a.notes + 1} (jamais raccourci après coup)")
        ids.append(x[:a.prefixe + a.notes + 1])
    mml = a.prefixe + a.notes + 64
    loaded = load_model(a.modele, dtype=torch.bfloat16, max_model_len=mml, max_concurrent_seqs=LOT)
    eng = Engine(loaded, tok, max_batch_size=LOT, max_model_len=mml, enable_cuda_graphs=True, enable_prefix_cache=False)
    eng._eos = set()
    eng.pipeline_actif = False                         # forçage pas à pas : aucune avance spéculative du graphe suivant
    etat = {f"r{i}": {"pos": a.prefixe - 1, "nll": [], "top": [], "kl": []} for i in range(LOT)}
    lots = []

    def force(logits, seqs, **_kw):
        lots.append(len(seqs))
        toks, lps = [], []
        for row, s in enumerate(seqs):
            e, x = etat[s.request_id], ids[int(s.request_id[1:])]
            p = e["pos"]; cible = x[p + 1]
            lp = torch.log_softmax(logits[row].to(torch.float32), dim=-1)
            e["nll"].append(round(float(-lp[cible]), 5))
            k = len(e["nll"]) - 1
            if ref is None:
                v, i = lp.topk(TOPK)
                e["top"].append([i.tolist(), [round(float(z), 5) for z in v]])
            else:
                ri, rv = ref["requetes"][s.request_id]["top"][k]
                lp_ref = torch.tensor(rv, dtype=torch.float64)
                lq = lp[torch.tensor(ri, device=lp.device)].double().cpu()
                pr, qr = lp_ref.exp(), lq.exp()
                reste_p, reste_q = max(1.0 - float(pr.sum()), 1e-12), max(1.0 - float(qr.sum()), 1e-12)
                e["kl"].append(round(float((pr * (lp_ref - lq)).sum()) + reste_p * math.log(reste_p / reste_q), 6))
            e["pos"] = p + 1
            toks.append(cible); lps.append(lp[cible])
        return torch.tensor(toks, device=logits.device, dtype=torch.long), torch.stack(lps)

    eng._sample_only = force
    for i in range(LOT):
        eng.add_request(ids[i][:a.prefixe], SamplingParams(temperature=0.0, max_tokens=a.notes), request_id=f"r{i}")
    while eng.running or eng.waiting:
        eng.step()
    statics = [st for l in loaded.model.modules() for st in getattr(l, "statics", []) if isinstance(st, dict)]
    gdn_int8 = sum(1 for st in statics if "Z" in st)
    if (a.bras == "int8") != (gdn_int8 > 0):
        sys.exit(f"REFUS : {gdn_int8} créneaux INT8 pour le bras {a.bras} — la configuration n'a pas pris")
    out = {"bras": a.bras, "regime_ligne": acvram.regime_ligne(), "engine_regime": eng.regime_ligne(),
           "modele": os.path.basename(os.path.normpath(a.modele)), "prefixe": a.prefixe, "notes": a.notes,
           "concat": a.concat, "creneaux_int8": gdn_int8, "lots": {str(n): lots.count(n) for n in sorted(set(lots))},
           "requetes": etat}
    json.dump(out, open(a.sortie, "w"))
    print("RESULTAT", json.dumps({"bras": a.bras, "lots": out["lots"], "creneaux_int8": gdn_int8,
                                  "nll_moy": round(statistics.fmean(v for e in etat.values() for v in e["nll"]), 5)}),
          flush=True)


def comparer(a):
    A, B = json.load(open(a.fp32)), json.load(open(a.int8))
    assert A["bras"] == "fp32" and B["bras"] == "int8" and A["prefixe"] == B["prefixe"]
    d_req, kls = [], []
    for r in A["requetes"]:
        na, nb = A["requetes"][r]["nll"], B["requetes"][r]["nll"]
        assert len(na) == len(nb)
        d_req.append(statistics.fmean(y - x for x, y in zip(na, nb)))   # Δ NLL moyen de la requête (int8 − fp32)
        kls += B["requetes"][r]["kl"]
    d = statistics.fmean(d_req)
    se = statistics.stdev(d_req) / math.sqrt(len(d_req))
    tenu_ppl = abs(d) <= 2 * se or d <= 0
    tenu_kl = max(kls) <= 0.74
    print(json.dumps({"prefixe": A["prefixe"], "dppl_geo_pct": round(100 * math.expm1(d), 3),
                      "se_pct": round(100 * se, 3), "kl_max": round(max(kls), 4), "kl_moy": round(statistics.fmean(kls), 5),
                      "lots_int8": B["lots"], "ppl_tenue": tenu_ppl, "kl_tenue": tenu_kl,
                      "verdict": "TENU" if tenu_ppl and tenu_kl else "FAUX"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="mode", required=True)
    m = sp.add_parser("mesurer")
    m.add_argument("--bras", choices=("fp32", "int8"), required=True)
    m.add_argument("--modele", required=True); m.add_argument("--sortie", required=True)
    m.add_argument("--prefixe", type=int, default=2048); m.add_argument("--notes", type=int, default=256)
    m.add_argument("--concat", type=int, default=1); m.add_argument("--reference")
    m.add_argument("--corpus", default=os.path.join(_REPO, "scratchpad", "corpus-prive", "tranches-128"))
    c = sp.add_parser("comparer"); c.add_argument("fp32"); c.add_argument("int8")
    a = ap.parse_args()
    mesurer(a) if a.mode == "mesurer" else comparer(a)
