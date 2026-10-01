"""p81 (poste1, 01/10) : part du KDA dans le pas de Kimi-Linear-35B, par phase, dans UNE trace nsys et UN chargement.
Prédictions scellées avant la prise (revue/poste1-p81-cake-kda-01-10.md) : `kda_decode_kernel` 5-12 % du pas b=1 (FAUX si
< 0,1 ms/pas) ; préfill de 8 k : KDA 5-15 % (FAUX hors 2-30 %) ; b=12 (fla fused_recurrent) ≥ 2,9 %.

Deux usages :
  piloter  (sous nsys, sous carte.sh) : trois plages NVTX — `p81:prefill8k` (une invite de 8 192 jetons, 1 jeton produit),
           `p81:decode_b1` (64 pas, 1 séquence), `p81:decode_b12` (64 pas, 12 séquences) — chacune après sa propre chauffe,
           régime servi (graphes, pipeline) ;
  analyser CSV (`nsys stats --report nvtx_kern_sum`) → par plage : temps GPU total, temps des noyaux KDA, part, par pas.
Noyaux KDA = nom contenant « kda » (notre kda_decode_kernel, chunk_kda* et fused_recurrent_kda* de fla).

  kda-part-p81.py piloter          (ACVRAM_MODELE_MESURE = dossier du converti)
  kda-part-p81.py analyser trace_nvtx_kern_sum.csv [--json sortie.json]
"""
import csv
import json
import os
import sys

# pièce 211 : la racine acvram vient de CE fichier (outils/gpu/mesure → racine), jamais du venv partagé
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

PAS = 64
PLAGES = ("p81:prefill8k", "p81:decode_b1", "p81:decode_b12")


def piloter() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    if not torch.cuda.is_available():
        sys.exit("kda-part-p81 : carte requise (sous outils/carte.sh)")
    modele = os.environ["ACVRAM_MODELE_MESURE"]
    ctx, b_max = 8192 + 256, 12
    loaded = load_model(modele, dtype=torch.bfloat16, max_model_len=ctx, max_concurrent_seqs=b_max)
    vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
    eng = Engine(loaded, None, max_batch_size=b_max, max_model_len=ctx, enable_cuda_graphs=True)
    eng._eos = set()
    eng.warm_graphs()
    print(eng.regime_ligne(), flush=True)

    def invite(k, n):
        return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(n)]

    def vider():
        while eng.running or eng.waiting:
            eng.step()

    nv = torch.cuda.nvtx
    # préfill : une chauffe de même forme, puis la mesurée
    for k, plage in ((0, None), (1, PLAGES[0])):
        eng.add_request(invite(k, 8192), SamplingParams(temperature=0.0, max_tokens=1))
        torch.cuda.synchronize()
        if plage:
            nv.range_push(plage)
        vider()
        torch.cuda.synchronize()
        if plage:
            nv.range_pop()
    # décodage : 16 pas de chauffe hors plage, puis PAS pas dans la plage
    for b, plage in ((1, PLAGES[1]), (12, PLAGES[2])):
        for s in range(b):
            eng.add_request(invite(10 + s, 256), SamplingParams(temperature=0.0, max_tokens=16 + PAS + 4))
        for _ in range(16):
            eng.step()
        torch.cuda.synchronize()
        nv.range_push(plage)
        for _ in range(PAS):
            eng.step()
        torch.cuda.synchronize()
        nv.range_pop()
        vider()
    print("kda-part-p81 : pilote terminé", flush=True)
    return 0


def _col(entete, *mots):
    for i, h in enumerate(entete):
        if all(m in h.lower() for m in mots):
            return i
    raise SystemExit(f"colonne {mots} absente : {entete}")


def analyser(chemin: str, sortie_json: str = "") -> dict:
    with open(chemin, newline="") as f:
        lignes = list(csv.reader(f))
    i0 = next(i for i, l in enumerate(lignes) if l and any("range" in c.lower() for c in l))
    entete, corps = lignes[i0], [l for l in lignes[i0 + 1:] if l]
    c_plage, c_tot, c_nom = _col(entete, "range"), _col(entete, "total", "time"), _col(entete, "kern", "name")
    res = {}
    for plage in PLAGES:
        tot = kda = 0.0
        for l in corps:
            if l[c_plage].lstrip(":").strip() != plage:
                continue
            t = float(l[c_tot]) / 1e6          # ns → ms
            tot += t
            if "kda" in l[c_nom].lower():
                kda += t
        n = 1 if plage == PLAGES[0] else PAS
        res[plage] = {"gpu_ms": round(tot, 3), "kda_ms": round(kda, 3), "part_kda": round(kda / tot, 4) if tot else None,
                      "kda_ms_par_pas": round(kda / n, 4), "gpu_ms_par_pas": round(tot / n, 4)}
    if not any(r["gpu_ms"] for r in res.values()):
        raise SystemExit("aucune plage p81 dans la trace : pilote non profilé ou plages NVTX perdues")
    for p, r in res.items():
        print(f"{p:16s} GPU {r['gpu_ms_par_pas']:.4f} ms/pas · KDA {r['kda_ms_par_pas']:.4f} ms/pas · part {r['part_kda']}")
    if sortie_json:
        json.dump(res, open(sortie_json, "w"), indent=1)
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["piloter"]:
        sys.exit(piloter())
    if sys.argv[1:2] == ["analyser"] and len(sys.argv) >= 3:
        js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else ""
        analyser(sys.argv[2], js)
        sys.exit(0)
    sys.exit(__doc__)
