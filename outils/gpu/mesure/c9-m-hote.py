"""C9 M-HÔTE (oceane-c9-conception-21-09 § 3.1) : combien coûte UN expert du
119B calculé sur le processeur ? Quatre experts réels lus aux shards du NVFP4
officiel (compressed-tensors nvfp4-pack-quantized, passage direct sans
requantifier), `nvfp4_matmul_cpu` (acvram_cpu.cpp, OpenMP), x [1, 4096] bf16,
w1/w3 [2048, 4096] puis w2 [4096, 2048] avec silu(w1·x)·(w3·x) entre les deux
— le MLP complet d un expert, comme en service.

Prédit (écrit avant) : 0,80 ± 0,15 ms les 4 experts (57 Mo à 71 Go/s), soit
0,20 ± 0,04 ms par expert. Seuil d arrêt : > 1,2 ms les 4 (< 47 Go/s :
l hôte ne bat plus llama.cpp, S2/S3 de la note tombent).

Contrôle avant tout chiffre : la sortie du noyau == linear(x, déquantifié
compressed-tensors) à ± 2^-6 relatif — sinon le temps mesuré est celui d un
calcul faux (ordre des quartets, échelle globale).

Usage : OMP_NUM_THREADS=16 python outils/gpu/mesure/c9-m-hote.py [--model DIR]
        [--couche 0] [--experts 0,1,2,3] [--rep 200] [--json SORTIE]
Carte : aucune allocation (CUDA_VISIBLE_DEVICES vide posé ici) ; sous carte.sh
seulement parce que le processeur d une fenêtre voisine compte.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# 21/09 : sans épinglage, 16 fils OpenMP rendaient 18 ms/expert contre 1,0 épinglés
# (fils migrants sur un poste chargé) — l épinglage fait partie du régime mesuré.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")
sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
import torch  # noqa: E402

MODELE_DEFAUT = "/mnt/2TO_2023_980PRO/Modeles/originaux/Mistral-Small-4-119B-2603-NVFP4"
PREDIT_MS_4, TOLERANCE_MS_4, SEUIL_ARRET_MS_4 = 0.80, 0.15, 1.2


def _en_tetes(dossier: str) -> dict[str, tuple[str, dict]]:
    """clé → (shard, entrée d en-tête) pour tous les `*.safetensors` du dossier."""
    out = {}
    for fn in sorted(os.listdir(dossier)):
        if not fn.endswith(".safetensors"):
            continue
        p = os.path.join(dossier, fn)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (p, v)
    return out


def charger_expert(dossier: str, couche: int, e: int, en_tetes=None) -> dict:
    """Les trois projections NVFP4 d un expert, passage direct (mêmes octets
    que la source, `HFQuantCheckpoint.nvfp4_direct`, 1/weight_global_scale)."""
    from safetensors import safe_open
    from acvram.quant.hfquant import HFQuantCheckpoint
    en_tetes = en_tetes or _en_tetes(dossier)
    out = {}
    for proj in ("w1", "w3", "w2"):
        base = f"layers.{couche}.experts.{e}.{proj}."
        shard = en_tetes[base + "weight_packed"][0]
        with safe_open(shard, framework="pt", device="cpu") as fh:
            packed = fh.get_tensor(base + "weight_packed")
            scale = fh.get_tensor(base + "weight_scale")
            g = fh.get_tensor(base + "weight_global_scale")
        out[proj] = HFQuantCheckpoint.nvfp4_direct(packed, scale, g, inverser_global=True)
    return out


def octets_expert(exp: dict) -> int:
    return sum(t.qweight.numel() + t.block_scale.numel() for t in exp.values())


def mlp_expert(x: torch.Tensor, exp: dict) -> torch.Tensor:
    from acvram.kernels.cpu import nvfp4_matmul_cpu
    h = torch.nn.functional.silu(nvfp4_matmul_cpu(x, exp["w1"])) * nvfp4_matmul_cpu(x, exp["w3"])
    return nvfp4_matmul_cpu(h, exp["w2"])


def controle_exactitude(x: torch.Tensor, exp: dict) -> float:
    """Écart relatif max entre le noyau et la référence déquantifiée
    (`_ct_nvfp4` : la déquantification compressed-tensors elle-même)."""
    from acvram.quant.hfquant import HFQuantCheckpoint
    from acvram.quant.nvfp4 import dequantize_nvfp4
    from acvram.kernels.cpu import nvfp4_matmul_cpu
    ecart = 0.0
    entrees = {"w1": x, "w3": x,
               "w2": (torch.randn(1, exp["w2"].shape[1], generator=torch.Generator().manual_seed(1)) * 0.5).to(x.dtype)}
    for proj, t in exp.items():
        xe = entrees[proj]
        ref = torch.nn.functional.linear(xe.float(), dequantize_nvfp4(t, torch.float32))
        y = nvfp4_matmul_cpu(xe, t).float()
        ecart = max(ecart, ((y - ref).abs().max() / (ref.abs().max() + 1e-9)).item())
        # la déquantification acvram doit aussi être celle de compressed-tensors (quartets, 1/g)
        ct = HFQuantCheckpoint._ct_nvfp4(t.qweight, t.block_scale, 1.0 / t.global_scale)
        ecart = max(ecart, ((dequantize_nvfp4(t, torch.bfloat16).float() - ct.float()).abs().max()
                            / (ct.float().abs().max() + 1e-9)).item())
    return ecart


def garde_charge() -> dict:
    """Même règle que `energie.py::relever_charge` (REGLES § 2, 18/09) : un
    voisin à plusieurs cœurs rend ce chiffre faux — 21/09, fumée sous un python
    à 783 % : 1,9 ms/expert à 8 fils, 1,0 à 16, rien de publiable. Refus si
    load1 > nproc/2 ou un voisin ≥ 300 % (il partage la DDR), sauf ACVRAM_CHARGE_OK=1 ; les trois premiers consommateurs
    vont dans le résultat quoi qu il arrive."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from energie import relever_charge
    ch = relever_charge()
    voisins = [pc for pc in ch.get("processus_charges", []) if pc[1] >= 300.0]
    if (ch["load1"] > ch["nproc"] / 2 or voisins) and not os.environ.get("ACVRAM_CHARGE_OK"):
        sys.exit(f"REFUS : load1 {ch['load1']} (nproc {ch['nproc']}), voisins ≥ 3 cœurs {voisins} — "
                 f"un voisin à plusieurs cœurs partage la DDR ; ACVRAM_CHARGE_OK=1 pour passer outre en le disant")
    return ch


def mesurer(dossier: str, couche: int, experts: list[int], rep: int, chauffe: int = 20) -> dict:
    charge = garde_charge() if rep >= 50 else {"load1": os.getloadavg()[0], "nproc": os.cpu_count()}
    en_tetes = _en_tetes(dossier)
    exps = [charger_expert(dossier, couche, e, en_tetes) for e in experts]
    K = exps[0]["w1"].shape[1]
    torch.manual_seed(0)
    x = (torch.randn(1, K) * 0.5).to(torch.bfloat16)
    exact = max(controle_exactitude(x, ex) for ex in exps)
    for _ in range(chauffe):
        for ex in exps:
            mlp_expert(x, ex)
    par_expert: list[float] = []
    par_tour: list[float] = []
    for _ in range(rep):
        t_tour = 0.0
        for ex in exps:
            t0 = time.perf_counter()
            mlp_expert(x, ex)
            dt = (time.perf_counter() - t0) * 1e3
            par_expert.append(dt)
            t_tour += dt
        par_tour.append(t_tour)
    octets = octets_expert(exps[0])
    med_e = statistics.median(par_expert)
    r = {
        "modele": dossier, "couche": couche, "experts": experts, "rep": rep, "charge": charge,
        "fils_omp": os.environ.get("OMP_NUM_THREADS"), "omp_bind": os.environ.get("OMP_PROC_BIND"), "torch_threads": torch.get_num_threads(),
        "octets_par_expert": octets, "formes": {p: list(t.shape) for p, t in exps[0].items()},
        "exactitude_rel_max": exact,
        "ms_par_expert": {"mediane": round(med_e, 4), "p90": round(sorted(par_expert)[int(0.9 * (len(par_expert) - 1))], 4),
                          "moyenne": round(statistics.fmean(par_expert), 4)},
        "ms_les_n": {"mediane": round(statistics.median(par_tour), 4),
                     "p90": round(sorted(par_tour)[int(0.9 * (len(par_tour) - 1))], 4)},
        "go_s_effectif": round(octets / (med_e * 1e-3) / 1e9, 1),
    }
    r.update(verdict(r["ms_les_n"]["mediane"], len(experts), exact))
    return r


def verdict(ms_les_n: float, n: int, exact: float) -> dict:
    ms_4 = ms_les_n * 4.0 / n                      # ramené à 4 experts, l unité de la prédiction
    if exact > 2 ** -6:
        v = "INVALIDE (noyau ≠ référence déquantifiée : le temps mesuré est celui d un calcul faux)"
    elif ms_4 > SEUIL_ARRET_MS_4:
        v = "RÉFUTÉ — arrêt C9 (a) : l hôte ne bat plus llama.cpp (< 47 Go/s)"
    elif abs(ms_4 - PREDIT_MS_4) <= TOLERANCE_MS_4:
        v = "TENU"
    elif ms_4 < PREDIT_MS_4:
        v = "TENU, mieux que prédit (bornes § 2 à recalculer à la valeur)"
    else:
        v = "RÉFUTÉ hors bande (0,95 < ms ≤ 1,2) : S2/S3 recalculées à la valeur, pas d arrêt"
    return {"predit_ms_4": PREDIT_MS_4, "tolerance_ms_4": TOLERANCE_MS_4, "seuil_arret_ms_4": SEUIL_ARRET_MS_4,
            "mesure_ms_4": round(ms_4, 4), "verdict": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODELE_DEFAUT)
    ap.add_argument("--couche", type=int, default=0)
    ap.add_argument("--experts", default="0,1,2,3")
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    experts = [int(e) for e in a.experts.split(",")]
    r = mesurer(a.model, a.couche, experts, a.rep)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[c9-m-hote] charge {r['charge']}" + (" — ACVRAM_CHARGE_OK posé : chiffre contaminé, pas un verdict" if os.environ.get("ACVRAM_CHARGE_OK") else ""))
    print(f"[c9-m-hote] {len(experts)} experts couche {a.couche}, {r['octets_par_expert'] / 1e6:.2f} Mo/expert, "
          f"OMP {r['fils_omp']} / torch {r['torch_threads']} fils, exactitude {r['exactitude_rel_max']:.2e}")
    print(f"  prédit {PREDIT_MS_4} ± {TOLERANCE_MS_4} ms les 4 · seuil d arrêt {SEUIL_ARRET_MS_4} ms")
    print(f"  mesuré : {r['ms_par_expert']['mediane']:.3f} ms/expert (p90 {r['ms_par_expert']['p90']:.3f}) · "
          f"{r['ms_les_n']['mediane']:.3f} ms les {len(experts)} → {r['mesure_ms_4']:.3f} ms les 4 · {r['go_s_effectif']} Go/s")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
