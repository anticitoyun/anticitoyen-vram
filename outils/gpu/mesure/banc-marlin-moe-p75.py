#!/usr/bin/env python3
"""Pièce 75 : notre port Marlin MoE contre l'op `fused_marlin_moe` de vLLM, au même banc, sur les mêmes octets
(revue/poste1-piece75-port-contre-op-vllm-23-09.md).

Trois sous-commandes, un seul harnais :
  preparer <dossier> <routages.pt>   (venv vLLM) une pile de 128 experts reconditionnée par LEUR repack, écrite une fois
  vllm <dossier>                     (venv vLLM) l'op `fused_marlin_moe`
  acvram <dossier>                   (notre venv) A2 = notre port avec les arguments de vLLM (2 GEMM, mêmes tenseurs),
                                     A3 = notre forme servie (`gemm_experts_tensor` + `moe_reduce`, gate/up découpées)

Les deux processus relisent le MÊME fichier : poids, x, routages et poids de routage sont les mêmes octets.
Harnais : graphe de COPIES appels, chacun sur une copie distincte (L2 froide) ou sur la même (L2 tiède), REP rejeux,
médiane ÷ COPIES ; profileur torch pour les µs par lancement du noyau Marlin, attribués par position dans l'appel.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

E, K, N, TOPK, B = 128, 2048, 768, 8, 12
COPIES, REP = 8, 100
dev = torch.device("cuda", 0)


def preparer(dossier: str, routages: str) -> None:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_nvfp4_like
    g = torch.Generator(device=dev).manual_seed(75)
    piles = {"w13": [], "s13": [], "g13": [], "w2": [], "s2": [], "g2": []}
    for _ in range(E):
        _, q, s, gs = rand_marlin_weight_nvfp4_like(torch.randn(2 * N, K, device=dev, dtype=torch.bfloat16, generator=g) * 0.02, 16)
        piles["w13"].append(q); piles["s13"].append(s); piles["g13"].append(gs.reshape(1))
        _, q, s, gs = rand_marlin_weight_nvfp4_like(torch.randn(K, N, device=dev, dtype=torch.bfloat16, generator=g) * 0.02, 16)
        piles["w2"].append(q); piles["s2"].append(s); piles["g2"].append(gs.reshape(1))
    d = {k: (torch.cat(v) if k.startswith("g") else torch.stack(v)).contiguous().cpu() for k, v in piles.items()}
    ids = [r.to(torch.int32).cpu() for r in torch.load(routages, weights_only=False)]
    gr = torch.Generator().manual_seed(76)
    d["topk_ids"] = ids
    d["topk_w"] = [(lambda w: w / w.sum(1, keepdim=True))(torch.rand(B, TOPK, generator=gr)).float() for _ in ids]
    d["x"] = (torch.randn(B, K, generator=torch.Generator().manual_seed(3)) * 0.5).to(torch.bfloat16)
    torch.save(d, os.path.join(dossier, "pile.pt"))
    print(json.dumps({k: list(v.shape) for k, v in d.items() if torch.is_tensor(v)}))


def charger(dossier: str):
    d = torch.load(os.path.join(dossier, "pile.pt"), weights_only=False)
    base = {k: d[k].to(dev) for k in ("w13", "s13", "g13", "w2", "s2", "g2")}
    copies = [base] + [{k: v.clone() for k, v in base.items()} for _ in range(COPIES - 1)]
    rts = [(w.to(dev), i.to(dev)) for w, i in zip(d["topk_w"], d["topk_ids"])]
    return copies, rts, d["x"].to(dev)


def chrono(appel, copies, rts) -> dict:
    """µs par couche (un appel = une couche), graphe de COPIES appels, médiane de REP rejeux, par routage."""
    out = {}
    for regime, idx in (("l2_froide", list(range(COPIES))), ("l2_tiede", [0] * COPIES)):
        par_routage = []
        for rt in rts:
            for i in idx:
                appel(copies[i], rt)
            torch.cuda.synchronize()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in idx:
                    appel(copies[i], rt)
            torch.cuda.current_stream().wait_stream(s)
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                for i in idx:
                    appel(copies[i], rt)
            gph.replay(); torch.cuda.synchronize()
            ts = []
            for _ in range(REP):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); gph.replay(); b.record(); torch.cuda.synchronize()
                ts.append(a.elapsed_time(b) * 1e3 / COPIES)
            par_routage.append(round(statistics.median(ts), 2))
            del gph
        out[regime] = {"us_couche_med": round(statistics.median(par_routage), 2), "par_routage": par_routage}
    return out


def profil(appel, copies, rts, n_marlin: int) -> dict:
    """µs par lancement Marlin, attribués par position (0..n_marlin-1) dans l'appel ; L2 froide ; eager."""
    from torch.profiler import ProfilerActivity, profile
    for rt in rts:
        for i in range(COPIES):
            appel(copies[i], rt)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for rt in rts:
            for i in range(COPIES):
                appel(copies[i], rt)
        torch.cuda.synchronize()
    evs = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                 key=lambda e: e.time_range.start)
    marlin = [e.device_time for e in evs if "Marlin" in e.name]
    autres = sum(e.device_time for e in evs if "Marlin" not in e.name)
    appels = len(rts) * COPIES
    assert len(marlin) == appels * n_marlin, (len(marlin), appels, n_marlin)
    pos = [round(statistics.median(marlin[j::n_marlin]), 2) for j in range(n_marlin)]
    noms = sorted({e.name[:90] for e in evs if "Marlin" in e.name})
    return {"marlin_us_par_position_med": pos, "marlin_us_couche": round(sum(marlin) / appels, 2),
            "hors_marlin_us_couche": round(autres / appels, 2), "noyaux": noms}


def bras_vllm(dossier: str) -> dict:
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.scalar_type import scalar_types
    copies, rts, x = charger(dossier)
    ws = marlin_make_workspace_new(dev, 4)
    qt = scalar_types.float4_e2m1f.id

    def appel(p, rt):
        return fused_marlin_moe(x, p["w13"], p["w2"], None, None, p["s13"], p["s2"], rt[0], rt[1], quant_type_id=qt,
                                global_num_experts=E, global_scale1=p["g13"], global_scale2=p["g2"], workspace=ws)
    torch.save([appel(copies[0], rt).cpu() for rt in rts], os.path.join(dossier, "sorties-vllm.pt"))
    return {"V_fused_marlin_moe": {**chrono(appel, copies, rts), **profil(appel, copies, rts, 2)}}


def bras_acvram(dossier: str) -> dict:
    sys.path.insert(0, os.getcwd())
    from acvram import kernels
    from acvram.engine.moe import gemm_experts_tensor
    from acvram.kernels import marlin_port as MP
    ext = kernels.get_extension()
    assert ext is not None and hasattr(ext, "moe_aligner_petit") and hasattr(ext, "moe_reduce"), kernels.build_info()
    copies, rts, x = charger(dossier)
    for p in copies:                       # disposition Marlin locale par tuile de 64 colonnes : découpe = mêmes octets
        p["gate"] = (p["w13"][..., : 2 * N].contiguous(), p["s13"][..., :N].contiguous(), p["g13"], K, N)
        p["up"] = (p["w13"][..., 2 * N:].contiguous(), p["s13"][..., N:].contiguous(), p["g13"], K, N)
        p["down"] = (p["w2"], p["s2"], p["g2"], N, K)
    ws = MP.espace_travail(dev)
    G = B * TOPK
    bloc = MP.choisir_block_size(B, TOPK, E)
    P = -(-(G + E * (bloc - 1)) // bloc) * bloc
    tampons = (torch.empty(P, dtype=torch.int32, device=dev), torch.empty(P // bloc, dtype=torch.int32, device=dev),
               torch.empty(1, dtype=torch.int32, device=dev))
    uns = torch.ones(G, 1, dtype=torch.float32, device=dev)
    c13 = torch.zeros(G, 2 * N, dtype=torch.bfloat16, device=dev)
    c2 = torch.zeros(G, K, dtype=torch.bfloat16, device=dev)
    tw_flat = [rt[0].reshape(-1).contiguous() for rt in rts]
    eids = [rt[1].reshape(-1).contiguous() for rt in rts]
    idx_rt = {id(rt): j for j, rt in enumerate(rts)}

    def a2(p, rt):                          # arguments de vLLM : w13 (N 1 536, top_k 8) puis down (mul_topk_weights)
        j = idx_rt[id(rt)]
        ext.moe_aligner_petit(eids[j], E, bloc, *tampons)
        s_ids, e_ids, n_post = tampons
        MP.gemm_moe(x, p["w13"], p["s13"], p["g13"], s_ids, e_ids, n_post, uns, bloc, TOPK, B, 2 * N, K, ws, c=c13)
        act = (torch.nn.functional.silu(c13[:, :N].float()) * c13[:, N:].float()).to(torch.bfloat16)
        MP.gemm_moe(act, p["w2"], p["s2"], p["g2"], s_ids, e_ids, n_post, tw_flat[j], bloc, 1, G, K, N, ws,
                    mul_topk_weights=True, c=c2)
        return c2.view(B, TOPK, K).sum(1)

    t3, s3 = {}, {}

    def a3(p, rt):                          # forme servie : gate, up, down + moe_act + moe_reduce
        j = idx_rt[id(rt)]
        d = gemm_experts_tensor(MP, ext, x, eids[j], {"gate_proj": p["gate"], "up_proj": p["up"], "down_proj": p["down"]},
                                TOPK, N, N, 0, ws, uns, t3, s3, fusion=True)
        return ext.moe_reduce(d.contiguous(), tw_flat[j], TOPK)

    ref = torch.load(os.path.join(dossier, "sorties-vllm.pt"))
    res = {}
    for nom, f, n_m in (("A2_port_arguments_vllm", a2, 2), ("A3_port_forme_servie", a3, 3)):
        ecarts = []
        for rt, y_v in zip(rts, ref):
            y = f(copies[0], rt).float().cpu()
            ecarts.append(((y - y_v.float()).abs().max() / y_v.float().abs().max()).item())
        res[nom] = {"ecart_rel_max_contre_V": float(f"{max(ecarts):.3g}"), **chrono(f, copies, rts), **profil(f, copies, rts, n_m)}
    return res


def main() -> int:
    cmd, dossier = sys.argv[1], sys.argv[2]
    if cmd == "preparer":
        preparer(dossier, sys.argv[3])
        return 0
    r = {"cmd": cmd, "E": E, "K": K, "N": N, "b": B, "k": TOPK, "copies": COPIES, "rep": REP, "torch": torch.__version__}
    r.update(bras_vllm(dossier) if cmd == "vllm" else bras_acvram(dossier))
    print("RESULTAT " + json.dumps(r, ensure_ascii=False), flush=True)
    with open(os.path.join(dossier, f"resultat-{cmd}.json"), "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
