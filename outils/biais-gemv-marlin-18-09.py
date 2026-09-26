"""Biais du GEMV lisant la disposition Marlin (forme (b)), par couche, sur les
PILES RÉELLES d'un converti (poste7-p1-situ-verdict-18-09 : ppl-decode-kv
+0,008 sur 1 024 pas, invisible au critère « 0 hors 2⁻⁷ » — REGLES § 4 bis : un
comptage ne détecte pas un biais, il faut la moyenne signée de Δ et un témoin
négatif).

Par couche MoE et par chemin (v1 = pile NVFP4, (b) = disposition Marlin,
témoin négatif = échelles tronquées d'un bit de mantisse, toujours ≤ la vraie :
DOIT rendre faux), contre la référence fp32 (poids déquantifiés, mêmes x et
routages) :
  - hors : nombre de valeurs |Δ| > 2⁻⁷·max|y| par ligne (le critère de P1) ;
  - biais_signe = moy(Δ) / moy|y|          (le critère de poste7 : ≤ 1e-4) ;
  - gain − 1   = Σ y·ref / Σ ref² − 1      (biais MULTIPLICATIF : invisible à
    la moyenne signée quand la sortie est centrée — c'est le cas d'un MoE —,
    le témoin le montre) ;
  - r_hf16     = la même chose pour un témoin d'accumulation fp16 (down seul).
Écrit un JSON par couche et une ligne par couche.

À sec (sans carte) : `--echelles-seulement` ne lit que les échelles de bloc et
compte, par couche, celles que la pile Marlin annule (s·facteur·2⁷ < 2,
traiter_echelles_nvfp4) — hypothèse 2 de poste7 (tables ≠ E4M3), chiffrée sur
le modèle réel, sans GPU.

    CUDA_VISIBLE_DEVICES="" python outils/biais-gemv-marlin-18-09.py <converti> --echelles-seulement
    outils/carte.sh python outils/biais-gemv-marlin-18-09.py <converti> [--couches 0,12,24,47] [--jetons 12] [--routages f.pt]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acvram.engine.loader import _ShardReader                                   # noqa: E402
from acvram.kernels import get_extension, marlin_port as MP                      # noqa: E402
from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4                     # noqa: E402

PROJS = ("gate_proj", "up_proj", "down_proj")
SEUIL_BIAIS = 1e-4


def lire_expert(reader, manifest, couche, e, proj):
    nom = f"model.layers.{couche}.mlp.experts.{e}.{proj}.weight"
    ent = manifest["tensors"][nom]
    sd = {k.rsplit(".", 1)[-1]: reader.get(k) for k in ent["keys"]}
    return NVFP4Tensor(sd["qweight"], sd["block_scale"].view(torch.float8_e4m3fn), sd["global_scale"],
                       tuple(ent["shape"]), sd["qweight"].shape[-1] * 2)


def couches_moe(manifest):
    cs = set()
    for n in manifest["tensors"]:
        if ".mlp.experts.0.gate_proj" in n:
            cs.add(int(n.split(".")[2]))
    return sorted(cs)


def nb_experts(manifest, couche):
    return 1 + max(int(n.split(".")[5]) for n in manifest["tensors"]
                   if n.startswith(f"model.layers.{couche}.mlp.experts."))


def echelles_annulees(bs_bf16):
    """Ce que fait traiter_echelles_nvfp4 : half(s·facteur)·2⁷ < 2 → 0."""
    facteur = MP.facteur_nvfp4(bs_bf16)
    s = bs_bf16.to(torch.half)
    if facteur > 1.0:
        s = (s.float() * facteur).to(torch.half)
    s = s * (2 ** 7)
    return int((s < 2).sum()), int(s.numel()), facteur, int((bs_bf16 == 0).sum())


def stats(y, ref):
    d = y.float() - ref
    hors = int((d.abs() > 2 ** -7 * ref.abs().amax(1, keepdim=True)).sum())
    return {"hors": hors, "n": ref.numel(),
            "biais_signe": float(d.mean() / ref.abs().mean()),
            "gain_moins_1": float((y.float() * ref).sum() / (ref * ref).sum() - 1.0)}


def temoin_echelles_tronquees(t: NVFP4Tensor) -> NVFP4Tensor:
    """Échelles E4M3 avec le bit bas de mantisse effacé : chaque échelle ≤ la
    vraie, jamais >, d'un pas moyen 2⁻⁴ relatif — un biais multiplicatif
    négatif certain, que l'instrument DOIT voir (gain − 1 ≈ −3 %)."""
    b = t.block_scale.view(torch.uint8) & 0xFE
    return NVFP4Tensor(t.qweight, b.view(torch.float8_e4m3fn), t.global_scale, t.shape, t.padded_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("converti")
    ap.add_argument("--couches", default="")
    ap.add_argument("--jetons", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--routages", default="")
    ap.add_argument("--graine", type=int, default=18)
    ap.add_argument("--echelles-seulement", action="store_true")
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(a.converti, "acvram_manifest.json")))
    reader = _ShardReader(a.converti, manifest["weight_map"])
    couches = [int(c) for c in a.couches.split(",")] if a.couches else couches_moe(manifest)
    out = {"converti": a.converti, "date": time.strftime("%Y-%m-%d %H:%M"), "couches": {}}
    sortie = os.path.join(os.path.dirname(__file__), "..", "scratchpad",
                          "biais-gemv-marlin-18-09" + ("-echelles" if a.echelles_seulement else "") + ".json")
    os.makedirs(os.path.dirname(sortie), exist_ok=True)

    if a.echelles_seulement:
        for c in couches:
            E = nb_experts(manifest, c)
            ligne = {}
            for proj in PROJS:
                bs = torch.stack([lire_expert(reader, manifest, c, e, proj).block_scale.to(torch.bfloat16)
                                  for e in range(E)])
                nz, n, facteur, zeros = echelles_annulees(bs)
                ligne[proj] = {"annulees": nz, "deja_nulles": zeros, "n": n, "facteur": facteur}
            out["couches"][c] = ligne
            print(f"couche {c:2d} : " + "  ".join(
                f"{p} annulées {ligne[p]['annulees'] - ligne[p]['deja_nulles']}/{ligne[p]['n']} (déjà nulles {ligne[p]['deja_nulles']}, facteur {ligne[p]['facteur']:g})"
                for p in PROJS), flush=True)
        json.dump(out, open(sortie, "w"), indent=1)
        print("JSON :", os.path.relpath(sortie))
        return

    from acvram.regime import regime_ligne
    ext = get_extension()
    assert ext is not None and hasattr(ext, "nvfp4_gemv_marlin_gateup"), "extension sans nvfp4_gemv_marlin"
    assert MP.charger(compiler=False) is not None, "extension Marlin non compilée à sec"
    dev = torch.device("cuda")
    print(regime_ligne(), "marlin_port_so=" + MP.sha_so(), flush=True)
    gen = torch.Generator().manual_seed(a.graine)
    reels = None
    if a.routages:
        reels = [t for t in torch.load(a.routages) if t.shape[1] == a.top_k and t.shape[0] >= a.jetons]
    B, TK = a.jetons, a.top_k
    for c in couches:
        E = nb_experts(manifest, c)
        ts = {p: [lire_expert(reader, manifest, c, e, p) for e in range(E)] for p in PROJS}
        piles, marlins, temoins = {}, {}, {}
        for p in PROJS:
            qw = torch.stack([t.qweight for t in ts[p]]).to(dev)
            bs = torch.stack([t.block_scale for t in ts[p]]).to(dev)
            gs = torch.tensor([t.global_scale_float() for t in ts[p]], dtype=torch.float32, device=dev)
            piles[p] = (qw, bs, gs)
            marlins[p] = MP.preparer_pile(qw, bs, gs)
            bt = torch.stack([temoin_echelles_tronquees(t).block_scale for t in ts[p]]).to(dev)
            temoins[p] = (qw, bt, gs)
        K, I = ts["gate_proj"][0].padded_in, ts["gate_proj"][0].shape[0]
        if reels is not None:
            topi = reels[c % len(reels)][:B].to(torch.long)
        else:
            topi = torch.stack([torch.randperm(E, generator=gen)[:TK] for _ in range(B)])
        eid = topi.reshape(-1).to(torch.int32).to(dev)
        tok = torch.arange(B, dtype=torch.int32, device=dev).repeat_interleave(TK)
        seq = torch.arange(B * TK, dtype=torch.int32, device=dev)
        x = (torch.randn(B, K, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        # référence fp32 : experts déquantifiés (ceux du routage seulement)
        w32 = {p: {} for p in PROJS}
        for e in torch.unique(topi).tolist():
            for p in PROJS:
                w32[p][e] = dequantize_nvfp4(ts[p][e], torch.float32).to(dev)
        xf = x.float()
        act_ref = torch.stack([torch.nn.functional.silu(xf[t] @ w32["gate_proj"][e].T) * (xf[t] @ w32["up_proj"][e].T)
                               for e, t in zip(eid.tolist(), tok.tolist())])
        ref = torch.stack([act_ref[p] @ w32["down_proj"][e].T for p, e in enumerate(eid.tolist())])

        def v1(pg, pu, pd):
            act = ext.nvfp4_gemv_grouped_gateup(pg[0], pg[1].view(torch.uint8), pg[2], pu[0], pu[1].view(torch.uint8), pu[2],
                                                eid, tok, x, K, 0)[:, :I]
            return act, ext.nvfp4_gemv_grouped(pd[0], pd[1].view(torch.uint8), pd[2], eid, seq, act, I)[:, :K]

        def forme_b(mg, mu, md):
            act = ext.nvfp4_gemv_marlin_gateup(*mg, *mu, eid, tok, x, K, I, 0)
            return act, ext.nvfp4_gemv_marlin(*md, eid, seq, act, I, K)

        bras = {"v1": v1(piles["gate_proj"], piles["up_proj"], piles["down_proj"]),
                "b": forme_b(marlins["gate_proj"], marlins["up_proj"], marlins["down_proj"]),
                "temoin_tronque_v1": v1(temoins["gate_proj"], temoins["up_proj"], temoins["down_proj"])}
        # témoin d'accumulation fp16 sur down : somme séquentielle par tranches de 16 en half
        acc = torch.zeros(ref.shape, dtype=torch.half, device=dev)
        for p, e in enumerate(eid.tolist()):
            prod = (act_ref[p].half()[None, :] * w32["down_proj"][e].half())          # [K, I]
            s = torch.zeros(K, dtype=torch.half, device=dev)
            for j in range(0, I, 16):
                s = (s + prod[:, j:j + 16].sum(-1, dtype=torch.half).half()).half()
            acc[p] = s
        ligne = {nom: {"act": stats(act, act_ref), "sortie": stats(y, ref)} for nom, (act, y) in bras.items()}
        ligne["temoin_hf16_down"] = {"sortie": stats(acc, ref)}
        out["couches"][c] = ligne
        fmt = lambda s: f"hors {s['hors']} biais {s['biais_signe']:+.1e} gain−1 {s['gain_moins_1']:+.1e}"
        print(f"couche {c:2d} E={E} | v1 {fmt(ligne['v1']['sortie'])} | (b) {fmt(ligne['b']['sortie'])} | "
              f"témoin tronqué {fmt(ligne['temoin_tronque_v1']['sortie'])} | témoin hf16 {fmt(ligne['temoin_hf16_down']['sortie'])}",
              flush=True)
        del ts, piles, marlins, temoins, w32
        torch.cuda.empty_cache()
    b_ok = all(abs(l["b"]["sortie"]["biais_signe"]) <= SEUIL_BIAIS and abs(l["b"]["sortie"]["gain_moins_1"]) <= SEUIL_BIAIS
               for l in out["couches"].values())
    temoin_vu = all(abs(l["temoin_tronque_v1"]["sortie"]["gain_moins_1"]) > SEUIL_BIAIS for l in out["couches"].values())
    out["verdict"] = ("instrument aveugle : le témoin tronqué passe" if not temoin_vu
                      else "(b) sans biais (|biais| et |gain−1| ≤ 1e-4 sur toutes les couches)" if b_ok
                      else "(b) BIAISÉ : voir les couches")
    print("verdict :", out["verdict"])
    json.dump(out, open(sortie, "w"), indent=1)
    print("JSON :", os.path.relpath(sortie))


if __name__ == "__main__":
    main()
