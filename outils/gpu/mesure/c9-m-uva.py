"""C9 M-UVA (poste1-c9-s1-reborne-22-09 § 3.0) : à quel débit un GEMV nvfp4
lit-il un expert du 119B laissé en hôte ÉPINGLÉ, par UVA, à travers le bus ?
C est le terme qui décide de C9-S1 (`pin` zéro-copie) : un noyau qui lit
l hôte n atteint pas le débit d une copie (22,6 Go/s, M0).

Méthode : 4 experts réels du 119B (w1 [2048, 4096] nvfp4, 4,72 Mo chacun
qweight + échelles), lus aux shards par `c9-m-hote.charger_expert` (passage
direct, mêmes octets que la source), épinglés (`pin_memory`) ; tables
d adresses int64 [E] = `data_ptr()` des tampons épinglés (le contrat de
`memory/table_adresses.py`) ; `nvfp4_gemv_grouped_table` avec G = 4 groupes
(un jeton, 4 experts actifs, comme le 119B top-4), 200 pas après 20 de
chauffe, événements CUDA : Go/s = octets lus par pas / temps médian. Témoin
VRAM : mêmes tables pointant sur des copies D2D des mêmes tampons, même
noyau → débit résident (la borne haute) ; contrôle d exactitude :
sorties UVA == VRAM au bit (même noyau, mêmes octets ; sinon la table
pointe ailleurs et le chiffre est faux).

Prédit (écrit avant) : 15-19 Go/s. Seuil « vaut » : ≥ 17 (S1 ≥ 30 j/s à
h ≥ 0,72) ; arrêt C9 : < 13 (S1 ≤ 24 j/s même à h = 0,75 : parité pour six
commits). Témoin VRAM attendu ≥ 1 200 Go/s (sinon le GEMV lui-même est
lent et le rapport UVA/VRAM ne dit rien).

Usage : outils/carte.sh python outils/gpu/mesure/c9-m-uva.py [--model DIR] [--couche 0]
        [--experts 0,1,2,3] [--rep 200] [--json SORTIE]   (≤ 3 min, chargement compris)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))

PREDIT = (15.0, 19.0)
SEUIL_VAUT, SEUIL_ARRET = 17.0, 13.0
TEMOIN_VRAM_MIN = 1200.0


def _hote():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c9-m-hote.py")
    spec = importlib.util.spec_from_file_location("c9_m_hote", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def octets_expert(t) -> int:
    """Octets qu un GEMV lit pour cet expert : qweight (E2M1 packé) + échelles E4M3."""
    return t.qweight.numel() * t.qweight.element_size() + t.block_scale.numel()


def tables_de(tenseurs: list, device) -> tuple:
    """(table_qw, table_bs, gscales) int64/int64/fp32 sur la carte, depuis des
    NVFP4Tensor dont les octets sont là où ils sont (épinglés ou VRAM) — la
    table porte l ADRESSE, jamais une copie (contrat memory/table_adresses)."""
    import torch
    tq = torch.tensor([t.qweight.data_ptr() for t in tenseurs], dtype=torch.int64, device=device)
    tb = torch.tensor([t.block_scale.data_ptr() for t in tenseurs], dtype=torch.int64, device=device)
    gs = torch.tensor([float(t.global_scale) for t in tenseurs], dtype=torch.float32, device=device)
    return tq, tb, gs


def verdict(gos_uva: float, gos_vram: float, exact: bool) -> dict:
    if not exact:
        v = "INVALIDE : sorties UVA ≠ VRAM au bit — la table ne pointe pas sur les mêmes octets, débit sans objet"
    elif gos_vram < TEMOIN_VRAM_MIN:
        v = f"INVALIDE : témoin VRAM {gos_vram:.0f} Go/s < {TEMOIN_VRAM_MIN:.0f} — le GEMV lui-même est lent, le rapport UVA/VRAM ne dit rien"
    elif gos_uva < SEUIL_ARRET:
        v = f"RÉFUTÉ — arrêt C9 (a) zéro-copie : {gos_uva:.1f} Go/s < {SEUIL_ARRET} (S1 ≤ 24 j/s même à h = 0,75)"
    elif gos_uva >= SEUIL_VAUT:
        v = f"TENU, vaut : {gos_uva:.1f} Go/s ≥ {SEUIL_VAUT} (S1 ≥ 30 j/s si h_pin(119B) ≥ 0,72 — reste la trace du 119B)"
    else:
        v = f"TENU, marginal : {SEUIL_ARRET} ≤ {gos_uva:.1f} < {SEUIL_VAUT} — S1 entre 24 et 30 j/s, exige h ≥ 0,78 : la trace du 119B tranche"
    return {"predit_go_s": list(PREDIT), "seuil_vaut": SEUIL_VAUT, "seuil_arret": SEUIL_ARRET,
            "temoin_vram_min": TEMOIN_VRAM_MIN, "verdict": v}


def mesurer(model: str, couche: int, experts: list[int], rep: int, chauffe: int = 20) -> dict:
    import torch
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_grouped_table"):
        sys.exit("ÉCHEC / CAUSE : extension sans nvfp4_gemv_grouped_table / SUITE : build")
    hote = _hote()
    en_tetes = hote._en_tetes(model)
    dev = torch.device("cuda:0")
    # w1 seul : un expert = trois projections, w1 suffit pour le débit (même octets par ligne que w3 ; w2 = K/2)
    epingles = []
    for e in experts:
        t = hote.charger_expert(model, couche, e, en_tetes)["w1"]
        t.qweight = t.qweight.contiguous().pin_memory()
        t.block_scale = t.block_scale.contiguous().pin_memory()
        epingles.append(t)
    residents = []
    for t in epingles:
        from acvram.quant.nvfp4 import NVFP4Tensor
        residents.append(NVFP4Tensor(qweight=t.qweight.to(dev), block_scale=t.block_scale.to(dev),
                                     global_scale=t.global_scale.clone(), shape=t.shape, padded_in=t.padded_in))
    M, K = epingles[0].shape
    G = len(experts)
    octets_pas = sum(octets_expert(t) for t in epingles)
    torch.manual_seed(0)
    x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
    expert_ids = torch.arange(G, dtype=torch.int32, device=dev)
    token_ids = torch.zeros(G, dtype=torch.int32, device=dev)
    t_uva = tables_de(epingles, dev)
    t_vram = tables_de(residents, dev)

    def gemv(tables):
        tq, tb, gs = tables
        return ext.nvfp4_gemv_grouped_table(tq, tb, gs, expert_ids, token_ids, x, M, K)

    def chrono(tables):
        for _ in range(chauffe):
            gemv(tables)
        torch.cuda.synchronize()
        temps = []
        for _ in range(rep):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); gemv(tables); b.record()
            torch.cuda.synchronize()
            temps.append(a.elapsed_time(b) * 1e3)
        temps.sort()
        return {"us_mediane": round(temps[len(temps) // 2], 1), "us_p90": round(temps[int(0.9 * (len(temps) - 1))], 1),
                "go_s_mediane": round(octets_pas / (temps[len(temps) // 2] * 1e-6) / 1e9, 2),
                "go_s_p90": round(octets_pas / (temps[int(0.9 * (len(temps) - 1))] * 1e-6) / 1e9, 2)}
    y_uva, y_vram = gemv(t_uva), gemv(t_vram)
    torch.cuda.synchronize()
    exact = bool(torch.equal(y_uva, y_vram))
    r_vram = chrono(t_vram)
    r_uva = chrono(t_uva)
    r = {"modele": model, "couche": couche, "experts": experts, "projection": "w1", "forme": [M, K], "G": G,
         "octets_par_expert": octets_expert(epingles[0]), "octets_par_pas": octets_pas, "rep": rep,
         "exact_uva_vs_vram": exact, "uva": r_uva, "vram": r_vram,
         "rapport_uva_vram": round(r_uva["go_s_mediane"] / r_vram["go_s_mediane"], 4) if r_vram["go_s_mediane"] else None}
    r.update(verdict(r_uva["go_s_mediane"], r_vram["go_s_mediane"], exact))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=_hote().MODELE_DEFAUT)
    ap.add_argument("--couche", type=int, default=0)
    ap.add_argument("--experts", default="0,1,2,3")
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    t0 = time.time()
    r = mesurer(a.model, a.couche, [int(e) for e in a.experts.split(",")], a.rep)
    r["duree_s"] = round(time.time() - t0, 1)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[c9-m-uva] {r['G']} experts w1 {r['forme']} · {r['octets_par_pas'] / 1e6:.1f} Mo/pas · exact {r['exact_uva_vs_vram']} · {r['duree_s']} s")
    print(f"  prédit {PREDIT[0]}-{PREDIT[1]} Go/s · vaut ≥ {SEUIL_VAUT} · arrêt < {SEUIL_ARRET}")
    print(f"  UVA  : {r['uva']['us_mediane']:8.1f} µs (p90 {r['uva']['us_p90']:.1f}) → {r['uva']['go_s_mediane']:.1f} Go/s (p90 {r['uva']['go_s_p90']:.1f})")
    print(f"  VRAM : {r['vram']['us_mediane']:8.1f} µs → {r['vram']['go_s_mediane']:.0f} Go/s · rapport UVA/VRAM {r['rapport_uva_vram']}")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
