#!/usr/bin/env python3
"""Sortie ET debit du GEMV nvfp4, sur les formes reelles de Qwen2.5-14B.

Deux precautions apprises aujourd hui :
  - le L2 de la 5090 fait 96 Mio : un tenseur unique repete y reste et la
    mesure rend jusqu a 2429 Go/s, soit 36 % AU-DESSUS du pic materiel de
    1792. On fait donc tourner assez de jeux de poids pour le deborder.
  - la sortie est enregistree pour comparaison bit a bit : le double tampon
    ne doit RIEN changer aux valeurs, seulement au moment des lectures.
"""
import os
import json, sys, time, torch
from acvram.kernels import get_extension
from acvram.quant.nvfp4 import quantize_nvfp4

ext = get_extension()
dev = "cuda:0"
L2 = 96 << 20
# (nom, sortie, entree) — formes reelles d une couche de Qwen2.5-Coder-14B
FORMES = [("qkv", 5120 + 1024 + 1024, 5120), ("o", 5120, 5120),
          ("gate_up", 2 * 13824, 5120), ("down", 5120, 13824)]

res, sorties = {}, {}
for nom, M, K in FORMES:
    octets = M * K // 2 + M * (K // 16)          # poids + echelles
    jeux = max(3, (2 * L2) // octets + 1)        # deborder le L2
    g = torch.Generator(device="cpu").manual_seed(1234)
    ts = []
    for _ in range(jeux):
        w = (torch.randn(M, K, generator=g) * 0.02).to(torch.bfloat16)
        ts.append(quantize_nvfp4(w).to(dev))
    x = (torch.randn(1, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)

    def un(t):
        gsr = getattr(t, "global_scale_rows", None)
        return ext.nvfp4_gemv(t.qweight.contiguous(),
                              t.block_scale.view(torch.uint8).contiguous(),
                              1.0 if gsr is not None else t.global_scale_float(),
                              x.contiguous(), t.padded_in, gsr)

    y0 = un(ts[0])
    sorties[nom] = y0.float().cpu()
    for _ in range(3):
        for t in ts: un(t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    TOURS = 20
    for _ in range(TOURS):
        for t in ts: un(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / (TOURS * len(ts))
    res[nom] = {"Go_s": octets / dt / 1e9, "us": dt * 1e6,
                "jeux": jeux, "Mio_par_jeu": octets / 2**20}

etq = sys.argv[1] if len(sys.argv) > 1 else "courant"
S = os.environ.get("ACVRAM_SCRATCH", "/tmp/acvram-mesures")
# Le chemin etait code en dur avec un scratchpad de session : il portait
# a la fois un nom d utilisateur et une mention de session, dans un
# fichier suivi. Un chemin de travail se passe par l environnement.
torch.save(sorties, f"{S}/gemv-sorties-{etq}.pt")
json.dump(res, open(f"{S}/gemv-debit-{etq}.json", "w"), indent=1)
for k, v in res.items():
    print(f"  {k:8s} {v['Go_s']:7.1f} Go/s  {v['us']:7.1f} us  "
          f"({v['jeux']} jeux x {v['Mio_par_jeu']:.0f} Mio)")
tot = sum(r["Go_s"] for r in res.values()) / len(res)
print(f"  moyenne  {tot:7.1f} Go/s   (pic materiel 1792)")
