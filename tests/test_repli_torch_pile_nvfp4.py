"""Repli torch de la déquantification NVFP4 sur des échelles de bloc en OCTETS
(Sage, verdict-repli-torch-passage-direct-17-09) : `ACVRAM_DISABLE_KERNELS=1`
rendait une PPL de 10⁸ — `MoEBlock._pile_bf16` empile les `block_scale` des
experts en uint8 (model.py) et le noyau CUDA les décode, mais le jumeau torch
`dequantize_nvfp4` convertissait l'octet en float (0-255) au lieu de l'E4M3.
Contrat : la référence décode les octets comme le noyau ; le repli
`kernels.nvfp4_dequant` (extension absente) sur une pile à échelles uint8 et
échelles globales par ligne rend la même chose que la déquantification expert
par expert. Casse si la vue E4M3 est retirée. Indépendant du passage direct :
tout converti y passait, seul le régime (piles construites + noyaux coupés) le
révélait."""
import torch

from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4


def _expert(graine, m=32, k=64):
    g = torch.Generator().manual_seed(graine)
    return quantize_nvfp4(torch.randn(m, k, generator=g) * 0.05)


def test_echelles_en_octets_decodees_comme_e4m3():
    t = _expert(1)
    octets = NVFP4Tensor(t.qweight, t.block_scale.view(torch.uint8), t.global_scale, t.shape, t.padded_in)
    assert octets.block_scale.dtype == torch.uint8
    a, b = dequantize_nvfp4(t, torch.bfloat16), dequantize_nvfp4(octets, torch.bfloat16)
    assert torch.equal(a, b), f"écart max {(a.float() - b.float()).abs().max().item():.3g}"
    # le témoin de faute : la valeur de l'octet à la place de l'E4M3 est loin
    faux = NVFP4Tensor(t.qweight, t.block_scale.view(torch.uint8).to(torch.float32).to(torch.float8_e4m3fn),
                       t.global_scale, t.shape, t.padded_in)
    assert not torch.equal(dequantize_nvfp4(faux, torch.bfloat16), a)


def test_repli_nvfp4_dequant_sur_une_pile_d_experts(monkeypatch):
    """La pile de `_pile_bf16` : qweight [E*M, K/2], block_scale uint8 [E*M, K/16],
    global 1 et une échelle globale PAR EXPERT (gscale_rows) — sans extension
    CUDA, le repli doit rendre expert par expert la déquantification de chacun."""
    from acvram import kernels
    monkeypatch.setattr(kernels, "get_extension", lambda: None)
    experts = [_expert(10 + e) for e in range(3)]
    E, M = len(experts), experts[0].shape[0]
    qw = torch.stack([w.qweight for w in experts])
    bs = torch.stack([w.block_scale.view(torch.uint8) for w in experts])
    gs = torch.tensor([float(w.global_scale) for w in experts], dtype=torch.float32)
    plat = NVFP4Tensor.__new__(NVFP4Tensor)
    plat.qweight = qw.view(E * M, -1)
    plat.block_scale = bs.view(E * M, -1)                 # OCTETS, comme la pile
    plat.global_scale = torch.ones(())
    plat.padded_in = experts[0].padded_in
    plat.shape = (E * M, experts[0].padded_in)
    plat.global_scale_rows = None
    w = kernels.nvfp4_dequant(plat, torch.bfloat16, gscale_rows=gs, rows_per_group=M).view(E, M, -1)
    for e, t in enumerate(experts):
        attendu = dequantize_nvfp4(t, torch.bfloat16)
        assert torch.equal(w[e, :, :attendu.shape[1]], attendu), f"expert {e} : repli ≠ référence"
