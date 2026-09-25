"""Pièce 182 (1) : au décodage GDN, z n'est plus casté en fp32 (48 `direct_copy` de [b, 6 144] par pas sur Qwen3.8,
≈ 96 µs, trace 173). La norme à porte F3 lit la vue bf16 [b, nv, dv] de la pile qkv‖gate (176) en place et convertit
au chargement : bf16 → fp32 est exact, donc la sortie de la couche est AU BIT de l'ancien chemin (z fp32 contigu).
Routage : le chemin servi (`decode_static_batch`) doit remettre à la norme la vue bf16, pas une copie fp32 — ce test
casse si l'on réintroduit le cast. Témoin : un ulp bf16 sur z doit se voir, sinon l'équivalence ne prouve rien."""
import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
H, NK, NV, DK, DV = 1024, 4, 8, 128, 128


def _lin(n, k, graine):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.formats import quantize
    g = torch.Generator(device="cuda").manual_seed(graine)
    w = torch.randn(n, k, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
    return QuantLinear(quantize(w, "int8", group_size=k, symmetric=True))


def _couche():
    from acvram import kernels
    from acvram.engine.gdn import GatedDeltaNet
    from acvram.kernels import gemm_etroit
    if kernels.get_extension() is None or not gemm_etroit.disponible():
        pytest.skip("extension ou Triton absents")
    kd, vd = NK * DK, NV * DV
    g = torch.Generator(device="cuda").manual_seed(3)
    la = GatedDeltaNet(_lin(2 * kd + vd, H, 1), _lin(vd, H, 2), _lin(NV, H, 3), _lin(NV, H, 4), _lin(H, vd, 5),
                       torch.randn(2 * kd + vd, 4, device="cuda", generator=g) * 0.3,
                       torch.randn(NV, device="cuda", generator=g), torch.randn(NV, device="cuda", generator=g) * 0.1,
                       torch.randn(DV, device="cuda", generator=g, dtype=torch.bfloat16), NK, NV, DK, DV)
    la.fuse()
    assert getattr(la, "qkv_gate", None) is not None, "montage : la pile 176 doit servir"
    return la


def _statics(la, b):
    """Créneaux 0..b-1 du lot 0, créés UNE fois par couche (un second `new_static` tomberait hors du lot contigu),
    rechargés à chaque appel avec les mêmes états de préfill."""
    from acvram.engine.gdn import GatedDeltaNet
    if not hasattr(la, "_t182"):
        etats = []
        for s in range(8):
            torch.manual_seed(50 + s)
            etats.append(la(torch.randn(12, H, device="cuda", dtype=torch.bfloat16), None)[1])
        la._t182 = ([la.new_static(torch.device("cuda")) for _ in range(8)], etats)
    sts, etats = la._t182
    for s in range(b):
        GatedDeltaNet.static_load(sts[s], etats[s])
    return sts[:b]


def _pas(la, h, projections=None, monkeypatch=None):
    sts = _statics(la, h.shape[0])                  # états du préfill, toujours par le chemin servi
    if projections is not None:
        monkeypatch.setattr(la, "_projections", projections)
    y = la.decode_static_batch(h, sts)
    if projections is not None:
        monkeypatch.undo()
    return y


def _ancien(la, ulp=False):
    """Le chemin d'avant la 182 : z casté en fp32 contigu (± un ulp bf16 pour le témoin)."""
    brut = type(la)._projections.__get__(la)

    def proj(x, qkv_brut=False):
        qkv, z, b, a = brut(x, qkv_brut)
        if ulp:
            z = (z.contiguous().view(torch.int16) + 1).view(torch.bfloat16)
        return qkv, z.to(torch.float32), b, a
    return proj


@carte
@pytest.mark.parametrize("b", [1, 3, 8])
def test_couche_au_bit_de_z_fp32(b, monkeypatch):
    la = _couche()
    h = torch.randn(b, H, device="cuda", dtype=torch.bfloat16)
    y = _pas(la, h)
    y_ref = _pas(la, h, _ancien(la), monkeypatch)
    assert torch.equal(y, y_ref), f"b={b} : {int((y != y_ref).sum())} valeurs diffèrent"
    y_ulp = _pas(la, h, _ancien(la, ulp=True), monkeypatch)
    assert not torch.equal(y_ulp, y_ref), "un ulp sur z ne se voit pas : l'équivalence est aveugle"


@carte
def test_decodage_servi_remet_la_vue_bf16(monkeypatch):
    from acvram.engine import gdn_norme
    la, vus = _couche(), []
    vrai = gdn_norme.norme_gated

    def espion(x, z, poids, eps):
        vus.append((z.dtype, z.dim(), z.is_contiguous()))
        return vrai(x, z, poids, eps)
    sts = _statics(la, 8)                           # préfill avant l'espion : seul le décodage est jugé
    monkeypatch.setattr(gdn_norme, "norme_gated", espion)
    la.decode_static_batch(torch.randn(8, H, device="cuda", dtype=torch.bfloat16), sts)
    assert vus == [(torch.bfloat16, 3, False)], f"la norme a reçu {vus} : z casté ou copié avant elle"
