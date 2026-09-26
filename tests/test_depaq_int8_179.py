"""Pièce 179 : B' (172) étendu à la déquantification int8 du préfill (`int8_matmul`, M > ACVRAM_INT8_GEMV_MAX : déquant
bf16 puis GEMM). AU BIT contre le chemin sans partage (`torch.equal`), formes réelles (6 144 × 5 120, séquences de 92
lignes > 80) ; témoin : la GEMM groupée DIFFÈRE, sinon le test casse (aveugle). Réserve : les octets retenus par le
partage d'une couche GDN aux dimensions de Qwen3.8 tiennent dans `ModelSpec.poids_bf16_couche_lineaire_bytes`, le
terme de la réserve de préfill (172) — mesurés, pas supposés. Bras cassant (prise) : partage retiré du chemin int8."""
import pytest
import torch

from acvram import kernels
from acvram.engine.layers import QuantLinear
from acvram.quant.formats import _quantize_int8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
DEV = torch.device("cuda:0")
LENS = [92] * 8


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _lin(n, k, graine=179):
    torch.manual_seed(graine)
    return QuantLinear(_quantize_int8(torch.randn(n, k, device=DEV) * 0.02, group_size=128))


def _x(k=5120):
    g = torch.Generator(device="cpu").manual_seed(92)
    return torch.randn(sum(LENS), k, generator=g).to(DEV, torch.bfloat16)


def _par_sequence(lin, x):
    out, d = [], 0
    for n in LENS:
        out.append(lin(x[d:d + n])); d += n
    return torch.cat(out)


def test_int8_partage_au_bit_et_temoin(monkeypatch):
    lin, x = _lin(6144, 5120), _x()
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", False)
    c = dict(kernels.CHEMINS_INT8)
    ref = _par_sequence(lin, x)
    assert kernels.CHEMINS_INT8["dequant"] - c.get("dequant", 0) == len(LENS), "le chemin déquant doit être pris"
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    c0 = dict(kernels.CHEMINS_NVFP4)
    with kernels.depaquetage_partage():
        bp = _par_sequence(lin, x)
    fab = kernels.CHEMINS_NVFP4["depaquetage_partage_fabrique"] - c0.get("depaquetage_partage_fabrique", 0)
    reu = kernels.CHEMINS_NVFP4["depaquetage_partage_reutilise"] - c0.get("depaquetage_partage_reutilise", 0)
    assert (fab, reu) == (1, len(LENS) - 1), (fab, reu)
    assert torch.equal(bp, ref)
    assert not torch.equal(lin(x), ref), "témoin aveugle : la GEMM groupée rend les mêmes bits sur ces formes"


def test_octets_retenus_dans_la_reserve(monkeypatch):
    """Couche GDN aux dimensions de Qwen3.8 (qkv 10 240, gate 6 144, alpha et beta 48, out 5 120 × 6 144) : ce que la
    portée garde vivant ≤ le terme de réserve de la 172."""
    from acvram.engine.config import ModelSpec
    spec = ModelSpec(name="q38", architecture="Qwen3_5ForConditionalGeneration", hidden_size=5120,
                     intermediate_size=17408, num_layers=64, num_attention_heads=24, num_key_value_heads=4,
                     vocab_size=248320, max_position_embeddings=262144, head_dim=256, linear_num_value_heads=48,
                     linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128)
    lins = [(_lin(10240, 5120, 1), 5120), (_lin(6144, 5120, 2), 5120), (_lin(48, 5120, 3), 5120),
            (_lin(48, 5120, 4), 5120), (_lin(5120, 6144, 5), 6144)]
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    # Retenu = ce que la SORTIE de la portée rend. « après − avant » comptait aussi les allocations persistantes
    # faites pendant la portée (espace de travail cuBLAS au premier GEMM d'une forme) : vert seul, rouge dans la
    # suite complète (179, 25/09).
    # Octets DEMANDÉS, pas `memory_allocated` (pièce 191) : l'allocateur caché ne découpe pas un bloc libre du grand
    # pool quand le reste est ≤ 1 Mio (`kSmallSize`) et compte alors le bloc entier. Ce reste dépend des segments
    # laissés par les tests précédents : +1 048 576 o, rouge dans la suite à b6c0d2c54, vert seul.
    def _demandes():
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()["requested_bytes.all.current"]

    with kernels.depaquetage_partage():
        for lin, k in lins:
            lin(_x(k)[:92]); lin(_x(k)[92:184])
        dedans = _demandes()
    retenu = dedans - _demandes()
    assert 0 < retenu <= spec.poids_bf16_couche_lineaire_bytes(), (retenu, spec.poids_bf16_couche_lineaire_bytes())
