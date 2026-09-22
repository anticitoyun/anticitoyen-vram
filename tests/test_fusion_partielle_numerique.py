"""La fusion partielle doit rendre EXACTEMENT ce que rendent trois GEMV.

Les épreuves de `test_fusion_partielle.py` lisent le source : elles vérifient
que la ligne `sorties[i], sorties[j], sorties[reste]` est écrite, pas ce
qu'elle calcule. Une inversion — `deux[1], deux[0]` — les laisserait toutes
passer et donnerait des jetons faux sans qu'aucune garde ne le voie.

Ici on compare des NOMBRES : le même `_proj`, avec et sans pile, sur les
mêmes poids. Les trois formes sont distinctes (q gros, k et v petits et
inégaux) pour qu'une permutation change le résultat au lieu de se cacher
derrière des tailles égales.
"""
import pytest
import torch

@pytest.fixture(autouse=True)
def _voie_partielle_active(monkeypatch):
    """La voie 2+1 est COUPEE par defaut depuis la mesure de Manon (-12,16 %).
    Ces epreuves portent sur sa justesse, pas sur son activation : elles
    l'allument explicitement. Le jour ou elle sera reactivee par defaut, ce
    reglage deviendra inutile sans rien casser."""
    monkeypatch.setenv("ACVRAM_FUSION_PARTIELLE", "1")

from acvram.engine.layers import QuantLinear
from acvram.engine.model import Attention
from acvram.quant import formats


cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="le chemin CPU d'une pile NVFP4 est faux "
                                 "(kernels/__init__.py:335) : mesurer ici "
                                 "testerait le défaut, pas la fusion")


def _lin(sortie, entree, fmt, graine, dev="cuda"):
    torch.manual_seed(graine)
    w = torch.randn(sortie, entree, dtype=torch.float32) * 0.02
    lin = QuantLinear(formats.quantize(w, fmt, group_size=128))
    lin.to_device(torch.device(dev))
    return lin


def _attention(formats_qkv):
    """q/k/v de tailles DIFFÉRENTES : une permutation ne peut pas se cacher."""
    from acvram.engine.config import ModelSpec
    spec = ModelSpec(name="t", architecture="llama", hidden_size=256,
                     intermediate_size=512, num_layers=1,
                     num_attention_heads=8, num_key_value_heads=2,
                     vocab_size=32, max_position_embeddings=64)
    fq, fk, fv = formats_qkv
    q = _lin(256, 256, fq, 1)      # 8 têtes x 32
    k = _lin(64, 256, fk, 2)       # 2 têtes x 32
    v = _lin(64, 256, fv, 3)
    o = _lin(256, 256, fq, 4)
    return Attention(spec, q, k, v, o, rope=None)


@cuda
@pytest.mark.parametrize("fmts", [
    ("nvfp4", "nvfp4", "int8"),    # le reste est v
    ("nvfp4", "int8", "nvfp4"),    # le reste est k
    ("int8", "nvfp4", "nvfp4"),    # le reste est q — la paire est (1, 2)
])
def test_le_partiel_rend_les_memes_nombres_que_trois_gemv(fmts):
    att = _attention(fmts)
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")

    att.qkv_proj = None
    att.qkv_partiel = None
    q_ref, k_ref, v_ref, _ = att._proj(x, x.shape[0])

    assert att.fuse() is True, "aucune pile formée sur un groupe mixte"
    assert att.qkv_proj is None, "le groupe mixte ne doit pas s'empiler en entier"
    assert att.qkv_partiel is not None, "la fusion partielle ne s'est pas armée"
    q, k, v, _ = att._proj(x, x.shape[0])

    for nom, a, b in (("q", q, q_ref), ("k", k, k_ref), ("v", v, v_ref)):
        assert a.shape == b.shape, f"{nom} : forme {a.shape} au lieu de {b.shape}"
        assert torch.equal(a, b), (
            f"{nom} diffère du chemin séparé — sorties permutées ou mal "
            f"découpées (écart max {(a - b).abs().max().item():.3e})")


@cuda
def test_une_inversion_des_sorties_serait_detectee():
    """Contrôle de l'épreuve elle-même : si le test ne voyait pas une
    permutation, il ne prouverait rien. On en fabrique une et on exige
    qu'elle soit vue — par un écart ou par une forme impossible."""
    att = _attention(("nvfp4", "nvfp4", "int8"))
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
    att.qkv_proj = att.qkv_partiel = None
    q_ref, k_ref, _, _ = att._proj(x, x.shape[0])
    assert att.fuse() is True
    pile, (i, j), reste, tailles = att.qkv_partiel
    att.qkv_partiel = (pile, (j, i), reste, tailles)      # inversion
    try:
        q, k, _, _ = att._proj(x, x.shape[0])
    except RuntimeError:
        return                                            # vue : forme impossible
    assert not (torch.equal(q, q_ref) and torch.equal(k, k_ref)), \
        "une permutation passe inaperçue : l'épreuve ne prouve rien"
