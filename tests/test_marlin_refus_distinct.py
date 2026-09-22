"""Disposition unique Marlin et MoE à gate/up distincts (GLM k48-calibA,
verdict-glm-b12-19-09) : `_construire_marlin` REFUSE (raison nommée), la pile
naturelle est gardée, un pas de décodage passe par le chemin d'avant sans
TypeError, `_chemin` publié — et aucun repli ne peut plus recevoir une pile
rendue (RuntimeError nommée, pas un None dans un noyau)."""
import sys
import pathlib

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from acvram.engine import model as MD                                           # noqa: E402
from acvram.engine import moe as MOE_D
from acvram.engine.layers import ChannelScaler, QuantLinear                      # noqa: E402
from acvram.engine.model import MLP, MoEBlock                                    # noqa: E402
from acvram.quant.formats import _quantize_int8                                 # noqa: E402
from acvram.quant.nvfp4 import quantize_nvfp4                                   # noqa: E402


def _bloc_distinct(E=8, H=256, I=128, top_k=4):
    """Comme `_bloc_moe_jouet(awq=True)` mais gate et up avec des échelles AWQ
    DIFFÉRENTES → `up_distinct` (la forme de GLM), top-4."""
    def lin(o, i, graine, sc=None):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(o, i, generator=g) * 0.02).to(torch.bfloat16)
        return QuantLinear(quantize_nvfp4(w), out_features=o, in_features=i, scaler=sc).to_device("cpu")

    def scaler(i, graine):
        g = torch.Generator().manual_seed(graine)
        return ChannelScaler((0.5 + torch.rand(i, generator=g) * 1.5).to(torch.bfloat16), 0)
    experts = []
    for e in range(E):
        experts.append(MLP(lin(I, H, 10 * e + 1, scaler(H, 100 + e)), lin(I, H, 10 * e + 2, scaler(H, 300 + e)),
                           lin(H, I, 10 * e + 3, scaler(I, 200 + e))))
    gen = torch.Generator().manual_seed(5)
    routeur = QuantLinear(_quantize_int8((torch.randn(E, H, generator=gen) * 0.02).to(torch.bfloat16), 128),
                          out_features=E, in_features=H).to_device("cpu")
    return MoEBlock(routeur, experts, top_k).to("cpu")


@pytest.mark.sans_extension
def test_refus_nomme_pile_gardee_et_decodage_sans_typeerror(monkeypatch):
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin")
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    bloc = _bloc_distinct()
    if not hasattr(bloc, "_construire_marlin"):
        pytest.skip("pas de disposition Marlin dans cet arbre")
    assert bloc._try_build_stacks(), bloc._raison_repli          # piles construites à sec (CPU)
    assert bloc._stacks_awq.get("up_distinct") is True
    assert getattr(bloc, "_stacks_marlin", None) is None            # refus
    assert bloc._stacks["gate_proj"][1] is not None                   # pile naturelle gardée
    # le chemin d'avant (`_grouped` → nvfp4_gemv_grouped) n'a pas de noyau à sec :
    # un stub à sa place, qui refuse une pile rendue et rend la bonne forme
    from acvram import kernels as K
    appels = []

    def stub(x32, qw, bs, gs, expert_ids, token_ids, k):
        assert qw is not None and bs is not None, "pile rendue passée au noyau"
        appels.append(tuple(qw.shape))
        return torch.zeros(expert_ids.shape[0], qw.shape[1], dtype=torch.float32)
    monkeypatch.setattr(K, "nvfp4_gemv_grouped", stub)
    x = (torch.randn(2, 256) * 0.5).to(torch.bfloat16)
    logits = bloc.router(x)
    topw, topi = torch.topk(torch.softmax(logits.float(), -1), bloc.top_k, dim=-1)
    y = bloc._forward_grouped(x, topw, topi)                          # un pas de décodage (t=2), chemin d'avant
    assert y.shape == (2, 256) and torch.isfinite(y.float()).all()
    assert len(appels) == 3                                           # gate, up, down : trois piles réelles
    assert "gemv_marlin" not in getattr(bloc, "chemins", {})


def test_raison_distincte_avant_la_raison_cuda():
    """À sec, un MoE non distinct est refusé « hors CUDA » ; un distinct l'est
    pour la forme — l'ordre des raisons rend la seconde lisible sans carte."""
    # scission 22/09 (module 4) : `_construire_marlin` vit dans engine/moe.py
    src = (pathlib.Path(__file__).resolve().parent.parent / "acvram" / "engine" / "moe.py").read_text()
    i_d = src.index('elif awq.get("up_distinct") and _MARLIN_DISTINCT != "1":')
    i_c = src.index('elif piles["gate_proj"][1].device.type != "cuda":')
    assert i_d < i_c


def test_plus_jamais_une_pile_rendue_dans_un_noyau(monkeypatch):
    """Le repli `_grouped` sur une pile rendue lève une RuntimeError nommée."""
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin")
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    bloc = _bloc_distinct(top_k=2)
    assert bloc._try_build_stacks(), bloc._raison_repli
    pile = bloc._stacks["gate_proj"]
    bloc._stacks["gate_proj"] = ("nvfp4", None, None, pile[3], pile[4], pile[5])     # pile rendue simulée
    bloc._stacks["up_proj"] = ("nvfp4", None, None, pile[3], pile[4], pile[5])
    x = (torch.randn(1, 256) * 0.5).to(torch.bfloat16)
    logits = bloc.router(x) if hasattr(bloc, "router") else bloc.routeur(x)
    topw, topi = torch.topk(torch.softmax(logits.float(), -1), bloc.top_k, dim=-1)
    with pytest.raises(RuntimeError, match="pile NVFP4 naturelle rendue"):
        bloc._forward_grouped(x, topw, topi)


def test_c10_leve_le_refus_sous_variable(monkeypatch):
    """ACVRAM_MARLIN_DISTINCT=1 (C10) : le refus « distinct » n'est plus la
    raison — à sec la raison suivante (hors CUDA) prend le relais, ce qui
    prouve que la forme distincte n'est plus exclue par elle-même."""
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin")
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    monkeypatch.setattr(MOE_D, "_MARLIN_DISTINCT", "1")
    MoEBlock._marlin_refus_dit = False
    bloc = _bloc_distinct(top_k=4)
    import io, contextlib
    tampon = io.StringIO()
    with contextlib.redirect_stdout(tampon):
        assert bloc._try_build_stacks(), bloc._raison_repli
    assert "distinctes" not in tampon.getvalue() and "hors CUDA" in tampon.getvalue()
    MoEBlock._marlin_refus_dit = False
