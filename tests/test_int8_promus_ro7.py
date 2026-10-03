"""ro7 (poste6 03/10, scellé revue/poste6-ro7-int8-promus-scelle-03-10.md) : les int8 « promus » de la conversion sont
g128 affines, donc inéligibles au GEMM int8 du préfill (`_i8c_eligible`) et déquantifiés en bf16 pour cuBLAS. Opt-in
`ACVRAM_INT8_PROMUS=canal` : re-quantification par canal symétrique au chargement. Cassant : sans la re-quantification le
poids reste inéligible ; une re-quantification qui perdrait les valeurs se voit (± 1 pas de l'échelle par canal)."""
import torch

from acvram import kernels as K
from acvram.engine import loader as LD
from acvram.quant.formats import _dequantize_int8, _quantize_int8


def _promu():
    torch.manual_seed(3)
    return _quantize_int8(torch.randn(96, 256) * 0.05, 128)                # g128 affine, comme la conversion


def test_la_requantification_rend_le_poids_eligible_a_un_pas_pres():
    t = _promu()
    assert not K._i8c_eligible(t)
    c = LD._requantifier_par_canal(t, pas=40)                              # 96 lignes : 40 + 40 + 16
    assert c.group_size == 256 and tuple(c.shape) == (96, 256) and c.scales.shape == (96, 1)
    assert bool((c.zeros == 128).all()) and K._i8c_eligible(c)
    avant, apres = _dequantize_int8(t, torch.float32), _dequantize_int8(c, torch.float32)
    pas = c.scales.to(torch.float32)                                        # amax / 127 par ligne
    assert bool(((avant - apres).abs() <= pas * 1.001).all()), "la re-quantification doit rester à un pas de l'origine"
    assert torch.allclose(avant, apres, atol=float(pas.max()) * 1.001)


class _Lecteur:
    def __init__(self, t):
        self.d = {"w.qweight": t.qweight, "w.scales": t.scales, "w.zeros": t.zeros}

    def get(self, k):
        return self.d[k]


def _entree(t, promu=True):
    e = {"format": "int8", "shape": list(t.shape), "group_size": t.group_size, "keys": list(_Lecteur(t).d)}
    if promu:
        e["promoted_from"] = "nvfp4"
    return e


def test_le_chargement_ne_requantifie_que_sous_l_opt_in_et_que_les_promus(monkeypatch):
    t = _promu()
    monkeypatch.setattr(LD, "_INT8_PROMUS", "")
    assert not K._i8c_eligible(LD._build_quant(_entree(t), "w", _Lecteur(t), 128))             # défaut : rien
    monkeypatch.setattr(LD, "_INT8_PROMUS", "canal")
    q = LD._build_quant(_entree(t), "w", _Lecteur(t), 128)
    assert K._i8c_eligible(q) and q.__dict__.get("requantifie_canal") is True                 # casse : branchement retiré
    assert not K._i8c_eligible(LD._build_quant(_entree(t, promu=False), "w", _Lecteur(t), 128))  # int8 natif : intact
    monkeypatch.setattr(LD, "_INT8_PROMUS_ROLES", ("down_proj",))                                 # filtre par rôle
    assert K._i8c_eligible(LD._build_quant(_entree(t), "layers.0.mlp.down_proj.weight", _Lecteur(t), 128))
    assert not K._i8c_eligible(LD._build_quant(_entree(t), "layers.0.self_attn.k_proj.weight", _Lecteur(t), 128))
    assert K.prefill_int8_regime.__doc__                                                          # la ligne de régime le compte
