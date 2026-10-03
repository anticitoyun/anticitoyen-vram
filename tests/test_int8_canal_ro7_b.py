"""ro7 option B (poste6 03/10, scellé revue/poste6-ro7-b-conversion-scelle-03-10.md) : `--int8-canal` envoie TOUT
tenseur int8 de la conversion en canal symétrique (éligible au GEMM int8 du préfill) ; sans lui, seuls q/k/v/o
(`--attn-qkvo-int8-canal`) et le GDN (`--gdn-int8-canal`) y vont, les promotions des autres rôles restent g128 affines.
Cassant : la décision `_int8_canal` est celle des deux sites de la conversion (:1891, :2051)."""
from types import SimpleNamespace

from acvram.quant.convert import _int8_canal


def _opts(**k):
    return SimpleNamespace(**{"attn_qkvo_int8_canal": False, "gdn_int8_canal": False, "int8_canal": False, **k})


DOWN, KPROJ, GDN = "model.layers.3.mlp.down_proj.weight", "model.layers.3.self_attn.k_proj.weight", "model.layers.3.linear_attn.in_proj_qkv.weight"


def test_sans_option_les_promus_hors_attention_restent_affines():
    o = _opts()
    assert not _int8_canal(o, DOWN, "int8") and not _int8_canal(o, KPROJ, "int8")


def test_attn_qkvo_ne_couvre_que_l_attention():
    o = _opts(attn_qkvo_int8_canal=True)
    assert _int8_canal(o, KPROJ, "int8") and not _int8_canal(o, DOWN, "int8")


def test_int8_canal_couvre_tout_int8_et_rien_d_autre():
    o = _opts(int8_canal=True)
    assert _int8_canal(o, DOWN, "int8") and _int8_canal(o, KPROJ, "int8") and _int8_canal(o, GDN, "int8")
    assert not _int8_canal(o, DOWN, "nvfp4") and not _int8_canal(o, KPROJ, "bf16")      # jamais hors int8


def test_le_cli_porte_l_option(monkeypatch):
    import acvram.cli as cli
    import inspect
    src = inspect.getsource(cli)
    assert '"--int8-canal"' in src and "int8_canal=args.int8_canal" in src
