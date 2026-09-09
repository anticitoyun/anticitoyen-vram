"""L'échelle se convertit une fois, pas à chaque appel.

`self.scale.to(x.dtype)` s'exécutait à chaque appel de chaque projection. Les
échelles sont en fp16, les activations en bf16 : la conversion est réelle et
lance un noyau — 337 par pas sur Qwen2.5-Coder-14B en nvfp4, exactement le
nombre de projections.

Les épreuves portent sur les deux propriétés qui comptent : le résultat est
**inchangé** (c'est la même conversion, faite une fois), et la conversion n'a
lieu **qu'une fois** par dtype.
"""
import torch

from acvram.quant.calibrate import ChannelScaler


def test_le_resultat_est_inchange():
    """Sans quoi le gain ne serait pas un gain mais un autre calcul."""
    s = torch.rand(64, dtype=torch.float16) + 0.5
    sc = ChannelScaler(s, 0)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    attendu = x / s.to(torch.bfloat16)          # ce que faisait l'ancien code
    assert torch.equal(sc.apply(x), attendu)


def test_l_echelle_n_est_convertie_qu_une_fois():
    """L'épreuve du gain lui-même : le second appel doit réutiliser l'objet,
    pas en refabriquer un. On compare l'identité mémoire, seul témoin qui ne
    mente pas sur « c'est le même tenseur »."""
    sc = ChannelScaler(torch.rand(32, dtype=torch.float16) + 0.5, 0)
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    sc.apply(x)
    premier = sc._au_dtype(torch.bfloat16)
    sc.apply(x)
    assert sc._au_dtype(torch.bfloat16) is premier


def test_deux_dtypes_ne_se_confondent_pas():
    """Un cache unique aurait servi une échelle bf16 à une activation fp16.
    Rien ne garantit qu'un modèle n'exécute qu'en un seul type."""
    sc = ChannelScaler(torch.rand(16, dtype=torch.float16) + 0.5, 0)
    a = sc._au_dtype(torch.bfloat16)
    b = sc._au_dtype(torch.float32)
    assert a.dtype == torch.bfloat16 and b.dtype == torch.float32
    assert torch.equal(sc.apply(torch.ones(1, 16, dtype=torch.float32)),
                       torch.ones(1, 16, dtype=torch.float32) / b)


def test_un_changement_de_peripherique_ne_reutilise_pas_le_cache():
    """Une échelle mise en cache pour un périphérique ne doit jamais servir
    sur un autre — `to()` rend un objet neuf, donc un cache neuf."""
    sc = ChannelScaler(torch.rand(8, dtype=torch.float16) + 0.5, 0)
    sc._au_dtype(torch.bfloat16)
    autre = sc.to("cpu")
    assert "_cache_dtype" not in autre.__dict__


def test_une_echelle_absente_ne_convertit_rien():
    """Le défaut par défaut : pas d'échelle, pas de cache, pas de noyau."""
    sc = ChannelScaler(None, 0)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    assert torch.equal(sc.apply(x), x)
