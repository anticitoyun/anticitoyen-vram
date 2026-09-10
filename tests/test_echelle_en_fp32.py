"""L'échelle d'activation se stocke en float32, jamais en fp16.

Le 9/09/2026 : sur Qwen2.5-Coder-14B, les 336 act_scale vont de 8,883e-4 a
4,002e+2. Marge x164 au plafond fp16 (65504), mais seulement **x15 au
plancher** des denormaux (6,104e-5). Un modele a echelles quinze fois plus
petites y tomberait, et ce ne seraient plus des valeurs imprecises mais des
valeurs FAUSSES, sans que rien ne le signale.

Un garde attraperait le defaut ; le float32 supprime la classe.
"""
import torch

from acvram.quant.calibrate import ChannelScaler


def test_l_echelle_part_en_float32_meme_venant_de_fp16():
    e = torch.rand(64, dtype=torch.float16) + 0.5
    sd = ChannelScaler(e, 0).state_dict("x.")
    assert sd["x.act_scale"].dtype == torch.float32


def test_une_echelle_sous_le_plancher_fp16_survit():
    """C'est le cas que le fp16 rendait faux : 1e-6 y est un denormal, 1e-9
    y est zero. En float32 les deux passent intacts."""
    e = torch.tensor([1e-6, 1e-9, 1e-12], dtype=torch.float32)
    sd = ChannelScaler(e, 0).state_dict()
    assert torch.equal(sd["act_scale"], e), "valeur alteree au stockage"
    # ce que l ancien stockage en aurait fait
    perdu = e.to(torch.float16).to(torch.float32)
    assert perdu[2] == 0, "le temoin ne demontre rien : 1e-12 devrait etre nul en fp16"
    assert sd["act_scale"][2] != 0, "la valeur aurait ete perdue"


def test_une_echelle_au_dessus_du_plafond_fp16_survit():
    e = torch.tensor([1e5, 7e4], dtype=torch.float32)     # > 65504
    sd = ChannelScaler(e, 0).state_dict()
    assert torch.isfinite(sd["act_scale"]).all()
    assert not torch.isfinite(e.to(torch.float16)).all(), \
        "le temoin ne demontre rien : ces valeurs devraient deborder le fp16"


def test_l_absence_d_echelle_n_ecrit_rien():
    assert ChannelScaler(None, 1024).state_dict() == {}


def test_la_recherche_rend_une_echelle_en_float32():
    """Le defaut etait A LA SOURCE, pas a l ecriture : `search_channel_scales`
    rabattait en fp16, donc ecrire du float32 au manifeste ne restaurait rien.
    C est le seul test qui attrape la vraie cause."""
    import torch

    from acvram.quant.calibrate import ActStats, search_channel_scales

    g = torch.Generator().manual_seed(9)
    w = torch.randn(64, 128, generator=g) * 0.02
    st = ActStats((torch.rand(128, generator=g) * 3 + 0.05), None, 128)
    sc, _ = search_channel_scales(w, st, "nvfp4", 128, 8)
    assert sc.scale is not None, "l echelle est vide, le test ne montre rien"
    assert sc.scale.dtype == torch.float32, \
        f"echelle rendue en {sc.scale.dtype}, la valeur est deja appauvrie"
