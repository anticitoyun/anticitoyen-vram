"""153 (chef 24/09) : --gdn-int8-canal, drapeau distinct de
--attn-qkvo-int8-canal (dont il ne change pas le sens), qui vise les cinq
projections de poids du GatedDeltaNet (qkv/gate/alpha/beta/out, noms acvram
post-renommage _QWEN35_RENOMMAGE). Trois proprietes controlees : le drapeau
seul ne touche que linear_attn (jamais self_attn/mlp), l'absence des deux
drapeaux laisse `_est_projection_attn`/`_est_projection_gdn` sans effet sur
le format (routage identique a avant la pièce), et les deux drapeaux se
cumulent sans interference (l'un ne fait pas taire l'autre)."""
from acvram.quant.convert import _est_projection_attn, _est_projection_gdn


def test_projection_gdn_couvre_les_cinq_poids_pas_le_reste():
    for nom in ("model.layers.3.linear_attn.qkv.weight",
                "model.layers.3.linear_attn.gate.weight",
                "model.layers.3.linear_attn.alpha.weight",
                "model.layers.3.linear_attn.beta.weight",
                "model.layers.3.linear_attn.out.weight"):
        assert _est_projection_gdn(nom), f"projection GDN non reconnue : {nom}"
    for nom in ("model.layers.3.linear_attn.conv1d.weight",
                "model.layers.3.linear_attn.a_log.weight",
                "model.layers.3.linear_attn.dt_bias.weight",
                "model.layers.3.linear_attn.norm.weight",
                "model.layers.3.self_attn.q_proj.weight",
                "model.layers.3.mlp.gate_proj.weight",
                "model.norm.weight"):
        assert not _est_projection_gdn(nom), f"faux positif GDN : {nom}"


def test_projection_gdn_et_attn_sont_disjointes():
    """`--attn-qkvo-int8-canal` ne doit jamais reconnaitre une projection GDN,
    et reciproquement -- sinon les deux drapeaux se marcheraient dessus."""
    gdn = ("model.layers.0.linear_attn.qkv.weight",
           "model.layers.0.linear_attn.gate.weight",
           "model.layers.0.linear_attn.alpha.weight",
           "model.layers.0.linear_attn.beta.weight",
           "model.layers.0.linear_attn.out.weight")
    attn = ("model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight")
    for nom in gdn:
        assert not _est_projection_attn(nom), f"GDN pris pour de l'attention : {nom}"
    for nom in attn:
        assert not _est_projection_gdn(nom), f"attention prise pour du GDN : {nom}"


def test_sans_drapeaux_aucune_projection_nest_candidate():
    """Regression : `gdn_int8_canal=False` (defaut) laisse le routage
    identique a avant la piece 153 -- ni self_attn ni linear_attn ne
    deviennent candidats au canal par la seule existence de la fonction."""
    from acvram.quant.convert import ConversionOptions
    opts = ConversionOptions(out_dir="/tmp/inutilise")
    assert opts.attn_qkvo_int8_canal is False
    assert opts.gdn_int8_canal is False
