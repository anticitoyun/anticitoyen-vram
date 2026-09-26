"""224 (chef, 26/09) : la commande de conversion attn+GDN int8-canal documentée dans
docs/CHOIX-FORMAT-GDN.md (pièce 153/153d) doit continuer à s'analyser par le parseur réel de
`acvram convert` — casse si un drapeau cité dans la doc disparaît ou change de type."""
from acvram.cli import build_parser

# Copie exacte de la ligne documentée (docs/CHOIX-FORMAT-GDN.md) : toute dérive entre les deux
# doit être visible en revue de diff, pas seulement ici.
COMMANDE_DOCUMENTEE = (
    "convert SOURCE -o SORTIE --dry-run "
    "--promotion-classes q_proj,k_proj,v_proj,o_proj,linear_attn.qkv,linear_attn.gate,"
    "linear_attn.alpha,linear_attn.beta,linear_attn.out "
    "--max-promotions 1.0 --snr-floor 99 --attn-qkvo-int8-canal --gdn-int8-canal --no-awq"
).replace("SOURCE", "/tmp/source").replace("SORTIE", "/tmp/sortie")


def test_commande_documentee_224_parse_sans_erreur():
    args = build_parser().parse_args(COMMANDE_DOCUMENTEE.split())
    assert args.command == "convert"
    assert args.dry_run is True
    assert args.no_awq is True
    assert args.attn_qkvo_int8_canal is True
    assert args.gdn_int8_canal is True
    assert args.max_promotions == 1.0
    assert args.snr_floor == 99.0
    assert args.promotion_classes == (
        "q_proj,k_proj,v_proj,o_proj,linear_attn.qkv,linear_attn.gate,"
        "linear_attn.alpha,linear_attn.beta,linear_attn.out"
    )


def test_gdn_int8_canal_reste_distinct_de_attn_qkvo_int8_canal():
    """Les deux drapeaux se cumulent (153) : aucun ne doit devenir un alias de l'autre."""
    p = build_parser()
    only_gdn = p.parse_args(["convert", "/tmp/s", "--gdn-int8-canal"])
    only_attn = p.parse_args(["convert", "/tmp/s", "--attn-qkvo-int8-canal"])
    assert only_gdn.gdn_int8_canal and not only_gdn.attn_qkvo_int8_canal
    assert only_attn.attn_qkvo_int8_canal and not only_attn.gdn_int8_canal
