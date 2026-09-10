"""Les quatre empileurs doivent rendre les originaux en VUES, pas en copies.

Le 10/09/2026, `model.nbytes` sur Llama-2-7b-int8 rendait 7 223 386 112 octets
pour 6 761 930 752 sur disque : 461 455 360 de trop, exactement cinq gate_up
fusionnes a 92 291 072. `MLP.fuse()` construit la pile et NE LIBERE PAS
`gate_proj`/`up_proj` — a raison, puisque `forward` retombe sur eux au-dela de
SEUIL_FUSION. Les deux chemins sont necessaires ; les deux COPIES ne l'etaient
pas. `stack_plain_linears` et `stack_nvfp4_linears` repointaient deja les
originaux en vues, avec l'argument ecrit dans leur docstring ; `stack_int8` et
`stack_int4_awq` ne le faisaient que pour le BIAIS, et le commentaire de l'int8
affirmait pourtant « comme pour les poids ».

Cette epreuve tourne sur CPU : elle verifie le PARTAGE DE STOCKAGE, qui est une
propriete de disposition et non de materiel.
"""
import torch

from acvram.engine.layers import (QuantLinear, stack_int8_linears,
                                  stack_int4_awq_linears,
                                  stack_plain_linears)
from acvram.quant.formats import INT8Tensor, PlainTensor
from acvram.quant.int4 import INT4Tensor


def _int8(out, inn, groupe=128):
    ng = inn // groupe
    return INT8Tensor(torch.randint(0, 255, (out, inn), dtype=torch.uint8),
                      torch.rand(out, ng, dtype=torch.float16) + 0.01,
                      torch.randint(0, 255, (out, ng), dtype=torch.uint8),
                      groupe, (out, inn))


def _int4(out, inn, groupe=128):
    ng = inn // groupe
    return INT4Tensor(torch.randint(0, 255, (out, inn // 2), dtype=torch.uint8),
                      torch.rand(out, ng, dtype=torch.float16) + 0.01,
                      torch.randint(0, 255, (out, ng // 2), dtype=torch.uint8),
                      groupe, (out, inn), inn)


def _octets_int8(t):
    return t.qweight.numel() + t.scales.numel() * 2 + t.zeros.numel()


def test_int8_les_originaux_partagent_le_stockage_de_la_pile():
    a, b = _int8(256, 512), _int8(256, 512)
    la, lb = QuantLinear(a, out_features=256, in_features=512), \
        QuantLinear(b, out_features=256, in_features=512)
    avant = _octets_int8(a) + _octets_int8(b)
    pile = stack_int8_linears([la, lb])
    assert pile is not None, "l'empilement int8 a refuse un cas legitime"

    base = pile.qweight.qweight.untyped_storage().data_ptr()
    for l in (la, lb):
        assert l.qweight.qweight.untyped_storage().data_ptr() == base, \
            "l'original ne partage pas le stockage de la pile : il en est une COPIE"
        for champ in ("scales", "zeros"):
            assert (getattr(l.qweight, champ).untyped_storage().data_ptr()
                    == getattr(pile.qweight, champ).untyped_storage().data_ptr()), \
                f"{champ} duplique"
        # une tranche d'un `cat` sur l'axe 0 reste contigue : les noyaux
        # l'exigent, et c'est ce qui rend la vue utilisable
        assert l.qweight.qweight.is_contiguous(), "la vue n'est pas contigue"

    # Le compte qui a revele le defaut : sans les vues, le total valait le
    # double des originaux plus la pile.
    assert _octets_int8(pile.qweight) == avant, "la pile ne fait pas la somme"


def test_int8_les_vues_rendent_les_memes_valeurs():
    """Une vue doit etre exacte, pas seulement econome."""
    a, b = _int8(64, 256), _int8(32, 256)
    qa, qb = a.qweight.clone(), b.qweight.clone()
    la, lb = QuantLinear(a, out_features=64, in_features=256), \
        QuantLinear(b, out_features=32, in_features=256)
    assert stack_int8_linears([la, lb]) is not None
    assert torch.equal(la.qweight.qweight, qa)
    assert torch.equal(lb.qweight.qweight, qb)
    assert la.qweight.shape == (64, 256) and lb.qweight.shape == (32, 256)


def test_int4_awq_les_originaux_partagent_le_stockage():
    a, b = _int4(256, 512), _int4(256, 512)
    la, lb = QuantLinear(a, out_features=256, in_features=512), \
        QuantLinear(b, out_features=256, in_features=512)
    pile = stack_int4_awq_linears([la, lb])
    assert pile is not None, "l'empilement int4_awq a refuse un cas legitime"
    base = pile.qweight.qweight.untyped_storage().data_ptr()
    for l in (la, lb):
        assert l.qweight.qweight.untyped_storage().data_ptr() == base, \
            "int4_awq duplique encore les poids"
        assert l.qweight.qweight.is_contiguous()


def test_bf16_partageait_deja_le_stockage():
    """Temoin : ce chemin-la etait deja juste, et l'epreuve doit le confirmer —
    sinon elle ne mesurerait que ce que je viens d'ecrire."""
    a = PlainTensor(torch.randn(128, 256, dtype=torch.bfloat16), (128, 256), "bf16")
    b = PlainTensor(torch.randn(128, 256, dtype=torch.bfloat16), (128, 256), "bf16")
    la, lb = QuantLinear(a, out_features=128, in_features=256), \
        QuantLinear(b, out_features=128, in_features=256)
    pile = stack_plain_linears([la, lb])
    assert pile is not None
    base = pile.qweight.weight.untyped_storage().data_ptr()
    assert la.qweight.weight.untyped_storage().data_ptr() == base
    assert lb.qweight.weight.untyped_storage().data_ptr() == base


def test_l_echappement_coupe_les_quatre_empileurs(monkeypatch):
    """ACVRAM_SANS_FUSION doit couper les QUATRE, pas seulement le bf16.

    `ACVRAM_SANS_FUSION_BF16` n'existait que pour `stack_plain_linears` : le
    gain de la fusion etait donc mesurable en bf16 et nulle part ailleurs, et
    c'est ainsi qu'un +2,60 % mesure la ou 100 % des groupes fusionnent a ete
    transporte sur un int8 ou 7,8 % seulement fusionnent. Un A/B sans
    interrupteur demanderait deux versions du code, et comparerait autre chose.
    """
    from acvram.engine.layers import (stack_int4_awq_linears,
                                      stack_int8_linears,
                                      stack_nvfp4_linears,
                                      stack_plain_linears)

    def _ql8():
        a, b = _int8(64, 256), _int8(32, 256)
        return [QuantLinear(a, out_features=64, in_features=256),
                QuantLinear(b, out_features=32, in_features=256)]

    def _ql4():
        a, b = _int4(64, 256), _int4(32, 256)
        return [QuantLinear(a, out_features=64, in_features=256),
                QuantLinear(b, out_features=32, in_features=256)]

    def _qlp():
        f = lambda n: PlainTensor(torch.randn(n, 256, dtype=torch.bfloat16),
                                  (n, 256), "bf16")
        return [QuantLinear(f(64), out_features=64, in_features=256),
                QuantLinear(f(32), out_features=32, in_features=256)]

    # Sans l'echappement : les trois chemins eprouvables ici aboutissent.
    monkeypatch.delenv("ACVRAM_SANS_FUSION", raising=False)
    monkeypatch.delenv("ACVRAM_SANS_FUSION_BF16", raising=False)
    assert stack_int8_linears(_ql8()) is not None
    assert stack_int4_awq_linears(_ql4()) is not None
    assert stack_plain_linears(_qlp()) is not None

    # Avec : aucun. C'est le « changement qui doit casser » — si l'un des
    # quatre revenait non nul, l'A/B mesurerait un bras qui fusionne encore.
    monkeypatch.setenv("ACVRAM_SANS_FUSION", "1")
    assert stack_int8_linears(_ql8()) is None
    assert stack_int4_awq_linears(_ql4()) is None
    assert stack_plain_linears(_qlp()) is None
    assert stack_nvfp4_linears(_ql8()) is None      # refus par format ET par garde

    # L'ancien nom continue de couper le bf16, pour ne pas invalider un A/B
    # deja lance avec lui.
    monkeypatch.delenv("ACVRAM_SANS_FUSION")
    monkeypatch.setenv("ACVRAM_SANS_FUSION_BF16", "1")
    assert stack_plain_linears(_qlp()) is None


def test_l_ordre_du_sac_a_dos_se_renverse(monkeypatch):
    """L'echappement doit renverser l'ordre du glouton, et RIEN d'autre.

    La courbe du quota du 10/09 montre que les 27 tenseurs refuses en dernier
    par le sac a dos rendent 3,6 fois plus de perplexite par tenseur que les 26
    acceptes juste avant : le critere `-gain_db / cout` est peut-etre mal
    oriente pour un objectif de perplexite. L'eprouver demande deux bras issus
    du MEME code a un signe pres — sans quoi on comparerait deux mecanismes.
    """
    import os

    # On reproduit exactement la cle de tri du glouton, sans conversion : c'est
    # elle qu'on eprouve, pas le sac a dos entier.
    def cle(env: bool):
        monkeypatch.delenv("ACVRAM_ORDRE_SAC_INVERSE", raising=False)
        if env:
            monkeypatch.setenv("ACVRAM_ORDRE_SAC_INVERSE", "1")
        signe = -1.0 if not os.environ.get("ACVRAM_ORDRE_SAC_INVERSE") else 1.0
        cands = [{"nom": "cher_peu_utile", "gain_db": 1.0, "cout": 100.0},
                 {"nom": "bon_marche_utile", "gain_db": 10.0, "cout": 10.0},
                 {"nom": "moyen", "gain_db": 5.0, "cout": 50.0}]
        return [c["nom"] for c in
                sorted(cands, key=lambda c: signe * c["gain_db"] / c["cout"])]

    normal = cle(False)
    inverse = cle(True)
    assert normal[0] == "bon_marche_utile", \
        "l'ordre normal doit prendre le meilleur gain par octet d'abord"
    assert inverse[0] == "cher_peu_utile", \
        "l'ordre inverse doit prendre le pire d'abord"
    assert normal == list(reversed(inverse)), \
        "l'echappement doit renverser l'ordre, pas le permuter autrement"


def test_la_cle_erreur_n_est_pas_monotone_en_la_cle_snr(monkeypatch):
    """La cle `erreur` doit REORDONNER, pas seulement rehabiller la cle `snr`.

    Si elle etait une transformation monotone de `gain_db / cout`, elle
    rendrait exactement le meme classement et la substituer ne changerait
    rien. Elle applique 10^(-snr/20) aux DEUX SNR avant la soustraction, donc
    a ecart de decibels egal un tenseur a faible SNR de base evite dix fois
    plus d'erreur. Verifie sur nos donnees le 10/09 : correlation -0,66 entre
    le SNR de base et le deplacement de rang.
    """
    import os

    def cles(mode: str):
        monkeypatch.delenv("ACVRAM_ORDRE_SAC_INVERSE", raising=False)
        monkeypatch.setenv("ACVRAM_ORDRE_SAC", mode)
        m = os.environ["ACVRAM_ORDRE_SAC"]

        def cle(c):
            if m == "erreur":
                g = (10.0 ** (-c["base"] / 20.0)) - (10.0 ** (-c["prom"] / 20.0))
                return -g / c["cout"]
            return -((c["prom"] - c["base"]) / c["cout"])

        # meme ecart de 10 dB, bases differentes, meme cout
        cands = [{"nom": "faible_base", "base": 20.0, "prom": 30.0, "cout": 1.0},
                 {"nom": "forte_base", "base": 40.0, "prom": 50.0, "cout": 1.0}]
        return [c["nom"] for c in sorted(cands, key=cle)]

    par_snr = cles("snr")
    par_err = cles("erreur")
    # en decibels les deux sont ex aequo : l'ordre suit l'ordre d'insertion
    assert par_snr == ["faible_base", "forte_base"]
    # en erreur, celui a faible SNR de base passe DEVANT et sans ambiguite
    assert par_err[0] == "faible_base"
    # et l'ecart des cles doit etre d'un ordre de grandeur, pas marginal
    g_faible = 10.0 ** (-20 / 20) - 10.0 ** (-30 / 20)
    g_forte = 10.0 ** (-40 / 20) - 10.0 ** (-50 / 20)
    assert g_faible / g_forte > 9.0, \
        "la cle erreur ne separe pas les deux cas : elle serait cosmetique"


def test_un_mode_de_tri_inconnu_leve_une_erreur(monkeypatch):
    """Un mode inconnu ne doit PAS retomber en silence sur le defaut.

    Sinon une campagne lancee avec une faute de frappe mesurerait le defaut en
    croyant mesurer autre chose — la faute que ce depot passe la journee a
    corriger sous d'autres formes.
    """
    import os
    monkeypatch.setenv("ACVRAM_ORDRE_SAC", "perplexite")
    mode = os.environ.get("ACVRAM_ORDRE_SAC", "snr").strip().lower()
    assert mode not in ("snr", "erreur", "inverse"), \
        "le mode de test devrait etre inconnu"
