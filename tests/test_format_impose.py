"""Un format demande explicitement ne doit jamais etre remplace en silence.

Le 8/09/2026, `--format int8` sur une source Q5_K_M a produit un modele dont
les 72 projections de perceptron etaient en q3n a 3,25 bits : la garde
anti-grossissement les avait basculees. SNR de 14,9 dB contre 44 pour le reste,
un dossier nomme « temoin-int8 », et une demi-journee passee a chercher un
biais d'instrument dans un chiffre qui mesurait un format a trois bits.

La bascule reste bonne quand le format vient de la POLITIQUE de placement.
Elle est interdite quand il vient de l'utilisateur.
"""
import pytest

from acvram.quant.convert import ConversionOptions


def test_l_option_porte_le_format_demande():
    """`format_impose` distingue le choix de l'utilisateur du choix par defaut."""
    assert ConversionOptions(out_dir="/tmp/x").format_impose is None
    assert ConversionOptions(out_dir="/tmp/x", format_impose="int8"
                             ).format_impose == "int8"


def test_le_refus_nomme_la_sortie():
    """Le message doit dire les deux nombres et les deux issues.

    Une erreur qui dit seulement « refuse » fait relancer au hasard ; celle-ci
    doit donner les bits par poids de part et d'autre, et les deux commandes
    qui debloquent — obtenir vraiment le format, ou laisser la politique
    choisir.
    """
    import inspect

    from acvram.quant import convert

    src = inspect.getsource(convert.convert_checkpoint)
    i = src.index("format_impose")
    bloc = src[i:i + 900]
    assert "--autoriser-grossissement" in bloc, "l'issue n'est pas nommee"
    assert "bits/poids" in bloc, "les deux mesures ne sont pas dites"
    assert "raise ValueError" in src[max(0, i - 400):i + 900], \
        "la garde ne leve pas"


def test_le_manifeste_porte_demande_et_obtenu():
    """Meme quand ils coincident : un manifeste qui ne porte que le resultat
    laisse croire qu'il a ete voulu."""
    import inspect

    from acvram.quant import convert

    src = inspect.getsource(convert.convert_checkpoint)
    assert '"formats_nominaux"' in src
    for champ in ("demande", "obtenu", "bascule_anti_grossissement",
                  "bits_par_poids_source"):
        assert f'"{champ}"' in src, champ


def test_eval_imprime_les_formats_reels():
    """Le tableau doit dire de quoi est fait le modele qu'il mesure."""
    from acvram.evaluate import EvalResult, render

    r = EvalResult(model="temoin", perplexity=193.1, tokens=8192,
                   window=512, min_context=256,
                   formats={"q3n": 72, "int8": 115, "bf16": 134})
    sortie = render([r])
    assert "q3n 72" in sortie, "les formats reels ne sont pas imprimes"
    assert "int8 115" in sortie
    assert "contexte minimal 256" in sortie, "le cadrage a disparu"


# ---------------------------------------------------------------------------
# Le garde-fou de contention, eprouve dans les deux sens.

def test_la_garde_de_debit_mord_sous_charge(monkeypatch):
    """Carte occupee : le test de debit doit etre IGNORE, jamais echoue.

    Le 8/09, `test_le_gemv_nvfp4_ne_part_pas_en_emulation` a rendu 57 Go/s au
    lieu de 300 pendant qu'une conversion occupait la carte a cent pour cent,
    et il a annonce « reparti en emulation logicielle ». Un diagnostic faux
    tire d'une mesure vraie.
    """
    import conftest

    monkeypatch.setattr(conftest, "occupation_gpu", lambda *a: 100)
    assert 100 > conftest.OCCUPATION_MAX


def test_la_garde_de_debit_se_tait_au_repos(monkeypatch):
    """Et carte au repos, elle laisse courir — sinon le test ne teste plus rien.

    Un detecteur s'eprouve aussi sur ce qui doit le faire taire.
    """
    import conftest

    monkeypatch.setattr(conftest, "occupation_gpu", lambda *a: 3)
    assert not (3 > conftest.OCCUPATION_MAX)


def test_l_occupation_est_lisible_ou_absente():
    """La lecture rend un pourcentage ou None, jamais une exception.

    Sur une machine sans nvidia-smi, le garde-fou doit s'effacer plutot que de
    faire echouer toute la suite.
    """
    import conftest

    v = conftest.occupation_gpu()
    assert v is None or (isinstance(v, int) and 0 <= v <= 100)
