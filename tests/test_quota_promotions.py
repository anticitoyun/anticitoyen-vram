"""Le quota de promotions doit dire quand il decide a la place du SNR.

Le 9/09/2026 : `max_promotions = 0,15` sature sur 27 modeles du parc sur 110,
a l'unite pres. Sur `qwen25-coder-14b`, 579 tenseurs donnent un plafond de
87,00 et il y a exactement 87 promotions.

Quand le quota sature, le critere n'est plus le SNR mais l'ORDRE DE PARCOURS
du checkpoint : deux tenseurs de SNR identique recoivent des sorts opposes
selon leur position. La signature est visible au manifeste — sur 48 groupes
q/k/v, AUCUN n'a ses trois membres promus quand 31 en ont exactement un. Une
difficulte propre a la couche produirait des 3/3 ; il n'y en a pas un seul.
"""
import inspect

from acvram.quant import convert


def test_le_plafond_utilise_le_meme_denominateur_que_la_condition():
    """La condition compte `keys` ; un plafond calcule sur `attendus`
    donnerait un seuil de saturation faux — la mauvaise population."""
    src = inspect.getsource(convert.convert_checkpoint)
    i = src.index("plafond = opts.max_promotions")
    assert "len(keys) + 1" in src[i:i + 120], \
        "le plafond n'utilise pas le denominateur de la condition"


def test_la_saturation_est_signalee_et_consignee():
    src = inspect.getsource(convert.convert_checkpoint)
    assert "quota_promotions_sature" in src, "la saturation n'entre pas au manifeste"
    i = src.index("quota de promotions SATURE")
    bloc = src[i - 200:i + 500]
    assert "ordre de" in bloc and "parcours" in bloc, \
        "le message ne dit pas ce qui decide a la place du SNR"


def test_le_snr_de_chaque_tenseur_est_consigne():
    """Sans le SNR de tous les tenseurs, on ne peut pas repondre a la question
    qui juge le quota : un NON promu est-il pire qu'un promu ?"""
    src = inspect.getsource(convert.convert_checkpoint)
    assert 'entry["snr_db"]' in src, "le SNR n'est pas consigne par tenseur"
    i = src.index('entry["snr_db"]')
    # il doit etre pose pour TOUS, pas seulement dans la branche de promotion
    assert 'if "out_snr_db" in metrics' in src[i - 200:i], \
        "le SNR n'est consigne que pour les promus"
