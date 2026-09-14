"""Garde de lot sur la speculation (chef, 14/09) : n-gram gagne a b_reel=1,
coute a b_reel=12 (revue/verdict-cout-verification-ngram-b12-14-09.md,
-49,9 % de debit). GardeSpeculation doit desactiver au-dela du seuil de lot
ET quand le gain reel mesure tombe sous 1,05, puis se reactiver quand le lot
redescend."""
from acvram.engine.speculative import GardeSpeculation


def test_eligible_sous_le_seuil_de_lot():
    g = GardeSpeculation(lot_max=2)
    assert g.eligible(1) is True
    assert g.eligible(2) is True


def test_inegible_au_dessus_du_seuil_de_lot():
    g = GardeSpeculation(lot_max=2)
    assert g.eligible(3) is False
    assert g.eligible(12) is False


def test_garde_glissante_desactive_sous_le_gain_minimal():
    g = GardeSpeculation(lot_max=12, fenetre=4, gain_min=1.05)
    assert g.eligible(12) is True
    # gain quasi nul (verification qui n'accepte presque rien de plus) :
    # jetons_emis == b_reel a chaque pas => ratio 1.0 < 1.05
    for _ in range(4):
        g.enregistrer(jetons_emis=12, b_reel=12)
    assert g.eligible(12) is False


def test_garde_glissante_reste_active_au_dessus_du_gain_minimal():
    g = GardeSpeculation(lot_max=12, fenetre=4, gain_min=1.05)
    for _ in range(4):
        g.enregistrer(jetons_emis=20, b_reel=12)  # ratio 1,667
    assert g.eligible(12) is True


def test_transition_1_vers_12_vers_1():
    """Le scenario demande par chef : b_reel 1 -> 12 -> 1."""
    g = GardeSpeculation(lot_max=2, fenetre=4, gain_min=1.05)
    # b_reel=1 : eligible, gain fort -> reste eligible
    assert g.eligible(1) is True
    g.enregistrer(jetons_emis=20, b_reel=1)
    # b_reel monte a 12 : inegible par le seuil de lot, quel que soit le gain
    for _ in range(5):
        assert g.eligible(12) is False
    # b_reel redescend a 1 : rearme (chance neuve), pas de memoire de l'echec
    # a b=12 puisque le seuil de lot l'excluait deja de la fenetre glissante
    assert g.eligible(1) is True


def test_garde_glissante_desactivee_persiste_tant_que_le_lot_ne_redescend_pas():
    g = GardeSpeculation(lot_max=12, fenetre=4, gain_min=1.05)
    for _ in range(4):
        g.eligible(12)
        g.enregistrer(jetons_emis=12, b_reel=12)
    assert g.eligible(12) is False
    # le lot reste haut : pas de rearmement
    assert g.eligible(12) is False


def test_garde_glissante_rearme_seulement_apres_une_montee_au_dessus_du_seuil():
    """Le rearmement suit une transition HAUTE -> BASSE (le lot est monte
    au-dessus du seuil puis en redescend) -- pas une simple repetition du
    meme lot sous le seuil, qui reste desactivee tant que le gain ne revient
    pas de lui-meme dans la fenetre glissante."""
    g = GardeSpeculation(lot_max=2, fenetre=4, gain_min=1.05)
    for _ in range(4):
        g.eligible(2)
        g.enregistrer(jetons_emis=2, b_reel=2)  # ratio 1.0 < 1.05
    assert g.eligible(2) is False
    # le lot reste sous le seuil : pas de transition haute -> basse, la
    # desactivation persiste malgre un nouvel appel au meme lot
    assert g.eligible(2) is False
    # le lot monte au-dessus du seuil puis en redescend : rearmement
    assert g.eligible(12) is False
    assert g.eligible(1) is True
