"""Quand le groupe entier ne s'empile pas, un sous-ensemble le peut.

Mesure du 9/09/2026 sur les formes de Qwen2.5-14B :

    3 appels separes             27,69 us
    1 appel empile               22,06 us
    2 empiles + 1 seul (k+v)     24,48 us   ->  57 % du gain
    2 empiles + 1 seul (q+k)     24,93 us   ->  49 %

**Zero octet, zero changement de qualite** — contrairement a l'uniformisation
de format, qui double les octets d'un tenseur promu, et au retrait de l'AWQ,
qui touche la qualite. Sur les 698 groupes du parc bloques par le seul format,
c'est la seule voie qui reste juste que la promotion ait ete meritee ou
fortuite — et le quota sature sur 27 modeles rend cette question ouverte.

L'ordre compte, a contre-sens : le gain vient du SAUVETAGE DES PETITS noyaux,
pas de l'agrandissement du gros. Empiler les deux plus petites en sauve deux ;
empiler la grosse avec une petite n'en sauve qu'une, et cela QUELLE QUE SOIT
la petite — q+k et q+v rendent 49 % a 0,01 us pres, ce qui etablit le
mecanisme et non une simple regle descriptive.
"""
import inspect

from acvram.engine import model


def test_la_fusion_partielle_choisit_la_paire_la_moins_couteuse():
    """Pas 'deux quelconques' : celle dont la somme des lignes est minimale."""
    src = inspect.getsource(model.Attention.fuse)
    assert "qkv_partiel" in src, "aucune fusion partielle"
    i = src.index("for i, j in ((0, 1), (0, 2), (1, 2))")
    bloc = src[i:i + 600]
    assert "shape[0] + " in bloc, "le cout n'est pas la somme des lignes"
    assert "cout < meilleur_cout" in bloc, "la paire minimale n'est pas retenue"


def test_les_trois_paires_sont_essayees():
    """Une seule paire essayee laisserait passer des groupes empilables."""
    src = inspect.getsource(model.Attention.fuse)
    assert "((0, 1), (0, 2), (1, 2))" in src, "les trois paires ne sont pas essayees"


def test_la_fusion_totale_reste_prioritaire():
    """Le partiel ne doit jamais remplacer le total quand celui-ci est
    possible : 57 % n'est pas 100 %."""
    src = inspect.getsource(model.Attention.fuse)
    i_total = src.index("self.qkv_proj = _empiler(lins)")
    i_partiel = src.index("qkv_partiel = (pile")
    assert i_total < i_partiel, "le partiel est tente avant le total"
    assert "return True" in src[i_total:i_partiel], \
        "la fusion totale ne sort pas immediatement"


def test_les_sorties_reviennent_dans_l_ordre_q_k_v():
    """Le partiel reordonne : une inversion donnerait des jetons faux sans
    qu'aucune garde ne le voie."""
    src = inspect.getsource(model.Attention._proj)
    i = src.index("qkv_partiel")
    bloc = src[i:i + 500]
    assert "sorties[i], sorties[j], sorties[reste]" in bloc, \
        "les sorties ne sont pas remises a leur place"
    assert "qr, kr, vr = sorties" in bloc


def test_le_partiel_ne_s_applique_pas_quand_k_egale_v():
    """Avec k_eq_v il n'y a que deux projections : rien a partitionner."""
    src = inspect.getsource(model.Attention.fuse)
    assert "if len(lins) < 3:" in src, "aucune garde sur le nombre de projections"
