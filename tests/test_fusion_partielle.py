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

import pytest

from acvram.engine import model


@pytest.fixture(autouse=True)
def _voie_partielle_active(monkeypatch):
    """La voie 2+1 est COUPEE par defaut depuis la mesure de poste2
    (-12,16 % sur qwen25-coder-14b). Ces epreuves portent sur sa justesse,
    pas sur son activation : elles l'allument explicitement."""
    monkeypatch.setenv("ACVRAM_FUSION_PARTIELLE", "1")


def _attention_trois_formats(formats_qkv):
    """Une Attention dont q, k et v ont des TAILLES differentes : une
    permutation des indices ne peut pas se cacher derriere une symetrie."""
    import torch

    from acvram.engine.config import ModelSpec
    from acvram.engine.layers import QuantLinear
    from acvram.quant import formats

    def _lin(sortie, entree, fmt, graine):
        torch.manual_seed(graine)
        w = torch.randn(sortie, entree, dtype=torch.float32) * 0.02
        return QuantLinear(formats.quantize(w, fmt, group_size=128))

    spec = ModelSpec(name="t", architecture="llama", hidden_size=256,
                     intermediate_size=512, num_layers=1,
                     num_attention_heads=8, num_key_value_heads=2,
                     vocab_size=32, max_position_embeddings=64)
    fq, fk, fv = formats_qkv
    return model.Attention(spec, _lin(256, 256, fq, 1), _lin(64, 256, fk, 2),
                           _lin(64, 256, fv, 3), _lin(256, 256, fq, 4),
                           rope=None)


def test_la_paire_retenue_est_celle_du_format_majoritaire():
    """Non pas 'la moins couteuse' : la SEULE possible.

    L'ancienne epreuve lisait le texte de `fuse` et exigeait d'y trouver une
    comparaison de couts. Elle etait juste sur le code et fausse sur le
    monde : un groupe a trois membres et deux formats n'offre JAMAIS deux
    paires empilables, puisque deux membres ne s'empilent que s'ils ont le
    meme format. Le tri par cout ne s'exerce donc jamais, et la mesure qui le
    justifiait — k+v a 57 % contre q+k a 49 % — decrivait un choix que le
    moteur n'a pas a faire. Ce que le code doit garantir est plus simple et
    verifiable : la paire retenue est celle du format majoritaire, et le
    membre laisse seul est le minoritaire.
    """
    att = _attention_trois_formats(("int8", "nvfp4", "nvfp4"))
    assert att.fuse() is True
    assert att.qkv_proj is None, "le total ne devrait pas etre possible"
    _, (i, j), reste, _ = att.qkv_partiel
    assert (i, j) == (1, 2), f"paire retenue {(i, j)} au lieu de k+v"
    assert reste == 0, f"membre isole {reste} au lieu de q"


def test_le_membre_isole_suit_le_format_minoritaire():
    """Meme groupe, minoritaire deplace : la paire suit, sans regle ecrite
    ailleurs que dans les formats."""
    att = _attention_trois_formats(("nvfp4", "int8", "nvfp4"))
    assert att.fuse() is True
    _, (i, j), reste, _ = att.qkv_partiel
    assert (i, j) == (0, 2) and reste == 1


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


# --- le compteur de refus garde la meme unite -------------------------------

def test_l_exploration_des_paires_ne_compte_pas_trois_refus():
    """Le 9/09/2026, une mesure a rendu « 240 refus » la ou il y a 80 groupes :
    la fusion partielle essaie TROIS paires par groupe bloque, et chaque echec
    etait compte. L'unite du compteur changeait selon le chemin — un lecteur y
    voyait un nombre de groupes."""
    import torch

    from acvram.engine.layers import (bilan_fusion_nvfp4, explorer_sans_compter,
                                      stack_nvfp4_linears)
    from acvram.quant.nvfp4 import quantize_nvfp4
    from acvram.engine.layers import QuantLinear

    def lin(sortie, entree, graine):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(sortie, entree, generator=g) * 0.02).to(torch.bfloat16)
        return QuantLinear(quantize_nvfp4(w))

    avant = sum(bilan_fusion_nvfp4().values())
    with explorer_sans_compter():
        for _ in range(3):
            stack_nvfp4_linears([lin(64, 128, 1), lin(64, 256, 2)])   # refus
    assert sum(bilan_fusion_nvfp4().values()) == avant, \
        "les tentatives d'exploration sont comptees comme des refus de groupe"


def test_hors_exploration_le_refus_est_bien_compte():
    """Le silence ne doit pas fuir hors du contexte."""
    import torch

    from acvram.engine.layers import bilan_fusion_nvfp4, stack_nvfp4_linears, QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4

    def lin(sortie, entree, graine):
        g = torch.Generator().manual_seed(graine)
        return QuantLinear(quantize_nvfp4(
            (torch.randn(sortie, entree, generator=g) * 0.02).to(torch.bfloat16)))

    avant = sum(bilan_fusion_nvfp4().values())
    stack_nvfp4_linears([lin(64, 128, 3), lin(64, 256, 4)])
    assert sum(bilan_fusion_nvfp4().values()) == avant + 1
