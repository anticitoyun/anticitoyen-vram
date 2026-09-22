"""Une disposition de RoPE inconnue doit refuser, pas servir en silence.

Le 8/09/2026, un qwen35 dont `rope.dimension_sections` vaut [11, 11, 10, 0] a
ete servi avec un RoPE NEOX standard. llama.cpp lui applique un M-RoPE
ENTRELACE : les frequences y sont permutees entre paires de dimensions, et meme
a positions egales — texte pur, trois axes identiques — le resultat differe. La
sortie est correcte au debut puis se degrade avec la distance entre positions.

Symptome mesure : la perplexite REMONTE quand on donne plus de contexte (85 a
128 jetons, 173 a 512), la ou un modele dense descend proprement jusqu'a 6,04.
Une demi-journee a ete passee a chercher ailleurs — couche lineaire, service
int8, bornes de notation, agregation — toutes innocentees.

La liste est blanche a dessein : la charge de la preuve va a l'architecture.
"""
import pytest

from acvram.quant.gguf import _ROPE_SECTIONS_CONTIGUES, _garde_rope_sections


def test_pas_de_sections_pas_de_garde():
    """Un RoPE ordinaire n'a pas de sections : rien a verifier."""
    _garde_rope_sections("qwen3", None)
    _garde_rope_sections("qwen3", [])


def test_une_seule_section_utile_est_un_rope_ordinaire():
    _garde_rope_sections("inconnu", [64, 0, 0, 0])


@pytest.mark.parametrize("arch", sorted(_ROPE_SECTIONS_CONTIGUES))
def test_les_dispositions_verifiees_passent(arch):
    """Les vision-langage servis en texte pur : sections contigues, demontre."""
    _garde_rope_sections(arch, [16, 24, 24, 0])


def test_qwen35_passe_car_sa_disposition_a_ete_verifiee():
    """Entrelace, mais equivalent a NEOX en texte pur — donc pas de refus.

    Le garde-fou avait d'abord ete pose en croyant l'inverse. La verification
    dans les deux sources a montre que l'entrelacement ne permute pas les
    frequences et que llama.cpp diffuse la meme position sur les quatre axes
    pour un lot de jetons. Un garde-fou doit refleter ce qu'on sait : le
    laisser refuser un modele sain sur une hypothese tombee serait pire que
    de ne pas l'avoir pose.
    """
    _garde_rope_sections("qwen35", [11, 11, 10, 0])
    _garde_rope_sections("qwen35moe", [11, 11, 10, 0])


def test_une_architecture_inconnue_est_refusee():
    """Et le message doit dire quoi faire, pas seulement que c'est refuse."""
    with pytest.raises(ValueError) as e:
        _garde_rope_sections("archi-de-demain", [16, 16, 16, 16])
    msg = str(e.value)
    assert "archi-de-demain" in msg and "[16, 16, 16, 16]" in msg
    assert "IMROPE" in msg, "le nom de la disposition manquante n'est pas dit"
    assert "_ROPE_SECTIONS_CONTIGUES" in msg, "l'issue n'est pas nommee"
    assert "se degrade avec la longueur" in msg, \
        "le symptome n'est pas decrit : c'est lui qui fait reconnaitre le cas"
