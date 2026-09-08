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


def test_une_architecture_inconnue_est_refusee():
    """Et le message doit dire quoi faire, pas seulement que c'est refuse."""
    with pytest.raises(ValueError) as e:
        _garde_rope_sections("qwen35", [11, 11, 10, 0])
    msg = str(e.value)
    assert "qwen35" in msg and "[11, 11, 10, 0]" in msg
    assert "IMROPE" in msg, "le nom de la disposition manquante n'est pas dit"
    assert "_ROPE_SECTIONS_CONTIGUES" in msg, "l'issue n'est pas nommee"
    assert "se degrade avec la longueur" in msg, \
        "le symptome n'est pas decrit : c'est lui qui fait reconnaitre le cas"
