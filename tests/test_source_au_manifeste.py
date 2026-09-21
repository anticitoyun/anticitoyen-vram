"""Un converti doit dire d ou il vient.

Le 9/09/2026, retrouver la source de `Qwen2.5-Coder-14B-bf16-pur` a demande
de comparer ses tenseurs bit a bit a un candidat : le manifeste portait
`options.out_dir` et aucune trace de l entree. Un couple a une variable —
meme modele converti en deux formats — exige la MEME origine ; sans cette
cle, on ne peut pas garantir qu on mesure deux formats plutot que deux
modeles.
"""
import json
import os

from acvram.quant.convert import _octets_du_checkpoint


def test_le_manifeste_porte_la_source():
    """chemin ET nom ET octets : le chemin seul se perd au premier
    deplacement, le nom seul ne distingue pas deux copies."""
    import inspect

    from acvram.quant import convert

    src = inspect.getsource(convert.convert_checkpoint)
    assert '"source"' in src, "le manifeste ne porte pas la source"
    for champ in ("chemin", "nom", "octets"):
        assert f'"{champ}"' in src, champ


def test_les_octets_ne_comptent_que_les_poids(tmp_path):
    """Un README ou un tokenizer ne doit pas entrer dans la somme : elle
    servira a reconnaitre une source deplacee, et le moindre fichier annexe
    ajoute ou retire la ferait mentir."""
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 1000)
    (tmp_path / "model-00002.safetensors").write_bytes(b"y" * 500)
    (tmp_path / "README.md").write_bytes(b"z" * 99999)
    (tmp_path / "tokenizer.json").write_bytes(b"w" * 77777)
    assert _octets_du_checkpoint(str(tmp_path)) == 1500


def test_les_octets_descendent_dans_les_sous_dossiers(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "part.bin").write_bytes(b"x" * 42)
    (tmp_path / "m.gguf").write_bytes(b"y" * 8)
    assert _octets_du_checkpoint(str(tmp_path)) == 50


def test_un_chemin_absent_ne_leve_pas(tmp_path):
    """Une conversion ne doit pas echouer parce que la somme est impossible :
    la cle est une trace, pas une garde."""
    assert _octets_du_checkpoint(str(tmp_path / "nexiste-pas")) == 0
