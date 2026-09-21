"""Le format declare doit correspondre a ce qui est ecrit.

Le 9/09/2026 : `Ornith-1.5-35B` declarait `format: nvfp4` sur trois blobs
d'experts groupes du MTP qui n'ont jamais ete quantifies — cle physique
directe, dtype reel F16. 1,158 Gio d'ecart, 5,85 % du modele, et tout calcul
de taille fonde sur le manifeste faux.

Un format declare qui ne correspond pas au stockage est PIRE qu'un format
absent : il fait croire qu'on sait.
"""
from acvram.quant.convert import _verifier_formats_declares


def test_un_tenseur_quantifie_ecrit_en_direct_est_signale(capsys):
    man = {"tensors": {"model.mtp.0.experts__exps__": {"format": "nvfp4"}}}
    wm = {"model.mtp.0.experts__exps__": "acvram-00000.safetensors"}
    _verifier_formats_declares(man, wm)
    sortie = capsys.readouterr().out
    assert "experts__exps__" in sortie
    assert "surestime" in sortie


def test_un_tenseur_quantifie_ecrit_en_morceaux_ne_l_est_pas(capsys):
    """Le cas normal ne doit pas crier : une garde qui crie toujours n'est
    plus lue."""
    man = {"tensors": {"model.layers.0.mlp.up_proj.weight": {"format": "nvfp4"}}}
    wm = {"model.layers.0.mlp.up_proj.weight.qweight": "a.safetensors",
          "model.layers.0.mlp.up_proj.weight.block_scale": "a.safetensors"}
    _verifier_formats_declares(man, wm)
    assert capsys.readouterr().out == ""


def test_un_tenseur_bf16_en_direct_est_normal(capsys):
    """bf16 n'est pas un format quantifie : une cle directe y est attendue."""
    man = {"tensors": {"model.norm.weight": {"format": "bf16"}}}
    _verifier_formats_declares(man, {"model.norm.weight": "a.safetensors"})
    assert capsys.readouterr().out == ""


def test_un_tenseur_absent_du_weight_map_n_est_pas_signale(capsys):
    """Absent n'est pas ecrit en direct : les deux cas ne se confondent pas."""
    man = {"tensors": {"model.x.weight": {"format": "nvfp4"}}}
    _verifier_formats_declares(man, {})
    assert capsys.readouterr().out == ""
