"""Garde `refus_nvfp4_sans_gpu` (chef 21/09) : un nom de sortie ou `--format nvfp4` sans palier GPU dans le plan
est refusé (rc 2), le message nomme memory/tiering.py:360-366. Le 30B « -nvfp4-vision » du 20/09 avait été converti
sans carte : int4_awq planifié processeur sous un nom nvfp4 (`poste1-hypotheses-qvl-30b-21-09`)."""
from types import SimpleNamespace
import pytest
from acvram.quant.convert import refus_nvfp4_sans_gpu


def _plan(*kinds):
    return SimpleNamespace(tiers=[SimpleNamespace(kind=k, weight_format=("nvfp4" if k == "gpu" else "int4_awq")) for k in kinds])


def test_nom_nvfp4_sans_palier_gpu_refuse():
    m = refus_nvfp4_sans_gpu(_plan("host"), "/parc/Qwen3-VL-30B-A3B-abl-nvfp4-vision", None)
    assert m and "aucun palier GPU" in m and "tiering.py:360-366" in m and "int4_awq" in m


def test_format_nvfp4_sans_palier_gpu_refuse_meme_nom_neutre():
    m = refus_nvfp4_sans_gpu(_plan("host"), "/parc/Modele-x", "nvfp4")
    assert m and "--format nvfp4" in m


def test_palier_gpu_ou_nom_neutre_passent():
    assert refus_nvfp4_sans_gpu(_plan("gpu", "host"), "/parc/Modele-nvfp4", "nvfp4") is None
    assert refus_nvfp4_sans_gpu(_plan("host"), "/parc/Modele-int4-awq", None) is None
    assert refus_nvfp4_sans_gpu(_plan(), "/parc/Modele", None) is None


def test_cmd_convert_refuse_avant_toute_conversion(tiny_checkpoint, tmp_path, monkeypatch):
    """La garde est BRANCHÉE dans cmd_convert : plan sans GPU (rig sans carte) + sortie « …-nvfp4 » → rc 2,
    convert_checkpoint jamais appelé. Casse si la garde ou son appel disparaît."""
    import io, contextlib
    from acvram import cli
    import acvram.hardware.detect as det
    import acvram.quant.convert as conv
    appels = []
    sans_carte = det.Rig(gpus=[], host=det.HostMemory(total=64 << 30, available=48 << 30))
    monkeypatch.setattr(det, "detect_rig", lambda profile=None: sans_carte)     # importé dans cmd_convert : patché à la source
    monkeypatch.setattr(conv, "convert_checkpoint", lambda *a, **k: appels.append(1))
    monkeypatch.setattr(cli, "convert_checkpoint", lambda *a, **k: appels.append(1), raising=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = cli.main(["convert", tiny_checkpoint, "--out", str(tmp_path / "tiny-nvfp4"), "--no-awq"])
    assert rc == 2 and appels == [] and "tiering.py:360-366" in out.getvalue(), out.getvalue()[-600:]
    # le même plan sous un nom qui dit son format passe la garde (la conversion factice est alors appelée)
    with contextlib.redirect_stdout(io.StringIO()):
        rc2 = cli.main(["convert", tiny_checkpoint, "--out", str(tmp_path / "tiny-int4-awq"), "--no-awq"])
    assert appels, "la conversion n a pas été appelée sur un nom neutre : la garde refuse trop"
