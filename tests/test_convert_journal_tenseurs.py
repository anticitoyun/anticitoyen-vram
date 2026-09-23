"""Manon 21/09 (31B `--quant-device cpu`, 96 min par shard, rien d incrémental) :
la conversion sur processeur journalise UNE ligne par tenseur sur stderr
(heure, rang, nom, forme, secondes), le dernier compris ; sur carte ou avec
`journal_tenseurs=False`, rien. Tests de conversion existants inchangés."""
import re

import torch

from acvram.quant.convert import ConversionOptions, _journal_tenseurs


def _lignes(capsys):
    return [l for l in capsys.readouterr().err.splitlines() if l.startswith("[convert] ")]


def test_journal_une_ligne_par_tenseur_dernier_compris(capsys):
    j = _journal_tenseurs(ConversionOptions(out_dir="x", quant_device="cpu"), torch.device("cpu"))
    assert j is not None
    j("model.layers.0.a", (4, 8), 0)
    j("model.layers.0.b", (16, 8), 1)
    j(None, None, 2)
    l = _lignes(capsys)
    assert len(l) == 2, l
    assert re.match(r"\[convert\] \d\d:\d\d:\d\d #0\s+model\.layers\.0\.a \[4, 8\] \d+\.\d\d s$", l[0]), l[0]
    assert "#1" in l[1] and "model.layers.0.b [16, 8]" in l[1]


def test_journal_automatique_processeur_seulement(capsys, monkeypatch):
    monkeypatch.delenv("ACVRAM_JOURNAL_TENSEURS", raising=False)
    assert _journal_tenseurs(ConversionOptions(out_dir="x"), torch.device("cuda", 0)) is None
    assert _journal_tenseurs(ConversionOptions(out_dir="x"), torch.device("cpu")) is not None
    assert _journal_tenseurs(ConversionOptions(out_dir="x", journal_tenseurs=False), torch.device("cpu")) is None
    monkeypatch.setenv("ACVRAM_JOURNAL_TENSEURS", "1")
    assert _journal_tenseurs(ConversionOptions(out_dir="x"), torch.device("cuda", 0)) is not None
    assert _lignes(capsys) == []


def test_conversion_reelle_sur_processeur_journalise_chaque_tenseur(tiny_checkpoint, target_rig, tmp_path, capsys):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import convert_checkpoint
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=64, max_concurrent_seqs=1))
    rapport = convert_checkpoint(tiny_checkpoint, plan,
                                 ConversionOptions(out_dir=str(tmp_path / "out"), quant_device="cpu"), spec=spec)
    l = _lignes(capsys)
    assert len(l) == rapport.tensors, (len(l), rapport.tensors)
    assert [int(re.search(r"#(\d+)", x).group(1)) for x in l] == list(range(rapport.tensors))
