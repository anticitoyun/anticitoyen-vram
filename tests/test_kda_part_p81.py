"""p81 (chef 01/10) : le classement des noyaux KDA de `outils/gpu/mesure/kda-part-p81.py` ne doit pas sous-compter le préfill.
`chunk_kda` de fla lance des noyaux génériques sans « kda » dans le nom ; un faux CSV en contient un, et l'ancien classement
(« kda » dans le nom) le rendait « autre » — ce test casse si l'on y revient. Les noms de la liste doivent exister dans le
fla installé : une mise à jour de fla qui les renomme rend ce test rouge au lieu de fausser la part en silence."""
import importlib.util
import os
import re
from pathlib import Path

import pytest

OUTIL = Path(__file__).resolve().parent.parent / "outils" / "gpu" / "mesure" / "kda-part-p81.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("kda_part_p81", OUTIL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_un_noyau_fla_sans_kda_compte_au_prefill(m, tmp_path, capsys):
    csv = tmp_path / "t.csv"
    csv.write_text(
        "NVTX Range,Style,PID,TID,NVTX Inst,Kern Inst,Total Time (ns),Kernel Name\n"
        ":p81:prefill8k,PushPop,1,1,1,40,1000000,chunk_kda_fwd_kernel_inter_solve_fused\n"
        ":p81:prefill8k,PushPop,1,1,1,40,2000000,chunk_gated_delta_rule_fwd_kernel_h_blockdim64\n"
        ":p81:prefill8k,PushPop,1,1,1,400,7000000,gemm_nvfp4_tensor\n"
        ":p81:decode_b1,PushPop,1,1,1,64,640000,void kda_decode_kernel<128>(...)\n"
        ":p81:decode_b1,PushPop,1,1,1,999,5760000,gemv_marlin\n")
    r = m.analyser(str(csv))
    assert r["p81:prefill8k"]["kda_ms"] == pytest.approx(3.0), "le noyau fla générique doit compter comme KDA"
    assert r["p81:prefill8k"]["part_kda"] == pytest.approx(0.3)
    assert r["p81:decode_b1"]["part_kda"] == pytest.approx(0.1)
    sortie = capsys.readouterr().out
    assert "KDA    chunk_gated_delta_rule_fwd_kernel_h_blockdim64" in sortie and "autre  gemm_nvfp4_tensor" in sortie
    assert "kda" not in "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "témoin : le nom ne porte pas « kda »"


def test_une_trace_sans_plage_p81_est_refusee(m, tmp_path):
    csv = tmp_path / "v.csv"
    csv.write_text("NVTX Range,Total Time (ns),Kernel Name\n:autre,5,x\n")
    with pytest.raises(SystemExit):
        m.analyser(str(csv))


def test_les_noms_fla_existent_dans_le_fla_installe(m):
    spec = importlib.util.find_spec("fla")
    if spec is None:
        pytest.skip("fla absent")
    racine = os.path.dirname(spec.origin)
    defs = set()
    for d, _, fs in os.walk(racine):
        for f in fs:
            if f.endswith(".py"):
                defs |= set(re.findall(r"^def (\w+)\(", open(os.path.join(d, f), encoding="utf-8").read(), re.M))
    absents = [n for n in m.NOYAUX_FLA_SANS_KDA if n not in defs]
    assert not absents, f"noyaux fla renommés ou retirés : {absents} — refaire l'inventaire (kda-part-p81.py)"


@pytest.mark.parametrize("model_type, types, accepte", [
    ("kimi_linear", ["linear_attention"] * 3 + ["full_attention"], True),
    ("qwen3_next", ["linear_attention"] * 3 + ["full_attention"], False),      # GDN : mêmes layer_types que Kimi
    ("qwen3_5_text", ["linear_attention", "full_attention"], False),
    ("kimi_linear", ["linear_attention", "full_attention", "mamba"], False),  # autre récurrence
    ("kimi_linear", ["full_attention"], False),                               # aucun KDA : part nulle par construction
])
def test_le_regime_refuse_tout_ce_qui_n_est_pas_kda_mla(m, model_type, types, accepte):
    """REGLES § 6 : la liste de noyaux fla ne vaut que pour un modèle KDA + MLA sans GDN."""
    if accepte:
        m.verifier_regime(model_type, types)
    else:
        with pytest.raises(SystemExit, match="régime refusé"):
            m.verifier_regime(model_type, types)


@pytest.mark.parametrize("dossier, accepte", [
    ("Kimi-Linear-35B-kda-nvfp4", True),
    ("Agents-A1-4B-kimi-nvfp4", False),       # « kimi » dans le nom, GDN (qwen3_5_text) dans la config
])
def test_le_regime_sur_les_convertis_du_parc(m, dossier, accepte):
    from acvram.engine.config import load_model_spec
    import sys
    sys.path.insert(0, str(OUTIL.parents[2]))           # outils/
    from racine_modeles import racine_modeles
    chemin = os.path.join(os.environ.get("ACVRAM_PARC") or racine_modeles(), dossier)
    if not os.path.isfile(f"{chemin}/config.json"):
        pytest.skip(f"{dossier} absent")
    spec = load_model_spec(chemin)
    if accepte:
        m.verifier_regime(spec.model_type, spec.layer_types)
    else:
        with pytest.raises(SystemExit, match="régime refusé"):
            m.verifier_regime(spec.model_type, spec.layer_types)
