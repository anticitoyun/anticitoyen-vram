"""`outils/gpu/mesure/familles-noyaux.py` : décomposition d une trace nsys par
famille et par pas (un pas = `couches` marqueurs `_route_fusee_kernel`).
Trace synthétique : 7 pas × 2 couches (6 fenêtres), fenêtres de tête (2) et de queue (1)
exclues → 3 pas jugés ; doit casser si une famille change de motif
(Marlin, étroit, attention…), si la médiane par pas se trompe de fenêtre, ou
si le trou (mur − noyaux) n est plus mur − Σ familles."""
import csv
import importlib.util
import os

ICI = os.path.dirname(os.path.abspath(__file__))


def _module():
    p = os.path.join(ICI, "..", "outils", "gpu", "mesure", "familles-noyaux.py")
    spec = importlib.util.spec_from_file_location("familles_noyaux", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _trace(chemin, pas=7, couches=2):
    """Par couche : route 7 µs, marlin 50 + 28, étroit 11, attention 4, rmsnorm 2,
    2 élémentaires torch de 2 µs ; par pas : tête 120 µs ; pas de 1 000 µs (trou = reste)."""
    t = 0
    lignes = []
    for p in range(pas):
        debut = t
        for c in range(couches):
            for nom, d in (("_route_fusee_kernel", 7_000), ("void marlin_moe_wna16::Marlin<...>", 50_000),
                           ("void nvfp4_gemv_marlin_kernel<float, ...>", 28_000), ("_etroit_kernel", 11_000),
                           ("_partiel_kernel", 4_000), ("rmsnorm_bf16_kernel", 2_000),
                           ("void at::native::vectorized_elementwise_kernel<...>", 2_000),
                           ("void at::native::reduce_kernel<...>", 2_000)):
                lignes.append((t, d, nom)); t += d + 1_000
        lignes.append((t, 120_000, "tete_wmma_kernel")); t += 120_000
        t = debut + 1_000_000
    with open(chemin, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start (ns)", "Duration (ns)", "CorrId", "Name"])
        for s, d, n in lignes:
            w.writerow([s, d, 0, n])


def test_familles_par_pas(tmp_path):
    m = _module()
    p = str(tmp_path / "trace.csv")
    _trace(p)
    r = m.decomposer(p, couches=2)
    assert r["pas_juges"] == 3
    f = r["familles"]
    assert f["experts_marlin"] == {"ms_pas": 0.156, "lancements_pas": 4, "part": f["experts_marlin"]["part"]}
    assert f["routage"]["ms_pas"] == 0.014 and f["proj_etroites_int8"]["ms_pas"] == 0.022
    assert f["attention"]["ms_pas"] == 0.008 and f["normes"]["ms_pas"] == 0.004
    assert f["glue_torch"] == {"ms_pas": 0.008, "lancements_pas": 4, "part": f["glue_torch"]["part"]}
    assert f["tete"]["ms_pas"] == 0.12 and "autres" not in f
    assert r["mur_ms_pas"] == 1.0 and r["noyaux_ms_pas"] == 0.332 and r["trou_ms_pas"] == 0.668
    assert abs(sum(t["part"] for t in f.values()) - 1.0) < 0.01
    assert r["lancements_pas"] == 17


def test_famille_de():
    m = _module()
    # 22/09 (verdict-nsys-familles) : l attention compacte tombait dans `autres`, le GEMM du routeur dans `tete`
    assert m.famille_de("_partiel_reduit_kernel") == "attention"
    assert m.famille_de("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>(...)") == "routeur_gemm"
    assert m.famille_de("nvfp4_quant_act_kernel(...)") == "experts_quant_a4"
    assert m.famille_de("void moe_route_pack_kernel<__nv_bfloat16, int>(...)") == "experts_glue"
    assert m.famille_de("moe_reduce_trie_kernel") == "experts_glue"
    assert m.famille_de("void nvfp4_gemm_grouped_mma_kernel<16>(...)") == "experts_marlin"
    assert m.famille_de("void marlin_moe_wna16::Marlin<(long)1>(const int4 *)") == "experts_marlin"
    assert m.famille_de("void <unnamed>::int8_gemv_kernel<...>") == "proj_etroites_int8"
    assert m.famille_de("void pytorch_flash::flash_fwd_splitkv_kernel<...>") == "attention"
    assert m.famille_de("[CUDA memcpy Device-to-Host]") == "copies"
    assert m.famille_de("kv_write_int8_kernel") == "rope_kv"
    assert m.famille_de("un_noyau_inconnu") == "autres"


def test_detail_par_noyau(tmp_path):
    m = _module()
    p = str(tmp_path / "trace.csv")
    _trace(p)
    d = m.detailler(p, "experts_marlin", couches=2)
    assert len(d) == 2
    marlin = next(v for k, v in d.items() if "marlin_moe_wna16" in k)
    assert marlin == {"ms_pas": 0.1, "lancements_pas": 2, "us_par_lancement": 50.0}
