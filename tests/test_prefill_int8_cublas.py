"""P2 (sage-reprise-rapide-19-09 § 2) : chemin `ACVRAM_PREFILL_INT8=cublas` —
`torch._int_mm` sur les poids INT8 symétriques par canal (convertis -qkvo-i8c).
À sec : `_int_mm` a une implémentation CPU, le produit entier est exact, donc
le chemin se juge contre sa propre arithmétique en fp64 (au bit près de la
sortie bf16) et contre la déquantification fp32 (erreur = celle de l'A8 par
jeton seulement). Témoins cassants : un poids affine par groupes n'est PAS
éligible (None, repli), un point-zéro ≠ 128 non plus, M ≤ 16 non plus."""
import os
import sys
import pathlib

import pytest
import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../outils'))
from racine_modeles import racine_modeles as _racine_modeles, alias_absent as _alias_absent  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from acvram.kernels import gemm_i8c_cublas, _i8c_poids                       # noqa: E402
from acvram.quant.formats import INT8Tensor, _dequantize_int8, _quantize_int8   # noqa: E402


def _poids(N=64, K=256, seed=1):
    torch.manual_seed(seed)
    w = torch.randn(N, K) * 0.02
    return w, _quantize_int8(w, group_size=K, symmetric=True)


def test_i8c_cublas_egal_a_son_arithmetique_et_proche_de_la_dequant():
    w, t = _poids()
    x = (torch.randn(40, 256) * 0.5).to(torch.bfloat16)
    y = gemm_i8c_cublas(x, t, sortie_fp32=True)
    assert y is not None and y.shape == (40, 64)
    # même arithmétique en fp64 : A8 par jeton (absmax/127), produit entier, échelles
    xf = x.float()                                    # l'A8 se calcule en fp32, comme le chemin
    sx = xf.abs().amax(1).clamp_min(1e-12) / 127.0
    a = torch.round(xf / sx[:, None]).clamp(-127, 127).double()
    ref = (a @ (t.qweight.double() - 128).T) * sx.double()[:, None] * t.scales[:, 0].double()[None, :]
    assert (y.double() - ref).norm() / ref.norm() < 1e-5          # fp32 vs fp64, même entier
    # contre la déquantification fp32 : seule l'erreur A8 sépare les deux
    deq = x.float() @ _dequantize_int8(t, torch.float32).T
    assert (y - deq).norm() / deq.norm() < 0.02
    # ce que l'ancien chemin rendait (bf16 déquant) reste la référence de forme/dtype
    yb = gemm_i8c_cublas(x, t)
    assert yb.dtype == torch.bfloat16 and yb.shape == (40, 64)


def test_temoins_cassants_affine_zero_et_petit_m():
    w, t = _poids()
    x = (torch.randn(40, 256) * 0.5).to(torch.bfloat16)
    assert gemm_i8c_cublas(x[:16], t) is None                      # M ≤ 16 : cuBLASLt refuse
    t_aff = _quantize_int8(w, group_size=128, symmetric=False)     # affine par groupes
    assert _i8c_poids(t_aff) is None and gemm_i8c_cublas(x, t_aff) is None
    z = t.zeros.clone(); z[3, 0] = 127                              # un zéro ≠ 128
    t_z = INT8Tensor(t.qweight, t.scales, z, t.group_size, t.shape, t.format)
    assert _i8c_poids(t_z) is None
    # la copie int8 est faite une fois et gardée
    assert _i8c_poids(t) is _i8c_poids(t) and _i8c_poids(t).dtype == torch.int8


def test_int8_matmul_prend_cublas_sous_regime(monkeypatch):
    """Le dispatcher `int8_matmul` prend le chemin cublas sous
    ACVRAM_PREFILL_INT8=cublas (variable lue à l'import : posée sur le
    module), et rend la même chose que l'appel direct."""
    import acvram.kernels as K
    w, t = _poids()
    x = (torch.randn(96, 256) * 0.5).to(torch.bfloat16)
    monkeypatch.setattr(K, "_PREFILL_INT8", "cublas")
    y = K.int8_matmul(x, t, gemv_threshold=1)
    assert torch.equal(y, gemm_i8c_cublas(x, t))
    monkeypatch.setattr(K, "_PREFILL_INT8", "bf16")
    yb = K.int8_matmul(x, t, gemv_threshold=1)
    assert (y.float() - yb.float()).norm() / yb.float().norm() < 0.02


def test_gemm_etroit_refuse_le_groupe_par_canal():
    """Le GEMM étroit Triton (Poste C) tuile K par groupe : G = K = 2048 le
    fait sortir de la mémoire partagée (19/09, i8c, godet 2). Refusé par
    `eligible`, accepté pour les groupes de 128 (sortie du classé inchangée)."""
    from acvram.kernels import gemm_etroit
    w, t = _poids(N=64, K=2048)
    assert not gemm_etroit.eligible(t)
    assert gemm_etroit.eligible(_quantize_int8(w, group_size=128, symmetric=False))


def test_ppl_prend_le_chemin_du_moteur_sous_cublas(monkeypatch):
    """19/09 (verdict-p2-moteur) : sous cublas, un poids i8c à M > seuil GEMV
    doit compter « cublas » et jamais « dequant » (compteur CHEMINS_INT8) ;
    et la tête INT8 ne reçoit x en bf16 que sous le seuil GEMV, sous tout
    régime (au-delà : conversion fp32 + déquant, jamais W8A8 à 2 048 lignes)."""
    import acvram.kernels as K
    w, t = _poids(N=64, K=2048)
    x = (torch.randn(96, 2048) * 0.5).to(torch.bfloat16)
    monkeypatch.setattr(K, "_PREFILL_INT8", "cublas")
    avant = dict(K.CHEMINS_INT8)
    K.int8_matmul(x, t, gemv_threshold=80)
    assert K.CHEMINS_INT8["cublas"] == avant.get("cublas", 0) + 1
    assert K.CHEMINS_INT8["dequant"] == avant.get("dequant", 0)
    # tête : jamais au-delà du seuil GEMV, quel que soit le régime (sage-p2-ppl-instrument-file-7h-19-09)
    assert K.tete_int8_entree_bf16(8) and not K.tete_int8_entree_bf16(2047)
    monkeypatch.setattr(K, "_PREFILL_INT8", "bf16")
    assert not K.tete_int8_entree_bf16(2047) and K.tete_int8_entree_bf16(K._INT8_GEMV_MAX)
    # témoin cassant : sous bf16 (témoin), le même appel compte la déquant
    K.int8_matmul(x, t, gemv_threshold=80)
    assert K.CHEMINS_INT8["dequant"] == avant.get("dequant", 0) + 1


def test_c11_vue_g128_du_poids_par_canal():
    """C11 : la vue g128 d'un i8c partage les codes, répète l'échelle de ligne
    sur K/128 groupes, zéros à 128 ; sa déquantification est IDENTIQUE à celle
    du tenseur d'origine (mêmes valeurs) ; construite une fois ; un g128 ou
    un affine par groupes est rendu tel quel (identité)."""
    from acvram.kernels import vue_g128
    w, t = _poids(N=64, K=2048)
    v = vue_g128(t)
    assert v is not t and v.group_size == 128 and v.qweight is t.qweight
    assert v.scales.shape == (64, 16) and torch.equal(v.scales[:, 0], t.scales[:, 0]) and torch.all(v.scales == t.scales)
    assert v.zeros.shape == (64, 16) and torch.all(v.zeros == 128)
    assert torch.equal(_dequantize_int8(v, torch.float32), _dequantize_int8(t, torch.float32))
    assert vue_g128(t) is v                                        # cache
    from acvram.kernels import gemm_etroit
    assert gemm_etroit.eligible(v) and not gemm_etroit.eligible(t)
    t128 = _quantize_int8(w, group_size=128, symmetric=False)
    assert vue_g128(t128) is t128                                  # identité pour le g128 du classé
    z = t.zeros.clone(); z[0, 0] = 127
    t_z = INT8Tensor(t.qweight, t.scales, z, t.group_size, t.shape, t.format)
    assert vue_g128(t_z) is t_z                                    # pas symétrique : pas de vue


@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise : gemm_etroit rend des valeurs fausses sous TRITON_INTERPRET (vérifié 19/09, aussi sur un g128 classé) — REGLES § 7, un Triton vert à sec ne prouve rien")
def test_c11_gemm_etroit_sur_la_vue_egale_la_reference_sur_carte():
    """Sur carte : le noyau étroit Triton sur la vue g128 d'un i8c rend x @ W.T
    à l'erreur bf16 près, ET la même chose qu'un vrai g128 aux mêmes codes et
    échelles (bit à bit : mêmes entrées, même noyau) ; témoin cassant :
    échelles décalées d'une ligne."""
    from acvram.kernels import gemm_etroit, vue_g128, get_extension
    if get_extension() is None or not gemm_etroit.disponible():
        pytest.skip("extension ou Triton absents")
    w, t = _poids(N=64, K=2048)
    t = INT8Tensor(t.qweight.cuda(), t.scales.cuda(), t.zeros.cuda(), t.group_size, t.shape, t.format)
    x = (torch.randn(8, 2048, device="cuda") * 0.5).to(torch.bfloat16)
    v = vue_g128(t)
    y = gemm_etroit.gemm_etroit(x, v, False)[:, :64]
    ref = x.float() @ _dequantize_int8(t, torch.float32).T
    assert (y.float() - ref).norm() / ref.norm() < 1e-2
    vrai = INT8Tensor(v.qweight, v.scales.clone(), v.zeros.clone(), 128, v.shape, v.format)
    assert torch.equal(y, gemm_etroit.gemm_etroit(x, vrai, False)[:, :64])
    v2 = INT8Tensor(v.qweight, torch.roll(v.scales, 1, dims=0).contiguous(), v.zeros, 128, v.shape, v.format)
    assert (gemm_etroit.gemm_etroit(x, v2, False)[:, :64].float() - ref).norm() / ref.norm() > 1e-2


CLASSES = [pytest.param(_RACINE + "/" + a, marks=pytest.mark.skipif(bool(_alias_absent(a)), reason=_alias_absent(a)))
           for a in ("Qwen3-Coder-30B-A3B-nvfp4", "GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA")]


def _premier_int8(dossier):
    """Le premier tenseur int8 du manifeste (une projection q/k/v/o promue),
    chargé paresseusement depuis ses safetensors — un seul tenseur lu."""
    import glob
    import json
    try:
        man = json.load(open(os.path.join(dossier, "acvram_manifest.json")))
    except OSError:
        return None
    for nom, e in man["tensors"].items():
        if e.get("format") == "int8" and "self_attn" in nom and "norm" not in nom:
            from safetensors import safe_open
            sd = {}
            for f in glob.glob(os.path.join(dossier, "*.safetensors")):
                with safe_open(f, "pt") as fh:
                    for k in e["keys"]:
                        if k in fh.keys():
                            sd[k.rsplit(".", 1)[-1]] = fh.get_tensor(k)
            if len(sd) == 3:
                return INT8Tensor(sd["qweight"], sd["scales"], sd["zeros"], e.get("group_size", 128), tuple(e["shape"]))
    return None


@pytest.mark.parametrize("dossier", CLASSES)
def test_defaut_cublas_ne_change_pas_un_classe(dossier, monkeypatch):
    """sage-p2-au-defaut-19-09 : sous le défaut `cublas`, un converti CLASSÉ
    (q_proj int8 g128 affine : Coder nvfp4, GLM calibA) ne prend jamais le
    chemin cublas (CHEMINS_INT8['cublas'] inchangé) ni a8, et rend la même
    sortie AU BIT que le témoin bf16 — sur le vrai tenseur du disque."""
    import acvram.kernels as K
    t = _premier_int8(dossier)
    if t is None:
        pytest.skip(f"converti absent ou sans q_proj int8 : {dossier}")
    assert t.group_size == 128 and t.zeros.shape[1] > 1                # g128 affine : pas un i8c
    x = (torch.randn(96, t.qweight.shape[1]) * 0.5).to(torch.bfloat16)
    monkeypatch.setattr(K, "_PREFILL_INT8", "cublas")
    avant = dict(K.CHEMINS_INT8)
    y_c = K.int8_matmul(x, t, gemv_threshold=80)
    assert K.CHEMINS_INT8["cublas"] == avant.get("cublas", 0)
    assert K.CHEMINS_INT8["a8"] == avant.get("a8", 0)
    assert K.CHEMINS_INT8["dequant"] == avant.get("dequant", 0) + 1
    monkeypatch.setattr(K, "_PREFILL_INT8", "bf16")
    y_b = K.int8_matmul(x, t, gemv_threshold=80)
    assert torch.equal(y_c, y_b)
    # et le régime par défaut, sans variable posée, est bien cublas
    from acvram import regime
    v = {z.env: z for z in regime.VARIABLES}["ACVRAM_PREFILL_INT8"]
    assert v.defaut == "cublas"
