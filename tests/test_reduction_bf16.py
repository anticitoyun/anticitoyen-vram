"""Réduction bf16 exacte de cuBLAS (poste6, 02/10, scellé revue/poste6-bf16-reduction-scelle-02-10.md, ordre chef : « une
sortie qui change avec le découpage du préfill est le bogue »). Cause mesurée sur carte : les projections étroites k_proj /
v_proj (N = 1 024) passent par `F.linear`, et cuBLAS bf16 y prend pour certaines formes une réduction à précision réduite —
15 % et 30 % de leurs éléments différaient entre 7 865 lignes d'un seul tenant et 4 096 + 3 769.
OPT-IN depuis le verdict carte (revue/poste6-bf16-reduction-verdict-carte-02-10.md) : +2,40 % de préfill à M = 4 096 pour
un seuil scellé de 2 %, et la sortie du seul tenant change aussi — le défaut reste `reduite` (tests/test_defaut_servi.py).
(1) sous `ACVRAM_BF16_REDUCTION=exacte` le chargement du moteur pose `allow_bf16_reduced_precision_reduction = False`
(cassant : retirer la pose) et `reduite` le retire ; (2) la ligne de régime dit le drapeau EN VIGUEUR, lu sur torch ; (3) sur carte,
à la forme mesurée, un seul tenant et un découpage rendent le même produit AU BIT sous le réglage, et pas sans lui — le
témoin : si cette carte ne diffère pas sans le réglage, le test ne prouve rien et s'ignore en le disant.
Pourquoi pas un préfill de moteur d'un seul tenant contre découpé : sur processeur le SDPA dépend de la longueur des clés
(REGLES § 4, 82 écarts sur 120 au jouet) et le drapeau n'y a aucun effet ; sur carte il faut un modèle dont les projections
atteignent les formes réduites — c'est la prise carte du scellé (chaîne dense de S1) qui le juge."""
import pytest
import torch

from acvram.engine import loader as LD


@pytest.fixture
def drapeau():
    m = torch.backends.cuda.matmul
    avant = m.allow_bf16_reduced_precision_reduction
    yield m
    m.allow_bf16_reduced_precision_reduction = avant


def test_le_chargement_pose_la_reduction_exacte(converted, drapeau, monkeypatch):
    from acvram.engine.loader import load_model
    drapeau.allow_bf16_reduced_precision_reduction = True                 # l'état de torch avant tout chargement
    monkeypatch.setattr(LD, "_BF16_REDUCTION", "exacte")
    load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=2)
    assert drapeau.allow_bf16_reduced_precision_reduction is False, "le chargement doit poser la réduction exacte"
    assert LD.reduction_bf16_en_vigueur() == "exacte"
    monkeypatch.setattr(LD, "_BF16_REDUCTION", "reduite")                 # retrait, pour la mesure
    load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=2)
    assert drapeau.allow_bf16_reduced_precision_reduction is True and LD.reduction_bf16_en_vigueur() == "reduite"


def test_la_ligne_de_regime_dit_le_drapeau_en_vigueur(converted, drapeau, monkeypatch):
    import acvram.engine.runner as R
    from acvram.engine.loader import load_model
    monkeypatch.setattr(LD, "_BF16_REDUCTION", "exacte")
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=2)
    eng = R.Engine(loaded, None, max_batch_size=2, max_model_len=256, enable_cuda_graphs=False)
    assert "reduction_bf16=exacte" in eng.regime_ligne()
    drapeau.allow_bf16_reduced_precision_reduction = True                 # quelqu'un d'autre le retire : la ligne le dit
    assert "reduction_bf16=reduite" in eng.regime_ligne()


@pytest.mark.gpu_requis
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuBLAS : carte requise")
def test_un_seul_tenant_egale_un_decoupage_au_bit_sous_le_reglage(drapeau, monkeypatch):
    """La forme mesurée le 02/10 : N = 1 024, K = 5 120, M = 7 865 contre 4 096 + 3 769."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7865, 5120, generator=g).to(torch.bfloat16).cuda()
    w = torch.randn(1024, 5120, generator=g).to(torch.bfloat16).cuda()

    def ecarts():
        un = torch.nn.functional.linear(x, w)
        deux = torch.cat([torch.nn.functional.linear(x[:4096], w), torch.nn.functional.linear(x[4096:], w)])
        return int((un != deux).sum()), un
    monkeypatch.setattr(LD, "_BF16_REDUCTION", "reduite"); LD.poser_reduction_bf16()
    sans, _ = ecarts()
    monkeypatch.setattr(LD, "_BF16_REDUCTION", "exacte"); LD.poser_reduction_bf16()
    avec, _ = ecarts()
    print(f"réduction bf16 : éléments différents entre un seul tenant et le découpage — reduite {sans}, exacte {avec} sur {x.shape[0] * w.shape[0]}")
    assert avec == 0, f"{avec} éléments diffèrent encore sous la réduction exacte : la sortie dépend du découpage"
    if sans == 0:
        pytest.skip("cette carte ne prend pas la réduction réduite à cette forme : le témoin ne prouve rien ici")


# ---- réduction ÉTROITE (03/10, scellé revue/poste6-bf16-etroite-scelle-02-10.md) : le drapeau n'est posé que le temps du
# `F.linear` du chemin NVFP4 naturel (`kernels._linear_naturel`), puis rendu ; `-tranches` découpe M par 1 024 lignes.
from acvram import kernels as K  # noqa: E402


@pytest.fixture
def etroite(drapeau, monkeypatch):
    """Sans carte : la portée s'applique quand même (le garde-fou cuBLAS est levé), les appels à F.linear sont observés."""
    monkeypatch.setattr(K, "_etroite_applicable", lambda x: K._REDUCTION_ETROITE is not None)
    vrai = torch.nn.functional.linear
    appels = []

    def espion(x, w, bias=None):
        appels.append((x.shape[0], drapeau.allow_bf16_reduced_precision_reduction))
        return vrai(x, w, bias)
    monkeypatch.setattr(torch.nn.functional, "linear", espion)
    drapeau.allow_bf16_reduced_precision_reduction = True
    return appels


def test_etroite_pose_pendant_l_appel_et_rend_apres(etroite, drapeau, monkeypatch):
    monkeypatch.setattr(K, "_REDUCTION_ETROITE", "appel")
    x, w = torch.randn(300, 64), torch.randn(32, 64)
    y = K._linear_naturel(x, w)
    assert etroite == [(300, False)], "le drapeau doit être à False le temps de l'appel"       # casse : pose retirée
    assert drapeau.allow_bf16_reduced_precision_reduction is True, "le drapeau doit être rendu"  # casse : rendu oublié
    assert torch.equal(y, torch.nn.functional.linear(x, w))


def test_etroite_rend_le_drapeau_meme_si_l_appel_leve(etroite, drapeau, monkeypatch):
    monkeypatch.setattr(K, "_REDUCTION_ETROITE", "appel")
    monkeypatch.setattr(torch.nn.functional, "linear", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuBLAS")))
    with pytest.raises(RuntimeError):
        K._linear_naturel(torch.randn(8, 64), torch.randn(32, 64))
    assert drapeau.allow_bf16_reduced_precision_reduction is True


def test_etroite_tranches_decoupe_m_par_1024_dans_la_portee(etroite, drapeau, monkeypatch):
    monkeypatch.setattr(K, "_REDUCTION_ETROITE", "tranches")
    x, w = torch.randn(2, 1250, 64), torch.randn(32, 64)                # 2 × 1 250 lignes : 1 024 + 1 024 + 452
    y = K._linear_naturel(x, w)
    assert y.shape == (2, 1250, 32) and etroite == [(1024, False), (1024, False), (452, False)]
    assert drapeau.allow_bf16_reduced_precision_reduction is True
    assert torch.allclose(y, torch.nn.functional.linear(x, w), atol=1e-4)      # fp32 processeur : ordre de K selon M
    etroite.clear()
    K._linear_naturel(torch.randn(1024, 64), w)                           # ≤ 1 024 lignes : un seul appel
    assert etroite == [(1024, False)]


def test_sans_mode_etroit_rien_ne_change(etroite, drapeau, monkeypatch):
    monkeypatch.setattr(K, "_REDUCTION_ETROITE", None)
    K._linear_naturel(torch.randn(3000, 64), torch.randn(32, 64))
    assert etroite == [(3000, True)]


def test_le_chargement_pose_le_mode_etroit_dans_les_noyaux(converted, drapeau, monkeypatch):
    from acvram.engine.loader import load_model
    for mode, portee, flag in (("etroite", "appel", True), ("etroite-tranches", "tranches", True),
                               ("exacte", None, False), ("reduite", None, True), ("n-importe-quoi", None, True)):
        monkeypatch.setattr(LD, "_BF16_REDUCTION", mode)
        load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=2)
        assert K._REDUCTION_ETROITE == portee and drapeau.allow_bf16_reduced_precision_reduction is flag, mode
        assert LD.reduction_bf16_en_vigueur() == (mode if mode != "n-importe-quoi" else "reduite")


@pytest.mark.gpu_requis
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuBLAS : carte requise")
def test_etroite_sur_carte_e1_e3(drapeau, monkeypatch):
    """E1 : sous `etroite`, 7 865 lignes contre 4 096 + 3 769 au bit (témoin `reduite` : des éléments diffèrent) ; E2 : au
    bit du drapeau global à chaque M ; E3 : `-tranches` au bit de `etroite` ; E0 : le drapeau est rendu à True."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7865, 5120, generator=g).to(torch.bfloat16).cuda()
    w = torch.randn(1024, 5120, generator=g).to(torch.bfloat16).cuda()
    Ms = (33, 512, 1024, 2048, 3769, 4096, 7865)

    def sous(mode, f):
        monkeypatch.setattr(LD, "_BF16_REDUCTION", mode); LD.poser_reduction_bf16()
        return f()
    un = sous("etroite", lambda: K._linear_naturel(x, w))
    deux = sous("etroite", lambda: torch.cat([K._linear_naturel(x[:4096], w), K._linear_naturel(x[4096:], w)]))
    assert drapeau.allow_bf16_reduced_precision_reduction is True, "E0 : le drapeau doit être rendu après l'appel"
    temoin = sous("reduite", lambda: int((K._linear_naturel(x, w) != torch.cat(
        [K._linear_naturel(x[:4096], w), K._linear_naturel(x[4096:], w)])).sum()))
    e1 = int((un != deux).sum())
    print(f"E1 étroite : {e1} éléments différents entre 7 865 et 4 096 + 3 769 (témoin reduite : {temoin}) sur {un.numel()}")
    assert e1 == 0, "E1 : la portée étroite doit rendre le produit indépendant du découpage"
    for M in Ms:
        glob = sous("exacte", lambda: torch.nn.functional.linear(x[:M], w))
        p = sous("etroite", lambda: K._linear_naturel(x[:M], w))
        t = sous("etroite-tranches", lambda: K._linear_naturel(x[:M], w))
        e2, e3 = int((p != glob).sum()), int((t != p).sum())
        print(f"E2 M={M} : {e2} éléments diffèrent du drapeau global ; E3 M={M} : tranches contre appel {e3}")
        assert e2 == 0 and e3 == 0, f"M={M} : E2 {e2}, E3 {e3}"
    if temoin == 0:
        pytest.skip("cette carte ne prend pas la réduction réduite à cette forme : le témoin ne prouve rien ici")
