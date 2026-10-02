"""Réduction bf16 exacte de cuBLAS (poste6, 02/10, scellé revue/poste6-bf16-reduction-scelle-02-10.md, ordre chef : « une
sortie qui change avec le découpage du préfill est le bogue »). Cause mesurée sur carte : les projections étroites k_proj /
v_proj (N = 1 024) passent par `F.linear`, et cuBLAS bf16 y prend pour certaines formes une réduction à précision réduite —
15 % et 30 % de leurs éléments différaient entre 7 865 lignes d'un seul tenant et 4 096 + 3 769.
(1) le chargement du moteur pose `allow_bf16_reduced_precision_reduction = False` (cassant : retirer la pose) et
`ACVRAM_BF16_REDUCTION=reduite` le retire ; (2) la ligne de régime dit le drapeau EN VIGUEUR, lu sur torch ; (3) sur carte,
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
