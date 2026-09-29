"""8fx (29/09) : un masque d'attention dense [q, kv] de 2,10 Gio (invite claude de ~30 k jetons, préfixe en cache) faisait
un OOM en service. `attention` garde le calcul d'un seul tenant (sortie inchangée là où il passait) et ne découpe par
blocs de lignes qu'après un OOM. Contrôles : sans OOM, un seul appel SDPA (rien ne change) ; avec OOM, découpé pour de
vrai, bf16 identique au bit et fp32 à ≤ 1e-6 près (quelques ulp) sur chaque chemin à masque (préfixe, fenêtre, images, GQA)."""
import pytest
import torch

from acvram.engine import layers

# cqy : blocs d au moins 1 024 lignes (plancher) ; les formes dépassent donc un bloc
CAS = [dict(q_offset=37, q_len=2100, kv_len=2137),
       dict(q_offset=0, q_len=2100, kv_len=2100, window=9),
       dict(q_offset=10, q_len=2100, kv_len=2110, images=[(12, 20), (1500, 1600)]),
       dict(q_offset=0, q_len=2100, kv_len=2100, images=[(5, 30)]),
       dict(q_offset=5, q_len=2100, kv_len=2105, n_rep=2)]


def _entrees(cas, dt):
    torch.manual_seed(0)
    n_rep = cas.get("n_rep", 1)
    q = torch.randn(cas["q_len"], 2, 8, dtype=dt)
    k = torch.randn(cas["kv_len"], 2 // n_rep, 8, dtype=dt)
    v = torch.randn(cas["kv_len"], 2 // n_rep, 8, dtype=dt)
    kw = dict(causal=True, scale=0.25, q_offset=cas["q_offset"], window=cas.get("window", 0), n_rep=n_rep,
              images=cas.get("images"))
    return q, k, v, kw


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cas", CAS)
def test_decoupe_seulement_apres_oom(monkeypatch, cas, dt):
    monkeypatch.setattr(layers, "_MASQUE_OCTETS_MAX", 0)        # témoin : pas de découpage d emblée (cqy)
    monkeypatch.setattr(layers, "_lignes_par_bloc", lambda kv, o: 1024)   # blocs au plancher (formes de test courtes)
    q, k, v, kw = _entrees(cas, dt)
    vrai = torch.nn.functional.scaled_dot_product_attention
    appels = []

    def sdpa(qh, *a, **k2):
        appels.append(qh.shape[2])
        return vrai(qh, *a, **k2)
    monkeypatch.setattr(layers.F, "scaled_dot_product_attention", sdpa)
    entier = layers.attention(q, k, v, **kw)
    assert appels == [cas["q_len"]]                      # sans OOM : un seul tenant, comme avant
    appels.clear()

    def sdpa_oom(qh, *a, **k2):                           # la carte ne tient qu un masque de 1 024 lignes
        appels.append(qh.shape[2])
        if qh.shape[2] > 1024:
            raise torch.OutOfMemoryError("masque trop grand (simulé)")
        return vrai(qh, *a, **k2)
    monkeypatch.setattr(layers.F, "scaled_dot_product_attention", sdpa_oom)
    blocs = layers.attention(q, k, v, **kw)
    assert appels[0] == cas["q_len"] and max(appels[1:]) <= 1024 and len(appels) == 1 + -(-cas["q_len"] // 1024), appels
    if dt == torch.bfloat16:
        assert torch.equal(entier, blocs)
    else:
        assert torch.allclose(entier, blocs, rtol=0, atol=1e-6), (entier - blocs).abs().max()


def test_sans_masque_pas_de_decoupage(monkeypatch):
    q, k, v = torch.randn(50, 2, 8), torch.randn(50, 2, 8), torch.randn(50, 2, 8)
    appels = []
    vrai = torch.nn.functional.scaled_dot_product_attention
    monkeypatch.setattr(layers.F, "scaled_dot_product_attention",
                        lambda qh, *a, **kw: appels.append(kw) or vrai(qh, *a, **kw))
    layers.attention(q, k, v, causal=True, q_offset=0)
    assert len(appels) == 1 and appels[0].get("is_causal") is True
