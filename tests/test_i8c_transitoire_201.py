"""Pièce 201 (B) : la copie int8 signée des poids par canal (`kernels._i8c_poids`) est TRANSITOIRE — fabriquée par
appel, ou une fois par portée `depaquetage_partage`, puis rendue. Gardée à vie, elle pesait 6,84 Gio sur
Qwen3.8-27B-nvfp4-attn-gdn-i8c, fabriqués au warm, après la borne du KV : OOM, service impossible.
(1) au bit contre la copie persistante d'avant, poids aléatoires ET réels (un q_proj du modèle, s'il est là) ;
(2) mémoire : après le préfill, seuls les octets de la sortie restent (octets demandés, pièce 191). Bras cassant
(prise) : copie regardée sur le tenseur → rouge."""
import json
import os
import sys

import pytest
import torch

from acvram import kernels
from acvram.kernels import gemm_i8c_cublas
from acvram.quant.formats import INT8Tensor, _quantize_int8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
DEV = torch.device("cuda:0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils"))
from racine_modeles import alias  # noqa: E402

# Le converti i8c de la 153 vit hors de la racine du parc (disque USB) : ACVRAM_MODELE_I8C le désigne, sinon l'alias.
I8C = os.environ.get("ACVRAM_MODELE_I8C") or alias("Qwen3.8-27B-nvfp4-attn-gdn-i8c")


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _aleatoire(N=6144, K=5120):
    torch.manual_seed(201)
    return _quantize_int8(torch.randn(N, K, device=DEV) * 0.02, group_size=K, symmetric=True)


def _reel():
    """Un poids int8 par canal du modèle i8c (le premier q_proj d'attention du manifeste), tel que le chargeur le lit."""
    man = os.path.join(I8C, "acvram_manifest.json")
    if not os.path.exists(man):
        pytest.skip(f"modèle i8c absent : {I8C}")
    from safetensors import safe_open
    m = json.load(open(man, encoding="utf-8"))
    nom = next(n for n, e in m["tensors"].items() if isinstance(e, dict) and e.get("format") == "int8"
               and e.get("symmetrique") and ".self_attn.q_proj." in n)
    e = m["tensors"][nom]
    sd = {}
    for cle in e["keys"]:
        with safe_open(os.path.join(I8C, m["weight_map"][cle]), framework="pt", device="cuda:0") as f:
            sd[cle.rsplit(".", 1)[1]] = f.get_tensor(cle)
    return INT8Tensor(sd["qweight"], sd["scales"], sd["zeros"], e["group_size"], tuple(e["shape"]))


def _ancienne_persistante(t):
    """Le chemin d'avant la 201 : la copie gardée sur le tenseur, rendue telle quelle à chaque appel."""
    w = (t.qweight.to(torch.int16) - 128).to(torch.int8).contiguous()
    return lambda u: w if u is t else None


@pytest.mark.parametrize("source", ["aleatoire", "reel"])
def test_au_bit_contre_la_copie_persistante(monkeypatch, source):
    t = _aleatoire() if source == "aleatoire" else _reel()
    assert kernels._i8c_eligible(t), "poids non éligible : le test ne jugerait rien"
    g = torch.Generator(device="cpu").manual_seed(92)
    x = (torch.randn(92, t.qweight.shape[1], generator=g) * 0.5).to(DEV, torch.bfloat16)
    y = gemm_i8c_cublas(x, t)
    with kernels.depaquetage_partage():
        yp = [gemm_i8c_cublas(x[:46], t), gemm_i8c_cublas(x[46:], t)]
    monkeypatch.setattr(kernels, "_i8c_poids", _ancienne_persistante(t))
    ref = gemm_i8c_cublas(x, t)
    refp = [gemm_i8c_cublas(x[:46], t), gemm_i8c_cublas(x[46:], t)]
    assert torch.equal(y, ref), "transitoire hors portée ≠ persistante"
    assert all(torch.equal(a, b) for a, b in zip(yp, refp)), "transitoire dans la portée ≠ persistante"


def _demandes():
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["requested_bytes.all.current"]


def test_rien_ne_reste_apres_le_prefill():
    """Octets demandés (pas `memory_allocated`, pièce 191) : après les appels, seules les sorties restent — hors
    portée comme à la sortie d'une portée de partage. La copie gardée laisserait N × K octets (31,5 Mio ici)."""
    t = _aleatoire()
    x = torch.randn(92, t.qweight.shape[1], device=DEV).to(torch.bfloat16)
    # Allocations persistantes légitimes faites au premier appel, amorcées SANS toucher la copie : l'éligibilité (un
    # booléen), les échelles fp32 par canal (`_i8c_echelles`, N × 4 o) et l'espace de travail cuBLASLt (`_int_mm`).
    kernels._i8c_eligible(t); kernels._i8c_echelles(t)
    torch._int_mm(torch.zeros(92, t.qweight.shape[1], dtype=torch.int8, device=DEV),
                  torch.zeros(t.qweight.shape[1], t.qweight.shape[0], dtype=torch.int8, device=DEV))
    avant = _demandes()
    y = gemm_i8c_cublas(x, t)
    assert _demandes() - avant == y.numel() * y.element_size(), "copie int8 gardée après l'appel"
    avant = _demandes()
    with kernels.depaquetage_partage():
        ys = [gemm_i8c_cublas(x[:46], t), gemm_i8c_cublas(x[46:], t)]
    assert _demandes() - avant == sum(v.numel() * v.element_size() for v in ys), "copie gardée après la portée"
