"""kv31b (poste6, 30/09, ordre chef) : 15 refus « budget KV insuffisant après 4 tours d'exil » de gemma-4-31B à 32 768
(edz définitif). Journal du préchargement : « 15,12 → 10,28 Gio (libre 28,1, poids 5,8, marge 12,0 dont préfill 10,42) » —
le plancher KV d'une séquence (60 couches × 16 têtes × 256, les 50 couches à fenêtre glissante stockées en pleine
longueur : la fenêtre n'est qu'un masque, layers.py:1108) vaut 15,1 Gio et la réserve d'activations d'un préfill d'un
seul tenant 10,4 Gio ; l'exil avait déjà sorti 13 des 19 Gio de poids (5,8 résidents, l'attention ne s'exile pas).
Le refus disait « réduire max_model_len » sans dire à combien : il dit désormais la fenêtre qui tient (ligne
`[acvram] fenêtre qui tient : N jetons` + message), que le lanceur relit. À sec (CUDA simulé comme test_annexes_201) :
(1) sur une réplique de gemma-4-31B (vraie ModelSpec, réserve réelle) le refus nomme N ∈ [20 000, 28 000] et N est
exact : N tient, N + 1 024 ne tient pas ; (2) une carte assez large ne refuse pas (témoin) ; (3) rien ne tient → 0.
Cassure : `_fenetre_qui_tient` rendant max_model_len → (1) rouge (N + 1 024 « tient »)."""
import pytest
import torch

from acvram.engine import loader as LD
from acvram.engine.config import ModelSpec
from acvram.memory.tiering import LayerPlacement, Plan, Tier

G, M = 2 ** 30, 2 ** 20
N_COUCHES = 60


def _spec():
    return ModelSpec(name="gemma31b", architecture="llama", hidden_size=5376, intermediate_size=21504,
                     num_layers=N_COUCHES, num_attention_heads=32, num_key_value_heads=16, vocab_size=262144,
                     max_position_embeddings=262144, head_dim=256, sliding_window=1024,
                     layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10)


def _manifest():
    t = {}
    for i in range(N_COUCHES):
        # attention 74 Mio + MLP 195 Mio en nvfp4 (chiffres du manifeste réel)
        t[f"model.layers.{i}.self_attn.q_proj.weight"] = {"format": "nvfp4", "bpw": 4.5, "shape": [8192, 16128]}
        t[f"model.layers.{i}.mlp.gate_proj.weight"] = {"format": "nvfp4", "bpw": 4.5, "shape": [21504, 16128]}
    t["model.embed_tokens.weight"] = {"format": "bf16", "shape": [262144, 5376]}      # 2,6 Gio
    return {"vision": "non", "tensors": t, "model": {}}


def _plan(cap):
    return Plan(model="gemma31b", tiers=[
        Tier(name="cuda:0", kind="gpu", device_index=0, capacity=cap, weight_format="nvfp4", kv_format="int8",
             read_bandwidth=1790.0, link_bandwidth=21.0),
        Tier(name="cpu", kind="host", device_index=-1, capacity=128 * G, weight_format="nvfp4", kv_format="int8",
             read_bandwidth=60.0, link_bandwidth=21.0)],
        layers=[LayerPlacement(index=i, exec_device="cuda:0", attn_storage="cuda:0", mlp_storage="cuda:0", fmt="nvfp4",
                               attn_bytes=74 * M, mlp_bytes=195 * M, mlp_active_bytes=195 * M) for i in range(N_COUCHES)],
        embed_device="cuda:0", lm_head_device="cuda:0")


def _refus(monkeypatch, libre, cap=34 * G, ctx=32768):
    monkeypatch.delenv("ACVRAM_MTP", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (libre, cap))
    spec, man, p = _spec(), _manifest(), _plan(cap)
    p.kv_bytes_per_token = spec.kv_bytes_per_token(8)
    p.kv_budget = {"cuda:0": p.kv_bytes_per_token * ctx}
    reserve = LD._reserve_prefill(spec, ctx, man, p)
    try:
        LD._borner_kv_avec_exil(p, man, lambda nom: nom, spec, max_model_len=ctx, reserve=reserve)
    except RuntimeError as e:
        return str(e), p, spec, man, reserve
    return None, p, spec, man, reserve


def test_le_refus_nomme_une_fenetre_exacte(monkeypatch, capsys):
    msg, p, spec, man, _ = _refus(monkeypatch, libre=int(28.1 * G))
    assert msg and "fenêtre qui tient : " in msg, msg
    n = int(msg.split("fenêtre qui tient : ")[1].split()[0])
    # levier 1 : la fenêtre annoncée compte la réserve PLAFONNÉE (MLP par tranches de 4 096 à la relance) : 31 744 à
    # 28,1 Gio libres (23 552 avec la réserve d'un seul tenant, chiffre du verdict kv31b avant le levier)
    assert 28000 <= n <= 32767, n
    assert f"[acvram] fenêtre qui tient : {n} jetons" in capsys.readouterr().err
    # exactitude : N tient, N + 1 024 ne tient pas — avec le plan tel qu'exilé au refus et la réserve plafonnée
    bornes = LD._borner_kv_par_la_vram(p, man, lambda nom: nom, reserve=0)
    base = int(bornes["cuda:0"])                                    # réserve 0 : libre − poids − marge de base
    spec.mlp_prefill_plafond = 4096
    cout = lambda k: LD._kv_plancher(p, spec, k, "cuda:0") + LD._reserve_prefill(spec, k, man, p)
    assert cout(n) <= base < cout(n + 1024), (cout(n) / G, base / G, cout(n + 1024) / G)
    spec.mlp_prefill_plafond = None
    assert n > 23552, "la réserve plafonnée doit faire tenir plus que la pleine"


def test_temoin_carte_large_ne_refuse_pas(monkeypatch):
    msg, *_ = _refus(monkeypatch, libre=60 * G, cap=64 * G)
    assert msg is None


def test_rien_ne_tient_rend_zero(monkeypatch):
    msg, *_ = _refus(monkeypatch, libre=6 * G, cap=8 * G)
    assert msg and "fenêtre qui tient : 0 jetons" in msg
