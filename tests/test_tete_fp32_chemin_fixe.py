"""La tête de sortie reçoit son entrée en fp32 sur TOUS les chemins (Sage,
sage-duel-verdict-16-09 § 13, REGLES §7) : a5fac1c avait porté le fp32 en
eager seulement ; le chemin à formes fixes (graphes, ACVRAM_GRAPHS_EAGER)
gardait `lm_head(h)` en bf16 — l'argmax basculait dès le 2e pas de décodage
(64-68 % de jetons justes à l'arbitre prefill = décodage, trois jours de
mesures). Contrat : `decode_fixed` et `forward` rendent des logits égaux au
bit sur un mini-modèle bf16, et le test casse si le fp32 est retiré d'un côté
(témoin ACVRAM_LOGITS_BF16 appliqué à un seul chemin)."""
import pytest
import torch

from acvram.engine.loader import load_model
from acvram.engine.model import ForwardBatch
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator


def _lot(model, prompt, blocks):
    n = len(prompt)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)])
    return ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n],
                        [torch.tensor(blocks)], slots, True)


def _charger(converted):
    try:
        loaded = load_model(converted, dtype=torch.bfloat16, device_override="cpu")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"mini-modèle bf16 sur processeur indisponible : {e}")
    return loaded


def test_decode_fixed_egal_forward_au_bit(converted, monkeypatch):
    loaded = _charger(converted)
    model = loaded.model
    prompt = [5, 42, 7, 99, 13]
    n = len(prompt)
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate((n + 2 + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    with torch.inference_mode():
        model(_lot(model, prompt, blocks))                     # prefill : KV des n jetons
        # pas de décodage : jeton 21 en position n, sur les deux chemins, même cache
        tok = torch.tensor([21]); pos = torch.tensor([n])
        slot = torch.tensor([blocks[n // BLOCK_SIZE] * BLOCK_SIZE + n % BLOCK_SIZE])
        tables = torch.tensor(blocks).unsqueeze(0)
        eager = ForwardBatch(tok, pos, [n + 1], [1], [torch.tensor(blocks)], slot, False)
        y_eager = model(eager).clone()
        x = torch.nn.functional.embedding(tok, model.embed_tokens).to(model.dtype)
        if model.spec.embedding_multiplier != 1.0:
            x = x * model.spec.embedding_multiplier
        y_fixe = model.decode_fixed(x, pos, slot, tables, torch.tensor([n + 1]),
                                    max_pos=n + 2, q_len=1)
        assert y_eager.dtype == torch.float32 and y_fixe.dtype == torch.float32
        assert torch.equal(y_eager, y_fixe), "decode_fixed et forward doivent rendre les mêmes logits au bit"
        # un changement qui doit casser : la tête en bf16 sur le chemin fixe seul
        tete = model._tete
        monkeypatch.setattr(model, "_tete", lambda h: model.lm_head(h))
        y_bf16 = model.decode_fixed(x, pos, slot, tables, torch.tensor([n + 1]),
                                    max_pos=n + 2, q_len=1)
        assert not torch.equal(y_bf16.float(), y_eager), "le témoin bf16 devait diverger du fp32"
        monkeypatch.setattr(model, "_tete", tete)
