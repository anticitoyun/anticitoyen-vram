"""aym (30/09, edz définitif) : 18 alias MoE 30B (qwen3-coder-30b-a3b ×16, qwen3-vl-30b-a3b ×2) morts au démarrage à
29 096-32 768. Les piles d experts et leur repack Marlin se construisaient APRÈS le KV (`GraphRunner._eligible`, ou
avant la chauffe depuis ya1), quand le budget KV avait rempli la carte jusqu à la marge : OOM dans `_construire_marlin`
(hors de la garde OOM de la pile naturelle) ou chauffe à 0 jeton. Le chargeur les construit désormais AVANT d allouer
le KV, et un OOM du repack devient un refus nommé (pile naturelle gardée). À sec, plus un bras carte (`-k carte`)."""
from __future__ import annotations

import pytest
import torch

from acvram.engine import contexte
from acvram.engine import loader as L
from acvram.engine import moe as MOE
from acvram.engine.loader import load_model
from test_moe_grouped import tiny_moe  # noqa: F401  (fixture de session réutilisée)


def _journal_de_chargement(tiny_moe, monkeypatch, **env):
    """Charge `tiny_moe` sur le processeur en notant l ordre « piles » / « kv ». La garde « couche sur la carte » est
    forcée vraie : à sec, la pile naturelle se construit (Marlin refusé hors CUDA, sans lever)."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(contexte, "_sur_carte", lambda couche: True)
    journal = []
    vraie_pile, vrai_kv = MOE.MoEBlock._try_build_stacks, L.PagedKVCache

    def pile(self):
        journal.append("piles")
        return vraie_pile(self)

    def kv(cfg):
        journal.append("kv")
        return vrai_kv(cfg)
    monkeypatch.setattr(MOE.MoEBlock, "_try_build_stacks", pile)
    monkeypatch.setattr(L, "PagedKVCache", kv)
    loaded = load_model(tiny_moe, dtype=torch.float32, device_override="cpu")
    return journal, [m for m in loaded.model.modules() if isinstance(m, MOE.MoEBlock)]


def test_piles_construites_avant_le_kv(tiny_moe, monkeypatch):
    # Casse sur main : aucune pile n est construite au chargement (journal = ["kv", …] seulement, blocs à « ? »)
    journal, blocs = _journal_de_chargement(tiny_moe, monkeypatch)
    assert "kv" in journal, "le chargeur n a alloué aucun KV : le test ne prouverait rien"
    assert journal.count("piles") == len(blocs) >= 1, journal
    assert max(i for i, e in enumerate(journal) if e == "piles") < journal.index("kv"), journal
    assert {b._stack_state for b in blocs} == {"oui"}


def test_temoin_construction_paresseuse(tiny_moe, monkeypatch):
    journal, blocs = _journal_de_chargement(tiny_moe, monkeypatch, ACVRAM_PILES_AU_CHARGEMENT="0")
    assert "piles" not in journal and {b._stack_state for b in blocs} == {"?"}


def test_hors_carte_rien_au_chargement(tiny_moe, monkeypatch):
    """Garde réelle (`couche.device`) : un chargement processeur ne construit rien, comme avant."""
    loaded = load_model(tiny_moe, dtype=torch.float32, device_override="cpu")
    assert {m._stack_state for m in loaded.model.modules() if isinstance(m, MOE.MoEBlock)} == {"?"}


def test_oom_du_repack_marlin_refus_nomme(tiny_moe, monkeypatch, capsys):
    """Casse sur main : l OutOfMemoryError de `_construire_marlin` traversait `_try_build_stacks` (moe.py:346) et
    tuait le serveur. Désormais : pile naturelle gardée, raison sur l instance, une ligne au journal."""
    loaded = load_model(tiny_moe, dtype=torch.float32, device_override="cpu")
    bloc = next(m for m in loaded.model.modules() if isinstance(m, MOE.MoEBlock))

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 48.00 MiB")
    monkeypatch.setattr(MOE.MoEBlock, "_construire_marlin", oom)
    monkeypatch.setattr(MOE.MoEBlock, "_marlin_oom_dit", False, raising=False)
    assert bloc._try_build_stacks() is True
    assert bloc._stacks_marlin is None
    assert bloc._raison_marlin == "mémoire GPU insuffisante pendant le repack Marlin"
    assert all(p[1] is not None for p in bloc._stacks.values()), "la pile naturelle doit rester servie"
    assert "repack Marlin" in capsys.readouterr().out


@pytest.mark.gpu_requis
@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte : piles réelles et repack Marlin")
def test_carte_meme_sortie_au_chargement_ou_paresseux(tiny_moe, monkeypatch):
    """Règle 9 : seul le MOMENT change — sorties des blocs MoE au bit et génération T=0 identiques entre piles au
    chargement et témoin paresseux (construites au 1er forward)."""
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    x = (torch.randn(5, 128, generator=torch.Generator().manual_seed(7)) * 0.3).to(torch.bfloat16).to("cuda:0")

    def bras(val):
        monkeypatch.setenv("ACVRAM_PILES_AU_CHARGEMENT", val)
        loaded = load_model(tiny_moe, dtype=torch.bfloat16, device_override="cuda:0")
        blocs = [m for m in loaded.model.modules() if isinstance(m, MOE.MoEBlock)]
        avant = {b._stack_state for b in blocs}
        ys = [b(x) for b in blocs]
        e = Engine(loaded, None, max_batch_size=2, max_model_len=128)
        jetons = [t for o in e.generate([3, 1, 4, 1, 5], SamplingParams(temperature=0.0, max_tokens=16))
                  for t in o.token_ids]
        return avant, ys, jetons

    avant1, ys1, j1 = bras("1")
    avant0, ys0, j0 = bras("0")
    assert avant1 == {"oui"} and avant0 == {"?"}, (avant1, avant0)
    assert all(torch.equal(a, b) for a, b in zip(ys1, ys0))
    assert j1 == j0
