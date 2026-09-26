"""`perplexity()` par tranches de tête (poste7-p2-ppl-instrument-file-7h-19-09) :
la NLL par position calculée par `_pertes_par_tranches` (états cachés → tête
par tranches) égale à 10⁻⁶ celle de l'ancien chemin (logits de la fenêtre
entière en un appel), quelle que soit la taille de tranche — sur le modèle
jouet, à sec. Témoin cassant : une tranche décalée d'une position change la
NLL (le test ne peut pas rendre « vrai » par construction)."""
import math
import os
import shutil
import sys
import pathlib

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from acvram.engine.loader import load_model                                     # noqa: E402
from acvram.engine.model import ForwardBatch                                    # noqa: E402
from acvram.evaluate import _pertes_par_tranches, perplexity                    # noqa: E402
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator                  # noqa: E402


def _batch(chunk, blocks_per_window):
    alloc = BlockAllocator(blocks_per_window, enable_prefix_cache=False)
    blocks = alloc.allocate(blocks_per_window)
    n = len(chunk)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)], dtype=torch.long)
    return ForwardBatch(tokens=torch.tensor(chunk, dtype=torch.long), positions=torch.arange(n, dtype=torch.long),
                        seq_lens=[n], query_lens=[n], block_tables=[torch.tensor(blocks, dtype=torch.long)],
                        slot_mapping=slots, is_prefill=True)


def test_tranches_egalent_la_fenetre_entiere(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_model_len=128)
    model = loaded.model
    torch.manual_seed(7)
    chunk = torch.randint(5, 200, (61,)).tolist()
    batch = _batch(chunk, (len(chunk) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    targets = torch.tensor(chunk[1:], dtype=torch.long)
    with torch.no_grad():
        ref = model(batch, logits_positions=batch.all_token_indices())[:-1].to(torch.float32)
        nll_ref = torch.nn.functional.cross_entropy(ref, targets, reduction="none")
        batch2 = _batch(chunk, (len(chunk) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
        h = model(batch2, return_hidden=True)
        # Pièce 267 (CI GitHub, runner CPU, `device_override="cpu"` ci-dessus force
        # d'ailleurs TOUJOURS ce test sur processeur, carte ou pas) : atol=1e-6
        # était franchi de ~2,6e-6 — valeur observée en CI 60,894676542969 contre
        # la référence 60,894673955078 (même calcul, même modèle jouet). Cause :
        # la tête entière (un seul appel) et la tête par tranches (plusieurs
        # appels plus petits) n'accumulent pas les produits matriciels dans le
        # même ordre en fp32 — le BLAS processeur choisit son propre découpage,
        # contrairement à la carte où l'ordre des accumulations est fixe pour
        # une même forme. 1e-5 couvre l'écart observé (2,6e-6) avec de la marge,
        # sans s'approcher d'une vraie divergence de chemin (le témoin cassant
        # plus bas diverge de ≥ 1e-3). CONDITIONNÉ au device réellement utilisé,
        # jamais élargi à l'aveugle si ce test tournait un jour aussi sur carte.
        atol = 1e-5 if h.device.type == "cpu" else 1e-6
        for tranche in (1, 7, 16, 256):
            nll = _pertes_par_tranches(model, h, targets, 0, tranche)
            assert nll.shape == nll_ref.shape
            assert torch.allclose(nll, nll_ref, rtol=0, atol=atol), (tranche, (nll - nll_ref).abs().max())
        nll3 = _pertes_par_tranches(model, h, targets, 3, 16)               # first_new respecté
        assert torch.allclose(nll3, nll_ref[3:], rtol=0, atol=atol)
        # témoin cassant : la même tête sur des positions décalées d'un cran
        faux = torch.nn.functional.cross_entropy(model._logits_finaux(model._tete(h[1:])).float()[:-1], targets[:-1], reduction="none")
        assert not torch.allclose(faux, nll_ref[:-1], atol=1e-3)


# Pièce 267b (chef, sur la CI locale/publication) : `assert r5.par_contexte ==
# r_tout.par_contexte` (égalité EXACTE d'un dict {contexte: (somme, compte)}) supposait
# que la somme des pertes ne dépend pas de la taille de tranche — faux en fp32 sur CPU,
# même mécanisme que `test_tranches_egalent_la_fenetre_entiere` ci-dessus (la tête par
# tranches de 5 et celle par tranches de 4096 n'accumulent pas les produits dans le même
# ordre), mais accumulé ici sur ~180 positions au lieu d'être comparé position par
# position. Observé en CI : {0: (60,894676542969, 180)} contre {0: (60,894673955078, 180)}
# — écart absolu 1,5e-5 sur une somme de 60,89, et {32: (1367,8499450684, 180)} contre
# {32: (1367,8499298096, 180)} — écart absolu 1,5e-5 sur 1367,85 (relatif ~1,1e-8).
# CAUSE probable de « ça passait avant, plus maintenant » : plusieurs fichiers de la
# suite (`test_c10_marlin_distinct_glm.py`, `test_godets_b.py`, `test_mla_1p_grille_c14.py`,
# `test_mla_glue_c15.py`, `test_mla_niveau2_jumeaux.py`) appellent
# `torch.set_num_threads(min(8, torch.get_num_threads()))` au niveau MODULE — exécuté à
# l'IMPORT, donc pendant la COLLECTE pytest, avant qu'aucun test ne tourne. Le nombre de
# fils BLAS est un état de PROCESSUS, partagé par toute la suite : quels fichiers sont
# collectés et dans quel ordre (alphabétique par défaut, tous les noms ci-dessus précèdent
# `test_ppl_tranches.py`) change ce nombre AVANT ce test, donc l'ordre d'accumulation
# fp32 de ses deux appels à `perplexity()`. Un run isolé de ce seul fichier (ce que j'ai
# vérifié ici) ne voit jamais cette clamp et ne peut donc pas reproduire le chiffre exact
# de la CI — le mécanisme (dépendance à l'ordre d'accumulation fp32) est cependant le
# même que celui déjà mesuré et corrigé plus haut dans ce fichier (256b/267 : ~2,6e-6 sur
# 60,89 avec seulement `atol`). rel_tol=1e-6 couvre l'écart observé (~1,1e-8) avec une
# marge large ; le COMPTE de positions, lui, ne dépend jamais de la taille de tranche —
# comparé à l'identique, jamais toléré.
def _par_contexte_proche(a: dict, b: dict, rel_tol: float = 1e-6) -> bool:
    if a.keys() != b.keys():
        return False
    for bas, (som_a, n_a) in a.items():
        som_b, n_b = b[bas]
        if n_a != n_b or not math.isclose(som_a, som_b, rel_tol=rel_tol, abs_tol=0):
            return False
    return True


def test_temoin_par_contexte_proche_detecte_un_vrai_ecart():
    """Le témoin qui casse : un écart 1000× plus grand que rel_tol n'est jamais toléré,
    et un compte différent ne l'est jamais non plus, même une somme identique."""
    ref = {0: (60.894676542969, 180), 32: (1367.8499450684, 180)}
    assert _par_contexte_proche(ref, {0: (60.894673955078, 180), 32: (1367.8499298096, 180)}), \
        "l'écart réellement observé en CI doit passer"
    assert not _par_contexte_proche(ref, {0: (60.894676542969 * 1.01, 180), 32: (1367.8499450684, 180)}), \
        "un écart 1 % (≈ 1e-2, bien au-delà de rel_tol) doit être refusé"
    assert not _par_contexte_proche(ref, {0: (60.894676542969, 179), 32: (1367.8499450684, 180)}), \
        "un compte de positions différent n'est jamais toléré, même somme égale"


def test_perplexity_de_bout_en_bout_inchangee(converted, tiny_checkpoint, monkeypatch):
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tiny_checkpoint, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(converted, fn))
    if not os.path.isfile(os.path.join(converted, "tokenizer.json")):
        pytest.skip("no tokenizer available")
    import acvram.evaluate as EV
    monkeypatch.setattr(EV, "_PPL_TRANCHE", 5)
    r5 = perplexity(converted, window=64, stride=32, max_tokens=256, device="cpu", dtype=torch.float32)
    monkeypatch.setattr(EV, "_PPL_TRANCHE", 4096)
    r_tout = perplexity(converted, window=64, stride=32, max_tokens=256, device="cpu", dtype=torch.float32)
    assert abs(r5.perplexity - r_tout.perplexity) <= 1e-6 * max(1.0, r_tout.perplexity)
    assert _par_contexte_proche(r5.par_contexte, r_tout.par_contexte), \
        (r5.par_contexte, r_tout.par_contexte)
