"""Voie d'exil (poids denses en flux, `layers.py::StreamedWeight`) : la sortie
DOIT être identique au résident — même noyau, seule l'adresse change.

Écrit sur ordre de Sage (`sage-exil-ppl-89-priorite-17-09.md`, à sec, avant
toute mesure), suite à `verdict-llama70b-snr-cause-17-09.md` : conversion
innocentée deux fois (SNR normal, lecture GGUF bit-exacte) sur Llama-3.3-70B
exilé, écart PPL +89 % restant à expliquer côté moteur. `ACVRAM_EXIL_COUCHES=N`
(loader.py:1636) force l'exil sur N'IMPORTE QUEL modèle — pas besoin du 70B
(508 s de chargement, 20 Gio d'arène) pour ce test.

Sans ce test, aucune correction de la voie d'exil n'entre dans main (règle
posée par Sage) : la sortie doit rester identique quel que soit le bras qui
finira par expliquer l'écart (course d'ordonnancement `layers.py:388-399`,
préchargement `layers.py:455`, ou une troisième cause).
"""
from __future__ import annotations

import os

import pytest
import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../outils'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")

# Petit modèle nvfp4 déjà classé, pour ne pas payer 508 s de chargement du
# 70B pour cette épreuve : le mécanisme d'exil (loader.py:1636) ne dépend pas
# de la taille du modèle.
MODELE = os.environ.get(
    "ACVRAM_TEST_EXIL_MODELE",
    _RACINE + "/Qwen3-14B-nvfp4")
CORPUS = os.environ.get(
    "ACVRAM_TEST_EXIL_CORPUS",
    "acvram/data/calibration-anglais.txt")
N_JETONS = 64
N_COUCHES_EXIL = 18


def _logits_64(model_dir: str, env_exil: dict) -> torch.Tensor:
    """Charge le modèle sous l'environnement donné, rend les logits de
    prefill sur les `N_JETONS` premiers jetons du corpus — même appel que
    `evaluate.perplexity` (`model(batch, logits_positions=...)`), sans passer
    par la fenêtre glissante ni la perte : on compare des LOGITS, pas une PPL
    agrégée, pour voir la première position qui diverge."""
    from acvram.engine.loader import load_model
    from acvram.engine.model import ForwardBatch
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    from acvram.server.chat import load_tokenizer

    anciens = {k: os.environ.get(k) for k in env_exil}
    try:
        for k, v in env_exil.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        loaded = load_model(model_dir, dtype=torch.bfloat16, device_override="cuda")
        tok = load_tokenizer(model_dir)
        texte = open(CORPUS, encoding="utf-8", errors="replace").read()
        ids = tok.encode(texte)[:N_JETONS]
        assert len(ids) == N_JETONS, f"corpus trop court pour {N_JETONS} jetons"

        n = len(ids)
        blocks_par_fenetre = (n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
        alloc = BlockAllocator(blocks_par_fenetre, enable_prefix_cache=False)
        blocks = alloc.allocate(blocks_par_fenetre)
        slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
                              for i in range(n)], dtype=torch.long)
        batch = ForwardBatch(
            tokens=torch.tensor(ids, dtype=torch.long),
            positions=torch.arange(n, dtype=torch.long),
            seq_lens=[n], query_lens=[n],
            block_tables=[torch.tensor(blocks, dtype=torch.long)],
            slot_mapping=slots, is_prefill=True)
        with torch.inference_mode():
            logits = loaded.model(batch, logits_positions=batch.all_token_indices())
        return logits.to(torch.float32).cpu()
    finally:
        for k, v in anciens.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@CUDA
@pytest.mark.skipif(not os.path.isdir(MODELE), reason=f"alias absent sous racine_modeles() : {MODELE}")
def test_exil_18_couches_rend_les_memes_logits_que_resident():
    """Témoin de Sage (REGLES §5) : E1 (exil seul) contre R (résident). Un
    écart désigne la voie d'exil elle-même (pas la conversion, déjà
    innocentée). Attendu : 0 exactement, même noyau -- toute divergence,
    même infime, est le signal que ce test existe pour capter."""
    if not os.path.isdir(MODELE):
        pytest.skip(f"modèle absent : {MODELE} (ACVRAM_TEST_EXIL_MODELE)")

    resident = _logits_64(MODELE, {"ACVRAM_EXIL_COUCHES": None})
    exile = _logits_64(MODELE, {"ACVRAM_EXIL_COUCHES": str(N_COUCHES_EXIL)})

    ecart = (resident - exile).abs()
    assert torch.isfinite(resident).all() and torch.isfinite(exile).all()
    assert torch.equal(resident, exile), (
        f"exil {N_COUCHES_EXIL} couches diverge du résident : "
        f"max|Δ|={ecart.max().item():.6g}, position pire={ecart.max(dim=-1).values.argmax().item()} "
        f"sur {N_JETONS} jetons — la voie d'exil change la sortie (REGLES 9)")


@CUDA
@pytest.mark.skipif(not os.path.isdir(MODELE), reason=f"alias absent sous racine_modeles() : {MODELE}")
def test_exil_d_une_seule_couche_isole_la_faute_par_couche():
    """E4 du protocole de Sage : une seule couche exilée doit aussi coller
    au résident. Si CE test passe mais le précédent (18 couches) échoue, la
    faute croît avec le nombre de couches (pool saturé, plan) plutôt que
    d'être une erreur de déquantification par couche (_rehydrate,
    layers.py:507-523)."""
    if not os.path.isdir(MODELE):
        pytest.skip(f"modèle absent : {MODELE} (ACVRAM_TEST_EXIL_MODELE)")

    resident = _logits_64(MODELE, {"ACVRAM_EXIL_COUCHES": None})
    exile_1 = _logits_64(MODELE, {"ACVRAM_EXIL_COUCHES": "1"})

    assert torch.equal(resident, exile_1), (
        "exil d'UNE SEULE couche diverge déjà du résident : la faute n'a "
        "pas besoin de plusieurs couches pour apparaître")
