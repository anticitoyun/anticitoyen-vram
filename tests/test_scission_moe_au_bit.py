"""Scission 22/09 (module 4) : le déplacement du bloc MoE vers `engine/moe.py`
ne change RIEN à la sortie.

Deux empreintes, figées avant le déplacement (calculées sur le dépôt à
30628c1c) : un bloc MoE jouet déterministe (préfill de 8 jetons puis un pas de
décodage, chemins par défaut) et la génération complète du modèle dense jouet
(ids ET logprobs, température 0). Les deux sont des sha256 des octets bruts
des tenseurs : un bit qui bouge les casse.

ACVRAM_EMPREINTE_FICHIER=<json> les écrit au lieu de les comparer — c est
ainsi que les valeurs de référence ont été relevées AVANT la scission, dans un
arbre à l état d origine (sha256 : des chaînes qu un filtre de terminal prend
pour des jetons, donc jamais imprimées).
"""
from __future__ import annotations

import hashlib
import os

import pytest
import torch

# Les empreintes ci-dessous ont été relevées avec les noyaux du dépôt (défaut) ; le chemin de
# référence (ACVRAM_DISABLE_KERNELS / _CPU_KERNELS, celui de la CI sans GPU) produit d'autres
# octets, donc d'autres sha256 : ici on ne juge que la scission, pas le chemin de référence.
pytestmark = pytest.mark.skipif(
    os.environ.get("ACVRAM_DISABLE_KERNELS") or os.environ.get("ACVRAM_DISABLE_CPU_KERNELS"),
    reason="empreintes relevées avec les noyaux du dépôt ; chemin de référence forcé",
)

# Relevées sur 30628c1c (avant le déplacement), processeur, fp32.
EMPREINTE_MOE = "523a51e1df3d0cb1289ddfa98f8d75059482f71c144eaacc04b7f9eaed784747"
EMPREINTE_DENSE = "8cdb9cf9607d18d064c9b726d9eed66baf1c31025ac76e3f1c7a89b2bfbf084e"


def _sha(*tenseurs) -> str:
    h = hashlib.sha256()
    for t in tenseurs:
        x = t.detach().to(torch.float32).contiguous().cpu()
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def _juger(nom: str, valeur: str, attendu: str) -> None:
    sortie = os.environ.get("ACVRAM_EMPREINTE_FICHIER")
    if sortie:
        import json
        d = json.load(open(sortie)) if os.path.exists(sortie) else {}
        d[nom] = valeur
        json.dump(d, open(sortie, "w"), indent=1)
        return
    assert valeur == attendu, f"{nom} : {valeur} ≠ {attendu} — la scission a changé la sortie"


def test_bloc_moe_au_bit():
    """Le bloc MoE lui-même : préfill puis décodage, chemins par défaut."""
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    bloc = _bloc_moe_jouet(E=4, H=128, I=64, top_k=2, dev="cpu")
    g = torch.Generator().manual_seed(20260922)
    xs = (torch.randn(8, 128, generator=g) * 0.3).to(torch.bfloat16)
    with torch.inference_mode():
        y_prefill = bloc(xs)
        y_decode = bloc(xs[:1])
    _juger("MOE", _sha(y_prefill, y_decode), EMPREINTE_MOE)


@pytest.fixture(scope="module")
def converted_cpu(tiny_checkpoint, target_rig, tmp_path_factory):
    """Comme la fixture `converted` (conftest.py), mais `quant_device="cpu"` figé.

    Pièce 154 (4) : `_resolve_quant_device("auto")` (convert.py:802-804) prend le
    GPU s'il est visible — la quantification GPU et CPU ne sont PAS au bit
    identiques (même défaut que 1/2, tests/test_mm_conversion.py). La fixture
    partagée `converted` sert ~35 fichiers dont beaucoup jugent le chemin GPU
    réel : la figer casserait leur couverture. Cette copie locale, réservée aux
    empreintes au bit de ce fichier, ne change qu'ici. Voir le bead qui nomme
    le défaut (GPU/CPU non bit-identiques en quantification)."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("acvram_cpu"))
    convert_checkpoint(tiny_checkpoint, plan,
                       ConversionOptions(out_dir=out, quant_device="cpu"), spec=spec)
    return out


def test_generation_dense_au_bit(converted_cpu):
    """Le reste du moteur : mêmes ids ET mêmes logprobs sur le jouet dense."""
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    engine = Engine(load_model(converted_cpu, dtype=torch.float32, device_override="cpu"),
                    None, max_batch_size=2, max_model_len=256)
    sorties = list(engine.generate([5, 42, 7, 99, 13], SamplingParams(temperature=0.0, max_tokens=12, logprobs=1)))
    ids = torch.tensor([t for o in sorties for t in o.token_ids], dtype=torch.long)
    lps = torch.tensor([o.logprob if o.logprob is not None else float("nan") for o in sorties])
    _juger("DENSE", _sha(ids, lps), EMPREINTE_DENSE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="démontre l'écart GPU/CPU : carte requise")
def test_quantification_gpu_diverge_du_cpu_au_bit(tiny_checkpoint, target_rig, tmp_path):
    """Témoin du défaut nommé (pièce 154 (4), bead) : `_resolve_quant_device("auto")`
    (convert.py:802-804) choisit le GPU quand il est visible, et la quantification
    GPU n'est PAS au bit celle du CPU — donc `ConversionOptions()` par défaut
    (quant_device="auto") rend un résultat qui dépend de la carte, jamais du
    seul modèle. Casserait si le calcul devenait un jour bit-identique aux deux
    (à réviser alors, pas un signe d'échec)."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512, max_concurrent_seqs=2))

    out_cpu = str(tmp_path / "out_cpu")
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=out_cpu, quant_device="cpu"), spec=spec)
    out_gpu = str(tmp_path / "out_gpu")
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=out_gpu, quant_device="cuda:0"), spec=spec)

    from safetensors import safe_open
    def empreinte(dossier):
        h = hashlib.sha256()
        import os as _os
        for fn in sorted(_os.listdir(dossier)):
            if fn.endswith(".safetensors"):
                with safe_open(_os.path.join(dossier, fn), framework="pt", device="cpu") as fh:
                    for k in sorted(fh.keys()):
                        t = fh.get_tensor(k).contiguous().reshape(-1)
                        h.update(k.encode())
                        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        return h.hexdigest()

    assert empreinte(out_cpu) != empreinte(out_gpu), (
        "GPU et CPU rendent maintenant le même octet en quantification : le défaut nommé "
        "en pièce 154 (4) n'existe plus, `converted_cpu` ci-dessus peut revenir à `converted`")
