"""Lecture des points de contrôle EXL3, comme source de conversion.

EXL3 (exllamav3) code chaque tuile de 16×16 poids par un treillis QTIP : un
codebook procédural (MCG ou MUL1) parcouru sous contrainte, encadré par des
transformées de Hadamard signées des deux côtés (``suh``, ``svh``). Réécrire ce
décodeur de zéro serait long et surtout invérifiable — un treillis mal parcouru
produit des poids plausibles et un modèle fou. On délègue donc la
reconstruction à ``exllamav3`` lui-même, dépendance *optionnelle et de
conversion seulement* : le modèle converti n'en dépend plus.

La reconstruction est un noyau CUDA d'exllamav3 : il faut un GPU, et son
extension se compile au premier import — avec le nvcc des roues pip si celui
du système est trop vieux, le même détour que pour nos propres noyaux.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterator, Optional

import torch

__all__ = ["EXL3Checkpoint", "is_exl3"]

_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")


def is_exl3(path: str) -> bool:
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, "r", encoding="utf-8") as fh:
            q = json.load(fh).get("quantization_config") or {}
        return q.get("quant_method") == "exl3"
    except (OSError, json.JSONDecodeError):
        return False


def _ext():
    """Importe exllamav3 en s'assurant qu'un nvcc capable existe."""
    from ..kernels import _MIN_CUDA_FOR_SM120, _ensure_cuda_home
    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    _ensure_cuda_home(_MIN_CUDA_FOR_SM120
                      if any(c >= (12, 0) for c in caps) else (11, 0))
    try:
        from exllamav3.modules.quant.exl3 import LinearEXL3
    except ImportError as exc:
        raise RuntimeError(
            "la lecture EXL3 exige exllamav3 (pip install exllamav3) ; "
            f"import impossible : {exc}") from exc
    return LinearEXL3


class EXL3Checkpoint:
    def __init__(self, path: str, device: str = "cuda:0") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("la reconstruction EXL3 est un noyau CUDA : "
                               "aucun peripherique disponible")
        self.path = path
        self.device = device
        with open(os.path.join(path, "config.json"), "r",
                  encoding="utf-8") as fh:
            self.config = json.load(fh)
        self.files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not self.files:
            raise FileNotFoundError(f"aucun .safetensors sous {path}")
        # nom -> (fichier, nom brut) ; les stems quantifiés sont regroupés.
        self.plain: dict[str, tuple[str, str]] = {}
        self.quant: dict[str, dict[str, tuple[str, str]]] = {}
        from safetensors import safe_open
        for f in self.files:
            with safe_open(f, framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    stem, _, leaf = k.rpartition(".")
                    if leaf in _SUFFIXES:
                        self.quant.setdefault(stem, {})[leaf] = (f, k)
                    elif not (leaf == "bias" and stem in self.quant):
                        self.plain[k] = (f, k)

    @staticmethod
    def _rename(name: str) -> str:
        """Les enrobages multimodaux rangent le modèle sous ``language_model``."""
        return name.replace("model.language_model.", "model.")

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        from safetensors import safe_open
        LinearEXL3 = _ext()

        handles: dict[str, Any] = {}

        def get(f: str, k: str) -> torch.Tensor:
            if f not in handles:
                handles[f] = safe_open(f, framework="pt", device="cpu")
            return handles[f].get_tensor(k)

        for name, (f, k) in self.plain.items():
            out = self._rename(name)
            if out.startswith(("visual.", "mmproj")):
                continue                       # `visual.` nu (Qwen2-VL) : texte seul
            # model.visual.* / model.vision_tower.* / model.embed_vision.* passent
            # tels quels : `_adapt_hf` (VISION_PREFIXES) les garde en bf16
            yield out, get(f, k)

        for stem, parts in self.quant.items():
            kw: dict[str, Any] = {}
            for leaf, (f, k) in parts.items():
                t = get(f, k)
                if leaf in ("trellis", "suh", "svh", "su", "sv"):
                    kw[leaf] = t.to(self.device)
                elif leaf == "mcg":
                    kw["mcg"] = t
                elif leaf == "mul1":
                    kw["mul1"] = t
            trellis = kw["trellis"]
            in_f = trellis.shape[0] * 16
            out_f = trellis.shape[1] * 16
            lin = LinearEXL3(None, in_f, out_f, **kw)
            w = lin.get_weight_tensor()        # [entree, sortie], fp16, GPU
            yield self._rename(stem) + ".weight", \
                w.t().contiguous().to(torch.float32).cpu()
            del lin, w, kw
            torch.cuda.empty_cache()

    def hf_config(self) -> dict:
        cfg = dict(self.config)
        if "text_config" in cfg and "hidden_size" not in cfg:
            inner = cfg.pop("text_config")
            cfg = {**cfg, **inner}
        cfg.pop("quantization_config", None)
        return cfg

    def export_sidecars(self, out_dir: str) -> None:
        """Réécrit un config.json épuré : le converti n'est plus de l'EXL3."""
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(self.hf_config(), fh, indent=1)
