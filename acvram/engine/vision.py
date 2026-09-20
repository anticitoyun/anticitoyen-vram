"""Tour de vision (multimodal P1, poste7-go-multimodal-organisation-20-09 § 2).

Une passe eager bf16 par image, sous ``torch.no_grad``, HORS graphes CUDA,
AVANT le prefill et jamais sur le chemin du décodage : la tour rend les traits
``[n, hidden]`` (projection ``embed_vision`` comprise) que ``ForwardBatch.images``
disperse à la place des embeddings des jetons image (engine/model.py,
``disperser_images``).

Les classes de la tour sont celles de ``transformers`` (Gemma 4 d'abord),
chargées depuis le dossier converti qui garde ``model.vision_tower.*`` et
``model.embed_vision.*`` en bf16 sous leur nom source (pièce (a)). Le calcul
lui-même est injectable (``TourVision(calcul=...)``) : les essais à sec posent
une mini-tour factice, sans ``transformers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

PREFIXES_TOUR = ("model.vision_tower.", "model.embed_vision.",
                 "model.multi_modal_projector.")

# La ligne de régime (acvram/regime.py) nomme la tour dès qu'une est chargée.
_CHARGEE: Optional[str] = None


class SansTourVision(ValueError):
    """Une image est arrivée sur un modèle sans tour de vision (manifeste
    ``vision: non``) : refus nommé, jamais un silence."""


@dataclass
class ImageRequete:
    """Une image d'une requête interne (frontière avec la pièce (b)).

    Tolérante : un objet à attributs ``debut``, ``fin``, ``pixel_values``,
    ``sha256``, ou un tuple ``(debut, fin, pixel_values, sha256)``.
    ``[debut, fin)`` sont les positions des jetons image dans l'invite,
    ``sha256`` celui des ``pixel_values`` (clé du cache de préfixe).
    """
    debut: int
    fin: int
    pixel_values: Any
    sha256: str

    @classmethod
    def depuis(cls, obj: Any) -> "ImageRequete":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, (tuple, list)):
            if len(obj) != 4:
                raise ValueError(f"image : tuple de {len(obj)} éléments, attendu "
                                 "(debut, fin, pixel_values, sha256)")
            d, f, pv, sha = obj
        else:
            try:
                d, f, pv, sha = obj.debut, obj.fin, obj.pixel_values, obj.sha256
            except AttributeError as exc:
                raise ValueError(f"image : objet {type(obj).__name__} sans attributs "
                                 "debut/fin/pixel_values/sha256") from exc
        d, f = int(d), int(f)
        if not (0 <= d < f):
            raise ValueError(f"image : plage [{d}, {f}) vide ou négative")
        return cls(d, f, pv, str(sha))


def verifier_plages(images: list[ImageRequete], n_prompt: int) -> None:
    """Plages dans l'invite, triées, sans chevauchement — sinon erreur nommée."""
    fin_prec = 0
    for im in sorted(images, key=lambda i: i.debut):
        if im.fin > n_prompt:
            raise ValueError(f"image : plage [{im.debut}, {im.fin}) au-delà de "
                             f"l'invite ({n_prompt} jetons)")
        if im.debut < fin_prec:
            raise ValueError(f"image : plage [{im.debut}, {im.fin}) chevauche la "
                             f"précédente (fin {fin_prec})")
        fin_prec = im.fin


class TourVision:
    """Tour de vision + projection : ``pixel_values`` → traits bf16 ``[n, h]``."""

    def __init__(self, calcul: Callable[[Any], torch.Tensor], device: torch.device,
                 nom: str = "transformers") -> None:
        self._calcul = calcul
        self.device = torch.device(device)
        self.nom = nom
        global _CHARGEE
        _CHARGEE = nom

    @torch.no_grad()
    def traits(self, pixel_values: Any, n_attendu: Optional[int] = None) -> torch.Tensor:
        """Une image → ``[n, hidden]`` bf16 ; ``n_attendu`` (fin − debut) vérifié."""
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device)
        out = self._calcul(pixel_values)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"tour de vision : sortie {type(out).__name__}, tenseur attendu")
        out = out.reshape(-1, out.shape[-1]).to(torch.bfloat16)
        if n_attendu is not None and out.shape[0] != n_attendu:
            raise ValueError(f"tour de vision : {out.shape[0]} traits pour une plage de "
                             f"{n_attendu} jetons image")
        return out

    @classmethod
    def depuis_dossier(cls, path: str, manifest: dict,
                       device: torch.device) -> Optional["TourVision"]:
        """None si le manifeste ne déclare pas de tour (``vision`` absent,
        faux ou « non ») ; sinon la tour transformers chargée en bf16 sur
        ``device``. Un manifeste qui déclare une tour introuvable est une
        erreur nommée, pas un modèle texte."""
        decl = manifest.get("vision")
        if not decl or (isinstance(decl, str) and decl.strip().lower() in ("non", "no", "false", "0")):
            return None
        try:
            import transformers
            from transformers import AutoConfig, AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError("tour de vision déclarée par le manifeste mais "
                               "transformers absent du venv") from exc
        from .loader import _ShardReader
        try:
            cfg = AutoConfig.from_pretrained(path)
        except Exception as exc:                            # noqa: BLE001
            raise RuntimeError(f"tour de vision : config.json illisible dans {path} "
                               f"({type(exc).__name__}: {exc})") from exc
        with torch.device("meta"):
            modele = AutoModelForImageTextToText.from_config(cfg, torch_dtype=torch.bfloat16)
        noms = [n for n in manifest.get("tensors", {}) if n.startswith(PREFIXES_TOUR)]
        if not noms:
            raise RuntimeError("tour de vision déclarée mais aucun tenseur "
                               f"{'|'.join(PREFIXES_TOUR)} dans le manifeste")
        reader = _ShardReader(path, manifest["weight_map"])
        # Seuls les sous-modules de la tour sont matérialisés ; le reste du
        # modèle HF reste sur « meta » et n'est jamais appelé.
        racine = getattr(modele, "model", modele)
        sous = [getattr(racine, p.split(".")[1]) for p in PREFIXES_TOUR
                if hasattr(racine, p.split(".")[1])]
        for sm in sous:
            sm.to_empty(device=device)
        etat = {}
        for n in noms:
            cle = n[len("model."):]
            etat[cle] = reader.get(n).to(device=device, dtype=torch.bfloat16)
        manque, inattendu = racine.load_state_dict(etat, strict=False)
        manque = [m for m in manque if m.startswith(tuple(p[len("model."):] for p in PREFIXES_TOUR))]
        if manque or inattendu:
            raise RuntimeError(f"tour de vision : {len(manque)} poids manquants, "
                               f"{len(inattendu)} inattendus (ex. {(manque + list(inattendu))[:3]})")
        for sm in sous:
            sm.eval()

        def calcul(pv: Any) -> torch.Tensor:
            pv = pv.to(device=device, dtype=torch.bfloat16)
            if pv.ndim == 3:
                pv = pv.unsqueeze(0)
            return modele.get_image_features(pixel_values=pv)

        return cls(calcul, device, nom=f"transformers {transformers.__version__}")


def regime_texte() -> str:
    """« vision=bf16(eager) » quand une tour est chargée, sinon ''."""
    return "vision=bf16(eager)" if _CHARGEE else ""
