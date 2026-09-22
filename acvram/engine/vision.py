"""Tour de vision (multimodal P1, sage-go-multimodal-organisation-20-09 § 2).

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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

PREFIXES_TOUR = ("model.vision_tower.", "model.embed_vision.",
                 "model.multi_modal_projector.", "model.vision_embedder.",   # vision_embedder : gemma4_unified
                 "model.visual.")                                            # Qwen3-VL : tour + merger + deepstack_merger_list

# La ligne de régime (acvram/regime.py) nomme la tour dès qu'une est chargée.
_CHARGEE: Optional[str] = None
# Dépendance ÉPINGLÉE du moteur (Sage 14 h 10) : la version qui charge gemma4 sur le poste, la même que
# le lanceur du .deb installe (tools/construire-deb.sh) — tests/test_paquet_charge_utile.py les tient égales.
VERSION_TRANSFORMERS = "5.17.0"


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
    supplement: dict = field(default_factory=dict)   # annexes du processeur PAR IMAGE (Gemma 4 : image_position_ids ;
                                                     # Qwen3-VL : image_grid_thw), passées telles quelles à la tour

    @classmethod
    def depuis(cls, obj: Any) -> "ImageRequete":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, (tuple, list)):
            if len(obj) != 4:
                raise ValueError(f"image : tuple de {len(obj)} éléments, attendu "
                                 "(debut, fin, pixel_values, sha256)")
            d, f, pv, sha = obj
            supp = {}
        else:
            try:
                d, f, pv, sha = obj.debut, obj.fin, obj.pixel_values, obj.sha256
            except AttributeError as exc:
                raise ValueError(f"image : objet {type(obj).__name__} sans attributs "
                                 "debut/fin/pixel_values/sha256") from exc
            supp = dict(getattr(obj, "supplement", None) or {})
        d, f = int(d), int(f)
        if not (0 <= d < f):
            raise ValueError(f"image : plage [{d}, {f}) vide ou négative")
        return cls(d, f, pv, str(sha), supp)


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
                 nom: str = "transformers", hidden: Optional[int] = None) -> None:
        self._calcul = calcul
        self.device = torch.device(device)
        self.nom = nom
        self.hidden = hidden          # dimension du LM : un trait d'une autre dimension (tour NON projetée) est refusé
        self.niveaux_deepstack = 0    # k niveaux deepstack (Qwen3-VL), posé par depuis_dossier depuis la config de la tour
        global _CHARGEE
        _CHARGEE = nom

    @torch.no_grad()
    def traits(self, pixel_values: Any, n_attendu: Optional[int] = None,
               supplement: Optional[dict] = None) -> torch.Tensor:
        """Une image → ``[n, hidden]`` bf16 ; ``n_attendu`` (fin − debut) vérifié ; ``supplement`` =
        les annexes du processeur pour CETTE image (Gemma 4 : ``image_position_ids`` — sans elles
        ``get_image_features`` casse, Manon 14 h 40 ; Qwen3-VL : ``image_grid_thw``), passées en kwargs
        au calcul, jamais nommées ici. Les niveaux deepstack, s'il y en a, sont ignorés ici :
        `traits_niveaux` les rend."""
        return self.traits_niveaux(pixel_values, n_attendu, supplement)[0]

    @torch.no_grad()
    def traits_niveaux(self, pixel_values: Any, n_attendu: Optional[int] = None,
                       supplement: Optional[dict] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Comme `traits`, et rend ``(traits [n, hidden], niveaux [k, n, hidden] bf16 | None)`` :
        les niveaux deepstack de Qwen3-VL (``deepstack_features`` de la sortie de la tour, un par
        ``deepstack_visual_indexes``, déjà projetés par leur merger), None pour une tour qui n'en
        rend pas (Gemma 4 : chemin inchangé)."""
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device)
        supp = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in (supplement or {}).items()}
        try:
            out = self._calcul(pixel_values, **supp) if supp else self._calcul(pixel_values)
        except TypeError as exc:
            if supp and "unexpected keyword" in str(exc):
                raise TypeError(f"tour de vision : le calcul refuse les annexes du processeur {sorted(supp)} ({exc})") from exc
            raise
        niveaux = niveaux_deepstack(out)
        out = traits_projetes(out)
        out = out.reshape(-1, out.shape[-1]).to(torch.bfloat16)
        if self.hidden is not None and out.shape[-1] != self.hidden:
            raise ValueError(f"tour de vision : traits de dimension {out.shape[-1]}, le LM attend {self.hidden} "
                             "(sortie de la tour non projetée : embed_vision / merger absent du calcul)")
        if n_attendu is not None and out.shape[0] != n_attendu:
            raise ValueError(f"tour de vision : {out.shape[0]} traits pour une plage de "
                             f"{n_attendu} jetons image")
        if niveaux is not None:
            niveaux = niveaux.to(torch.bfloat16)
            if niveaux.shape[1:] != out.shape:
                raise ValueError(f"tour de vision : niveaux deepstack {tuple(niveaux.shape)} pour des traits "
                                 f"{tuple(out.shape)} (attendu (k, {out.shape[0]}, {out.shape[1]}))")
        return out, niveaux

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
        if transformers.__version__ != VERSION_TRANSFORMERS:
            print(f"[acvram] transformers {transformers.__version__} au lieu de {VERSION_TRANSFORMERS} "
                  f"(épinglé) : la ligne de régime porte la version relevée", flush=True)
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
        archi = (getattr(cfg, "architectures", None) or [None])[0]
        if archi and type(modele).__name__ != archi:
            raise RuntimeError(f"tour de vision : config.json déclare {archi}, transformers a construit "
                               f"{type(modele).__name__} — chemin get_image_features inconnu, tour refusée")
        racine = getattr(modele, "model", modele)
        # Les noms SOURCE (gardés tels quels par la conversion) deviennent les noms des modules HF par
        # le mapping de transformers (gemma4_unified : model.vision_embedder.* → embed_vision.*,
        # model.embed_vision.embedding_projection → embed_vision.multimodal_embedder.…) ; sans mapping,
        # le nom source privé de « model. » est le nom du module (Gemma4 SigLIP).
        renommeurs = []
        try:
            from transformers import conversion_mapping as _cm
            renommeurs = list(_cm.get_checkpoint_conversion_mapping(str(getattr(cfg, "model_type", ""))) or [])
        except Exception:                                   # noqa: BLE001
            renommeurs = []

        def nom_module(n: str) -> str:
            for r in renommeurs:
                try:
                    neuf = r.rename_source_key(n)
                except Exception:                           # noqa: BLE001
                    continue
                neuf = neuf[0] if isinstance(neuf, tuple) else neuf
                if neuf and neuf != n:
                    n = neuf
                    break
            return n[len("model."):] if n.startswith("model.") else n
        cles = {n: nom_module(n) for n in noms}
        sous_noms = sorted({c.split(".")[0] for c in cles.values()})
        sous = [getattr(racine, s) for s in sous_noms if hasattr(racine, s)]
        if len(sous) != len(sous_noms):
            raise RuntimeError(f"tour de vision : modules {sous_noms} attendus sur {type(racine).__name__}, "
                               f"présents {[n for n, _ in racine.named_children()]}")
        for sm in sous:
            sm.to_empty(device=device)
            rematerialiser_tampons(sm, device)
        etat = {}
        for n in noms:
            etat[cles[n]] = reader.get(n).to(device=device, dtype=torch.bfloat16)
        manque, inattendu = racine.load_state_dict(etat, strict=False)
        manque = [m for m in manque if m.startswith(tuple(f"{s}." for s in sous_noms))]
        if manque or inattendu:
            raise RuntimeError(f"tour de vision : {len(manque)} poids manquants, "
                               f"{len(inattendu)} inattendus (ex. {(manque + list(inattendu))[:3]})")
        for sm in sous:
            sm.eval()

        def calcul(pv: Any, **supplement: Any) -> torch.Tensor:
            pv = forme_pour_la_tour(pv.to(device=device, dtype=torch.bfloat16))
            # les annexes du processeur (image_position_ids Gemma 4, image_grid_thw Qwen3-VL) vont à
            # get_image_features telles quelles : c'est le modèle qui sait ce qu'il lui faut
            return modele.get_image_features(pixel_values=pv, **supplement)

        tcfg = getattr(cfg, "text_config", None) or cfg
        # Deepstack (Qwen3-VL) : le nombre de niveaux est celui de la config de la tour
        # (`deepstack_visual_indexes`), déclaré à la ligne de régime ; la config prime sur le manifeste
        vcfg = getattr(cfg, "vision_config", None)
        idx = getattr(vcfg, "deepstack_visual_indexes", None) if vcfg is not None else None
        if idx is not None:
            from .. import regime as _regime
            _regime.declarer_deepstack(len(idx))
        tour = cls(calcul, device, nom=f"transformers {transformers.__version__}",
                   hidden=int(getattr(tcfg, "hidden_size", 0)) or None)
        tour.niveaux_deepstack = len(idx) if idx is not None else 0       # lu sur la CONFIG de la tour chargée
        return tour


class MasqueImageInconnu(RuntimeError):
    """Famille de modèle à images dont le masque du LM sur la plage image n'est pas connu : refus nommé."""


# Masque du LM sur la plage image, PAR FAMILLE (Sage, sage-p3-1-scelle-temoin-20-09) : Gemma 3/4 — les jetons image
# se voient tous (bloc bidirectionnel, HF token_type_ids) ; Qwen2-VL / Qwen2.5-VL / Qwen3-VL — causal (HF
# create_causal_mask, la non-causalité est dans la tour seulement). Trouvé le 20/09 18:57 : le bloc de Gemma
# appliqué à Qwen3-VL rendait des lignes image fausses dès la couche 0 (diff-couches img02, au bit sous masque
# causal). Une famille absente = refus nommé, jamais un défaut de classe.
_MASQUE_PAR_FAMILLE = (("Gemma3", "bidir"), ("Gemma4", "bidir"),
                       ("Qwen2VL", "causal"), ("Qwen2_5_VL", "causal"), ("Qwen3VL", "causal"))


def masque_images_famille(spec: Any) -> str:
    """'bidir' | 'causal' d'après ``architectures`` du config.json du modèle chargé (spec.raw) ; MasqueImageInconnu sinon."""
    raw = getattr(spec, "raw", None) or {}
    archs = [str(a) for a in (raw.get("architectures") or [])]
    for a in archs:
        for prefixe, masque in _MASQUE_PAR_FAMILLE:
            if a.startswith(prefixe):
                return masque
    raise MasqueImageInconnu(f"masque de la plage image inconnu pour architectures={archs or '(absent)'} "
                             f"(familles connues : {', '.join(p for p, _ in _MASQUE_PAR_FAMILLE)}) — alias à images refusé")


def rematerialiser_tampons(module: torch.nn.Module, device: torch.device) -> list[str]:
    """Un module construit sur « meta » puis ``to_empty`` garde ses tampons NON persistants (absents du
    state_dict : ``inv_freq`` du RoPE de la tour Gemma 4) en mémoire NON initialisée — la tour tourne et
    rend des traits faux en silence (cos 0,13-0,53 contre la référence, 20/09 14 h 48). Chaque sous-module
    qui en possède est reconstruit sur ``device`` par son propre constructeur (``type(mod)(config,
    device=)`` — la convention des modules de position de transformers) et remplacé chez son parent ;
    un module qu'on ne sait pas reconstruire est un refus nommé, jamais un tampon aléatoire. Rend les
    noms des modules reconstruits."""
    refaits = []
    for nom, mod in list(module.named_modules()):
        persistants = set(mod.state_dict(keep_vars=True).keys())
        propres = [n for n, _ in mod.named_buffers(recurse=False) if n not in persistants]
        if not propres:
            continue
        cfg = getattr(mod, "config", None)
        neuf = None
        for essai in ((lambda: type(mod)(cfg, device=device)), (lambda: type(mod)(cfg)), (lambda: type(mod)(config=cfg))):
            try:
                neuf = essai()
                break
            except TypeError:
                continue
            except Exception as exc:                        # noqa: BLE001
                raise RuntimeError(f"tour de vision : tampons non persistants {propres} de {nom or '<racine>'} "
                                   f"({type(mod).__name__}) non reconstruits ({type(exc).__name__}: {exc})") from exc
        if neuf is None:
            raise RuntimeError(f"tour de vision : tampons non persistants {propres} de {nom or '<racine>'} "
                               f"({type(mod).__name__}) : constructeur inconnu, tour refusée")
        neuf = neuf.to(device)
        if nom == "":
            raise RuntimeError("tour de vision : le module racine porte des tampons non persistants, non reconstruisible")
        parent_nom, _, feuille = nom.rpartition(".")
        parent = module.get_submodule(parent_nom) if parent_nom else module
        setattr(parent, feuille, neuf)
        refaits.append(nom)
    return refaits


def traits_projetes(out: Any) -> torch.Tensor:
    """Ce qui ENTRE dans le LM, quelle que soit la forme de sortie de la tour : Gemma 4
    ``get_image_features`` rend un ``BaseModelOutputWithPooling`` dont ``pooler_output`` = tuple par
    image des traits PROJETÉS par ``embed_vision`` ([n, 5376]) et ``last_hidden_state`` = la tour brute
    ([1, patches, 1152], jamais servie au LM — références de Manon, references-tour.py) ; un tuple/liste
    (Qwen3-VL, transformers ≤ 4) = premier élément ; un tenseur = tel quel."""
    po = getattr(out, "pooler_output", None)
    if po is not None:
        return torch.cat([t.reshape(-1, t.shape[-1]) for t in po], 0) if isinstance(po, (tuple, list)) else po
    if isinstance(out, (tuple, list)):
        out = out[0]
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None and not isinstance(out, torch.Tensor):
        return lhs
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"tour de vision : sortie {type(out).__name__}, tenseur attendu")
    return out


def niveaux_deepstack(out: Any) -> Optional[torch.Tensor]:
    """Les niveaux deepstack de la sortie de la tour, empilés ``[k, n, h]``, ou None : Qwen3-VL
    (transformers 5) rend ``BaseModelOutputWithDeepstackFeatures.deepstack_features`` = liste de k
    tenseurs ``[n, h]`` (un par ``deepstack_visual_indexes``, chacun passé par son merger) ; un tuple
    ``(embeds, deepstack)`` (forme ancienne) porte la liste en second ; Gemma 4 (pooler seul) → None.
    Une liste vide → None (tour sans deepstack, jamais un tenseur à zéro niveau)."""
    ds = getattr(out, "deepstack_features", None)
    if ds is None and isinstance(out, (tuple, list)) and len(out) == 2 \
            and isinstance(out[1], (tuple, list)) and out[1] \
            and all(isinstance(t, torch.Tensor) for t in out[1]):
        ds = out[1]
    if ds is None or (isinstance(ds, (tuple, list)) and not ds):
        return None
    if isinstance(ds, torch.Tensor):
        return ds if ds.ndim == 3 else ds.reshape(1, -1, ds.shape[-1])
    return torch.stack([t.reshape(-1, t.shape[-1]) for t in ds], 0)


def forme_pour_la_tour(pv: torch.Tensor) -> torch.Tensor:
    """Le fragment garde la dimension image du processeur (``pv[i:i+1]``) : Gemma 4 rend des PATCHES
    déjà lotis ``[images, patches, dim]`` = [1, 2520, 768] — un unsqueeze « si 3-D » en faisait une image
    4-D fausse (« size of tensor a (16) must match … (2520) », Manon 14 h 55, 3e essai de (c)). Seule une
    image CHW NUE ``[3, H, W]`` reçoit sa dimension image ; aucune autre heuristique de forme."""
    if pv.ndim == 3 and pv.shape[0] == 3 and pv.shape[1] > 3 and pv.shape[2] > 3:
        return pv.unsqueeze(0)
    return pv


def oublier_la_tour() -> None:
    """Aucune tour chargée : appelé quand un modèle est déclaré sans vision (ou aucun modèle) — sinon le nom de la
    dernière tour (réelle ou factice) survivait au modèle et la ligne disait « vision=bf16(eager,…) » sur un alias
    texte (rouge test_mm_conversion dans la suite complète, 21/09)."""
    global _CHARGEE
    _CHARGEE = None


def regime_texte() -> str:
    """« vision=bf16(eager,transformers=5.17.0) » quand une tour est chargée (la version RELEVÉE sur le
    module importé, Sage 14 h 10 : transformers est une dépendance épinglée du moteur, jamais devinée),
    « vision=bf16(eager,<nom de la tour factice>) » pour une tour de test, sinon ''."""
    if not _CHARGEE:
        return ""
    nom = str(_CHARGEE)
    if nom.startswith("transformers "):
        nom = "transformers=" + nom[len("transformers "):]
    return f"vision=bf16(eager,{nom})"
