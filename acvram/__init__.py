"""anticitoyen VRAM/RAM — inférence quantifiée par GPU, sur mémoire étagée.

Deux idées tiennent le projet ensemble :

1. Un GPU doit stocker ses poids dans le format que son propre silicium lit le
   mieux. Blackwell a des tensor cores FP4, Ampere n'en a pas : le même modèle
   est donc écrit en NVFP4 pour l'une des cartes et en INT4 pour l'autre,
   plutôt que de rabaisser les deux à un dénominateur commun.

2. La mémoire est une hiérarchie, pas un mur. La VRAM de la carte rapide, puis
   celle de la carte lente, puis la mémoire vive atteinte par le PCIe — avec un
   placement choisi en mesurant ce que coûte chaque étage, et non en espérant
   que le modèle tienne.

La surface publique est la ligne de commande (`acvram`) et le serveur
compatible avec l'API OpenAI.
"""

__version__ = "0.7.4"

# Segments extensibles de l'allocateur CUDA, sur demande seulement. Posé ici
# parce que la variable n'est lue qu'une fois, à la première allocation, avant
# que le moindre import ne crée le contexte CUDA.
#
# Pourquoi c'est utile : le planificateur remplit la carte à quelques pour cent
# près, et avec l'allocateur par blocs la mémoire rendue par un tenseur
# intermédiaire reste prisonnière du bloc où elle a été prise. Un chargement a
# échoué sur les 20 derniers Mio d'un modèle de 30 milliards alors que 2,55 Gio
# étaient réservés et inutilisés — de la fragmentation, pas un manque de place.
#
# Pourquoi ce n'est pas le défaut : le réglage est expérimental chez PyTorch, et
# un allocateur qui étend ses segments est en tension avec la capture de graphes
# CUDA, qui exige des adresses figées — nous capturons des graphes par défaut
# (`graphs.py`). vLLM et TensorRT-LLM rapportent des échecs d'initialisation
# dans cette combinaison, le contournement documenté chez le second étant de
# désactiver les graphes. Tant que la capture n'a pas été éprouvée sous ce
# réglage sur nos deux cartes, le défaut reste l'allocateur par blocs : la
# fragmentation coûte un chargement, une capture perdue coûte le débit de
# toutes les requêtes.
#
# `ACVRAM_ALLOC_EXTENSIBLE=1` l'active.
import os as _os


def _garde_arbre() -> None:
    """Refuse l'import quand le cwd est dans un AUTRE arbre acvram que celui importé.

    Quatre fois (20/09, 22/09, 23/09, 24/09 — MECANISMES.md) un poste a mesuré ou testé depuis
    son worktree le code de l'arbre principal, installé en éditable : `python script.py` met le
    dossier du script en tête de sys.path, pas la racine du worktree. La règle écrite reposait
    sur la vigilance ; la garde la porte. Contournement nommé : ACVRAM_ARBRE_LIBRE=1."""
    if _os.environ.get("ACVRAM_ARBRE_LIBRE") == "1":
        return
    ici = _os.path.realpath(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    try:
        d = _os.path.realpath(_os.getcwd())
    except OSError:
        return
    while True:
        if (_os.path.isfile(_os.path.join(d, "acvram", "__init__.py"))
                and _os.path.exists(_os.path.join(d, ".git"))):
            if d != ici:
                raise ImportError(
                    f"acvram importé depuis {ici} alors que le cwd est dans l'arbre {d} : "
                    f"PYTHONPATH={d} (ou python -m depuis {d}), ou ACVRAM_ARBRE_LIBRE=1 si voulu")
            return
        parent = _os.path.dirname(d)
        if parent == d:
            return
        d = parent


_garde_arbre()
# 0.6.31 : réglages hôte génériques (THP, OMP 8, affinité optionnelle) posés AVANT torch, par le paquet
# — donc par tout lanceur, serveur ou instrument (acvram/hote.py) ; rejoués à la construction d'Engine.
from .hote import regler_hote as _regler_hote  # noqa: E402
_regler_hote()

if _os.environ.get("ACVRAM_ALLOC_EXTENSIBLE"):
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__all__ = ["__version__"]


def regime_noyaux() -> dict:
    """Régime effectif des noyaux (variables de chemin, extension, masques) —
    voir `acvram.regime`. Importé paresseusement : ce module reste léger."""
    from .regime import regime_noyaux as f
    return f()


def regime_ligne() -> str:
    from .regime import regime_ligne as f
    return f()
