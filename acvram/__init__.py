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

__version__ = "0.4.96"

# Segments extensibles de l'allocateur CUDA. Posé ici, avant que le moindre
# import ne crée le contexte CUDA : la variable n'est lue qu'une fois, à la
# première allocation, et n'a plus aucun effet ensuite.
#
# Le planificateur remplit la carte à quelques pour cent près. Avec
# l'allocateur par blocs, la mémoire rendue par un tenseur intermédiaire reste
# prisonnière du bloc où elle a été prise : un chargement a échoué sur les
# 20 derniers Mio d'un modèle de 30 milliards alors que 2,55 Gio étaient
# réservés et inutilisés — de la fragmentation, pas un manque de place. Les
# segments extensibles rendent cette réserve fongible.
#
# `ACVRAM_ALLOC_BLOCS=1` revient à l'allocateur historique.
import os as _os

if not _os.environ.get("ACVRAM_ALLOC_BLOCS"):
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__all__ = ["__version__"]
