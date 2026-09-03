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

__version__ = "0.4.28"

__all__ = ["__version__"]
