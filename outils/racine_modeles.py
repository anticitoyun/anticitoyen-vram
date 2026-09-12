"""Racine du parc de modeles convertis, surchargeable par ACVRAM_MODELES.

Le SSD a change de point de montage au passage Ubuntu (12/09) : plutot que
suivre le mouvement en 41 endroits, un seul litteral vit ici, tous les
scripts de outils/ importent MODELES. Un utilisateur pointe son SSD par la
variable d\'environnement, sans toucher au code.
"""
import os

_DEFAUT = "/run/media/anticitoyenu/2TO_2023_980PRO/Modeles/models_acvram"
MODELES = os.environ.get("ACVRAM_MODELES", _DEFAUT)
