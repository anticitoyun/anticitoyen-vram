#!/usr/bin/env python3
"""Racine du parc de modeles convertis, surchargeable par ACVRAM_MODELES.

Le SSD a change de point de montage au passage Ubuntu (12/09) : plutot que
suivre le mouvement en 41 endroits, un seul litteral vit ici, tous les
scripts de outils/ importent MODELES. Un utilisateur pointe son SSD par la
variable d'environnement, sans toucher au code.

Ordre de lecture (20/09, poste=20-09-1030 : GNOME n'herite pas de environment.d,
les chaines lancees depuis un terminal graphique ne voyaient pas ACVRAM_MODELES) :
1. ACVRAM_MODELES (puis ACVRAM_MODELS_DIR, heritage) ;
2. ~/.config/acvram/modeles — meme lecture que acvram/cli.py (premiere ligne non
   commentee qui est un dossier existant ; XDG_CONFIG_HOME respecte) ;
3. le litteral USB d'avant (un poste sans configuration garde son comportement).
"""
import os

_DEFAUT = "/mnt/2TO_2023_980PRO/Modeles/models_acvram"


def _depuis_config() -> str | None:
    conf = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "acvram", "modeles")
    try:
        with open(conf, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne and not ligne.startswith("#") and os.path.isdir(ligne):
                    return ligne
    except OSError:
        return None
    return None


def racine_modeles() -> str:
    """Racine effective : variable, sinon ~/.config/acvram/modeles, sinon le litteral."""
    return (os.environ.get("ACVRAM_MODELES") or os.environ.get("ACVRAM_MODELS_DIR")
            or _depuis_config() or _DEFAUT)


MODELES = racine_modeles()


def alias(nom: str) -> str:
    """Chemin d'un converti nommé sous la racine du moment."""
    return os.path.join(racine_modeles(), nom)


def alias_absent(nom: str) -> str:
    """Raison de skip pytest, vide si le dossier existe (nom stable : grep « alias absent »)."""
    return "" if os.path.isdir(alias(nom)) else f"alias absent sous racine_modeles() : {alias(nom)}"


if __name__ == "__main__":
    print(MODELES)
