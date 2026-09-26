#!/usr/bin/env python3
"""La tour visuelle des modeles VL est-elle comptee dans le plan ?

Mesure du 10/09/2026 sur `Jan-v2-VL-max-srcQ4_K_M-nvfp4` : sous charge, les
tenseurs REELLEMENT detenus pesent **21,71 Gio pour 17,93 annonces** — +3,78,
soit 21 % — et le bras a frole l'OOM a 5 Mo pres. Sur `GLM-4.7-Flash`, MoE lui
aussi, l'ecart valait -0,23 : ce n'est donc pas le MoE. Le seul trait qui
distingue le premier est d'etre un modele VISION-langage.

Ce balayage ne charge rien et ne demande pas la carte. Il lit l'entete des
`.safetensors` — qui donne les octets EXACTS de chaque tenseur par ses
`data_offsets`, sans decompresser — et le compare a ce que le manifeste
annonce.

**Le test qui tranche** : `total_weight_bytes` colle-t-il a la somme de TOUS
les tenseurs, ou seulement a celle des tenseurs NON visuels ? Dans le second
cas la tour existe sur la carte sans figurer au plan.
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prefixes de tour visuelle rencontres dans le parc. Volontairement larges :
# manquer un prefixe ferait passer un modele pour sain, ce qui est le sens
# dangereux de l'erreur.
VISUELS = ("visual", "vision_tower", "vision_model", "mm_projector",
           "multi_modal_projector", "image_newline", "vision_encoder",
           "audio_tower", "aligner")


def _entete(chemin: str) -> dict:
    """Tenseurs d'un .safetensors, par leur entete seule."""
    with open(chemin, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def _octets(e: dict) -> int:
    d = e.get("data_offsets")
    return (d[1] - d[0]) if d else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tout", action="store_true",
                    help="lister aussi les modeles sans tour visuelle")
    ns = ap.parse_args(argv[1:])

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parc import get_all_model_paths

    g = 2**30
    print("modele\ttotal_G\tvisuel_G\tpart_%\tannonce_G\tannonce_colle_a")
    n_vl = 0
    for c in sorted(get_all_model_paths()):
        mf = os.path.join(c, "acvram_manifest.json")
        if not os.path.exists(mf):
            continue
        try:
            m = json.load(open(mf))
        except Exception:                                  # noqa: BLE001
            continue
        total = visuel = 0
        for f in sorted(os.listdir(c)):
            if not f.endswith(".safetensors"):
                continue
            try:
                h = _entete(os.path.join(c, f))
            except Exception as exc:                       # noqa: BLE001
                print(f"{os.path.basename(c)}\tENTETE ILLISIBLE\t"
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                h = {}
            for nom, e in h.items():
                if nom == "__metadata__" or not isinstance(e, dict):
                    continue
                o = _octets(e)
                total += o
                if any(p in nom for p in VISUELS):
                    visuel += o
        if not total:
            continue
        if not visuel and not ns.tout:
            continue
        n_vl += 1 if visuel else 0
        annonce = int(m.get("plan", {}).get("total_weight_bytes", 0))
        # A quoi l'annonce colle-t-elle ? Marge de 3 % : l'annonce compte des
        # postes que le disque n'a pas (tete liee fabriquee au chargement).
        def _proche(a, b):
            return b and abs(a - b) / b < 0.03
        colle = ("TOUT" if _proche(annonce, total)
                 else "SANS LE VISUEL" if _proche(annonce, total - visuel)
                 else "ni l un ni l autre")
        print(f"{os.path.basename(c)}\t{total/g:.2f}\t{visuel/g:.2f}\t"
              f"{100*visuel/total:.1f}\t{annonce/g:.2f}\t{colle}")
    print(f"\n{n_vl} modeles portent une tour visuelle", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
