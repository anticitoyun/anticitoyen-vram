#!/usr/bin/env python3
"""La VRAM annoncée par le plan correspond-elle à la VRAM réellement prise ?

Trois défauts du budget mémoire ont été corrigés le 9/09/2026 : le cache KV
comptait des couches qui n'en allouent aucune, l'état récurrent n'était
budgété nulle part, et les modèles à latent compressé recevaient une formule
fausse — dans les deux sens selon le modèle. Tout est vérifié par le calcul et
par 428 épreuves. **Rien ne l'est sur une carte**, et c'est la seule chose qui
compte : un plan juste sur le papier qui réserve mal en pratique laisse le
défaut entier.

LES DEUX ÉCARTS N'ONT PAS LE MÊME NOM, et c'est ce que ce script rend :

    reel > annonce    SOUS-PROVISION : le moteur prend ce que le plan n'a pas
                      reserve. C'est le defaut d'origine, celui qui fait
                      manquer la VRAM sous charge, tard, loin du chargement.

    reel < annonce    RESERVATION INUTILE : le plan met de cote ce que
                      personne ne prend. Ca ne casse rien tout de suite, ca
                      pousse un MLP en RAM hote — et une couche exilee coute
                      70 % du debit.

Le TÉMOIN QUADRATIQUE est le plus important des trois et c'est celui qu'on
saute quand on est pressé : sans lui, deux VRAM correctes sur les cas
particuliers ne disent pas si le cas général a été déréglé.
"""
import argparse
import gc
import json
import os
import sys

import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


A = _RACINE
CIBLES = [
    ("hybride", "Nemotron-Nano-9B-int8"),
    ("MLA", "GLM-4.7-Flash-nvfp4"),
    ("temoin quadratique", "Qwen3-4B-srcgguf-nvfp4"),
]


def _vram_prise() -> int:
    """Ce que le pilote voit, pas ce que l'allocateur de torch a mis de côté.

    `memory_allocated` ignore les blocs que l'allocateur garde en réserve ;
    c'est `mem_get_info` qui dit ce qui manque vraiment aux autres."""
    libre, total = torch.cuda.mem_get_info(0)
    return total - libre


def _exiger_une_seule_carte() -> None:
    """Refuse de mesurer si plus d'une carte est visible.

    Le plan est recalculé selon les cartes visibles : sans épinglage, le
    planificateur peut recruter la seconde — qui porte déjà un service
    permanent — et la VRAM relevée n'est plus celle qu'on croit lire. Une
    mesure fausse qui s'annonce vraie coûte plus qu'une mesure absente.
    """
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(
            f"{n} cartes visibles : relancez avec CUDA_VISIBLE_DEVICES=0. "
            f"Le plan depend des cartes visibles, la mesure serait fausse "
            f"sans le dire.")


def _autres_processus() -> list[str]:
    """Processus de calcul sur la carte visible, LE NÔTRE EXCLU.

    Compter les OCTETS pris ne distingue pas un voisin au travail de notre
    propre contexte CUDA, qui pèse à lui seul un demi-gibioctet dès que torch
    s'initialise. La première version de cette garde refusait de mesurer sur
    « 521 Mio pris » alors que la carte était à 15 Mio et que les 521 étaient
    les siens : elle lisait sa propre présence comme celle d'un autre.

    Ce qu'on veut savoir n'est pas « combien est pris » mais « quelqu'un
    d'autre travaille-t-il ici », et cette question se pose aux processus.
    """
    import subprocess
    moi = str(os.getpid())
    # `-i 0` : SANS lui, nvidia-smi liste les processus de TOUTES les cartes, et
    # la garde refusait de mesurer sur la 5090 libre parce qu'un service
    # permanent occupe la 3080 Ti. Troisieme fois de la journee qu'un controle
    # lit la propriete voisine de celle qui decide — ici « un processus GPU
    # existe » au lieu de « un processus occupe MA carte ».
    #
    # `-i 0` designe la carte PHYSIQUE 0 : nvidia-smi ignore
    # CUDA_VISIBLE_DEVICES, c'est donc bien la 5090 meme sous epinglage.
    try:
        sortie = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:                                       # noqa: BLE001
        return []                                           # pas d'avis : pas d'alarme
    autres = []
    for ligne in sortie.splitlines():
        if not ligne.strip():
            continue
        pid = ligne.split(",")[0].strip()
        if pid and pid != moi:
            autres.append(ligne.strip())
    return autres


def _attendre_carte_libre(patience_s: int = 180) -> None:
    """Attend que les AUTRES processus rendent la carte, jamais un délai fixe.

    Un `sleep` calibré sur une observation passée ne garantit rien : le
    voisin rend sa VRAM quand il la rend. Constaté le 9/09/2026, une manche de
    comparatif refusée pour « carte non libre (1176 Mio) » parce que le délai
    fixe qui la précédait ne suffisait plus ce soir-là.
    """
    import time
    debut = time.monotonic()
    while True:
        autres = _autres_processus()
        if not autres:
            return
        if time.monotonic() - debut > patience_s:
            raise SystemExit(
                f"carte toujours occupee apres {patience_s} s : {autres} — "
                f"quelqu'un travaille dessus, ne mesurez pas par-dessus.")
        time.sleep(2)


def mesurer(chemin: str, max_model_len: int) -> dict:
    from acvram.engine.loader import load_model

    gc.collect()
    torch.cuda.empty_cache()
    _attendre_carte_libre()
    avant = _vram_prise()

    charge = load_model(chemin, max_model_len=max_model_len)
    torch.cuda.synchronize()
    apres = _vram_prise()

    plan = charge.plan
    annonce = (plan.total_weight_bytes
               + sum(plan.kv_budget.values())
               + getattr(plan, "etat_recurrent_bytes", 0)
               + sum(plan.expert_cache_bytes.values()))
    reel = apres - avant

    del charge
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "annonce": annonce, "reel": reel, "ecart": reel - annonce,
        "poids": plan.total_weight_bytes,
        "kv": sum(plan.kv_budget.values()),
        "etat": getattr(plan, "etat_recurrent_bytes", 0),
        "experts": sum(plan.expert_cache_bytes.values()),
        "kv_par_jeton": plan.kv_bytes_per_token,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tolerance", type=float, default=5.0,
                    help="ecart tolere en %% de la VRAM annoncee")
    ap.add_argument("--role", choices=[r for r, _ in CIBLES],
                    help="ne mesurer qu'un role. Enchainer les trois dans un "
                         "seul processus a fait tuer la tache pour pression "
                         "memoire : lire un modele de 17 Gio remplit le cache "
                         "de pages, et trois lectures d'affilee ne le rendent "
                         "pas assez vite. Un modele par lancement, le plus "
                         "petit d'abord.")
    ns = ap.parse_args(argv[1:])

    if not torch.cuda.is_available():
        print("aucune carte : cette verification n'a pas de sens sans elle",
              file=sys.stderr)
        return 2
    _exiger_une_seule_carte()

    echecs = []
    print(f"{'role':<20}{'modele':<26}{'annonce':>9}{'reel':>9}"
          f"{'ecart':>9}{'':>3}verdict")
    for role, nom in CIBLES:
        if ns.role and role != ns.role:
            continue
        chemin = os.path.join(A, nom)
        if not os.path.isdir(chemin):
            print(f"{role:<20}{nom[:24]:<26}   ABSENT DU DISQUE")
            continue
        try:
            r = mesurer(chemin, ns.max_model_len)
        except Exception as exc:                            # noqa: BLE001
            print(f"{role:<20}{nom[:24]:<26}   ECHEC DE CHARGEMENT : "
                  f"{type(exc).__name__}: {str(exc)[:60]}")
            echecs.append((role, nom, "ne charge pas"))
            continue
        g = 2**30
        pc = 100 * r["ecart"] / max(1, r["annonce"])
        if abs(pc) <= ns.tolerance:
            verdict = "conforme"
        elif r["ecart"] > 0:
            verdict = "SOUS-PROVISION"
            echecs.append((role, nom, verdict))
        else:
            verdict = "RESERVATION INUTILE"
            echecs.append((role, nom, verdict))
        print(f"{role:<20}{nom[:24]:<26}{r['annonce']/g:>8.2f}G"
              f"{r['reel']/g:>8.2f}G{pc:>+8.1f}%   {verdict}")
        print(f"{'':<46}poids {r['poids']/g:.2f}G  kv {r['kv']/g:.2f}G  "
              f"etat {r['etat']/g:.2f}G  experts {r['experts']/g:.2f}G  "
              f"kv/jeton {r['kv_par_jeton']}")

    print()
    if echecs:
        print("ECHECS :")
        for role, nom, quoi in echecs:
            print(f"  {role} — {nom} : {quoi}")
        print("\nUne SOUS-PROVISION est le defaut d'origine : la VRAM manquera")
        print("sous charge, tard. Une RESERVATION INUTILE pousse un MLP en RAM")
        print("hote, et une couche exilee coute 70 % du debit. Les deux sont")
        print("des echecs, ils ne se corrigent pas du meme cote.")
        return 1
    print("les trois sont conformes : le plan annonce ce que le moteur prend.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
