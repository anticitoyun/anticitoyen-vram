#!/usr/bin/env python3
"""La VRAM au PIC, sous charge — ce que le chargement seul ne pouvait pas voir.

Le protocole précédent (`verifier-budget-vram.py`) relevait la VRAM juste après
le chargement, et ne pouvait donc pas vérifier ce qu'il annonçait : les postes
corrigés le 9/09/2026 ne sont **pas alloués au chargement**.

    etat recurrent (KDA, GDN, Mamba2, LFM2)   naît PAR SEQUENCE, dans le forward
    cache latent MLA                          CROIT avec les jetons produits
    blocs KV pagines                          alloues au chargement, eux

Mesurer au chargement une VRAM prise à l'exécution, c'est lire la propriété
voisine de celle qui décide. Ici on charge, on sert N séquences en parallèle,
et on relève le MAXIMUM atteint pendant le décodage.

CE QUE LE RESULTAT VEUT DIRE, fixé avant de le voir :

    pic > annonce    SOUS-PROVISION. Le moteur prend ce que le plan n a pas
                     reserve. C est le defaut d origine, celui qui fait manquer
                     la VRAM tard, sous charge — exactement le regime teste ici.
    pic < annonce    RESERVATION INUTILE. Le plan met de cote ce que personne
                     ne prend : un MLP part en RAM hote, et une couche exilee
                     coute 70 % du debit.

Le TEMOIN QUADRATIQUE reste le plus important : il ne porte aucun des postes
corrigés, donc son écart mesure le biais qui existait AVANT — 32,1 % relevés
au chargement, et poste2 a vu le même facteur indépendamment sur un autre
modèle. Sans lui, on attribue à ses propres correctifs un défaut plus ancien.
"""
import argparse
import gc
import os
import sys
import threading
import time

import torch

A = "/media/anticitoyenlm/2TO_2023_980PRO1/Modeles/models_acvram"
CIBLES = [
    ("hybride", "Nemotron-Nano-9B-int8"),
    ("MLA", "GLM-4.7-Flash-nvfp4"),
    ("temoin quadratique", "Qwen3-4B-srcgguf-nvfp4"),
]


def _vram_prise() -> int:
    libre, total = torch.cuda.mem_get_info(0)
    return total - libre


def _autres_processus() -> list[str]:
    """Processus de calcul sur la carte 0, LE NOTRE EXCLU.

    `-i 0` n'est pas un détail : sans lui, nvidia-smi liste toutes les cartes,
    et un service permanent sur la seconde faisait refuser la première alors
    qu'elle était libre. Et compter les OCTETS plutôt que les processus faisait
    prendre notre propre contexte CUDA — un demi-gibioctet — pour un intrus.
    """
    import subprocess
    moi = str(os.getpid())
    try:
        sortie = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:                                       # noqa: BLE001
        return []
    return [l.strip() for l in sortie.splitlines()
            if l.strip() and l.split(",")[0].strip() != moi]


class Guetteur(threading.Thread):
    """Echantillonne la VRAM pendant que le moteur travaille.

    `torch.cuda.max_memory_allocated` ne suffirait pas : il ignore ce que
    l'allocateur garde en reserve et tout ce qui est pris hors de lui. Seul
    `mem_get_info` dit ce qui manque vraiment aux autres.
    """

    def __init__(self, periode: float = 0.02) -> None:
        super().__init__(daemon=True)
        self.periode = periode
        self.pic = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.pic = max(self.pic, _vram_prise())
            time.sleep(self.periode)

    def arreter(self) -> int:
        self._stop.set()
        self.join(timeout=2.0)
        return max(self.pic, _vram_prise())


def mesurer(chemin: str, max_model_len: int, n_seqs: int,
            n_jetons: int, invite: int) -> dict:
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    gc.collect()
    torch.cuda.empty_cache()
    avant = _vram_prise()

    charge = load_model(chemin, max_model_len=max_model_len)
    moteur = Engine(charge, None, max_batch_size=n_seqs,
                    max_model_len=max_model_len)
    torch.cuda.synchronize()
    apres_chargement = _vram_prise()

    # Invites distinctes : des sequences identiques partageraient leurs blocs
    # par le cache de prefixe, et N sequences ne couteraient que la place d une.
    # Le test mesurerait alors la deduplication, pas la concurrence.
    params = SamplingParams(temperature=0.0, max_tokens=n_jetons)
    for i in range(n_seqs):
        ids = [(i * 7919 + j * 31 + 11) % 30000 + 1 for j in range(invite)]
        moteur.add_request(ids, params, request_id=f"r{i}")

    guetteur = Guetteur()
    guetteur.start()
    pas = 0
    while pas < n_jetons + invite + 8:
        sorties = moteur.step()
        pas += 1
        if not moteur.running and not moteur.waiting:
            break
    torch.cuda.synchronize()
    pic = guetteur.arreter()

    plan = charge.plan
    annonce = (plan.total_weight_bytes + sum(plan.kv_budget.values())
               + getattr(plan, "etat_recurrent_bytes", 0)
               + sum(plan.expert_cache_bytes.values()))
    r = {
        "annonce": annonce,
        "chargement": apres_chargement - avant,
        "pic": pic - avant,
        "pas": pas,
        "poids": plan.total_weight_bytes,
        "kv": sum(plan.kv_budget.values()),
        "etat": getattr(plan, "etat_recurrent_bytes", 0),
    }
    del moteur, charge
    gc.collect()
    torch.cuda.empty_cache()
    return r


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", choices=[r for r, _ in CIBLES], required=True,
                    help="un modele par lancement : enchainer trois modeles "
                         "dans un processus fait tuer la tache pour pression "
                         "memoire (le cache de pages fait tomber MemFree)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seqs", type=int, default=16,
                    help="sequences concurrentes ; c est ce nombre qui "
                         "multiplie l etat recurrent")
    ap.add_argument("--jetons", type=int, default=64)
    ap.add_argument("--invite", type=int, default=256)
    ns = ap.parse_args(argv[1:])

    if not torch.cuda.is_available():
        print("aucune carte", file=sys.stderr)
        return 2
    if torch.cuda.device_count() != 1:
        raise SystemExit("relancez avec CUDA_VISIBLE_DEVICES=0 : le plan depend "
                         "des cartes visibles, la mesure serait fausse.")
    autres = _autres_processus()
    if autres:
        raise SystemExit(f"carte occupee : {autres} — ne mesurez pas par-dessus.")

    nom = dict(CIBLES)[ns.role]
    r = mesurer(os.path.join(A, nom), ns.max_model_len, ns.seqs,
                ns.jetons, ns.invite)
    g = 2**30
    pc = 100 * (r["pic"] - r["annonce"]) / max(1, r["annonce"])
    verdict = ("conforme" if abs(pc) <= 5 else
               "SOUS-PROVISION" if pc > 0 else "RESERVATION INUTILE")
    print(f"{ns.role} — {nom}   {ns.seqs} sequences, {r['pas']} pas")
    print(f"  annonce au plan  {r['annonce']/g:6.2f} G   "
          f"(poids {r['poids']/g:.2f}  kv {r['kv']/g:.2f}  etat {r['etat']/g:.2f})")
    print(f"  apres chargement {r['chargement']/g:6.2f} G")
    print(f"  PIC sous charge  {r['pic']/g:6.2f} G   {pc:+.1f} %   {verdict}")
    print(f"  ce que la charge ajoute : "
          f"{(r['pic'] - r['chargement'])/g:+.2f} G")
    return 0 if verdict == "conforme" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
