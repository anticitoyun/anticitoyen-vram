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
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


A = _RACINE
CIBLES = [
    ("hybride", "Nemotron-Nano-9B-int8"),
    ("MLA", "GLM-4.7-Flash-nvfp4"),
    ("temoin quadratique", "Qwen3-4B-srcgguf-nvfp4"),
    # LE BRAS QUI LEVE UN CONFONDANT, ajoute le 10/09. « Le MLA fragmente trois
    # fois plus » etait une attribution, pas une mesure : GLM-4.7-Flash est
    # MLA **et** MoE (64 experts, 4 par jeton), les deux autres cibles ne sont
    # ni l un ni l autre. Le parc ne contient AUCUN modele MLA sans MoE (0 sur
    # 120) mais trente MoE sans MLA ; celui-ci pese 16,41 Gio contre 15,79 au
    # GLM, donc a taille comparable. S il fragmente comme le GLM, la cause est
    # le routage d experts ; s il fragmente comme le temoin, c est le MLA.
    ("MoE sans MLA", "Jan-v2-VL-max-srcQ4_K_M-nvfp4"),
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
    """Echantillonne la VRAM DEPUIS UN AUTRE PROCESSUS, et ce n est pas un detail.

    La premiere version appelait `torch.cuda.mem_get_info` toutes les 20 ms
    depuis un thread du meme processus. Un appel CUDA concurrent INVALIDE une
    capture de graphe en cours : le moteur echouait par
    `cudaErrorStreamCaptureInvalidated` des que plusieurs sequences
    demarraient ensemble. L instrument cassait la mesure qu il devait prendre.

    `nvidia-smi` s execute hors du contexte CUDA du moteur, donc il n y touche
    pas. Il coute une cinquantaine de millisecondes par releve — assez fin
    pour un pic qui dure le temps d un decodage, et sans effet sur lui.
    """

    def __init__(self, periode: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.periode = periode
        self.pic = 0
        self._fin = threading.Event()

    @staticmethod
    def _lire() -> int:
        import subprocess
        try:
            s = subprocess.run(
                ["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            return int(float(s.splitlines()[0])) * 2**20
        except Exception:                                   # noqa: BLE001
            return 0

    def run(self) -> None:
        while not self._fin.is_set():
            self.pic = max(self.pic, self._lire())
            time.sleep(self.periode)

    def arreter(self) -> int:
        self._fin.set()
        self.join(timeout=3.0)
        return max(self.pic, self._lire())


def _memoire_hote() -> tuple[int, int]:
    """`MemFree` ET `MemAvailable`, en Kio. Jamais l un sans l autre.

    Mesure d poste1 le 10/09 : le superviseur a tue son banc deux fois alors
    que la machine avait 83 Go DISPONIBLES. `MemFree` tombait de 1,8 Gio,
    `Cached` montait d autant, `MemAvailable` ne bougeait pas — la memoire
    etait pretee au cache de pages, pas consommee. Publier `MemFree` seul,
    c est lire une propriete voisine de celle qui decide.
    """
    d = {}
    with open("/proc/meminfo") as fh:
        for ligne in fh:
            k, _, v = ligne.partition(":")
            if k in ("MemFree", "MemAvailable"):
                d[k] = int(v.split()[0])
    return d.get("MemFree", 0), d.get("MemAvailable", 0)


def mesurer(chemin: str, max_model_len: int, n_seqs: int,
            n_jetons: int, invite: int, sans_graphes: bool = False) -> dict:
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    gc.collect()
    torch.cuda.empty_cache()
    avant = Guetteur._lire()
    # LOCALISER le pic, au lieu de l attribuer. `_try_build_stacks` empile les
    # experts par `torch.stack(...)` et ne libere les anciens stockages
    # qu apres : « la pile d une projection double transitoirement sa memoire »
    # (model.py:612). Reste a savoir OU ce doublement tombe — a la construction
    # du moteur, a l echauffement des graphes, ou dans la boucle. Trois relevés
    # repondent ; une explication n aurait fait que deplacer la question.
    torch.cuda.reset_peak_memory_stats()

    charge = load_model(chemin, max_model_len=max_model_len)
    moteur = Engine(charge, None, max_batch_size=n_seqs,
                    max_model_len=max_model_len,
                    enable_cuda_graphs=not sans_graphes)
    # Prechauffer les graphes comme le fait le serveur : sans cela, la capture
    # se declenche au milieu du premier pas et echoue par
    # `cudaErrorStreamCaptureInvalidated` des que plusieurs sequences allouent
    # ensemble. Le sauter ne mesurerait pas le regime de production — et les
    # graphes retiennent de la VRAM, donc c est bien du poste mesure.
    torch.cuda.synchronize()
    vivant_moteur = torch.cuda.max_memory_allocated()
    if moteur.graphs is not None and not sans_graphes:
        moteur.warm_graphs(max_model_len)
    torch.cuda.synchronize()
    vivant_warm = torch.cuda.max_memory_allocated()
    apres_chargement = Guetteur._lire()

    # Invites distinctes : des sequences identiques partageraient leurs blocs
    # par le cache de prefixe, et N sequences ne couteraient que la place d une.
    # Le test mesurerait alors la deduplication, pas la concurrence.
    params = SamplingParams(temperature=0.0, max_tokens=n_jetons)
    for i in range(n_seqs):
        ids = [(i * 7919 + j * 31 + 11) % 30000 + 1 for j in range(invite)]
        moteur.add_request(ids, params, request_id=f"r{i}")

    # LE BRAS QUI SEPARE les deux candidats restants pour les ~1,8 Gio.
    #
    # `pic` est un chiffre du PILOTE : il contient tout. Trois compteurs de
    # l allocateur le decomposent exactement, et ils ne coutent rien :
    #   vivant        = max_memory_allocated : les tenseurs reellement detenus
    #                   (poids + KV + intermediaires en cours)
    #   fragmentation = max_memory_reserved - max_memory_allocated : ce que
    #                   l allocateur garde au pilote sans le preter
    #   hors_torch    = pic - max_memory_reserved : contexte CUDA et tout ce
    #                   que PyTorch ne compte pas
    # Sans cette decomposition on ne peut qu ATTRIBUER l ecart ; avec elle on
    # le lit. C est la difference entre une hypothese et une mesure.
    torch.cuda.reset_peak_memory_stats()
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
    vivant = torch.cuda.max_memory_allocated()
    reserve = torch.cuda.max_memory_reserved()
    # `max_memory_reserved` et `max_memory_allocated` sont DEUX MAXIMA
    # INDEPENDANTS : leur difference combine des valeurs atteintes a des
    # instants differents et ne mesure donc pas une fragmentation. Le champ qui
    # la mesure vraiment est `inactive_split_bytes` — les octets pris dans des
    # blocs decoupes et inutilisables. On publie aussi les courants au MEME
    # instant synchronise, seuls comparables entre eux.
    st = torch.cuda.memory_stats()
    fragm = st.get("inactive_split_bytes.all.peak", 0)
    res_cur = st.get("reserved_bytes.all.current", 0)
    all_cur = st.get("allocated_bytes.all.current", 0)

    from acvram.engine.layers import QuantLinear
    exiles = sum(1 for m_ in charge.model.modules()
                 if isinstance(m_, QuantLinear) and m_.streamed is not None)
    g_ = getattr(moteur, "graphs", None)
    graphes_actifs = bool(getattr(g_, "enabled", False))
    raison_graphes = (getattr(g_, "raison", "") or "")[:80]
    libre_h, dispo_h = _memoire_hote()
    n_graphes = len(getattr(g_, "graphs", {}) or {}) if g_ is not None else 0

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
        "exiles": exiles,
        "graphes": graphes_actifs,
        "raison_graphes": raison_graphes,
        "hote_libre": libre_h,
        "hote_dispo": dispo_h,
        "n_graphes": n_graphes,
        "vivant": vivant,
        "vivant_moteur": vivant_moteur,
        "vivant_warm": vivant_warm,
        "reserve": reserve,
        "fragm_reelle": fragm,
        "res_cur": res_cur,
        "all_cur": all_cur,
        "sans_graphes": sans_graphes,
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
    ap.add_argument("--sans-graphes", action="store_true",
                    help="le bras qui discrimine : l ecart annonce/pic est-il "
                         "la reserve des graphes CUDA, ou l aire de travail ?")
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
                ns.jetons, ns.invite, ns.sans_graphes)
    g = 2**30
    pc = 100 * (r["pic"] - r["annonce"]) / max(1, r["annonce"])
    verdict = ("conforme" if abs(pc) <= 5 else
               "SOUS-PROVISION" if pc > 0 else "RESERVATION INUTILE")
    print(f"{ns.role} — {nom}   {ns.seqs} sequences, {r['pas']} pas")
    print(f"  annonce au plan  {r['annonce']/g:6.2f} G   "
          f"(poids {r['poids']/g:.2f}  kv {r['kv']/g:.2f}  etat {r['etat']/g:.2f})")
    print(f"  apres chargement {r['chargement']/g:6.2f} G")
    # Un pourcentage deplace le coupable vers le plus petit denominateur :
    # +39,0 % et +9,9 % etaient le MEME 1,4 Gio sur deux modeles differant d un
    # facteur 5,3 en poids. La valeur absolue se publie a cote, toujours.
    print(f"  PIC sous charge  {r['pic']/g:6.2f} G   "
          f"{(r['pic']-r['annonce'])/g:+.2f} G   {pc:+.1f} %   {verdict}")
    print(f"  ce que la charge ajoute : "
          f"{(r['pic'] - r['chargement'])/g:+.2f} G")
    # CONTROLES publies A COTE du verdict, jamais a sa place.
    print(f"  MLP exiles {r['exiles']}   graphes "
          f"{'coupes (bras temoin)' if r['sans_graphes'] else
             ('actifs, %d vivants' % r['n_graphes']) if r['graphes']
             else 'INACTIFS'}"
          + (f" ({r['raison_graphes']})" if not r['graphes'] else ""))
    print(f"  ou naît le pic : moteur {r['vivant_moteur']/g:.2f} G   "
          f"echauffement {r['vivant_warm']/g:.2f} G   "
          f"boucle {r['vivant']/g:.2f} G")
    print(f"  decomposition du pic : vivant {r['vivant']/g:.2f} G   "
          f"fragmentation {(r['reserve']-r['vivant'])/g:+.2f} G   "
          f"hors torch {(r['pic']-r['reserve'])/g:+.2f} G")
    print(f"  fragmentation REELLE (inactive_split, pic) {r['fragm_reelle']/g:.2f} G"
          f"   —   courants au meme instant : reserve {r['res_cur']/g:.2f} G "
          f"alloue {r['all_cur']/g:.2f} G   ecart {(r['res_cur']-r['all_cur'])/g:.2f} G")
    print(f"  hote : MemFree {r['hote_libre']/2**20:.1f} G   "
          f"MemAvailable {r['hote_dispo']/2**20:.1f} G")
    if r["exiles"] or (not r["graphes"] and not r["sans_graphes"]):
        print("  MANCHE SANS OBJET : un MLP exile ou des graphes inactifs "
              "changent le regime — ce chiffre ne mesure pas le budget KV.")
        return 2
    return 0 if verdict == "conforme" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
