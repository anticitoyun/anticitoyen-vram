"""Graphes CUDA pour le pas de décodage.

Le profil de Qwen3-14B montrait ~68 ms de Python par jeton pour ~30 ms de
calcul GPU : plus de la moitié du temps partait en lancements de noyaux et en
navette d'objets Python. Un graphe CUDA capture une fois la séquence complète
des noyaux d'un pas de décodage, puis la rejoue pour le prix d'un seul appel.

Ce que la capture exige — et comment on l'obtient :

* **Des formes fixes.** Un graphe est capturé par *godet* ``(lot, blocs KV)`` :
  le lot ET le nombre de blocs sont arrondis à la puissance de deux supérieure
  (lot : `bucket_batch`, depuis le 11/09 ; ``ACVRAM_GODETS_B=0`` rend le lot
  exact, témoin de mesure). La table de blocs est complétée avec le bloc 0 —
  lu pour rien, masqué par ``seq_lens`` ; les lignes de rembourrage du lot
  portent x = 0, slot -1, seq_len 0 (`_fill`) et, sur un hybride, un créneau
  d'état lié à une sentinelle (`_bind_hybrid`).
* **Des adresses stables.** Les entrées vivent dans des tampons statiques dont
  seul le contenu change avant chaque rejeu ; le cache RoPE est étendu à la
  longueur maximale *avant* la capture pour ne jamais être réalloué.
* **Aucun scalaire tiré des données.** C'est le rôle des chemins
  ``decode_fixed`` : longueurs et positions restent des tenseurs.

L'écriture du cache KV pendant l'échauffement et la capture est volontairement
idempotente — mêmes emplacements, mêmes valeurs — si bien qu'exécuter le pas
deux fois puis le rejouer produit exactement l'état qu'un pas ordinaire aurait
produit. Un test l'affirme jeton par jeton.

Ce qui n'est **pas** capturé, à dessein : le prefill (formes libres), la
spéculation (plusieurs positions par séquence), les couches réparties sur
plusieurs appareils, et les poids streamés dont le préchargement change
l'ADRESSE d'un rejeu à l'autre (chemin par expert / `ExpertPool`, tampons
rotatifs). EXCEPTION (bead pds, 14/09) : un poids streamé lu par TABLE
(`table_qw[e]`, jamais gravée dans le graphe — adresse du tampon stable,
seul son contenu change) reste capturable ; `_eligible()` le distingue.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch

from ..memory.kvcache import BLOCK_SIZE, bucket_blocks


def bucket_batch(n: int) -> int:
    """Arrondit un lot au godet supérieur (puissances de deux, minimum 1)."""
    b = 1
    while b < n:
        b <<= 1
    return b


# ACVRAM_GODETS_B = 1 (défaut) | 0 : la dimension `b` de la clé de graphe est
# arrondie au godet (`bucket_batch`, puissances de deux, plafonnée à
# ACVRAM_HYBRID_SLOTS sur un hybride — en place depuis le 11/09, 70c10a3 et
# 1213554) ou prise EXACTE (un graphe par taille de lot réelle). Le « 0 » est
# le témoin de mesure du chantier C4 (revue/chantier-c4-19-09) : ce que les
# godets rapportent quand un lot se vide séquence par séquence (12 → 1 : douze
# clés exactes contre cinq godets, sous le plafond MAX_GRAPHS=16) ne se lit
# que contre un bras sans godets. Lu une fois, à l'import ; `regime.masquer`
# réécrit `_GODETS_B` sur le module déjà importé.
_GODETS_B = os.environ.get("ACVRAM_GODETS_B", "1") != "0"


def godet_lot(b_reel: int) -> int:
    """La dimension `b` de la clé d'un lot dense : le godet, ou le lot exact
    sous ``ACVRAM_GODETS_B=0``."""
    return bucket_batch(b_reel) if _GODETS_B else b_reel


def godet_hybride(b_reel: int, max_slots: int) -> Optional[int]:
    """Le godet à utiliser pour un lot hybride, ou ``None`` s'il refuse.

    Sous ``ACVRAM_GODETS_B=0`` : le lot exact tant qu'il tient dans les
    créneaux (``b_reel <= max_slots``), refus au-delà, comme au défaut.

    Bug du 13/09 (bead anticitoyen-vram-x0s) : comparer ``bucket_batch(b_reel)``
    (une puissance de deux) à ``max_slots`` faisait tomber en eager,
    silencieusement et en permanence, tout lot de
    ``(puissance_de_deux_inférieure, max_slots]`` dès que ``max_slots`` n'est
    pas lui-même une puissance de deux — à ``max_slots=12``,
    ``bucket_batch(9..12) = 16 > 12``, et aucun lot de 9 à 12 séquences ne
    passait jamais par le graphe. ``max_slots`` borne les tampons RÉELS par
    créneau (``self.statics``) : un lot qui y tient doit toujours pouvoir
    être rejoué, quel que soit son godet naturel — ``max_slots`` devient donc
    lui-même un godet valide au-delà de la puissance de deux qui le précède.
    """
    if b_reel > max_slots:
        return None
    if not _GODETS_B:
        return b_reel
    return min(bucket_batch(b_reel), max_slots)
from .layers import QuantLinear
from .model import DecoderLayerGDN, ForwardBatch, MoEBlock

from .mla import MLA_BUCKET, godet_mla   # un graphe par palier de cache latent

__all__ = ["GraphRunner", "bucket_batch", "godet_hybride", "godet_lot"]

# Plafond du NOMBRE TOTAL de graphes captures. Chaque graphe retient sa memoire
# d'activations, d'ou un plafond ; mais il faut lire ce qu'il fait vraiment.
#
# LE COMMENTAIRE PRECEDENT DISAIT « les godets les moins recents ne sont plus
# captures », ce qui decrit une eviction LRU. IL N'Y EN A AUCUNE. Une fois les
# seize places prises, toute forme nouvelle est refusee DEFINITIVEMENT et
# repasse en eager, quelle que soit la frequence a laquelle elle revient. Les
# seize premieres formes rencontrees gardent leur place pour la vie du
# serveur, meme si elles ne reviennent jamais.
#
# La cle etant (b, ql, nblk, lb), le nombre de formes croit avec la variete
# des tailles de lot ET des longueurs de contexte : un lot de douze qui se
# vide sequence par sequence parcourt a lui seul douze valeurs de b. Seize
# places se remplissent donc en quelques tours, et les refus qui suivent
# dependent du texte genere — c'est-a-dire qu'ils varient d'un essai a
# l'autre. Piste mesuree le 10/09 pour notre dispersion de 26 a 37 % contre
# 1,5 a 1,8 % chez llama.cpp.
#
# Configurable pour pouvoir mesurer ce que coute ce plafond, sans le deplacer
# par defaut : le defaut reste 16, et une campagne qui le bouge doit le dire.
MAX_GRAPHS = int(os.environ.get("ACVRAM_MAX_GRAPHS", "16"))


def _empreinte_adresses(runner, entry: dict) -> dict:
    """Adresse et taille de chaque tenseur qu'un graphe peut avoir figées :
    tampons du godet, puis tout tenseur atteignable depuis le modèle par ses
    attributs, sous-modules, listes et dictionnaires (états récurrents,
    historiques, caches KV, caches RoPE, piles d'experts, poids). Sert au
    débogage (``ACVRAM_TRACE_PTRS``) : un tenseur dont l'adresse change entre
    la capture et un rejeu est un accès fantôme assuré."""
    out = {}
    for k, v in entry.items():
        if torch.is_tensor(v):
            out[f"entry.{k}"] = (v.data_ptr(), v.numel() * v.element_size())
    vu = set()

    def visiter(obj, chemin, prof):
        if obj is None or prof > 14 or id(obj) in vu:
            return
        if torch.is_tensor(obj):
            out[chemin] = (obj.data_ptr(), obj.numel() * obj.element_size()); return
        if isinstance(obj, (str, bytes, int, float, bool, type)):
            return
        vu.add(id(obj))
        if isinstance(obj, dict):
            for k, v in obj.items():
                visiter(v, f"{chemin}[{k}]", prof + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visiter(v, f"{chemin}[{i}]", prof + 1)
        elif hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if k in ("_backward_hooks", "_forward_hooks", "_forward_pre_hooks",
                         "_state_dict_hooks", "_load_state_dict_pre_hooks", "training"):
                    continue
                visiter(v, f"{chemin}.{k}", prof + 1)
    visiter(runner.model, "model", 0)
    return out



_AVERTI_SAMPLER_GRAPHE = False


def sampler_graphe_actif() -> bool:
    """Défaut : glouton dans le graphe. Opt-out `ACVRAM_SAMPLER_LENT=1`
    (témoin, ancien chemin hôte). L ancien opt-in `ACVRAM_SAMPLER_GRAPHE`
    (21-22/09, chaînes de Manon) reste lu un temps : `=1` ne fait rien de
    plus que le défaut, `=0` vaut le témoin — avec un avertissement une fois,
    pour qu aucune chaîne ne porte un nom mort sans le savoir."""
    global _AVERTI_SAMPLER_GRAPHE
    ancien = os.environ.get("ACVRAM_SAMPLER_GRAPHE")
    if ancien is not None and not _AVERTI_SAMPLER_GRAPHE:
        _AVERTI_SAMPLER_GRAPHE = True
        print(f"[acvram] ACVRAM_SAMPLER_GRAPHE={ancien} : variable remplacée — le graphe est le défaut, "
              f"le témoin s obtient par ACVRAM_SAMPLER_LENT=1", flush=True)
    if os.environ.get("ACVRAM_SAMPLER_LENT", "0") == "1":
        return False
    return ancien != "0"


_AVERTI_RAPATRIEMENT = False


def rapatriement_epingle_actif() -> bool:
    """Défaut : rapatriement épinglé (levier 2). Opt-out `ACVRAM_RAPATRIEMENT_FLUX=1`
    (témoin : `.tolist()` sur le flux). L ancien opt-in `ACVRAM_RAPATRIEMENT_EPINGLE`
    (22/09, chaînes de Manon) reste lu un temps : `=1` = le défaut, `=0` = le
    témoin, avec un avertissement une fois."""
    global _AVERTI_RAPATRIEMENT
    ancien = os.environ.get("ACVRAM_RAPATRIEMENT_EPINGLE")
    if ancien is not None and not _AVERTI_RAPATRIEMENT:
        _AVERTI_RAPATRIEMENT = True
        print(f"[acvram] ACVRAM_RAPATRIEMENT_EPINGLE={ancien} : variable remplacée — l épinglé est le défaut, "
              f"le témoin s obtient par ACVRAM_RAPATRIEMENT_FLUX=1", flush=True)
    if os.environ.get("ACVRAM_RAPATRIEMENT_FLUX", "0") == "1":
        return False
    return ancien != "0"


def echantillon_glouton_dans(sortie: torch.Tensor, logits: torch.Tensor) -> None:
    """Le glouton de `sampler._sample_lent` (lignes `logits.to(float32)`,
    `argmax`, `gather − logsumexp`), MÊMES noyaux torch dans le MÊME ordre :
    ids et logprobs au bit avec le chemin hôte. Écrit dans les `n` premières
    colonnes d un tampon statique [2, b_godet·ql] : ligne 0 = ids, ligne 1 =
    bits fp32 des logprobs élargis en int64 (un seul tampon, un seul clone,
    un seul rapatriement ; `depaqueter_logprobs` les relit). Les colonnes au
    delà de `n` (fantômes du godet) gardent leur dernière valeur et ne sont
    jamais lues."""
    l32 = logits.to(torch.float32)
    ids = l32.argmax(dim=-1)
    lp = l32.gather(1, ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(l32, dim=-1)
    n = ids.shape[0]
    sortie[0, :n].copy_(ids)
    sortie[1, :n].copy_(lp.view(torch.int32).to(torch.int64))


def depaqueter_logprobs(bits: list[int]) -> list[float]:
    """Inverse de la ligne 1 de `echantillon_glouton_dans`, côté hôte, après
    `.tolist()` : int64 → int32 → fp32, sans tenseur."""
    import struct
    return [struct.unpack("<f", struct.pack("<i", int(b)))[0] for b in bits]


def _avec_tokens(batch: "ForwardBatch", tokens: torch.Tensor) -> "ForwardBatch":
    """Copie superficielle du lot avec d'autres jetons (épinglés ou device)."""
    import copy
    b2 = copy.copy(batch)
    b2.tokens = tokens
    return b2


class GraphRunner:
    max_ql = 1          # > 1 : lots de vérification spéculative sur hybrides
    # ACVRAM_HYBRID_SLOTS : plafond du nombre de séquences hybrides (GDN/KDA/
    # MLA) rejouées dans un même graphe — dimensionne aussi `self.statics`,
    # les tampons à états fixes, un par créneau. N'IMPORTE QUELLE valeur
    # entière >= 1 est valable depuis le correctif du 13/09 (bead
    # anticitoyen-vram-x0s, cf. `godet_hybride`) : elle n'a plus besoin
    # d'être une puissance de deux. Avant ce correctif, une valeur qui n'en
    # était pas une (12 par ex.) faisait tomber en eager, silencieusement et
    # en permanence, tout lot concurrent de (puissance_de_deux_inférieure,
    # ACVRAM_HYBRID_SLOTS] — ici, 9 à 12 séquences n'empruntaient jamais le
    # graphe. Lu une seule fois, à l'IMPORT du module : le changer en cours
    # de session n'a aucun effet sur un GraphRunner déjà construit.
    max_slots = int(os.environ.get("ACVRAM_HYBRID_SLOTS", "4"))   # séquences par graphe

    """Capture paresseuse et rejeu des pas de décodage purs."""

    @staticmethod
    def plafond_hybride(max_batch_size: Optional[int], env: Optional[str] = None) -> int:
        """Plafond effectif des créneaux hybrides (GDN/KDA/Mamba2/MLA) d'un
        graphe : ACVRAM_HYBRID_SLOTS s'il est posé, sinon le --max-batch du
        moteur (au moins 4). Le défaut fixe à 4 faisait tomber GLM (MLA compte
        comme hybride) en eager dès b=5 sous `acvram serve` — b=12 à 155 t/s
        quand `certifie`, qui pose HYBRID_SLOTS=12 avant l'import, rendait 568
        (Manon, G1 19/09) : un régime par défaut qui masquait le régime
        mesuré. Le plafond dimensionne `statics` (un tampon d'états par
        créneau) : il suit la taille de lot servie, pas une constante."""
        if env is None:
            env = os.environ.get("ACVRAM_HYBRID_SLOTS", "")
        if env:
            return max(1, int(env))
        return max(4, int(max_batch_size or 0))

    def __init__(self, model, max_model_len: int, max_batch_size: Optional[int] = None) -> None:
        self.model = model
        self.max_model_len = max_model_len
        self.max_slots = self.plafond_hybride(max_batch_size)
        self.device: Optional[torch.device] = None
        self.graphs: dict[tuple[int, int], dict] = {}
        # pipeline (runner, ACVRAM_PIPELINE=1) : lot préparé en attente de
        # rejeu, et événement enregistré après chaque rejeu
        self._prepare = None
        self.evenement_jetons = torch.cuda.Event()
        self._pool = None
        # Levier 1 (oceane-levier-1-conception-21-09) : le glouton de
        # `_sample_lent` (argmax, gather, logsumexp sur les logits fp32) est
        # CAPTURÉ dans le graphe et écrit dans `entry["sortie"]` [2, b_godet·ql]
        # int64 (ids ; bits fp32 des logprobs) ; le runner le prend par
        # `prendre_echantillon` (un clone, une seule fois par rejeu). DÉFAUT
        # depuis le verdict Laurine d145bf0d (22/09 : ids/logprobs au bit
        # b=1/b=12, frontière −54 µs, ABBA B/A 1,0124) ; `ACVRAM_SAMPLER_LENT=1`
        # = témoin (ancien chemin hôte), voir `sampler_graphe_actif`.
        self.sampler_graphe = sampler_graphe_actif()
        self._echantillon = None
        self.paged_ok = False
        self.hybrid_layers: list = []
        self.replays = 0
        self.captures = 0
        self._last_key: Optional[tuple[int, int]] = None
        self._raisons_eager_vues: set = set()
        self.replis_eager = 0                  # pas retombés en eager depuis le démarrage
        # Instrument (gemma-4-26B-A4B, capture du godet 1 en OOM alors que les
        # godets 2/8/16 se capturent : chantier-gemma-capture-godet1-20-09) :
        # photographie de la VRAM juste AVANT la première capture du moteur, et
        # juste après un échec — libre (pilote), réservé et alloué (allocateur).
        # Un OOM de capture avec `réservé − alloué` grand n'est pas un manque
        # de VRAM : c'est du cache que l'allocateur ne rend pas pendant une
        # capture (le bassin privé du graphe exige des segments neufs).
        self.memoire_avant_capture: Optional[dict] = None
        self.memoire_apres_echec: Optional[dict] = None
        # Posée AVANT _eligible, qui la remplit : l'initialiser après
        # l'effacerait à chaque fois, et le message aurait annoncé
        # « raison non nommée » pour tous les cas nommés.
        self.raison = ""
        self.enabled = self._eligible()
        if not self.enabled and not os.environ.get("ACVRAM_GRAPHES_MUETS"):
            print(f"[acvram] graphes CUDA désactivés : "
                  f"{self.raison or 'raison non nommée'}", flush=True)

    # -- éligibilité -----------------------------------------------------
    def _eligible(self) -> bool:
        """Vrai si les graphes sont capturables, et ``self.raison`` dit pourquoi
        pas sinon.

        Un refus n'annonçait rien. Sur un hybride cela coûte 11,9 % de débit
        (mesuré le 9 septembre 2026 : 44,28 contre 49,53 pas/s) et le moteur
        servait sans qu'aucune ligne ne le signale — le banc mesurait un moteur
        diminué en croyant mesurer le moteur. Un exil d'un seul MLP de 0,51 Gio
        suffit à faire basculer les quarante couches. La raison est donc
        conservée et affichée : une désactivation silencieuse se lit comme une
        absence de problème.
        """
        if os.environ.get("ACVRAM_DISABLE_CUDA_GRAPHS"):
            self.raison = "demandé par ACVRAM_DISABLE_CUDA_GRAPHS"
            return False
        if not torch.cuda.is_available():
            self.raison = "pas de GPU"
            return False
        m = self.model
        # Hybrides à états (GDN/KDA/MLA) : capturés par couche via des
        # tampons fixes, une séquence par graphe (b = 1, q_len = 1).
        self.hybrid_layers = [l for l in m.layers
                              if isinstance(l, DecoderLayerGDN)]
        if self.hybrid_layers and any(
                not hasattr(l.linear_attn, "decode_static")
                for l in self.hybrid_layers):
            self.raison = "couche hybride sans chemin à formes fixes"
            return False
        devs = {l.device for l in m.layers} | {l.mlp_device for l in m.layers}
        devs.add(m.norm.weight.device)
        head = getattr(m.lm_head.qweight, "qweight", None)
        devs.add(head.device if head is not None else m.norm.weight.device)
        if len(devs) != 1 or next(iter(devs)).type != "cuda":
            self.raison = ("modèle réparti sur "
                           + ", ".join(sorted(str(d) for d in devs)))
            return False                     # pipeline multi-appareils : eager
        # Chemin table (bead pds, point 4 suite — Jérôme 14/09) : un
        # `QuantLinear` streamed dont le noyau lit son adresse par TABLE
        # (`table_qw[e]`/`table_bscale[e]`, relue à CHAQUE lancement, jamais
        # gravée dans les arguments capturés) tolère un REPIN entre deux
        # rejeux — seul le CONTENU du tampon change, pas l'adresse du
        # tampon lui-même (stable pour la vie du modèle, `construire_table`
        # au chargement, jamais réalloué ensuite). Preuve :
        # test_repin_sous_graphe_cuda (3 conditions : visibilité rejeu,
        # formes stables, un désync doit casser). Le chemin par expert /
        # `ExpertPool` reste exclu : lui fait tourner un pointeur à travers
        # des tampons ROTATIFS — l'adresse que le graphe a capturée
        # resterait celle du créneau, quel que soit l'expert qui l'occupe
        # réellement au rejeu suivant.
        # Défaut = garde active : mesuré le 14/09 (revue/graphes-table-
        # regression-b12-14-09.md), b=12 RÉGRESSE de 37 % (30,3→19,0 j/s)
        # une fois la garde levée pour le chemin table, malgré les trois
        # conditions de sécurité prouvées — cause non expliquée (pas une
        # recapture : 5 captures, 203 rejeux). Tant qu'elle ne l'est pas,
        # le chemin table reste hors des graphes par défaut ;
        # ACVRAM_GRAPHES_TABLE=1 pour l'activer en connaissance de cause
        # (mesure, débogage).
        table_graphes_ok = bool(os.environ.get("ACVRAM_GRAPHES_TABLE"))
        surs_table: set = set()
        for mod in m.modules():
            if isinstance(mod, MoEBlock):
                # Le chemin groupé est à formes fixes : le routage ne varie
                # que par le contenu du tenseur d'indices, pas par les formes.
                # Les piles doivent exister avant la capture — les construire
                # pendant remplacerait des adresses que le graphe a retenues.
                if mod._stack_state == "?":
                    dev = next(iter({l.device for l in m.layers}))
                    if dev.type == "cuda":
                        mod._stack_state = ("oui" if mod._try_build_stacks()
                                            else "non")
                if mod._stack_state != "oui":
                    # « pile d'experts hétérogène » recouvrait trois causes
                    # distinctes sans les distinguer (14/09, Qwen3-Coder-30B-A3B
                    # reconverti a snr_floor=0 : le message pointait a tort
                    # vers une hétérogénéité de format alors que le manifeste
                    # était uniforme — la vraie cause était une échelle AWQ
                    # posée sur certains experts seulement). `_raison_repli`
                    # nomme la cause reelle, deja imprimee par
                    # `_try_build_stacks` au moment du repli.
                    self.raison = (getattr(mod, "_raison_repli", "")
                                  or "pile d'experts non homogène (cause non identifiée)")
                    return False             # pile heterogene : eager
                if table_graphes_ok and all(
                        p[0] in ("nvfp4", "nvfp4_table")
                        for p in mod._stacks.values()):
                    for exp in mod.experts:
                        for nom in ("gate_proj", "up_proj", "down_proj"):
                            lin = getattr(exp, nom, None)
                            if lin is not None:
                                surs_table.add(id(lin))
            if isinstance(mod, QuantLinear) and mod.streamed is not None:
                if id(mod) in surs_table:
                    continue                 # table : adresse relue en direct
                # Le cas le plus couteux et le moins visible : il suffit d'un
                # poids exile en RAM hote pour que TOUT le modele passe en
                # eager. On nomme lequel.
                self.raison = (f"poids en flux depuis la RAM hôte "
                               f"({self._nom_du_module(mod)}) — un seul suffit "
                               f"à désactiver les graphes de tout le modèle")
                return False                 # les adresses changent en vol
        pleines = [i for i, l in enumerate(m.layers)
                   if getattr(l, "self_attn", None) is not None]
        if any(i not in m.caches for i in pleines):
            self.raison = "couche pleine sans cache KV"
            return False
        if any(m.caches[i].k.device != next(iter(devs)) for i in pleines):
            return False
        self.device = next(iter(devs))
        from .. import kernels
        c0 = m.caches.get(pleines[0]) if pleines else None
        self.paged_ok = (c0 is not None and c0.k_scale is not None
                         and c0.cfg.dtype == "int8"
                         and all(m.caches[i].cfg.head_dim in (32, 64, 128, 256, 512)
                                 for i in pleines)
                         and kernels.get_extension() is not None)
        return True

    def _nom_du_module(self, cible) -> str:
        for nom, mod in self.model.named_modules():
            if mod is cible:
                return nom
        return "module inconnu"

    # -- mémoire autour de la première capture ----------------------------
    def _photo_memoire(self) -> Optional[dict]:
        """Octets : ``libre`` (pilote, `mem_get_info`), ``reserve`` et
        ``alloue`` (allocateur PyTorch) sur l'appareil des graphes ; ``None``
        sans CUDA (à sec) — la photo n'est jamais un chiffre inventé."""
        if self.device is None or not torch.cuda.is_available():
            return None
        try:
            libre, total = torch.cuda.mem_get_info(self.device)
            return {"libre": int(libre), "total": int(total),
                    "reserve": int(torch.cuda.memory_reserved(self.device)),
                    "alloue": int(torch.cuda.memory_allocated(self.device))}
        except Exception:                                   # noqa: BLE001
            return None

    @staticmethod
    def _ligne_memoire(photo: dict) -> str:
        g = 2 ** 30
        return (f"libre {photo['libre'] / g:.2f} Gio, réservé {photo['reserve'] / g:.2f}, "
                f"alloué {photo['alloue'] / g:.2f} "
                f"(cache non rendu {(photo['reserve'] - photo['alloue']) / g:.2f})")

    def _avant_premiere_capture(self) -> None:
        """Une ligne au journal avant la PREMIÈRE capture du moteur, et la photo
        sous `regime()['graphes_memoire_avant_capture']`. Une seule fois."""
        if self.captures or self.memoire_avant_capture is not None:
            return
        photo = self._photo_memoire()
        if photo is None:
            return
        self.memoire_avant_capture = photo
        print(f"[graphe] mémoire avant capture : {self._ligne_memoire(photo)}", flush=True)

    def _apres_echec_capture(self) -> None:
        """Même photo juste après un échec de capture, AVANT `empty_cache` :
        c'est l'état qui a fait échouer, pas celui d'après le ménage."""
        photo = self._photo_memoire()
        if photo is None:
            return
        self.memoire_apres_echec = photo
        print(f"[graphe] mémoire après l'échec : {self._ligne_memoire(photo)}", flush=True)

    # -- exécution -------------------------------------------------------
    def _eager(self, raison: str) -> None:
        """Un lot qui retombe en eager le dit — une fois par raison distincte.

        Bug du 13/09 (bead anticitoyen-vram-x0s) : un repli silencieux se lit
        comme une absence de problème, exactement comme `_eligible()` le dit
        déjà pour la désactivation globale. Une seule fois par raison : un
        serveur qui replie à chaque pas sur le même motif ne doit pas noyer
        sa propre sortie.
        """
        # compte TOUT repli en service, pas seulement la première raison :
        # trois replis silencieux en une soirée (plafond hybride, capture
        # aveugle, MLA spéculatif — 19/09) ; `/metrics.repli_eager`,
        # `regime_ligne()` porte `repli_eager=N`, certifie/capture rendent
        # faux si N > 0 (Sage)
        self.replis_eager = getattr(self, "replis_eager", 0) + 1
        if raison in self._raisons_eager_vues:
            return
        self._raisons_eager_vues.add(raison)
        print(f"[graphe] repli eager — {raison}", flush=True)

    def run(self, batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Logits du lot, ou None si ce lot n'est pas rejouable en graphe.

        Chemin historique (ACVRAM_PIPELINE=0) : préparer, rejouer, copier.
        Le chemin recouvert (runner, ACVRAM_PIPELINE=1) appelle ``preparer``
        et ``rejouer_suivant`` séparément pour enfiler le pas n+1 pendant
        que le rejeu n tourne encore."""
        if not self.preparer(batch):
            return None
        out = self.rejouer_suivant()
        return out if out is None else out.clone()

    def preparer(self, batch: ForwardBatch) -> bool:
        """Lie les créneaux, capture au besoin, remplit les tampons d'entrée
        du graphe pour ce lot. Rend False si le lot n'est pas rejouable en
        graphe (le runner replie en eager). Aucune attente de l'hôte : les
        copies vers les tampons d'entrée sont enfilées sur le flux courant
        APRÈS le rejeu précédent — l'ordre du flux garantit que ce rejeu a
        fini de les lire. ``batch.tokens`` peut être un tenseur device (les
        jetons échantillonnés du pas précédent, jamais rapatriés)."""
        self._prepare = None
        if not self.enabled or batch.is_prefill:
            return False
        ql = batch.query_lens[0]
        if any(q_ != ql for q_ in batch.query_lens):
            return False                      # longueurs mixtes : eager
        if ql != 1 and not self.paged_ok:
            return False                      # la verification exige le noyau
        b_reel = batch.batch_size
        b = godet_lot(b_reel)                 # ACVRAM_GODETS_B : godet ou lot exact
        nblk = bucket_blocks(max(t.shape[0] for t in batch.block_tables))
        if nblk * BLOCK_SIZE > self.max_model_len + BLOCK_SIZE:
            nblk = bucket_blocks((self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if any(t.shape[0] > nblk for t in batch.block_tables):
            # une table plus longue que le godet ne se copie pas (a (256) ≠ b (257)) : un pas eager nommé, pas un 500
            self._eager(f"table de {max(t.shape[0] for t in batch.block_tables)} blocs au-dela du godet {nblk}")
            return False
        lb = 0
        trace = bool(os.environ.get("ACVRAM_TRACE_STEPS"))
        t0 = time.perf_counter()
        if self.hybrid_layers:
            godet = godet_hybride(b_reel, self.max_slots)
            if godet is None:
                self._eager(f"lot hybride {b_reel} au-dela du plafond "
                            f"ACVRAM_HYBRID_SLOTS={self.max_slots}")
                return False
            b = godet
            if batch.gdn_store is None:
                self._eager("gdn_store absent (etat recurrent non initialise)")
                return False
            if ql > self.max_ql:
                self._eager(f"ql={ql} au-dela de max_ql={self.max_ql}")
                return False
            if b_reel > 1 and ql != 1:
                self._eager("lot multi-sequences en verification speculative "
                             "(ql != 1) : non supporte sous graphe")
                return False
            lb = godet_mla(max(batch.seq_lens))
            self._bind_hybrid(batch, lb, b_godet=b)
        key = (b, ql, nblk, lb)
        self._last_key = key
        # UNE BORNE PAR FRONTIERE, SINON LA DECOMPOSITION RESTE FAUSSE. Un seul
        # `synchronize` apres le rejeu ferait absorber a `replay` le travail de
        # `bind` et de `fill` : les trois chronos changeraient de valeur sans
        # cesser de mentir. Le garde est le meme, eteint par defaut.
        if trace and os.environ.get("ACVRAM_CHRONO_SYNC"):
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()

        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= MAX_GRAPHS:
                if trace:
                    print(f"[graphe] limite {MAX_GRAPHS} atteinte, clé {key} : eager",
                          flush=True)
                return False
            self._avant_premiere_capture()
            try:
                entry = self._capture(b, ql, nblk, batch)
            except Exception as e:                        # noqa: BLE001
                # UNE CAPTURE QUI ECHOUE NE DOIT PAS TUER LE SERVEUR, quelle
                # qu'en soit la cause. Le filtre precedent ne relachait que
                # l'OOM, reconnu au TEXTE du message : toute autre defaillance
                # etait relevee et arretait le moteur. Mesure du 10/09 :
                # cudaErrorStreamCaptureInvalidated a neuf sequences arretait
                # `acvram serve` au lieu de le degrader — une panne la ou le
                # commentaire promettait une degradation.
                #
                # ET LA CAUSE EST JOURNALISEE, jamais tue : un echec de capture
                # silencieux est precisement ce qui nous a coute la journee. On
                # imprime le type et le message, puis on replie en eager.
                # UNE SEULE FOIS PAR SESSION : `enabled` passant a False, on
                # ne repasse normalement pas ici — mais si un jour un chemin
                # reessaie, un serveur qui refuse la capture a chaque pas
                # noierait sa propre sortie. Le drapeau coute un attribut.
                self.raison = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                # NOMMER L'ALLOCATEUR quand il est en cause, apport de main a
                # ne pas perdre : les segments extensibles sont en tension avec
                # la capture, qui exige des adresses figees. Sans cette
                # mention, une capture perdue sous ACVRAM_ALLOC_EXTENSIBLE ne
                # se lit que comme un manque de VRAM.
                extensible = "expandable_segments" in os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", "")
                if not getattr(self, "_dit_raison", False):
                    self._dit_raison = True
                    print(f"[acvram] graphes CUDA desactives, decodage en eager "
                          f"— capture impossible : {self.raison}"
                          + (" (allocateur a segments extensibles actif)"
                             if extensible else ""), flush=True)
                self.enabled = False
                self.graphs.clear()
                self._apres_echec_capture()           # avant le ménage : l'état fautif
                torch.cuda.empty_cache()
                # gemma-4-26B-A4B (verdict-capture-parc-19-09) : capture du godet 1
                # impossible (OOM), le service tournait en eager 2,9 × plus lent
                # sans que `repli_eager` ni ses raisons ne le portent — la ligne de
                # régime disait `graphes=off` sans cause, /metrics `repli_eager=0`.
                # C'est un repli comme les autres : compté, nommé, une fois.
                self._eager(f"capture impossible (godet b={b}, ql={ql}) : {self.raison}")
                return False
            self.graphs[key] = entry
            if os.environ.get("ACVRAM_TRACE_PTRS"):
                entry["ptrs"] = _empreinte_adresses(self, entry)
                entry["ne"] = self.replays
            self.replays += 1                # la capture rejoue deja une fois
            if trace:
                print(f"[graphe] capture clé {key} : "
                      f"{(time.perf_counter()-t1)*1000:.1f} ms", flush=True)
            # la capture a rempli ET rejoué ce lot : rien à relancer
            self._prepare = (entry, key, b_reel, ql, t0, t1, time.perf_counter(), True)
            return True

        self._fill(entry, batch)
        if os.environ.get("ACVRAM_TRACE_ENTREES"):
            self._sonde_entrees(entry, batch, key)
        if trace and os.environ.get("ACVRAM_CHRONO_SYNC"):
            torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        self._prepare = (entry, key, b_reel, ql, t0, t1, t2, False)
        return True

    def _sonde_entrees(self, entry: dict, batch: ForwardBatch, key) -> None:
        """ACVRAM_TRACE_ENTREES=1 (Sage § 12) : ce que le pas à formes fixes va
        lire — jetons, positions, seq_lens du lot ET les tampons device
        remplis (x = plongement, positions, slots, seq_lens) — à comparer au
        journal du chemin eager (runner._sonde_eager) sur la même séquence."""
        torch.cuda.synchronize(self.device)
        n = min(4, batch.batch_size)
        tok = batch.tokens[:n].tolist() if torch.is_tensor(batch.tokens) else list(batch.tokens[:n])
        x = entry["x"][:n, :4].float().tolist()
        print(f"[ENTREES-FIXES] rejeu n°{self.replays} clé {key} jetons={tok} "
              f"positions={batch.positions[:n].tolist()} seq_lens={list(batch.seq_lens[:n])} | "
              f"x[:4]={[[round(v, 5) for v in r] for r in x]} "
              f"positions_dev={entry['positions'][:n].tolist()} slots_dev={entry['slots'][:n].tolist()} "
              f"seq_lens_dev={entry['seq_lens'][:n].tolist()}", flush=True)

    def prendre_echantillon(self) -> Optional[torch.Tensor]:
        """Le paquet [2, n] (ids ; bits fp32 des logprobs) du DERNIER rejeu,
        CLONÉ sur le flux courant — le clone est enfilé après les noyaux du
        graphe et avant le rejeu suivant, donc l ordre du flux garantit qu il
        lit ce rejeu-ci et pas le prochain (§ 7.3 de la note : invariant de
        flux, pas une marge). None si le rejeu n a pas écrit de sortie
        (opt-in absent, entrée capturée avant l opt-in). Une seule prise par
        rejeu : la deuxième rend None, jamais un paquet périmé."""
        e = self._echantillon
        self._echantillon = None
        if e is None or e[0] is None:
            return None
        sortie, n = e
        return sortie[:, :n].clone()

    def rejouer_suivant(self) -> Optional[torch.Tensor]:
        """Rejoue le lot préparé par ``preparer`` et rend une VUE des logits
        (``entry["out"][:b*ql]`` : adresse stable pour une clé de godet
        donnée, réécrite par le rejeu suivant — tout ce qui la lit doit être
        enfilé avant le prochain ``rejouer_suivant``). Enregistre
        ``evenement_jetons`` juste après le rejeu ; le runner l'attend
        seulement quand il consomme les jetons de ce pas, un pas plus tard."""
        if self._prepare is None:
            return None
        entry, key, b_reel, ql, t0, t1, t2, deja = self._prepare
        self._prepare = None
        trace = bool(os.environ.get("ACVRAM_TRACE_STEPS"))
        if deja:
            self.evenement_jetons.record()
            self._echantillon = (entry.get("sortie"), b_reel * ql)
            return entry["out"][:b_reel * ql]
        if trace and os.environ.get("ACVRAM_TRACE_CRENEAUX") and self.hybrid_layers:
            # Sonde (Sage § 11) : ce que le rejeu va lire, couche 0 hybride —
            # longueurs, palier, adresses des créneaux contre la table _mla_lot
            l0 = self.hybrid_layers[0]
            sts = l0.statics[:b_reel]
            lens = [int(st["len"].item()) for st in sts]
            lot = l0.__dict__.get("_mla_lots", {})
            cle = tuple((st["cache"].data_ptr(), st["len"].data_ptr()) for st in l0.statics[:key[0]])
            print(f"[graphe-CRENEAUX] rejeu n°{self.replays} clé {key} palier {l0.static_bucket} "
                  f"lens={lens} proprietaires={l0.static_owners[:b_reel]} "
                  f"caches={[st['cache'].dtype for st in sts][:1]} "
                  f"table_mla_lot={'ok' if cle in lot else 'ABSENTE'}", flush=True)
        if "step" in entry:                  # ACVRAM_GRAPHS_EAGER : sans capture
            with torch.inference_mode():
                entry["out"] = entry["step"]()
        else:
            if "ptrs" in entry:
                print(f"[graphe-REJEU] clé {key} capturée au rejeu n°{entry.get('ne', '?')}, "
                      f"rejeu n°{self.replays}, {len(entry['ptrs'])} adresses surveillées", flush=True)
                actuel = _empreinte_adresses(self, entry)
                bouge = [k for k, v in entry["ptrs"].items() if actuel.get(k) != v]
                if bouge:
                    print(f"[graphe-ADRESSES] clé {key} : {len(bouge)} tenseur(s) ont "
                          f"changé d'adresse depuis la capture : {bouge[:12]}", flush=True)
                    entry["ptrs"] = actuel
            entry["graph"].replay()
            if "ptrs" in entry:
                torch.cuda.synchronize(self.device)   # débogage : faute attribuée au bon rejeu
            elif os.environ.get("ACVRAM_CHRONO_SYNC"):
                # Instrumentation temporaire (Laure, mandat Jerome) : sans
                # synchronisation, .replay() est un lancement asynchrone --
                # perf_counter() juste apres ne mesure QUE le lancement, pas
                # l'execution reelle sur le GPU. Sous garde d'env car cette
                # synchronisation elle-meme serialise le pipeline et FAUSSE
                # le debit si elle reste active en permanence. Non committe.
                torch.cuda.synchronize(self.device)
        self.replays += 1
        self.evenement_jetons.record()
        self._echantillon = (entry.get("sortie"), b_reel * ql)
        out = entry["out"][:b_reel * ql]
        t3 = time.perf_counter()
        getattr(self, "temps_bind", None) is None and setattr(self, "temps_bind", [])
        getattr(self, "temps_fill", None) is None and setattr(self, "temps_fill", [])
        getattr(self, "temps_replay", None) is None and setattr(self, "temps_replay", [])
        self.temps_bind.append((t1 - t0) * 1000)
        self.temps_fill.append((t2 - t1) * 1000)
        self.temps_replay.append((t3 - t2) * 1000)
        if trace and (t3 - t0) * 1000 > 20:
            print(f"[graphe-lent] bind {(t1-t0)*1000:.1f} fill "
                  f"{(t2-t1)*1000:.1f} replay {(t3-t2)*1000:.1f} ms clé {key}",
                  flush=True)
        return out
        return out

    # -- hybrides ----------------------------------------------------------
    _SID_REMBOURRAGE = -1

    def _bind_hybrid(self, batch: ForwardBatch, lb: int,
                     b_godet: int = 0) -> None:
        """Lie les ``b_godet`` créneaux de chaque couche hybride : les
        ``len(sids)`` premiers aux séquences du lot, les suivants (godet >
        lot réel) à une sentinelle chacun.

        LIER `range(godet)` ET NON `sids` (1213554, 11/09 ; MECANISMES
        « prérequis bloquant ») : un créneau de rembourrage non lié garderait
        l'état récurrent de son propriétaire précédent, `_la_decode`
        (model.py) l'avancerait avec les entrées factices du godet — la
        récurrence mute en place (gdn.py `decode_static`) — et `static_bind`
        l'exporterait ensuite sous l'identifiant de ce propriétaire : un état
        corrompu rentre dans une séquence vivante, sans planter. Lié à une
        sentinelle, le créneau part à zéro et son état n'est jamais exporté
        (`static_bind`, sid négatif : `_sid_fantome`)."""
        sids = batch.seq_ids or list(range(batch.batch_size))
        b = b_godet or len(sids)
        m = self.model
        n_reel = len(sids)
        for layer in self.hybrid_layers:
            store = batch.gdn_store.setdefault(layer.index, {})
            for slot in range(b):
                # UN SID DISTINCT PAR CRENEAU DE REMBOURRAGE, PAS UN SID
                # PARTAGE. `static_bind` (model.py) suppose qu'un sid ne vit
                # que dans UN SEUL creneau a la fois : le partager entre deux
                # rembourrages fait croire au second que le sid « vit deja
                # ailleurs » (`store[sid] is _STATIC`), declenche un export
                # du premier creneau (`static_owners[premier] = None`,
                # `_reprendre`) et lui fait potentiellement heriter l'etat
                # exporte du premier au lieu d'un etat neuf a zero. Rare tant
                # que le godet ne depasse le lot reel que d'un seul creneau —
                # devenu frequent depuis le correctif du bug x0s
                # (godet_hybride), qui autorise plusieurs creneaux de
                # rembourrage simultanes (ex. b_reel=9, max_slots=12 : trois).
                sid = (sids[slot] if slot < n_reel
                      else self._SID_REMBOURRAGE - (slot - n_reel))
                layer.static_bind(slot, sid, store, godet_mla(self.max_model_len) + MLA_BUCKET,
                                  m.dtype)
            layer.static_bucket = lb
            if self.max_ql > 1:
                layer.ensure_hist(self.max_ql)

    # -- tampons ---------------------------------------------------------
    def rollback_hybrid(self, n_consumed: int) -> None:
        """Après une vérification spéculative partiellement rejetée : l'état
        de chaque couche hybride revient au dernier jeton consommé."""
        for layer in self.hybrid_layers:
            layer.rollback(n_consumed)

    def _embed(self, batch: ForwardBatch) -> torch.Tensor:
        """Le plongement, hors graphe, sur l'appareil où réside la table."""
        m = self.model
        idx = batch.tokens
        if idx.device != m.embed_tokens.device:
            idx = idx.to(m.embed_tokens.device, non_blocking=idx.is_pinned())
        x = torch.nn.functional.embedding(idx, m.embed_tokens).to(m.dtype)
        return x if m.spec.embedding_multiplier == 1.0 else x * m.spec.embedding_multiplier

    def _etage(self, entry: dict) -> dict:
        """Tampons hôte ÉPINGLÉS de l'entrée, à deux parités : le pas n écrit
        la parité n % 2 pendant que la copie asynchrone du pas n-1 lit
        l'autre. Une copie depuis de la mémoire paginable bloque l'hôte le
        temps de la mise en attente ; depuis l'épinglée, elle est enfilée et
        l'hôte continue."""
        et = entry.get("etage")
        if et is None:
            b, ql, nblk = entry["key"][:3]
            def deux(*forme):
                return [torch.zeros(*forme, dtype=torch.long).pin_memory() for _ in range(2)]
            et = entry["etage"] = {"tour": 0, "positions": deux(b * ql), "slots": deux(b * ql),
                                   "seq_lens": deux(b), "tables": deux(b, nblk),
                                   "tokens": deux(b * ql)}
        return et

    def _fill(self, entry: dict, batch: ForwardBatch) -> None:
        b, _ql, nblk = entry["key"][:3]
        b_reel = batch.batch_size
        n = b_reel * _ql
        et = self._etage(entry)
        k = et["tour"] & 1
        et["tour"] += 1
        if not batch.tokens.is_cuda:
            # jetons hôte : passés par l'épinglé pour que l'embedding parte
            # sans bloquer ; jetons device (pipeline) : rien à faire
            tk = et["tokens"][k]
            tk[:n].copy_(batch.tokens)
            batch = _avec_tokens(batch, tk[:n])
        emb = self._embed(batch).to(self.device)
        pos, sl, sq, tb = et["positions"][k], et["slots"][k], et["seq_lens"][k], et["tables"][k]
        pos.zero_(); pos[:n].copy_(batch.positions)
        sl.fill_(-1); sl[:n].copy_(batch.slot_mapping)
        sq.zero_(); sq[:b_reel] = torch.tensor(batch.seq_lens, dtype=torch.long)
        # Table completee au godet avec le bloc 0 : lu, dequantifie, masque.
        tb.zero_()
        for i, t in enumerate(batch.block_tables):
            tb[i, : t.shape[0]].copy_(t)
        if b_reel < b:
            entry["x"].zero_()
        entry["x"][:n].copy_(emb, non_blocking=True)
        entry["positions"].copy_(pos, non_blocking=True)
        entry["slots"].copy_(sl, non_blocking=True)
        entry["seq_lens"].copy_(sq, non_blocking=True)
        entry["tables"].copy_(tb, non_blocking=True)
        if os.environ.get("ACVRAM_TRACE_PTRS"):
            sl = batch.slot_mapping.tolist(); po = batch.positions.tolist()
            print(f"[graphe-FORMES] clé {entry['key']} seq_lens={batch.seq_lens} "
                  f"blocs={[int(t.shape[0]) for t in batch.block_tables]} "
                  f"tables_max={[int(t.max()) for t in batch.block_tables]} "
                  f"slots={min(sl)}..{max(sl)} positions={min(po)}..{max(po)} "
                  f"nblk={nblk} q_len={_ql}", flush=True)

    def _capture(self, b: int, ql: int, nblk: int,
                 batch: ForwardBatch) -> dict:
        m = self.model
        d = self.device
        h = m.spec.hidden_size
        entry = {
            "key": (b, ql, nblk, self._last_key[3] if self._last_key else 0),
            "ql": ql,
            "x": torch.zeros(b * ql, h, dtype=m.dtype, device=d),
            "positions": torch.zeros(b * ql, dtype=torch.long, device=d),
            "slots": torch.zeros(b * ql, dtype=torch.long, device=d),
            "tables": torch.zeros(b, nblk, dtype=torch.long, device=d),
            "seq_lens": torch.zeros(b, dtype=torch.long, device=d),
        }
        if self.sampler_graphe:
            entry["sortie"] = torch.zeros(2, b * ql, dtype=torch.int64, device=d)
        self._fill(entry, batch)
        if os.environ.get("ACVRAM_TRACE_ENTREES"):
            self._sonde_entrees(entry, batch, entry["key"])
        max_pos = min(self.max_model_len + 1, nblk * BLOCK_SIZE + 1)

        # Tout cache RoPE doit exister à sa taille finale avant la capture :
        # étendu ensuite, il laisserait aux graphes déjà capturés l'adresse
        # d'un tenseur abandonné. Toutes les RoPE du modèle sont concernées,
        # pas seulement celles de l'attention : la RoPE du chemin MLA s'étend
        # au godet courant (jusqu'à max_model_len arrondi au MLA_BUCKET).
        from .layers import RotaryEmbedding
        for mod in m.modules():
            if isinstance(mod, RotaryEmbedding):
                mod.reserver(godet_mla(self.max_model_len) + MLA_BUCKET + 1, d, m.dtype)
        # REGLES § 6 (remède C15 niveau 2) : les caches paresseux du chemin MLA (k_b contigu,
        # v_b fp32, tables) sont matérialisés hors capture ; toute allocation pendant une
        # capture lève ensuite (`_refuser_en_capture`), au lieu de corrompre en silence
        for mod in m.modules():
            if hasattr(mod, "chauffer") and hasattr(mod, "rank"):
                mod.chauffer(d, godet_mla(self.max_model_len) + MLA_BUCKET + 1)

        # Toute echelle globale NVFP4 doit etre LUE avant la capture.
        #
        # `NVFP4Tensor.global_scale_float()` memorise sa valeur dans `_gs_f`,
        # mais le premier appel fait `.item()` — une synchronisation hote,
        # `cudaErrorStreamCaptureUnsupported` si elle tombe dans la capture.
        # Mesure du 10/09/2026 : a douze sequences, `v_proj` franchit le seuil
        # de lot, prend le chemin W4A8 (`nvfp4_mm_w4a8`), qui dequantifie et
        # lit cette echelle pour la premiere fois — dans la capture. A huit
        # sequences le seuil n'est pas franchi, le GEMV ne lit pas cette
        # valeur la, et la capture reussit. Meme classe que la reservation des
        # caches RoPE juste au-dessus : ce qui doit exister avant la capture
        # doit etre FABRIQUE avant elle, pas rencontre pendant.
        for mod in m.modules():
            w_ = getattr(mod, "qweight", None)
            if hasattr(w_, "global_scale_float"):
                w_.global_scale_float()

        def step() -> torch.Tensor:
            out = m.decode_fixed(entry["x"], entry["positions"],
                                 entry["slots"], entry["tables"],
                                 entry["seq_lens"], max_pos, q_len=ql)
            if "sortie" in entry:
                echantillon_glouton_dans(entry["sortie"], out)
            return out

        # Echauffement sur un flux annexe (exige par la capture), puis capture.
        # Les ecritures KV de ces passes sont identiques a celle du pas reel :
        # les rejouer n'ajoute rien, n'efface rien.
        if os.environ.get("ACVRAM_GRAPHS_EAGER"):
            # Débogage : le chemin à formes fixes, exécuté sans capture.
            entry["step"] = step
            with torch.inference_mode():
                entry["out"] = step()
            self.captures += 1
            return entry

        # Les états récurrents ne sont pas idempotents : l'échauffement et le
        # rejeu de capture les feraient avancer trois fois pour un jeton.
        # On les photographie avant, on les restaure avant le vrai rejeu.
        torch.cuda.synchronize(d)
        instantane = [(l, [l.linear_attn.static_export(st) for st in l.statics[:b]])
                      for l in self.hybrid_layers]
        side = torch.cuda.Stream(d)
        side.wait_stream(torch.cuda.current_stream(d))
        with torch.cuda.stream(side):
            with torch.inference_mode():
                for _ in range(2):
                    step()
        torch.cuda.current_stream(d).wait_stream(side)

        # Le tampon de l'état caché MTP doit exister à sa capacité finale avant
        # la capture : alloué pendant, il appartiendrait au pool du graphe.
        if getattr(m, "mtp", None) is not None:
            m.reserver_hidden(b * ql, entry["x"])
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            if self._pool is None:
                with torch.cuda.graph(graph):
                    entry["out"] = step()
                self._pool = graph.pool()
            else:
                with torch.cuda.graph(graph, pool=self._pool):
                    entry["out"] = step()
        entry["graph"] = graph
        self.captures += 1
        for l, es in instantane:
            for st, e in zip(l.statics, es):
                l.linear_attn.static_load(st, e)
        torch.cuda.synchronize(d)
        graph.replay()                       # la capture n'execute pas : rejouer
        return entry
