"""Briques de transformeur conscientes de la quantification et des étages.

Chaque couche linéaire du modèle est un :class:`QuantLinear`. Elle détient son
poids dans le format qu'a choisi son appareil, applique la mise à l'échelle de
calibration produite par le convertisseur, et aiguille vers le noyau fusionné
lorsqu'il en existe un. Une couche dont les poids résident en mémoire vive les
enveloppe dans un :class:`StreamedWeight`, qui les copie vers le GPU sur un flux
annexe, de sorte que le transfert de la couche i+1 recouvre le calcul de la
couche i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels
from ..quant.calibrate import ChannelScaler
from ..quant.formats import INT8Tensor, PlainTensor, dequantize
from ..quant.int4 import INT4Tensor
from ..quant.nvfp4 import NVFP4Tensor

__all__ = ["QuantLinear", "StreamedWeight", "RMSNorm", "RotaryEmbedding",
           "apply_rope", "repeat_kv", "repeat_kv_batched", "attention",
           "batched_decode_attention", "causal_mask"]


class PoolHote:
    """Arène de mémoire hôte épinglée où les tampons plats sont découpés.

    L'allocateur hôte de PyTorch arrondit toute demande à la puissance de 2
    supérieure — à toutes les échelles. Nos experts pèsent exactement 3,00 Mio
    (768 x 2048 en bf16) et en coûtent 4 : +33,3 % sur toute la mémoire
    verrouillée, soit 46,01 Gio mesurés pour 33,76 de poids réels. Mesuré le
    8/09/2026 sur 200 allocations — sur une seule, un surcoût fixe de première
    allocation donne un facteur 3,9 qui n'est pas le régime.

    Une arène dont la taille EST une puissance de 2 ne paie rien (facteur
    1,000 relevé à 2, 4, 512 et 1024 Mio), et une vue prise dedans est
    elle-même épinglée. Grossir les tampons ne suffirait pas : un tampon par
    couche ferait 1152 Mio, arrondis à 2048, soit +77,8 % — le pire point de
    l'échelle. C'est l'alignement qui compte, pas la taille.

    Les blocs sont demandés par puissances de 2 décroissantes, avec bissection
    quand l'allocation échoue : le surcoût reste nul à chaque étape, et une
    machine trop chargée dégrade en blocs plus petits au lieu de tuer le
    chargement entier — ce qu'une arène unique ferait.
    """

    ALIGNEMENT = 256

    def __init__(self, octets: int, plancher: int = 64 * 2 ** 20,
                 bloc_max: int = 2 ** 30, souffle: float = 0.0) -> None:
        """`bloc_max` plafonne la taille d'UNE demande, `souffle` l'espace dans
        le temps. Les deux lissent la pression, mais par des mécanismes
        différents et il ne faut pas les changer ensemble.

        Le 8/09/2026, réserver 33,76 Gio en un bloc de 32 a fait tuer le
        chargement par la garde : trois secondes pendant lesquelles aucune
        tâche de la machine n'avançait. Les mêmes octets pris en 11 550
        morceaux sur soixante-dix secondes ne produisaient aucune pression —
        `MemoryPeak` 70,5 Gio sans pression contre 57,2 Gio avec.

        **Hypothèse retenue, et une seule est active par défaut** : c'est la
        TAILLE de la demande qui pèse, pas sa cadence. Pour trouver 32 Gio
        épinglables d'un coup, le noyau doit réclamer 32 Gio d'un coup —
        éviction de cache, voire swap — et cette réclamation est synchrone :
        tout le reste attend pendant ce temps. `bloc_max` à 1 Gio divise donc
        chaque réclamation par 32, en gardant toutes les tailles à des
        puissances de 2, donc le surcoût toujours nul.

        `souffle` (secondes entre deux blocs) reste à **zéro par défaut** :
        c'est l'hypothèse concurrente, et l'activer en même temps que
        `bloc_max` rendrait impossible de dire laquelle a agi. À mesurer
        séparément si le plafonnement ne suffit pas.
        """
        self.blocs: list[torch.Tensor] = []
        self._curseurs: list[int] = []
        reste = int(octets)
        while reste >= plancher:
            taille = min(1 << (reste.bit_length() - 1), bloc_max)
            while taille >= plancher:
                try:
                    self.blocs.append(
                        torch.empty(taille, dtype=torch.uint8).pin_memory())
                    self._curseurs.append(0)
                    reste -= taille
                    if souffle > 0:
                        time.sleep(souffle)
                    break
                except (RuntimeError, MemoryError):
                    taille >>= 1                        # bissection
            else:
                break                                   # même le plancher échoue

    @property
    def octets(self) -> int:
        return sum(b.numel() for b in self.blocs)

    def tranche(self, n: int) -> "Optional[torch.Tensor]":
        """Une vue de n octets, ou None s'il ne reste pas la place.

        Une tranche ne chevauche jamais deux blocs : elle tient entière dans
        l'un d'eux, sinon on essaie le suivant.
        """
        besoin = (n + self.ALIGNEMENT - 1) // self.ALIGNEMENT * self.ALIGNEMENT
        for i, bloc in enumerate(self.blocs):
            debut = self._curseurs[i]
            if debut + besoin <= bloc.numel():
                self._curseurs[i] = debut + besoin
                return bloc[debut:debut + n]
        return None


_POOL: "Optional[PoolHote]" = None


def reserver_pool(octets: int) -> "Optional[PoolHote]":
    """Réserve l'arène. À appeler APRÈS le calcul du plan, jamais avant.

    Avant le plan, la taille exilée n'est pas connue : un pool dimensionné là
    serait un chiffre deviné.
    """
    global _POOL
    _POOL = PoolHote(octets) if octets > 0 else None
    return _POOL


def liberer_pool() -> None:
    """Rend l'arène entière.

    C'est le seul moment où la mémoire épinglée revient au système :
    l'allocateur hôte de PyTorch ne rend jamais ce qu'il a pris, et 11 550
    tampons individuels ne peuvent donc pas être rendus. Une arène, si.
    """
    global _POOL
    _POOL = None
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()


def _tranche_pool(n: int) -> "Optional[torch.Tensor]":
    return _POOL.tranche(n) if _POOL is not None else None


def _emballer(host: dict[str, torch.Tensor]):
    """Copie les tenseurs dans un tampon uint8 épinglé contigu ; rend le
    tampon et la découpe ``{clé: (décalage, forme, dtype)}``, décalages
    alignés sur 256 octets pour les copies et les vues."""
    decoupe = {}; off = 0
    for k in sorted(host):
        v = host[k]
        n = v.numel() * v.element_size()
        decoupe[k] = (off, tuple(v.shape), v.dtype, n)
        off += (n + 255) // 256 * 256
    plat = _tranche_pool(max(off, 1))
    if plat is None:
        t = torch.empty(max(off, 1), dtype=torch.uint8)
        plat = t.pin_memory() if torch.cuda.is_available() else t
    for k, (o, forme, dt, n) in decoupe.items():
        plat[o:o + n].view(dt).view(forme).copy_(host[k])
    return plat, decoupe


def _decouper(plat: torch.Tensor, decoupe: dict) -> dict[str, torch.Tensor]:
    return {k: plat[o:o + n].view(dt).view(forme) for k, (o, forme, dt, n) in decoupe.items()}

class StreamedWeight:
    """Un poids qui vit en mémoire hôte épinglée et ne visite le GPU qu'à la demande.

    C'est la mémoire épinglée qui rend la copie asynchrone : une source
    paginable forcerait le pilote à la sérialiser et le recouvrement
    disparaîtrait. Le double tampon fait que le transfert de la couche i+1 est
    déjà en vol pendant que la couche i calcule, si bien qu'une couche
    transférée coûte ``max(copie, calcul)`` et non leur somme — ce que suppose
    exactement le modèle de coût du planificateur.
    """

    def __init__(self, host_tensors: dict[str, torch.Tensor], device: torch.device,
                 n_buffers: int = 2, pool: "Optional[ExpertPool]" = None) -> None:
        # Un seul tampon épinglé contigu par poids : la copie hôte→carte se
        # fait en UN lancement au lieu d'un par tenseur (trois pour NVFP4 :
        # qweight, block_scale, global_scale). La courbe du 8/09 a montré un
        # transfert borné par la latence des copies, pas par le débit du bus —
        # 5,7 Go/s effectifs sur 18,7 —, avec quatre-vingt-dix copies par
        # couche exilée. Le dictionnaire `host` reste la vue par clé.
        #
        # Les tenseurs source sont passés TELS QUELS, paginables. Ils l'étaient
        # épinglés un à un avant l'emballage, et c'était inutile deux fois : la
        # copie vers `plat` est CPU→CPU (`_emballer`), elle n'exige rien de sa
        # source ; et seul `plat` sert au DMA.
        #
        # Inutile, mais pas gratuit. L'allocateur hôte épinglé de PyTorch ne
        # rend JAMAIS au système ce qu'il a pris : il le garde en cache pour
        # réemploi. Chaque tenseur épinglé ici, aussitôt déréférencé par le
        # `_decouper` plus bas, restait donc verrouillé pour la vie du
        # processus, invisible à `Unevictable` comme à `Mlocked` — seul
        # `nr_foll_pin_acquired − nr_foll_pin_released` de /proc/vmstat le
        # voyait. Mesure du 8/09/2026 sur le témoin MoE, 30 couches exilées :
        # 46 Gio réellement épinglés pour 33,75 attendus, soit ~12 Gio de
        # résidu que le correctif précédent n'avait pas touchés — il avait
        # supprimé la double RÉFÉRENCE, pas la double ÉPINGLURE.
        self.plat, self.decoupe = _emballer(host_tensors)
        # `host` redevient ce que la ligne au-dessus annonce : des VUES sur le
        # tampon plat. Il en était une COPIE épinglée indépendante, du même
        # contenu, gardée vivante pour deux usages qui n'ont besoin ni de copie
        # ni d'épinglage — le repli sans GPU et `nbytes`. Le transfert, lui,
        # part de `plat`.
        # La mémoire épinglée n'est ni évinçable ni swappable : la doubler
        # doublait ce que le noyau doit chasser ailleurs. Sur un MoE de 30
        # milliards à 30 couches exilées, c'était 67,5 Gio verrouillés au lieu
        # de 33,75 sur 93,98 de RAM — la machine devenait inutilisable, souris
        # comprise, et `memory.peak` de la session a touché 89,42 Gio.
        self.host = _decouper(self.plat, self.decoupe)
        self.device = device
        self.pool = pool
        self.stream = (pool.stream if pool is not None else
                       torch.cuda.Stream(device=device) if device.type == "cuda" else None)
        self._buffers: list[dict[str, torch.Tensor]] = []
        self._events: list[Any] = []
        self._libres: list[Any] = []
        self._slot = 0
        self.n_buffers = n_buffers

    def _ensure(self) -> None:
        if self._buffers:
            return
        self._plats = []
        for _ in range(self.n_buffers):
            plat = torch.empty_like(self.plat, device=self.device)
            self._plats.append(plat)
            self._buffers.append(_decouper(plat, self.decoupe))
            self._events.append(
                torch.cuda.Event() if self.device.type == "cuda" else None)
            self._libres.append(
                torch.cuda.Event() if self.device.type == "cuda" else None)
        if self.stream is not None:
            # L'allocation est ordonnée sur le flux courant : le bloc rendu par
            # l'allocateur peut encore être lu par un noyau de ce flux (une
            # temporaire de la couche précédente). Écrire dessus depuis le
            # flux annexe sans attendre corrompait cette lecture — c'est la
            # course qui faisait planter Qwen3-Coder-Next au premier prompt
            # long, et que CUDA_LAUNCH_BLOCKING masquait.
            alloue = torch.cuda.Event()
            alloue.record(torch.cuda.current_stream(self.device))
            alloue.wait(self.stream)

    def prefetch(self) -> int:
        """Lance la copie vers le tampon suivant ; rend son emplacement."""
        if self.device.type != "cuda":
            return 0
        if self.pool is not None:
            return self.pool.copier(self.plat, self.decoupe)
        self._ensure()
        slot = self._slot
        self._slot = (self._slot + 1) % self.n_buffers
        with torch.cuda.stream(self.stream):
            self._libres[slot].wait(self.stream)      # le calcul précédent a fini de lire
            self._plats[slot].copy_(self.plat, non_blocking=True)   # une seule copie
            self._events[slot].record(self.stream)
        return slot

    def wait(self, slot: int) -> dict[str, torch.Tensor]:
        if self.device.type != "cuda":
            return self.host
        if self.pool is not None:
            return self.pool.attendre(slot)
        self._events[slot].wait(torch.cuda.current_stream(self.device))
        return self._buffers[slot]

    def release(self, slot: int) -> None:
        """Après le calcul : l'emplacement peut être réécrit."""
        if self.device.type != "cuda":
            return
        if self.pool is not None:
            self.pool.liberer(slot)
        elif self._libres:
            self._libres[slot].record(torch.cuda.current_stream(self.device))

    @property
    def nbytes(self) -> int:
        """Taille des POIDS. C'est ce que l'affichage appelle « poids » et ce
        qui divise les temps pour donner des Go/s : elle ne doit pas compter le
        rembourrage."""
        return sum(t.numel() * t.element_size() for t in self.host.values())

    @property
    def octets_verrouilles(self) -> int:
        """Mémoire hôte réellement RÉSERVÉE et épinglée, rembourrage compris.

        Distincte de `nbytes` : `_emballer` aligne chaque tenseur sur 256
        octets, d'autant plus visible que les tenseurs sont petits et nombreux
        — trois par poids en NVFP4. Les confondre donnerait un nom à deux
        propriétés ; `nbytes` sous-estimerait ce que la machine subit, et
        `octets_verrouilles` mentirait sur ce qu'est un poids.
        """
        return self.plat.numel()


class ExpertPool:
    """Tampons GPU partagés par tous les experts exilés d'une même couche.

    Le double tampon de `StreamedWeight` convient à un poids dense : deux
    copies d'un seul tenseur. Appliqué à un mélange de 512 experts, il
    allouait deux copies de *chaque* expert dès le premier préchargement —
    deux fois la couche entière sur la carte, par couche exilée. Sur
    Qwen3-Coder-Next cela dépassait la 5090 à la première requête.

    Ici la couche possède `n_slots` jeux de tampons par disposition (gate et
    up d'un côté, down de l'autre) ; dix experts routés par jeton n'en
    occupent jamais plus. Un emplacement est réécrit seulement après que le
    calcul qui le lisait a été enregistré comme fini sur le flux de calcul.
    Rien n'est préchargé d'avance : on ne connaît les experts à copier
    qu'après le routage.
    """

    def __init__(self, device: torch.device, n_slots: int, dense: bool = False) -> None:
        self.device = device
        self.dense = dense              # pool partagé des poids denses exilés : préchargeable
        self.n_slots = max(2, n_slots)
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._par_disposition: dict[tuple, dict] = {}
        self._slots: list[tuple[tuple, int]] = []      # slot global -> (disposition, index)

    @staticmethod
    def _disposition(decoupe: dict, nbytes: int) -> tuple:
        return (nbytes,) + tuple((k, o, forme, dt, n) for k, (o, forme, dt, n) in sorted(decoupe.items()))

    def _jeu(self, plat: torch.Tensor, decoupe: dict) -> dict:
        d = self._disposition(decoupe, plat.numel())
        jeu = self._par_disposition.get(d)
        if jeu is None:
            plats = [torch.empty_like(plat, device=self.device) for _ in range(self.n_slots)]
            jeu = {
                "plats": plats,
                "tampons": [_decouper(pl, decoupe) for pl in plats],
                "pret": [torch.cuda.Event() for _ in range(self.n_slots)],
                "libre": [torch.cuda.Event() for _ in range(self.n_slots)],
                # Un emplacement distribue et pas encore rendu ne doit jamais
                # etre reecrit. L'evenement `libre` ne le protege pas : tant
                # qu'il n'a jamais ete enregistre, l'attendre ne fait rien.
                "distribue": [False] * self.n_slots,
                "prochain": 0,
                "base": len(self._slots),
            }
            for i in range(self.n_slots):
                self._slots.append((d, i))
            self._par_disposition[d] = jeu
            if self.stream is not None:
                alloue = torch.cuda.Event()
                alloue.record(torch.cuda.current_stream(self.device))
                alloue.wait(self.stream)
        return jeu

    # ACVRAM_POOL_SYNC=1 : copies sur le flux de calcul, sans flux annexe ni
    # événement. Témoin de diagnostic : si une course disparaît avec lui, elle
    # est dans l'ordonnancement ci-dessous.
    SYNC = bool(os.environ.get("ACVRAM_POOL_SYNC"))

    def copier(self, plat: torch.Tensor, decoupe: dict) -> int:
        jeu = self._jeu(plat, decoupe)
        # Premier emplacement LIBRE à partir du tour de rôle — pas le tour de
        # rôle seul : en exil total (36/36, poste3 02b316d), la couche 0 se
        # copie à la demande APRÈS que la couche 1 a été préchargée, l'ordre
        # des rendus n'est plus celui des prises, et le tour de rôle tombait
        # sur un emplacement en vol alors que deux étaient rendus.
        i = jeu["prochain"]
        for k in range(self.n_slots):
            j = (i + k) % self.n_slots
            if not jeu["distribue"][j]:
                i = j
                break
        if jeu["distribue"][i]:
            # Tous les emplacements de cette disposition sont en vol. Continuer
            # ecraserait les octets d'un expert qu'un calcul n'a pas encore lu,
            # et le calcul lirait alors un autre expert — sans erreur, sans
            # trace, avec pour seul symptome une sortie qui degenere. Mesure du
            # 8/09/2026 : dix experts pour quatre emplacements, l'echelle
            # globale relue appartenait a un autre expert.
            raise RuntimeError(
                f"ExpertPool sature : {self.n_slots} emplacements, tous "
                f"distribues et non rendus pour cette disposition. Augmenter "
                f"n_slots (2 x experts_par_jeton + 2) ou liberer avant de "
                f"copier.")
        jeu["distribue"][i] = True
        jeu["prochain"] = (i + 1) % self.n_slots
        if self.SYNC:
            jeu["plats"][i].copy_(plat, non_blocking=True)
            return jeu["base"] + i
        with torch.cuda.stream(self.stream):
            # ne pas écraser un emplacement qu'un calcul lit encore
            jeu["libre"][i].wait(self.stream)
            jeu["plats"][i].copy_(plat, non_blocking=True)      # une seule copie
            jeu["pret"][i].record(self.stream)
        return jeu["base"] + i

    def attendre(self, slot: int) -> dict[str, torch.Tensor]:
        d, i = self._slots[slot]
        jeu = self._par_disposition[d]
        if not self.SYNC:
            jeu["pret"][i].wait(torch.cuda.current_stream(self.device))
        return jeu["tampons"][i]

    def liberer(self, slot: int) -> None:
        d, i = self._slots[slot]
        jeu = self._par_disposition[d]
        jeu["distribue"][i] = False
        if self.SYNC:
            return
        jeu["libre"][i].record(torch.cuda.current_stream(self.device))

    @property
    def nbytes(self) -> int:
        return sum(pl.numel() for jeu in self._par_disposition.values() for pl in jeu["plats"])


class QuantLinear(nn.Module):
    """``y = x @ W.T (+ b)`` où W est stocké quantifié.

    La mise à l'échelle s'applique à l'*entrée*, jamais repliée dans le poids :
    le convertisseur l'a choisie précisément pour que le poids se quantifie bien
    une fois mis à l'échelle, et la replier défairait cela.
    """

    def __init__(self, qweight: Any, bias: Optional[torch.Tensor] = None,
                 scaler: Optional[ChannelScaler] = None,
                 out_features: Optional[int] = None,
                 in_features: Optional[int] = None) -> None:
        super().__init__()
        self.qweight = qweight
        self.scaler = scaler
        self.bias = bias
        shape = getattr(qweight, "shape", None)
        self.out_features = out_features or (shape[0] if shape else 0)
        self.in_features = in_features or (shape[1] if shape else 0)
        self.streamed: Optional[StreamedWeight] = None
        self._pending_slot: Optional[int] = None

    # -- placement -------------------------------------------------------
    def to_device(self, device: torch.device, streamed: bool = False,
                  pool: "Optional[ExpertPool]" = None) -> "QuantLinear":
        if streamed:
            self.streamed = StreamedWeight(
                dict(self.qweight.state_dict()), device, pool=pool)
        else:
            self.qweight = self.qweight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        if self.scaler is not None:
            self.scaler = self.scaler.to(device)
        return self

    def prefetch(self) -> None:
        # Un poids d'un pool n'est jamais préchargé d'avance : on ne sait
        # quels experts serviront qu'après le routage, et précharger les 512
        # d'une couche remplissait la carte.
        if os.environ.get("ACVRAM_SANS_PRECHARGE"):
            return                       # diagnostic : tout se copie à la demande
        if self.streamed is not None and (self.streamed.pool is None
                                          or getattr(self.streamed.pool, "dense", False)):
            # sans pool, ou pool DENSE partagé (17/09) : un poids par couche,
            # le pool garde ses emplacements par événements — jamais pour un
            # pool d'experts (les 512 d'une couche rempliraient la carte)
            self._pending_slot = self.streamed.prefetch()

    def precharger(self) -> None:
        """Préchargement volontaire, pool compris — pour un expert dont le
        routage vient de désigner qu'il servira. Lancer toutes les copies
        d'une couche avant le premier calcul les met en file sur le flux de
        copie, qui prend de l'avance pendant que les GEMV s'exécutent : c'est
        le recouvrement que le profil du 8/09 montrait absent (bus muet
        pendant le calcul, calcul muet pendant le bus)."""
        if self.streamed is not None and self._pending_slot is None:
            self._pending_slot = self.streamed.prefetch()

    # -- forward ---------------------------------------------------------
    def _resolved_weight(self) -> tuple[Any, Optional[int]]:
        if self.streamed is None:
            return self.qweight, None
        slot = self._pending_slot if self._pending_slot is not None \
            else self.streamed.prefetch()
        tensors = self.streamed.wait(slot)
        self._pending_slot = None
        return _rehydrate(self.qweight, tensors), slot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None and not self.scaler.is_identity:
            x = self.scaler.apply(x)
        w, slot = self._resolved_weight()
        # Le registre choisit le backend par (format, peripherique) ; voir
        # kernels/backends.py. Un acces de dictionnaire memoise, rien de plus.
        y = kernels.matmul(x, w)
        if slot is not None:
            self.streamed.release(slot)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    @property
    def nbytes(self) -> int:
        return getattr(self.qweight, "nbytes", 0)

    def extra_repr(self) -> str:
        fmt = getattr(self.qweight, "format", "?")
        return (f"in={self.in_features}, out={self.out_features}, fmt={fmt}"
                f"{', streamed' if self.streamed else ''}")


def _rehydrate(template: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Reconstruit un objet tenseur quantifié autour de tampons GPU fraîchement copiés.

    L'echelle globale memoisee du modele est RECOPIEE depuis le template. Sans
    cela, chaque transfert d'expert rend un objet neuf dont le cache est vide,
    et le premier GEMV le relit par `.item()` : une synchronisation du flux
    CUDA par expert et par couche, la ou la memoisation existe justement pour
    l'eviter — ce meme appel pesait 62 % du temps de decodage au profil NVFP4,
    et une synchronisation rend le noyau incapturable dans un graphe. La valeur
    est identique par construction : le tampon GPU est la copie du tenseur hote
    que le template decrit.
    """
    objet = _rehydrate_brut(template, tensors)
    gs = template.__dict__.get("_gs_f")
    if gs is not None:
        objet.__dict__["_gs_f"] = gs
    return objet


def _rehydrate_brut(template: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(template, NVFP4Tensor):
        return NVFP4Tensor(
            tensors["qweight"], tensors["block_scale"].view(torch.float8_e4m3fn),
            tensors["global_scale"], template.shape, template.padded_in)
    if isinstance(template, INT4Tensor):
        return INT4Tensor(tensors["qweight"], tensors["scales"], tensors["zeros"],
                          template.group_size, template.shape, template.padded_in)
    if isinstance(template, PlainTensor):
        return PlainTensor(tensors["weight"], template.shape, template.format)
    if isinstance(template, INT8Tensor):
        # Les tenseurs promus en 8 bits (experts partagés, attention linéaire)
        # suivent le perceptron exilé : sans cette branche, le premier
        # perceptron transféré d'un MoE plantait la requête.
        return INT8Tensor(tensors["qweight"], tensors["scales"], tensors["zeros"],
                          template.group_size, template.shape)
    from ..quant.q3n import Q3NTensor
    if isinstance(template, Q3NTensor):
        # La table suit le gabarit, comme block et shape : identique pour tout
        # le modèle, elle n'a rien à faire dans les tampons transférés.
        return Q3NTensor(tensors["qweight"],
                         tensors["block_scale"].view(torch.float8_e4m3fn),
                         tensors["global_scale"], template.block,
                         template.shape, template.format, template.table)
    raise TypeError(f"impossible de reconstruire {type(template)!r}")


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # On accumule la variance en fp32 : avec des poids sur 4 bits, les
        # activations sont déjà bruitées, et une réduction en demi-précision sur
        # 8192 canaux ajoute de l'erreur pour un gain de vitesse dérisoire.
        if dtype == torch.bfloat16 and x.is_cuda \
                and self.weight.dtype == torch.bfloat16:
            ext = kernels.get_extension()
            if ext is not None and hasattr(ext, "rmsnorm_bf16"):
                if _norme_warp(ext, x):
                    return ext.rmsnorm_bf16_warp(x, self.weight, self.eps)[0]
                if _norme_reg(ext, x):
                    return ext.rmsnorm_bf16_reg(x, self.weight, self.eps)[0]
                return ext.rmsnorm_bf16(x, self.weight, self.eps)[0]  # 1 lancement
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32.to(dtype) * self.weight.to(dtype))


def add_norm(residu: torch.Tensor, y: torch.Tensor, norme, mult: float = 1.0):
    """``x = residu + mult*y`` puis ``norme(x)``, en un lancement.

    Renvoie (x, normalisé). Repli PyTorch si le noyau manque."""
    if (type(norme).__name__ == "RMSNorm" and residu.is_cuda
            and residu.dtype == torch.bfloat16 and y.dtype == torch.bfloat16
            and norme.weight.dtype == torch.bfloat16):
        ext = kernels.get_extension()
        if ext is not None and hasattr(ext, "rmsnorm_bf16"):
            if _norme_warp(ext, y):
                h, x = ext.rmsnorm_bf16_warp(y, norme.weight, norme.eps, residu, mult)
                return x, h
            if _norme_reg(ext, y):
                h, x = ext.rmsnorm_bf16_reg(y, norme.weight, norme.eps, residu, mult)
                return x, h
            h, x = ext.rmsnorm_bf16(y, norme.weight, norme.eps, residu, mult)
            return x, h
    x = residu + (y if mult == 1.0 else y * mult)
    return x, norme(x)


# C15-prefill (fusion « norm ») : au-delà de ce nombre de lignes la RMSNorm prend
# le noyau à un warp par ligne (rmsnorm_bf16_warp, acvram_kernels.cu : même
# ordre de somme, au bit). En dessous — le décodage, sous graphes — le noyau à
# bloc reste : un nœud capturé ne change pas de noyau.
NORME_WARP_MIN_LIGNES = 256
NORME_WARP_H_MAX = 2048          # la ligne tient en registres (64 par lane) ; au-delà, le bloc


# Pièce 156 F6 (DÉFAUT depuis 156 d ; 0 = témoin) : hors du noyau à warp (préfill), le noyau à bloc dont la ligne
# reste en registres (rmsnorm_bf16_reg, acvram_kernels.cu) — même découpe, même ordre de somme : au bit (test et KL
# de la 156 d : 16 fenêtres × 512 pas identiques).
_NORME_REGISTRES = os.environ.get("ACVRAM_NORME_REGISTRES", "1") == "1"
NORME_REG_H_MAX = 8192           # EPT = ceil(H / 1024) ≤ 8 éléments par fil


def _norme_reg(ext, x: torch.Tensor) -> bool:
    return _NORME_REGISTRES and hasattr(ext, "rmsnorm_bf16_reg") and x.shape[-1] <= NORME_REG_H_MAX


def _norme_warp(ext, x: torch.Tensor) -> bool:
    return (kernels.prefill_compact("norm") and hasattr(ext, "rmsnorm_bf16_warp")
            and x.shape[-1] <= NORME_WARP_H_MAX
            and x.numel() // x.shape[-1] >= NORME_WARP_MIN_LIGNES)


class LayerNorm(nn.Module):
    """LayerNorm classique (moyenne centrée, biais) — starcoder2."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor],
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        y = F.layer_norm(x32, (x32.shape[-1],), self.weight.to(torch.float32),
                         None if self.bias is None else self.bias.to(torch.float32), self.eps)
        return y.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE, avec les variantes de mise à l'échelle que les modèles actuels embarquent."""

    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0,
                 scaling: Optional[dict] = None, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32,
                 n_active: Optional[int] = None) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.scaling = scaling or {}
        inv_freq = self._build_inv_freq(device)
        if n_active is not None and n_active < inv_freq.shape[0]:
            # RoPE « proportionnel » (Gemma 4) : fréquences calculées sur la
            # tête entière, seules les n_active premières paires tournent
            inv_freq = inv_freq.clone()
            inv_freq[n_active:] = 0.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._dtype = dtype
        # M-RoPE (Qwen3-VL, engine/mrope) : axe (t, h, w) de chaque fréquence,
        # sections ENTRELACÉES ; None = RoPE 1-D, rien ne change.
        from .mrope import axes_interleaved, section_depuis
        section = section_depuis(self.scaling)
        self.mrope_section: Optional[list[int]] = section
        self._mrope_axes: Optional[torch.Tensor] = (
            None if section is None else axes_interleaved(section, inv_freq.shape[0]))

    def _build_inv_freq(self, device) -> torch.Tensor:
        dim = self.head_dim
        inv = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        rtype = str(self.scaling.get("rope_type") or self.scaling.get("type") or "")
        factor = float(self.scaling.get("factor", 1.0) or 1.0)
        if rtype in ("linear",):
            return inv / factor
        if rtype in ("dynamic", "ntk"):
            base = self.base * (factor ** (dim / (dim - 2)))
            return 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        if rtype in ("yarn",):
            # YaRN (DeepSeek-V2/V3) : interpolation des basses fréquences,
            # extrapolation des hautes, rampe entre beta_fast et beta_slow
            import math
            factor = max(factor, 1.0)
            orig = float(self.scaling.get("original_max_position_embeddings") or 4096)
            bf = float(self.scaling.get("beta_fast", 32)); bs = float(self.scaling.get("beta_slow", 1))
            def corr(nrot: float) -> float:
                return (dim * math.log(orig / (nrot * 2 * math.pi))) / (2 * math.log(self.base))
            low = max(math.floor(corr(bf)), 0); high = min(math.ceil(corr(bs)), dim - 1)
            rampe = (torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / max(high - low, 0.001)
            masque = 1.0 - rampe.clamp(0.0, 1.0)      # 1 = extrapolé (hautes fréquences)
            return (inv / factor) * (1.0 - masque) + inv * masque
        if rtype in ("llama3",):
            low = float(self.scaling.get("low_freq_factor", 1.0))
            high = float(self.scaling.get("high_freq_factor", 4.0))
            orig = float(self.scaling.get("original_max_position_embeddings", 8192))
            wavelen = 2 * math.pi / inv
            low_wl, high_wl = orig / low, orig / high
            smooth = ((orig / wavelen) - low) / max(1e-6, (high - low))
            smoothed = (1 - smooth) * (inv / factor) + smooth * inv
            inv = torch.where(wavelen > low_wl, inv / factor, inv)
            inv = torch.where((wavelen <= low_wl) & (wavelen >= high_wl), smoothed, inv)
            return inv
        return inv

    def _ensure(self, seq_len: int, device, dtype) -> None:
        """Tables cos/sin au dtype FIXE du module (`self._dtype`), jamais à celui du premier appelant.

        Pièce 213 : le dtype de l'appel (`dtype`) n'entre plus dans la table. Avant, la table prenait le dtype du
        PREMIER appelant et n'était reconstruite que sur la longueur : le RoPE principal, construit sans dtype
        (fp32 par défaut), était d'abord rempli par `tables32` (préfill fusionné) en fp32 NON arrondi, puis
        reconstruit en bf16 par `reserver` — le 1er lot d'un processus tournait sur d'autres tables que tous les
        suivants (contam2 de la 213 : Y seul ≠ Y après n'importe quel lot, défaut présent en 0.6.38). Le dtype
        est aussi une condition de reconstruction, par défense."""
        dtype = self._dtype
        if self._cos is not None and seq_len <= self._cache_len \
                and self._cos.device == device and self._cos.dtype == dtype:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # Un graphe capturé garde l'adresse des tables : les remplacer
            # pendant une capture — ou après, pour un graphe déjà capturé —
            # laisse ce graphe lire une page morte. Vu sur GLM-4.7-Flash (RoPE
            # du chemin MLA étendue de 1025 à 2049 lignes entre deux godets
            # précapturés). Le moteur réserve la taille finale avant toute
            # capture ; arriver ici est une erreur de programmation.
            raise RuntimeError(f"cache RoPE étendu à {seq_len} pendant une capture de "
                               f"graphe (réservé : {self._cache_len}) — appeler "
                               f"reserver() avant la capture")
        n = max(seq_len, 1024)
        t = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cache_len = n
        self._generation = getattr(self, "_generation", 0) + 1     # tables dérivées : reconstruites si périmées

    def reserver(self, max_pos: int, device, dtype) -> None:
        """Amène les tables à leur taille finale, hors de toute capture, pour
        qu'aucun graphe n'ait à les étendre."""
        self._ensure(max_pos, device, dtype)
        # REGLES § 6, remède C15 niveau 2 : les tables dérivées (fp32 pour le noyau de
        # préparation, demi-tables C15) sont matérialisées ICI, sans condition — avant,
        # `reserver` ne les réservait que si elles existaient déjà, et un chemin qui les
        # touche pour la première fois SOUS capture (b=1 au niveau 2 : tables32 n'est
        # lue que par le chemin de lot) les allouait dans le bassin privé du graphe.
        self.tables32(max_pos, device)
        self.tables_demi(None, device, dtype, max_pos)

    def tables_demi(self, positions: Optional[torch.Tensor], device, dtype,
                    max_pos: int) -> Optional[torch.Tensor]:
        """C15 (MLA, RoPE « norm » : paires 2i, 2i+1) : les demi-tables
        ``cos[:, :d/2]`` et ``sin[:, :d/2]`` empilées en ``[n, 2, d/2]``, pour
        UNE indexation par pas au lieu de deux — mêmes valeurs bf16 que
        ``forward`` (découpées après coup). ``positions`` None : construit
        seulement (``reserver``, hors capture) ; rend la ligne indexée sinon."""
        self._ensure(max_pos, device, dtype)
        cs = getattr(self, "_cs_demi", None)
        if (cs is None or cs.shape[0] != self._cos.shape[0] or cs.device != device or cs.dtype != self._cos.dtype
                or getattr(self, "_gen_demi", None) != self._generation):
            if cs is not None and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("demi-tables RoPE réallouées pendant une capture de "
                                   "graphe — appeler reserver() avant la capture")
            half = self._cos.shape[-1] // 2
            self._cs_demi = torch.stack((self._cos[:, :half], self._sin[:, :half]), dim=1).contiguous()
            self._gen_demi = self._generation
            cs = self._cs_demi
        return None if positions is None else cs[positions]

    def tables32(self, max_pos: int, device):
        """Tables cos/sin complètes en fp32, pour le noyau fusionné : lui
        indexe par position, ce qui épargne deux index_select et deux
        conversions par couche et par jeton."""
        self._ensure(max_pos, device, self._dtype)
        c32 = getattr(self, "_cos32", None)
        if (c32 is None or c32.shape[0] != self._cos.shape[0] or c32.device != device
                or getattr(self, "_gen32", None) != self._generation):
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                # première allocation comprise : un tenseur créé pendant une capture vit
                # dans le bassin du graphe et meurt avec la capture suivante (REGLES § 6)
                raise RuntimeError("tables RoPE fp32 allouées pendant une capture de "
                                   "graphe — appeler reserver() avant la capture")
            if os.environ.get("ACVRAM_TRACE_PTRS"):
                print(f"[rope32] (ré)allocation des tables fp32 : {None if c32 is None else tuple(c32.shape)}"
                      f" -> {tuple(self._cos.shape)} (max_pos demandé {max_pos}), pendant une capture : "
                      f"{torch.cuda.is_current_stream_capturing()}", flush=True)
            self._cos32 = self._cos.to(torch.float32).contiguous()
            self._sin32 = self._sin.to(torch.float32).contiguous()
            self._gen32 = self._generation
        return self._cos32, self._sin32

    def forward(self, positions: torch.Tensor, device, dtype,
                max_pos: Optional[int] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``positions.max()`` vit sur le GPU : le rapatrier synchronise tout le
        # flux, a chaque couche, a chaque jeton — 40 synchronisations par jeton
        # sur un modele de 40 couches. L'appelant connait deja la longueur de
        # contexte en Python ; qu'il la donne, et le cache s'etend sans jamais
        # attendre le GPU.
        if max_pos is None:
            max_pos = int(positions.max().item()) + 1 if positions.numel() else 1
        self._ensure(max_pos, device, dtype)
        if self._mrope_axes is not None and positions.dim() == 2:
            return self.forward_mrope(positions, device, dtype, max_pos)
        cos, sin = self._cos[positions], self._sin[positions]
        # table au dtype du module ; l'appelant d'un autre dtype reçoit une conversion, jamais une autre table
        return (cos, sin) if cos.dtype == dtype else (cos.to(dtype), sin.to(dtype))

    def forward_mrope(self, positions: torch.Tensor, device, dtype,
                      max_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """M-RoPE : ``positions`` [3, t] (axes t, h, w) → cos/sin [t, dim].

        Chaque fréquence lit la position de SON axe (sections entrelacées,
        engine/mrope.axes_interleaved) dans les tables 1-D — les mêmes valeurs
        que ``cos(p · inv_freq)`` de transformers (``Qwen3VLTextRotaryEmbedding``
        puis ``recomposition_frequencies``), et, quand les trois axes sont égaux
        (texte), exactement la ligne ``_cos[p]`` du chemin 1-D. ``max_pos`` majore
        les trois axes : une position M-RoPE ne dépasse jamais la position 1-D."""
        if self._mrope_axes is None:
            raise ValueError("positions [3, t] sur un RoPE sans mrope_section")
        if positions.shape[0] != 3:
            raise ValueError(f"positions M-RoPE attendues en [3, t], reçu {tuple(positions.shape)}")
        self._ensure(max_pos, device, dtype)
        axes = self._mrope_axes
        if axes.device != device:
            axes = self._mrope_axes = axes.to(device)
        half = self.inv_freq.shape[0]
        # idx[t, i] = position de l'axe de la fréquence i au jeton t
        idx = positions.to(device)[axes].t()                   # [t, half]
        col = torch.arange(half, device=device).unsqueeze(0)     # [1, half]
        cos = self._cos[idx, col]
        sin = self._sin[idx, col]
        if cos.dtype != dtype:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_fusee(q: torch.Tensor, k: torch.Tensor, rope, positions: torch.Tensor,
               max_pos: int, q_norm=None, k_norm=None):
    """Normalisation par tête et RoPE en un lancement, tables indexées dans le
    noyau. None si inéligible — l'appelant applique alors le chemin PyTorch,
    normalisations comprises."""
    from .. import kernels as _k
    ext = _k.get_extension()
    if (ext is None or not hasattr(ext, "rope_inplace") or not q.is_cuda
            or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16
            or q.dim() != 3 or k.dim() != 3):
        return None
    # les deux normes doivent être des RMSNorm bf16 de la taille d'une tête,
    # et partager epsilon (elles le font toujours dans les modèles servis)
    normes = [n for n in (q_norm, k_norm) if n is not None]
    if any(type(n).__name__ != "RMSNorm" or n.weight.dtype != torch.bfloat16
           for n in normes):
        return None
    if q_norm is not None and q_norm.weight.shape[-1] != q.shape[-1]:
        return None
    if k_norm is not None and k_norm.weight.shape[-1] != k.shape[-1]:
        return None
    if len(normes) == 2 and abs(q_norm.eps - k_norm.eps) > 1e-12:
        return None
    cos32, sin32 = rope.tables32(max_pos, q.device)
    d = cos32.shape[-1]
    if d % 2 or d > q.shape[-1] or d > k.shape[-1]:
        return None
    # tranches d'une projection empilée : le noyau suit leur pas, pas de copie
    if q.stride(2) != 1 or q.stride(1) != q.shape[2]:
        q = q.contiguous()
    if k.stride(2) != 1 or k.stride(1) != k.shape[2]:
        k = k.contiguous()
    qc, kc = q, k
    pos = positions if positions.dtype == torch.int64 else positions.to(torch.int64)
    ext.rope_inplace(qc, kc, cos32, sin32, pos,
                     None if q_norm is None else q_norm.weight,
                     None if k_norm is None else k_norm.weight,
                     normes[0].eps if normes else 1e-6)
    return qc, kc


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Chemin fusionné : un lancement au lieu d'une dizaine de petits noyaux
    # (tranches, négations, concaténations, produits) qui ne font qu'une
    # multiplication par élément. Le RoPE partiel est géré par le noyau, qui
    # ne touche que les cos.shape[-1] premières dimensions.
    from .. import kernels as _k
    ext = _k.get_extension()
    if (ext is not None and hasattr(ext, "rope_inplace") and q.is_cuda
            and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            and q.dim() == 3 and k.dim() == 3 and cos.dim() == 2
            and cos.shape[0] == q.shape[0] and cos.shape[-1] % 2 == 0
            and cos.shape[-1] <= q.shape[-1] and cos.shape[-1] <= k.shape[-1]):
        qc = q if q.is_contiguous() else q.contiguous()
        kc = k if k.is_contiguous() else k.contiguous()
        ext.rope_inplace(qc, kc, cos.float().contiguous(), sin.float().contiguous())
        return qc, kc
    if cos.shape[-1] < q.shape[-1]:
        # RoPE partiel (qwen3-next : 64 dims tournées sur 256) : la tranche
        # au-delà passe telle quelle.
        d = cos.shape[-1]
        q1, q2 = q[..., :d], q[..., d:]
        k1, k2 = k[..., :d], k[..., d:]
        q1, k1 = apply_rope(q1, k1, cos, sin)
        return torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)
    """``q`` et ``k`` valent [jetons, têtes, dim] ; cos et sin valent [jetons, dim]."""
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Étend les têtes clé/valeur de la GQA au nombre de têtes de requête."""
    if n_rep == 1:
        return x
    t, h, d = x.shape
    return x.unsqueeze(2).expand(t, h, n_rep, d).reshape(t, h * n_rep, d)


def causal_mask(q_len: int, kv_len: int, q_offset: int, device,
                dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Masque pour un bloc de requêtes commençant à la position absolue ``q_offset``.

    ``F.scaled_dot_product_attention(is_causal=True)`` aligne le triangle en
    *haut à gauche*, ce qui n'est correct que si la requête couvre toute la
    séquence. Dès qu'un prefill est découpé — ou qu'un préfixe est servi depuis
    le cache et que seule la queue est précalculée — le bloc de requêtes commence
    en cours de séquence et le drapeau intégré masque silencieusement les
    mauvaises cellules. Le cache de préfixe et le prefill par morceaux dépendent
    tous deux de ce détail, d'où un masque construit explicitement dès que la
    requête est décalée.
    """
    if q_len == 1:
        return None                       # le décodage attend sur tout
    if q_offset == 0 and q_len == kv_len:
        return None                       # le drapeau causal intégré est correct
    rows = torch.arange(q_offset, q_offset + q_len, device=device).unsqueeze(1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    allowed = cols <= rows
    mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


def masque_images(q_len: int, kv_len: int, q_offset: int,
                  images: Sequence[tuple[int, int]], device) -> Optional[torch.Tensor]:
    """Cellules (requête, clé) que les plages image ouvrent EN PLUS du causal.

    Gemma 4 (multimodal P1, poste7-go-multimodal-organisation-20-09 § 2) : les
    jetons d'une même image [debut, fin) se voient tous entre eux, y compris
    vers l'avant ; deux images distinctes ne s'ouvrent pas l'une à l'autre.
    Positions absolues dans la séquence ; les clés sont les positions
    [0, kv_len), les requêtes [q_offset, q_offset + q_len). Rend None quand
    aucune plage ne touche le bloc de requêtes (le masque causal suffit).

    Prefill par morceaux : une requête image dont l'image se prolonge AU-DELÀ
    des clés existantes (fin > kv_len) ne peut pas voir ses jetons à venir ;
    le masque serait faux en silence — refus nommé, c'est au moteur de ne pas
    couper un morceau dans une image (`Engine._eviter_coupe_image`).
    """
    plages = [(int(d), int(f)) for d, f in images
              if int(d) < q_offset + q_len and int(f) > q_offset]
    if not plages:
        return None
    for d, f in plages:
        if f > kv_len:
            raise RuntimeError(
                f"plage image [{d}, {f}) coupée par un morceau de prefill : clés "
                f"jusqu'à {kv_len} seulement, requêtes [{q_offset}, {q_offset + q_len})")
    rows = torch.arange(q_offset, q_offset + q_len, device=device).unsqueeze(1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    ouvert = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
    for d, f in plages:
        ouvert |= ((rows >= d) & (rows < f)) & ((cols >= d) & (cols < f))
    return ouvert


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True, scale: Optional[float] = None,
              q_offset: int = 0, window: int = 0, n_rep: int = 1,
              images: Optional[Sequence[tuple[int, int]]] = None) -> torch.Tensor:
    """Attention par produit scalaire normalisé sur des tenseurs ``[jetons, têtes, dim]``.

    Délègue au SDPA de PyTorch, qui choisit FlashAttention sur tout GPU qui le
    gère. Les transpositions sont des vues, pas des copies. ``n_rep`` > 1
    (C15-prefill, fusion « attn ») : k et v portent les têtes KV seules et SDPA
    les diffuse (``enable_gqa``) — la tête h lit la tête KV h // n_rep, la même
    que `repeat_kv` matérialisait (deux copies de [t, têtes, d] par couche).
    """
    qh = q.transpose(0, 1).unsqueeze(0)          # [1, heads, tq, dim]
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    gqa = {"enable_gqa": True} if n_rep > 1 else {}
    q_len, kv_len = q.shape[0], k.shape[0]
    ouvert = (masque_images(q_len, kv_len, q_offset, images, q.device)
              if images and causal else None)
    if window > 0:
        # fenêtre glissante : chaque requête ne voit que les `window` derniers
        qpos = torch.arange(q_len, device=q.device) + q_offset
        kpos = torch.arange(kv_len, device=q.device)
        mask = ((kpos[None, :] <= qpos[:, None])
                & (kpos[None, :] > qpos[:, None] - window))
        if ouvert is not None:
            mask = mask | ouvert           # HF Gemma : or_mask sur la fenêtre aussi
    elif ouvert is not None:
        rows = torch.arange(q_offset, q_offset + q_len, device=q.device).unsqueeze(1)
        cols = torch.arange(kv_len, device=q.device).unsqueeze(0)
        allowed = (cols <= rows) | ouvert
        mask = torch.zeros(q_len, kv_len, device=q.device, dtype=q.dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
    else:
        mask = causal_mask(q_len, kv_len, q_offset, q.device, q.dtype) if causal else None
    if mask is not None:
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale, **gqa)
    else:
        use_causal = bool(causal and q_len > 1 and q_offset == 0
                          and q_len == kv_len)
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=use_causal,
                                             scale=scale, **gqa)
    return out.squeeze(0).transpose(0, 1).contiguous()


def batched_decode_attention(q: torch.Tensor, keys: list[torch.Tensor],
                             values: list[torch.Tensor], n_rep: int,
                             scale: float) -> torch.Tensor:
    """Un seul appel SDPA pour tout un lot de décodage, au lieu d'un par séquence.

    Les séquences ont des longueurs de contexte différentes : les clés sont donc
    complétées à droite jusqu'à la plus longue, et ce remplissage est masqué.
    Cela coûte ``lot × (longueur_max − longueur)`` emplacements de clé gâchés ;
    en face, la boucle Python disparaît et le GPU ne voit qu'un lancement au
    lieu de ``lot``. À un lot de 16, le seul surcoût de lancement était déjà le
    plus grand des deux.
    """
    b = len(keys)
    lens = [kk.shape[0] for kk in keys]
    max_len = max(lens)
    hq, d = q.shape[1], q.shape[2]
    hkv = keys[0].shape[1]
    device, dtype = q.device, q.dtype

    kpad = torch.zeros(b, max_len, hkv, d, device=device, dtype=dtype)
    vpad = torch.zeros(b, max_len, hkv, d, device=device, dtype=dtype)
    for i, (kk, vv) in enumerate(zip(keys, values)):
        kpad[i, : lens[i]] = kk
        vpad[i, : lens[i]] = vv

    kh = repeat_kv_batched(kpad, n_rep).permute(0, 2, 1, 3)   # [b, hq, s, d]
    vh = repeat_kv_batched(vpad, n_rep).permute(0, 2, 1, 3)
    qh = q.unsqueeze(2)                                       # [b, hq, 1, d]

    valid = torch.arange(max_len, device=device).unsqueeze(0) < \
        torch.tensor(lens, device=device).unsqueeze(1)        # [b, s]
    mask = torch.zeros(b, 1, 1, max_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~valid[:, None, None, :], float("-inf"))

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    return out.squeeze(2)                                     # [b, hq, d]


def decode_attention_fixed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           seq_lens: torch.Tensor, n_rep: int,
                           scale: float, window: int = 0) -> torch.Tensor:
    """Attention de décodage à formes fixes, pour la capture en graphe CUDA.

    ``q`` vaut ``[lot, têtes, dim]`` (un jeton par séquence), ``k``/``v``
    ``[lot, S, têtes_kv, dim]`` avec ``S`` fixé par le godet de capture, et
    ``seq_lens`` est un tenseur — jamais une liste Python : la frontière vit
    sur le GPU, seul le masque en dépend.
    """
    b, s = k.shape[0], k.shape[1]
    # enable_gqa laisse SDPA diffuser les têtes KV vers les têtes de requête :
    # l'ancien repeat_kv passait par reshape-sur-expand, qui matérialise une
    # copie ×n_rep de K et de V à chaque couche, à chaque pas.
    kh = k.permute(0, 2, 1, 3)                                # [b, hkv, S, d]
    vh = v.permute(0, 2, 1, 3)
    pos = torch.arange(s, device=q.device)[None, :]
    mask = pos < seq_lens[:, None]
    if window > 0:
        mask = mask & (pos >= seq_lens[:, None] - window)
    mask = mask.view(b, 1, 1, s)
    out = F.scaled_dot_product_attention(
        q.unsqueeze(2), kh, vh,
        attn_mask=mask, scale=scale, enable_gqa=(n_rep > 1))  # [b, hq, 1, d]
    return out.squeeze(2)                                     # [b, hq, d]


def repeat_kv_batched(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[lot, s, têtes_kv, d]`` -> ``[lot, s, têtes_kv × n_rep, d]``."""
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x.unsqueeze(3).expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)


# Hauteur de bloc du noyau nvfp4_gemv (ROWS_PER_BLOCK dans le .cu).
ROWS_PAR_BLOC = 4


# Chaque refus de fusion se comptait a zero : la fonction rendait None et
# personne ne savait si les piles avaient ete construites. Le 9/09/2026, une
# session a du envisager de compter les noyaux sous `ncu` pour repondre a
# « les fusions s appliquent-elles au nvfp4 ? » — une question que le code
# pouvait dire lui-meme. Une absence n est un resultat que si l instrument
# pouvait rendre autre chose.
_REFUS_FUSION: dict[str, int] = {}


def _sans_fusion() -> bool:
    """Echappement de MESURE, commun aux quatre empileurs.

    `ACVRAM_SANS_FUSION_BF16` n'existait que pour `stack_plain_linears` : le
    gain de la fusion etait donc mesurable en bf16 et NULLE PART AILLEURS. D'ou
    le +2,60 % du 9/09, mesure en bf16 pur — ou 100 % des groupes fusionnent —
    et transporte a tort sur un int8 calibre, ou `_scaler_commun` n'en accepte
    que 7,8 % (5 groupes sur 64 sur Llama-2-7b-int8, parce que la recherche AWQ
    choisit un exposant `alpha` par tenseur).

    Sans interrupteur, l'A/B demanderait deux versions du code — et comparerait
    autre chose que la fusion.
    """
    return bool(os.environ.get("ACVRAM_SANS_FUSION")
                or os.environ.get("ACVRAM_SANS_FUSION_BF16"))


def _scaler_commun(lins: list):
    """Rend (utilisable, scaler a porter par la pile).

    Un scaler s applique a l ENTREE. Des projections d un meme groupe lisent
    la meme entree : si leurs scalers sont identiques — a fortiori s ils sont
    tous l identite — la pile peut porter ce scaler unique et le resultat est
    exactement celui des appels separes.

    La fusion refusait sur `l.scaler is not None`, c est-a-dire sur
    l EXISTENCE de l objet, alors que l execution teste son CONTENU
    (`not self.scaler.is_identity`, forward l.479). Un scaler identite —
    scale None et hadamard_block 0, ce qu une conversion sans calibration
    produit pour TOUS les tenseurs — bloquait donc les 96 fusions d un modele
    qui n avait aucune mise a l echelle a concilier.
    """
    scs = [getattr(l, "scaler", None) for l in lins]
    vifs = [s for s in scs if s is not None and not s.is_identity]
    if not vifs:
        return True, None                      # tous identite : rien a porter
    if len(vifs) != len(scs):
        return False, None                     # certains actifs, d autres non
    tete = vifs[0]
    for s in vifs[1:]:
        if s.hadamard_block != tete.hadamard_block:
            return False, None
        if (s.scale is None) != (tete.scale is None):
            return False, None
        if s.scale is not None and not torch.equal(s.scale, tete.scale):
            return False, None
    return True, tete                           # tous egaux : la pile le porte


# Pendant l'exploration des paires de la fusion PARTIELLE, trois tentatives
# ont lieu par groupe. Les compter toutes ferait dire au bilan « 240 refus »
# la ou il y a 80 groupes — l'unite du compteur changerait selon le chemin,
# et un lecteur y verrait un nombre de groupes. Le compteur se tait donc
# pendant l'exploration ; seul le verdict du groupe est enregistre.
_EXPLORATION = False


def explorer_sans_compter():
    """Contexte ou les refus ne sont pas comptes — voir `_EXPLORATION`."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        global _EXPLORATION
        avant, _EXPLORATION = _EXPLORATION, True
        try:
            yield
        finally:
            _EXPLORATION = avant
    return _ctx()


def _refus_fusion(raison: str) -> None:
    """Enregistre pourquoi une fusion NVFP4 n a pas eu lieu, et rend None."""
    if not _EXPLORATION:
        _REFUS_FUSION[raison] = _REFUS_FUSION.get(raison, 0) + 1
    return None


def bilan_fusion_nvfp4() -> dict[str, int]:
    """Refus par raison depuis le chargement. Vide = aucune fusion refusee."""
    return dict(_REFUS_FUSION)


def stack_nvfp4_linears(lins: list) -> Optional["QuantLinear"]:
    """Empile des QuantLinear NVFP4 de même entrée en un seul.

    Trois projections q, k et v lisent la même activation mais lancent trois
    GEMV, chacune payant sa latence : mesuré sur Qwen3-Coder-30B, 28,6 µs à
    trois contre 9,6 une fois empilées, soit 0,9 ms par jeton sur 48 couches.
    Leurs échelles globales diffèrent et les ramener à une seule les ferait
    passer par un arrondi e4m3 à 6 % — le noyau accepte donc une échelle par
    ligne de sortie, que chaque segment garde intacte.
    """
    import os
    if os.environ.get("ACVRAM_FUSION_NVFP4") == "0":     # témoin de mesure
        return _refus_fusion("temoin ACVRAM_FUSION_NVFP4=0")
    from ..quant.nvfp4 import NVFP4Tensor
    if _sans_fusion():
        return None
    ts = [getattr(l, "qweight", None) for l in lins]
    if not all(isinstance(t, NVFP4Tensor) for t in ts):
        return _refus_fusion("un des poids n est pas NVFP4")
    if len({(t.padded_in, t.qweight.shape[1], t.block_scale.shape[1]) for t in ts}) != 1:
        return _refus_fusion("entrees de tailles differentes")
    # Un biais s applique a la SORTIE : le concatener sur l axe 0 est exact,
    # exactement comme les lignes de poids. Les refuser coutait les 48
    # attentions de Qwen2.5 — q, k et v y portent un biais — soit 96 GEMV par
    # pas, alors que `stack_plain_linears` les empile deja en bf16.
    biais = [l.bias for l in lins]
    if any((b is None) != (biais[0] is None) for b in biais):
        return _refus_fusion("biais present sur une partie du groupe")
    if any(l.streamed is not None for l in lins):
        return _refus_fusion("poids en flux")
    ok_scaler, scaler_pile = _scaler_commun(lins)
    if not ok_scaler:
        return _refus_fusion("scalers differents entre projections")
    if any(getattr(t, "global_scale_rows", None) is not None for t in ts):
        return _refus_fusion("deja empile")
    # Le noyau lit l'échelle de la première ligne de chaque bloc : les segments
    # doivent commencer sur un multiple de la hauteur de bloc.
    if any(t.qweight.shape[0] % ROWS_PAR_BLOC for t in ts[:-1]):
        return _refus_fusion(f"segment non multiple de {ROWS_PAR_BLOC} lignes")
    lignes = torch.cat([
        torch.full((t.qweight.shape[0],), t.global_scale_float(),
                   dtype=torch.float32, device=t.qweight.device) for t in ts])
    fus = NVFP4Tensor(
        torch.cat([t.qweight for t in ts]).contiguous(),
        torch.cat([t.block_scale for t in ts]).contiguous(),
        ts[0].global_scale,
        (sum(t.shape[0] for t in ts), ts[0].shape[1]),
        ts[0].padded_in,
        global_scale_rows=lignes.contiguous())
    # Les originaux deviennent des vues de la pile : le prefill continue de les
    # appeler séparément, mais plus un octet de VRAM n'est dupliqué — sur un
    # dense de 27 milliards de paramètres, gate et up recopiés coûteraient
    # plusieurs gibioctets pour rien.
    d = 0
    for l, t in zip(lins, ts):
        n = t.qweight.shape[0]
        l.qweight = NVFP4Tensor(fus.qweight[d:d + n], fus.block_scale[d:d + n],
                                t.global_scale, t.shape, t.padded_in)
        d += n
    pbiais = torch.cat(biais) if biais[0] is not None else None
    if pbiais is not None:
        # Les originaux deviennent des vues du biais empile, comme les poids :
        # le prefill continue de les appeler separement sans qu un octet soit
        # duplique.
        o = 0
        for l, b in zip(lins, biais):
            l.bias = pbiais.narrow(0, o, b.shape[0]); o += b.shape[0]
    return QuantLinear(fus, bias=pbiais, scaler=scaler_pile)


def stack_plain_linears(lins: list) -> Optional["QuantLinear"]:
    """Empile des poids bf16 de même entrée en un seul, SANS les dupliquer.

    Les modèles bf16 purs ne passaient par aucune des deux fusions existantes
    (int8, nvfp4) : ni q/k/v ni gate/up n'y étaient empilés, soit **sept GEMV
    par couche** au décodage là où llama.cpp en lance six (mesuré le 9 septembre
    2026 sous ncu : 337 noyaux GEMV par pas contre 289).

    Empiler naïvement doublerait les poids concernés — q, k, v, gate et up font
    environ 70 % d'un modèle dense, soit 13,6 Gio de copie sur Qwen2.5-14B, que
    la carte n'a pas. Les originaux sont donc **remplacés par des vues** dans
    l'empilement : ``narrow`` sur la dimension 0 d'un tenseur contigu reste
    contigu, le chemin de prefill continue de fonctionner, et il ne survit
    qu'une seule copie des poids. Le pic transitoire est celui d'une couche.
    """
    from ..quant.formats import PlainTensor
    # Echappement : sert a mesurer le gain de la fusion sur la meme binaire, et
    # a comparer les jetons emis avec et sans elle. Sans interrupteur, l'A/B
    # demanderait deux versions du code -- et comparerait autre chose.
    if _sans_fusion():
        return None
    ts = [getattr(l, "qweight", None) for l in lins]
    if not all(isinstance(t, PlainTensor) for t in ts):
        return None
    if len({t.weight.shape[1] for t in ts}) != 1:
        return None
    if len({(t.weight.dtype, str(t.weight.device)) for t in ts}) != 1:
        return None
    if any(l.scaler is not None or l.streamed is not None for l in lins):
        return None
    # Les biais s'empilent comme les poids -- Qwen2.5 en porte sur q, k et v, et
    # les refuser laissait ses 48 attentions sur le chemin a trois GEMV. Tout ou
    # rien : un empilement partiel decalerait les lignes de sortie.
    biais = [l.bias for l in lins]
    if any(b is not None for b in biais) and any(b is None for b in biais):
        return None
    # ORDRE DES ALLOCATIONS : liberer AVANT d'allouer, pas l'inverse.
    #
    # `torch.cat` alloue le tenseur concatene pendant que les deux sources
    # vivent encore : 0,355 Gio de pic par fusion, 95 fois sur un 14B. Les
    # blocs liberes ensuite retombent dans le cache de l'allocateur, entrelaces
    # avec des blocs vivants, donc irrecuperables par `empty_cache` -- 1,28 Gio
    # de reserve non alloue restaient apres purge, et le banc, qui dimensionne
    # ses caches plus largement que le chargement nu, tombait en OOM sur une
    # demande de 2 Mio.
    #
    # On passe donc par la RAM hote pour les gros tenseurs : descendre, liberer
    # la VRAM, allouer le concatene, remonter. Le pic VRAM devient NUL -- le
    # bloc libere est exactement celui que l'allocateur reutilise -- au prix
    # d'un aller-retour PCIe au chargement. Les petits (q/k/v) passent par
    # `cat` : leur pic ne fragmente pas et le transfert coute plus qu'il ne
    # rapporte.
    SEUIL_HOTE = 64 * 2 ** 20
    octets = sum(t.weight.numel() * t.weight.element_size() for t in ts)

    def _concatener(tenseurs):
        if octets < SEUIL_HOTE or not tenseurs[0].is_cuda:
            return torch.cat(tenseurs)
        hote = [x.to("cpu", copy=True) for x in tenseurs]
        dev, dt = tenseurs[0].device, tenseurs[0].dtype
        formes = [x.shape for x in tenseurs]
        del tenseurs[:]                      # plus aucune reference VRAM ici
        for l, t in zip(lins, ts):
            t.weight = None
            l.qweight = None
        plein = torch.empty((sum(f[0] for f in formes),) + tuple(formes[0][1:]),
                            dtype=dt, device=dev)
        o = 0
        for x, f in zip(hote, formes):
            plein.narrow(0, o, f[0]).copy_(x)
            o += f[0]
        return plein

    poids_src = [t.weight for t in ts]
    formes_src = [(t.weight.shape[0], t.shape, t.format) for t in ts]
    plat = _concatener(poids_src)
    pbiais = torch.cat(biais) if biais[0] is not None else None
    off = 0
    for l, (n, forme, fmt) in zip(lins, formes_src):
        l.qweight = PlainTensor(plat.narrow(0, off, n), forme, fmt)
        if pbiais is not None:
            l.bias = pbiais.narrow(0, off, n)
        off += n
    return QuantLinear(PlainTensor(plat, tuple(plat.shape), formes_src[0][2]),
                       bias=pbiais)


def stack_int8_linears(lins: list) -> Optional["QuantLinear"]:
    """Empile des QuantLinear INT8 de même entrée en un seul (lignes
    concaténées) : une GEMV au lieu de n au décodage. None si inapplicable."""
    from ..quant.formats import INT8Tensor
    if _sans_fusion():
        return None
    ts = [getattr(l, "qweight", None) for l in lins]
    if not all(isinstance(t, INT8Tensor) for t in ts):
        return None
    if len({(t.qweight.shape[1], t.group_size) for t in ts}) != 1:
        return None
    if any(l.streamed is not None for l in lins):
        return None
    # Un biais s'applique a la SORTIE : le concatener sur l'axe 0 est exact,
    # exactement comme les lignes de poids. Le refuser ici coutait TOUS les
    # groupes tout-int8 des modeles a biais — Qwen2.5 en porte sur q, k et v —
    # alors que `stack_nvfp4_linears` les accepte depuis ce matin avec la meme
    # justification, 150 lignes plus haut. Deux fonctions voisines, deux
    # regles opposees sur le meme objet : releve en dressant la
    # table de verite des quatre cas (int8/nvfp4 x avec/sans biais).
    biais = [l.bias for l in lins]
    if any((b is None) != (biais[0] is None) for b in biais):
        return None
    ok_scaler, scaler_pile = _scaler_commun(lins)
    if not ok_scaler:
        return None
    t = INT8Tensor(torch.cat([t.qweight for t in ts]).contiguous(),
                   torch.cat([t.scales for t in ts]).contiguous(),
                   torch.cat([t.zeros for t in ts]).contiguous(),
                   ts[0].group_size,
                   (sum(t.shape[0] for t in ts), ts[0].shape[1]))
    # Pièce 139 : la marque « préfill bf16 » (int8 d'origine fp8, loader._build_quant) suit la pile et ses vues —
    # perdue ici, la pile gate_up des couches 56-63 reprenait la copie signée du chemin cublas (OOM du 24/09 08:26)
    marque = any(src.__dict__.get("prefill_bf16") for src in ts)
    if marque:
        t.__dict__["prefill_bf16"] = True
    # LES ORIGINAUX DEVIENNENT DES VUES DE LA PILE. Le commentaire ci-dessous
    # affirmait « comme pour les poids » alors que seul le BIAIS etait repointe :
    # les poids restaient dupliques, et le `forward` garde les deux chemins
    # (fusion sous SEUIL_FUSION, projections separees au-dela), donc les deux
    # copies survivaient au chargement. Mesure du 10/09 sur Llama-2-7b-int8 :
    # model.nbytes rendait 7 223 386 112 octets pour 6 761 930 752 sur disque,
    # soit 461 455 360 de trop — exactement 5 x 92 291 072, la taille de cinq
    # gate_up fusionnes. 6,4 % de VRAM pour +2,6 % de debit, sur la grandeur
    # meme qui declenche l'exil d'une couche (61 a 70 % du debit).
    #
    # La disposition le permettait depuis toujours : `cat` sur l'axe 0 rend un
    # tenseur contigu dont chaque tranche `[d:d+n]` est contigue elle aussi.
    # `stack_plain_linears` et `stack_nvfp4_linears` le faisaient deja, avec
    # l'argument ecrit ; seuls int8 et int4_awq ne l'avaient pas.
    d = 0
    for l, src in zip(lins, ts):
        n = src.qweight.shape[0]
        l.qweight = INT8Tensor(t.qweight[d:d + n], t.scales[d:d + n],
                               t.zeros[d:d + n], src.group_size, src.shape)
        if marque:
            l.qweight.__dict__["prefill_bf16"] = True
        d += n
    pbiais = torch.cat(biais) if biais[0] is not None else None
    if pbiais is not None:
        o = 0
        for l, b in zip(lins, biais):
            l.bias = pbiais.narrow(0, o, b.shape[0]); o += b.shape[0]
    return QuantLinear(t, bias=pbiais, scaler=scaler_pile)


def stack_int4_awq_linears(lins: list) -> Optional["QuantLinear"]:
    """Empile des QuantLinear INT4-AWQ de meme entree en un seul.

    Le format n'a PAS eu besoin de l'equivalent de `global_scale_rows` :
    `INT4Tensor.scales` est deja `[out, in//group]`, une echelle par ligne de
    sortie et par groupe. Il n'y a donc aucune echelle globale a concilier
    entre les segments, et la concatenation sur l'axe 0 est exacte — le
    packing uint4 de `qweight` et de `zeros` porte sur l'axe 1, jamais entre
    deux lignes de sortie. Mesure du 9/09/2026 sur (256|64|64) x 256 : une
    pile contre trois appels separes, ecart 0,000e+00, sur GPU comme sur
    processeur.

    Sans cette fonction, `fuse()` n'essayait qu'int8, nvfp4 et bf16 : 135
    groupes du parc, parfaitement homogenes en int4_awq, restaient decoupes
    en GEMV separees faute d'empileur — 64 q/k/v et 71 gate/up, soit 199
    lancements par pas de decodage.
    """
    from ..quant.int4 import INT4Tensor
    if _sans_fusion():
        return None
    ts = [getattr(l, "qweight", None) for l in lins]
    if not all(isinstance(t, INT4Tensor) for t in ts):
        return _refus_fusion("un des poids n est pas INT4-AWQ")
    # `padded_in` en plus de l'entree et du groupe : deux tenseurs de meme
    # entree logique peuvent avoir ete rembourres differemment, et le noyau
    # lit la largeur rembourree.
    if len({(t.padded_in, t.qweight.shape[1], t.group_size) for t in ts}) != 1:
        return _refus_fusion("entrees de tailles differentes")
    if any(l.streamed is not None for l in lins):
        return _refus_fusion("poids en flux")
    biais = [l.bias for l in lins]
    if any((b is None) != (biais[0] is None) for b in biais):
        return _refus_fusion("biais present sur une partie du groupe")
    ok_scaler, scaler_pile = _scaler_commun(lins)
    if not ok_scaler:
        return _refus_fusion("scalers differents entre projections")
    t = INT4Tensor(torch.cat([t.qweight for t in ts]).contiguous(),
                   torch.cat([t.scales for t in ts]).contiguous(),
                   torch.cat([t.zeros for t in ts]).contiguous(),
                   ts[0].group_size,
                   (sum(t.shape[0] for t in ts), ts[0].shape[1]),
                   ts[0].padded_in)
    # Vues, meme raison qu'en int8 ci-dessous : le prefill continue d'appeler
    # les projections separement, sans qu'un octet soit duplique.
    d = 0
    for l, src in zip(lins, ts):
        n = src.qweight.shape[0]
        l.qweight = INT4Tensor(t.qweight[d:d + n], t.scales[d:d + n],
                               t.zeros[d:d + n], src.group_size, src.shape,
                               src.padded_in)
        d += n
    pbiais = torch.cat(biais) if biais[0] is not None else None
    if pbiais is not None:
        o = 0
        for l, b in zip(lins, biais):
            l.bias = pbiais.narrow(0, o, b.shape[0]); o += b.shape[0]
    return QuantLinear(t, bias=pbiais, scaler=scaler_pile)
