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
from typing import Any, Optional

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
        plat = torch.empty(max(off, 1), dtype=torch.uint8).pin_memory()
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

    def __init__(self, device: torch.device, n_slots: int) -> None:
        self.device = device
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
        i = jeu["prochain"]
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
        if self.streamed is not None and self.streamed.pool is None:
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
            h, x = ext.rmsnorm_bf16(y, norme.weight, norme.eps, residu, mult)
            return x, h
    x = residu + (y if mult == 1.0 else y * mult)
    return x, norme(x)


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
        if self._cos is not None and seq_len <= self._cache_len \
                and self._cos.device == device:
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

    def reserver(self, max_pos: int, device, dtype) -> None:
        """Amène les tables à leur taille finale, hors de toute capture, pour
        qu'aucun graphe n'ait à les étendre."""
        self._ensure(max_pos, device, dtype)
        if getattr(self, "_cos32", None) is not None:
            self.tables32(max_pos, device)

    def tables32(self, max_pos: int, device):
        """Tables cos/sin complètes en fp32, pour le noyau fusionné : lui
        indexe par position, ce qui épargne deux index_select et deux
        conversions par couche et par jeton."""
        self._ensure(max_pos, device, self._dtype)
        c32 = getattr(self, "_cos32", None)
        if c32 is None or c32.shape[0] != self._cos.shape[0] or c32.device != device:
            if c32 is not None and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("tables RoPE fp32 réallouées pendant une capture de "
                                   "graphe — appeler reserver() avant la capture")
            if os.environ.get("ACVRAM_TRACE_PTRS"):
                print(f"[rope32] (ré)allocation des tables fp32 : {None if c32 is None else tuple(c32.shape)}"
                      f" -> {tuple(self._cos.shape)} (max_pos demandé {max_pos}), pendant une capture : "
                      f"{torch.cuda.is_current_stream_capturing()}", flush=True)
            self._cos32 = self._cos.to(torch.float32).contiguous()
            self._sin32 = self._sin.to(torch.float32).contiguous()
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
        return self._cos[positions], self._sin[positions]


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


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True, scale: Optional[float] = None,
              q_offset: int = 0, window: int = 0) -> torch.Tensor:
    """Attention par produit scalaire normalisé sur des tenseurs ``[jetons, têtes, dim]``.

    Délègue au SDPA de PyTorch, qui choisit FlashAttention sur tout GPU qui le
    gère. Les transpositions sont des vues, pas des copies.
    """
    qh = q.transpose(0, 1).unsqueeze(0)          # [1, heads, tq, dim]
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    q_len, kv_len = q.shape[0], k.shape[0]
    if window > 0:
        # fenêtre glissante : chaque requête ne voit que les `window` derniers
        qpos = torch.arange(q_len, device=q.device) + q_offset
        kpos = torch.arange(kv_len, device=q.device)
        mask = ((kpos[None, :] <= qpos[:, None])
                & (kpos[None, :] > qpos[:, None] - window))
    else:
        mask = causal_mask(q_len, kv_len, q_offset, q.device, q.dtype) if causal else None
    if mask is not None:
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    else:
        use_causal = bool(causal and q_len > 1 and q_offset == 0
                          and q_len == kv_len)
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=use_causal,
                                             scale=scale)
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
        return None
    from ..quant.nvfp4 import NVFP4Tensor
    ts = [getattr(l, "qweight", None) for l in lins]
    if not all(isinstance(t, NVFP4Tensor) for t in ts):
        return None
    if len({(t.padded_in, t.qweight.shape[1], t.block_scale.shape[1]) for t in ts}) != 1:
        return None
    if any(l.bias is not None or l.scaler is not None or l.streamed is not None
           for l in lins):
        return None
    if any(getattr(t, "global_scale_rows", None) is not None for t in ts):
        return None                       # déjà empilé : on n'empile pas deux fois
    # Le noyau lit l'échelle de la première ligne de chaque bloc : les segments
    # doivent commencer sur un multiple de la hauteur de bloc.
    if any(t.qweight.shape[0] % ROWS_PAR_BLOC for t in ts[:-1]):
        return None
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
    return QuantLinear(fus)


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
    if os.environ.get("ACVRAM_SANS_FUSION_BF16"):
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
    plat = torch.cat([t.weight for t in ts]).contiguous()
    pbiais = None
    if biais[0] is not None:
        pbiais = torch.cat([b for b in biais]).contiguous()
    off = 0
    for l, t in zip(lins, ts):
        n = t.weight.shape[0]
        l.qweight = PlainTensor(plat.narrow(0, off, n), t.shape, t.format)
        if pbiais is not None:
            l.bias = pbiais.narrow(0, off, n)
        off += n
    return QuantLinear(PlainTensor(plat, tuple(plat.shape), ts[0].format),
                       bias=pbiais)


def stack_int8_linears(lins: list) -> Optional["QuantLinear"]:
    """Empile des QuantLinear INT8 de même entrée en un seul (lignes
    concaténées) : une GEMV au lieu de n au décodage. None si inapplicable."""
    from ..quant.formats import INT8Tensor
    ts = [getattr(l, "qweight", None) for l in lins]
    if not all(isinstance(t, INT8Tensor) for t in ts):
        return None
    if len({(t.qweight.shape[1], t.group_size) for t in ts}) != 1:
        return None
    if any(l.bias is not None or l.scaler is not None or l.streamed is not None
           for l in lins):
        return None
    t = INT8Tensor(torch.cat([t.qweight for t in ts]).contiguous(),
                   torch.cat([t.scales for t in ts]).contiguous(),
                   torch.cat([t.zeros for t in ts]).contiguous(),
                   ts[0].group_size,
                   (sum(t.shape[0] for t in ts), ts[0].shape[1]))
    return QuantLinear(t)
