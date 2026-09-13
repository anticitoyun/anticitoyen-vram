"""Graphes CUDA pour le pas de décodage.

Le profil de Qwen3-14B montrait ~68 ms de Python par jeton pour ~30 ms de
calcul GPU : plus de la moitié du temps partait en lancements de noyaux et en
navette d'objets Python. Un graphe CUDA capture une fois la séquence complète
des noyaux d'un pas de décodage, puis la rejoue pour le prix d'un seul appel.

Ce que la capture exige — et comment on l'obtient :

* **Des formes fixes.** Un graphe est capturé par *godet* ``(lot, blocs KV)`` :
  le lot est pris tel quel (un serveur local décode presque toujours à 1), le
  nombre de blocs est arrondi à la puissance de deux supérieure. La table de
  blocs est complétée avec le bloc 0 — lu pour rien, masqué par ``seq_lens``.
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


def godet_hybride(b_reel: int, max_slots: int) -> Optional[int]:
    """Le godet à utiliser pour un lot hybride, ou ``None`` s'il refuse.

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
    return min(bucket_batch(b_reel), max_slots)
from .layers import QuantLinear
from .model import DecoderLayerGDN, ForwardBatch, MoEBlock

from .mla import MLA_BUCKET, godet_mla   # un graphe par palier de cache latent

__all__ = ["GraphRunner", "bucket_batch", "godet_hybride"]

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

    def __init__(self, model, max_model_len: int) -> None:
        self.model = model
        self.max_model_len = max_model_len
        self.device: Optional[torch.device] = None
        self.graphs: dict[tuple[int, int], dict] = {}
        self._pool = None
        self.paged_ok = False
        self.hybrid_layers: list = []
        self.replays = 0
        self.captures = 0
        self._last_key: Optional[tuple[int, int]] = None
        self._raisons_eager_vues: set = set()
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
        # Chemin table (bead pds, point 4 suite — chef 14/09) : un
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
                    self.raison = "pile d'experts hétérogène"
                    return False             # pile heterogene : eager
                if all(p[0] in ("nvfp4", "nvfp4_table")
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

    # -- exécution -------------------------------------------------------
    def _eager(self, raison: str) -> None:
        """Un lot qui retombe en eager le dit — une fois par raison distincte.

        Bug du 13/09 (bead anticitoyen-vram-x0s) : un repli silencieux se lit
        comme une absence de problème, exactement comme `_eligible()` le dit
        déjà pour la désactivation globale. Une seule fois par raison : un
        serveur qui replie à chaque pas sur le même motif ne doit pas noyer
        sa propre sortie.
        """
        if raison in self._raisons_eager_vues:
            return
        self._raisons_eager_vues.add(raison)
        print(f"[graphe] repli eager — {raison}", flush=True)

    def run(self, batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Logits du lot, ou None si ce lot n'est pas rejouable en graphe."""
        if not self.enabled or batch.is_prefill:
            return None
        ql = batch.query_lens[0]
        if any(q_ != ql for q_ in batch.query_lens):
            return None                      # longueurs mixtes : eager
        if ql != 1 and not self.paged_ok:
            return None                      # la verification exige le noyau
        b_reel = batch.batch_size
        b = bucket_batch(b_reel)
        nblk = bucket_blocks(max(t.shape[0] for t in batch.block_tables))
        if nblk * BLOCK_SIZE > self.max_model_len + BLOCK_SIZE:
            nblk = bucket_blocks((self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        lb = 0
        trace = bool(os.environ.get("ACVRAM_TRACE_STEPS"))
        t0 = time.perf_counter()
        if self.hybrid_layers:
            godet = godet_hybride(b_reel, self.max_slots)
            if godet is None:
                self._eager(f"lot hybride {b_reel} au-dela du plafond "
                            f"ACVRAM_HYBRID_SLOTS={self.max_slots}")
                return None
            b = godet
            if batch.gdn_store is None:
                self._eager("gdn_store absent (etat recurrent non initialise)")
                return None
            if ql > self.max_ql:
                self._eager(f"ql={ql} au-dela de max_ql={self.max_ql}")
                return None
            if b_reel > 1 and ql != 1:
                self._eager("lot multi-sequences en verification speculative "
                             "(ql != 1) : non supporte sous graphe")
                return None
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
                return None
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
                torch.cuda.empty_cache()
                return None
            self.graphs[key] = entry
            if os.environ.get("ACVRAM_TRACE_PTRS"):
                entry["ptrs"] = _empreinte_adresses(self, entry)
                entry["ne"] = self.replays
            self.replays += 1                # la capture rejoue deja une fois
            if trace:
                print(f"[graphe] capture clé {key} : "
                      f"{(time.perf_counter()-t1)*1000:.1f} ms", flush=True)
            out = entry["out"][:b_reel * ql].clone()
            return out

        self._fill(entry, batch)
        if trace and os.environ.get("ACVRAM_CHRONO_SYNC"):
            torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
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
                # Instrumentation temporaire (poste3, mandat chef) : sans
                # synchronisation, .replay() est un lancement asynchrone --
                # perf_counter() juste apres ne mesure QUE le lancement, pas
                # l'execution reelle sur le GPU. Sous garde d'env car cette
                # synchronisation elle-meme serialise le pipeline et FAUSSE
                # le debit si elle reste active en permanence. Non committe.
                torch.cuda.synchronize(self.device)
        self.replays += 1
        out = entry["out"][:b_reel * ql].clone()
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

    # -- hybrides ----------------------------------------------------------
    _SID_REMBOURRAGE = -1

    def _bind_hybrid(self, batch: ForwardBatch, lb: int,
                     b_godet: int = 0) -> None:
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
        idx = batch.tokens.to(m.embed_tokens.device)
        x = torch.nn.functional.embedding(idx, m.embed_tokens).to(m.dtype)
        return x if m.spec.embedding_multiplier == 1.0 else x * m.spec.embedding_multiplier

    def _fill(self, entry: dict, batch: ForwardBatch) -> None:
        b, _ql, nblk = entry["key"][:3]
        b_reel = batch.batch_size
        emb = self._embed(batch).to(self.device)
        if b_reel < b:
            n_pad = (b - b_reel) * _ql
            entry["x"].zero_()
            entry["x"][:b_reel * _ql].copy_(emb, non_blocking=True)
            entry["positions"].zero_()
            entry["positions"][:b_reel * _ql].copy_(batch.positions, non_blocking=True)
            entry["slots"].fill_(-1)
            entry["slots"][:b_reel * _ql].copy_(batch.slot_mapping, non_blocking=True)
            sl = torch.zeros(b, dtype=torch.long)
            sl[:b_reel] = torch.tensor(batch.seq_lens, dtype=torch.long)
            entry["seq_lens"].copy_(sl, non_blocking=True)
        else:
            entry["x"].copy_(emb, non_blocking=True)
            entry["positions"].copy_(batch.positions, non_blocking=True)
            entry["slots"].copy_(batch.slot_mapping, non_blocking=True)
            entry["seq_lens"].copy_(
                torch.tensor(batch.seq_lens, dtype=torch.long), non_blocking=True)
        # Table completee au godet avec le bloc 0 : lu, dequantifie, masque.
        tables = entry["tables"]
        tables.zero_()
        for i, t in enumerate(batch.block_tables):
            tables[i, : t.shape[0]].copy_(t, non_blocking=True)
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
        self._fill(entry, batch)
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
            return m.decode_fixed(entry["x"], entry["positions"],
                                  entry["slots"], entry["tables"],
                                  entry["seq_lens"], max_pos, q_len=ql)

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
