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
spéculation (plusieurs positions par séquence), les modèles à experts (le
routage dépend des données), les couches réparties sur plusieurs appareils et
les poids streamés (le préchargement change les adresses).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch

from ..memory.kvcache import BLOCK_SIZE, bucket_blocks
from .layers import QuantLinear
from .model import DecoderLayerGDN, ForwardBatch, MoEBlock

from .mla import MLA_BUCKET, godet_mla   # un graphe par palier de cache latent

__all__ = ["GraphRunner"]

# Au-delà, les godets les moins récents ne sont plus capturés : chaque graphe
# retient sa mémoire d'activations, et un serveur qui voit trente formes de
# lot différentes est un serveur de lots — le prefill y domine de toute façon.
MAX_GRAPHS = 16


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
        self.enabled = self._eligible()
        self.replays = 0
        self.captures = 0
        self._last_key: Optional[tuple[int, int]] = None

    # -- éligibilité -----------------------------------------------------
    def _eligible(self) -> bool:
        if os.environ.get("ACVRAM_DISABLE_CUDA_GRAPHS"):
            return False
        if not torch.cuda.is_available():
            return False
        m = self.model
        # Hybrides à états (GDN/KDA/MLA) : capturés par couche via des
        # tampons fixes, une séquence par graphe (b = 1, q_len = 1).
        self.hybrid_layers = [l for l in m.layers
                              if isinstance(l, DecoderLayerGDN)]
        if self.hybrid_layers and any(
                not hasattr(l.linear_attn, "decode_static")
                for l in self.hybrid_layers):
            return False
        devs = {l.device for l in m.layers} | {l.mlp_device for l in m.layers}
        devs.add(m.norm.weight.device)
        head = getattr(m.lm_head.qweight, "qweight", None)
        devs.add(head.device if head is not None else m.norm.weight.device)
        if len(devs) != 1 or next(iter(devs)).type != "cuda":
            return False                     # pipeline multi-appareils : eager
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
                    return False             # pile heterogene : eager
            if isinstance(mod, QuantLinear) and mod.streamed is not None:
                return False                 # les adresses changent en vol
        pleines = [i for i, l in enumerate(m.layers)
                   if getattr(l, "self_attn", None) is not None]
        if any(i not in m.caches for i in pleines):
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

    # -- exécution -------------------------------------------------------
    def run(self, batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Logits du lot, ou None si ce lot n'est pas rejouable en graphe."""
        if not self.enabled or batch.is_prefill:
            return None
        ql = batch.query_lens[0]
        if any(q_ != ql for q_ in batch.query_lens):
            return None                      # longueurs mixtes : eager
        if ql != 1 and not self.paged_ok:
            return None                      # la verification exige le noyau
        b = batch.batch_size
        nblk = bucket_blocks(max(t.shape[0] for t in batch.block_tables))
        if nblk * BLOCK_SIZE > self.max_model_len + BLOCK_SIZE:
            nblk = bucket_blocks((self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        lb = 0
        trace = bool(os.environ.get("ACVRAM_TRACE_STEPS"))
        t0 = time.perf_counter()
        if self.hybrid_layers:
            if (batch.gdn_store is None or ql > self.max_ql
                    or (b > 1 and ql != 1) or b > self.max_slots):
                return None                  # spéculation : une séquence
            lb = godet_mla(max(batch.seq_lens))
            self._bind_hybrid(batch, lb)
        key = (b, ql, nblk, lb)
        self._last_key = key
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
                if not getattr(self, "_dit_raison", False):
                    self._dit_raison = True
                    print(f"[acvram] graphes CUDA desactives, decodage en eager "
                          f"— capture impossible : {self.raison}", flush=True)
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
            return entry["out"].clone()

        self._fill(entry, batch)
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
        self.replays += 1
        out = entry["out"].clone()
        if trace:
            t3 = time.perf_counter()
            if (t3 - t0) * 1000 > 20:
                print(f"[graphe-lent] bind {(t1-t0)*1000:.1f} fill "
                      f"{(t2-t1)*1000:.1f} replay {(t3-t2)*1000:.1f} ms clé {key}",
                      flush=True)
        return out

    # -- hybrides ----------------------------------------------------------
    def _bind_hybrid(self, batch: ForwardBatch, lb: int) -> None:
        sids = batch.seq_ids or list(range(batch.batch_size))
        m = self.model
        for layer in self.hybrid_layers:
            store = batch.gdn_store.setdefault(layer.index, {})
            for slot, sid in enumerate(sids):
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
        entry["x"].copy_(self._embed(batch).to(self.device), non_blocking=True)
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
