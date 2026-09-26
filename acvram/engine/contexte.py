"""Contexte servi : chauffe au chargement, clamp nommé, séquence de service (plan → clamp → dichotomie sans graphes →
capture au tenu → confirmation avec graphes). Déplacement PUR depuis `runner.py` (4 bis, 21/09) : `ChauffeContexte` est
un mixin d `Engine`, les corps sont ceux de runner.py au commit précédent, octet pour octet ; `ContexteNonTenu`,
`CTX_TENU_MIN` et `_ctx_texte` vivent ici et restent réexportés par `runner`."""

from __future__ import annotations

import os
import time
from typing import Optional

import torch

from ..memory.kvcache import BlockAllocator
from .sampler import SamplingParams

__all__ = ["ChauffeContexte", "ContexteNonTenu", "CTX_TENU_MIN", "_ctx_texte"]


class ContexteNonTenu(RuntimeError):
    """`max_model_len` demandé mais non tenu par la chauffe au chargement : refus nommé, pas un 500 à la requête.
    Levé seulement si le contexte tenu est < 4096 ou sous ``--ctx-strict`` ; sinon le moteur se CLAMPE (poste7
    poste7-s2-k48-feu-vert-21-09 § 2 (c)) et la ligne de régime dit ``ctx_tenu=N(demandé M)``."""

    def __init__(self, demande: int, tenu: int) -> None:
        self.demande, self.tenu = int(demande), int(tenu)
        super().__init__(f"contexte non tenu : max_model_len={self.demande} demandé, {self.tenu} jetons tenus par la "
                         f"chauffe (prefill d'une séquence pleine dans le régime servi) — relancer avec "
                         f"--max-model-len {self.tenu} ou moins ; ACVRAM_CHAUFFE_CTX=0 pour un banc qui pose son plan")


CTX_TENU_MIN = 4096                    # en dessous, un clamp n'est plus un service : refus nommé (poste7 § 2 (c))


def _ctx_texte(engine) -> str:
    """`ctx_tenu=N` (chauffe tenue) | `ctx_tenu=non-verifie` (opt-out nommé) | `ctx_tenu=non-chauffe` (pas encore)."""
    v = getattr(engine, "ctx_tenu", "absent")
    if v is None:
        return " ctx_tenu=non-verifie"
    if v == "absent":
        return " ctx_tenu=non-chauffe"
    demande = getattr(engine, "ctx_demande", v)
    return f" ctx_tenu={v}" if demande == v else f" ctx_tenu={v}(demandé {demande})"


class ChauffeContexte:
    """Mixin d `Engine` : les méthodes de chauffe du contexte et de démarrage du service (voir l en-tête du module)."""

    def sequence_de_chauffe(self, L: int) -> list[int]:
        """Séquence pseudo-aléatoire sur le vocabulaire, graine fixe 0, longueur ``L`` (poste7 § 2 (a)) : une séquence
        homogène ``[1]×N`` route tous les jetons vers les mêmes experts et sous-estime la crête d'un vrai prefill
        (GLM k48 : chauffe TENUE 32768/32768 puis 500 CUDA OOM réel à ctx − 64, 318/298 Mio)."""
        g = torch.Generator().manual_seed(0)
        vocab = int(getattr(self.spec, "vocab_size", 0) or 0) or 2
        return torch.randint(0, vocab, (L,), generator=g).tolist()

    def _avant_essai_de_chauffe(self) -> None:
        """Entre deux pas de la dichotomie, les segments réservés par le pas précédent (plus grand, ou tenu sans
        réserve) restent dans l allocateur CUDA : le pas suivant lit moins de libre et la recherche descend trop
        (Coder i8c : 15 360 hier avec [1]×N, 4 096 ce matin — chef 21/09 (3)). Synchroniser, vider."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _essai_de_chauffe(self, L: int, max_tokens: int = 1) -> bool:
        """Une passe de chauffe de ``L`` jetons dans le régime courant : tenue ssi pas d OOM ET ≥ max(5 %, 64 Mio)
        libres après elle. ``max_tokens=2`` force un pas de décodage (les graphes, s ils sont actifs, rejouent)."""
        self._avant_essai_de_chauffe()                                   # (3) chaque pas part d un allocateur vide
        try:
            for _ in self.generate(self.sequence_de_chauffe(L - 2), SamplingParams(max_tokens=max_tokens, temperature=0.0)):
                pass
        except Exception as exc:                                         # noqa: BLE001
            oom = isinstance(exc, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(exc).lower() \
                or "blocs KV" in str(exc)
            if not oom:
                raise
            self._apres_oom_de_chauffe()
            return False
        libre, total = self._libre_apres_chauffe()
        seuil = max(total * 5 // 100, 64 << 20)
        self._oublier_la_chauffe()
        if libre < seuil:                                                # (b) : tenu sans réserve = non tenu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        self.reserve_chauffe = (libre, seuil)
        return True

    def _libre_apres_chauffe(self) -> tuple[int, int]:
        """(libre, total) octets du pilote après la passe, avant tout `empty_cache` : le réservé du prefill y est
        encore compté. Hors carte : (total, total) — la réserve ne se juge que sur carte."""
        dev = self.model.embed_tokens.device
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.synchronize(dev)                                  # la crête est passée avant la lecture
            libre, total = torch.cuda.mem_get_info(dev)
            return int(libre), int(total)
        return (1 << 40, 1 << 40)

    def _experts_sollicites(self, seq: list[int], params) -> int:
        """Prefill de ``seq`` en comptant les experts DISTINCTS touchés par couche MoE (somme sur les couches) :
        c'est la grandeur qui sépare ``[1]×N`` (top_k experts par couche) d'une séquence réelle (jusqu'à tous) —
        chaque expert sollicité a ses poids mis en jeu (Marlin nvfp4 : atelier par expert actif). 0 = modèle dense."""
        from .model import MoEBlock
        blocs = [m for m in self.model.modules() if isinstance(m, MoEBlock)]
        vus: dict[int, set] = {}
        originaux = []
        for i, m in enumerate(blocs):
            orig = m._route

            def route(x, _o=orig, _i=i):
                topw, topi = _o(x)
                vus.setdefault(_i, set()).update(int(e) for e in torch.unique(topi.detach().cpu()).tolist())
                return topw, topi
            originaux.append((m, orig)); m._route = route
        try:
            for _ in self.generate(seq, params):
                pass
        finally:
            for m, orig in originaux:
                m._route = orig
        return sum(len(v) for v in vus.values())

    def chauffer_contexte(self, pas: int = 1024, strict: bool = False) -> Optional[int]:
        """Prouve ``max_model_len`` AU CHARGEMENT (poste7, poste7-3b-lanceur-contexte-20-09 (ii), resserré par
        poste7-s2-k48-feu-vert-21-09 § 2) : une séquence PSEUDO-ALÉATOIRE (graine 0, ``sequence_de_chauffe``) de
        ``max_model_len − 2`` jetons passe le prefill dans le RÉGIME SERVI, puis ses blocs sont rendus et le cache
        de préfixe ne la garde pas. Tenu(L) = la passe ne lève pas d'OOM ET laisse ≥ max(5 %, 64 Mio) du pilote
        libres après elle (réserve pour la crête d'une vraie requête). ``ctx_tenu`` = plus grand multiple de
        ``pas`` tenu (dichotomie depuis ``max_model_len``). Non tenu → CLAMP : ``max_model_len`` prend
        ``ctx_tenu``, la ligne dit ``ctx_tenu=N(demandé M)``, une invite au-delà reçoit un 400 nommé ; refus
        ``ContexteNonTenu`` seulement si ``ctx_tenu < CTX_TENU_MIN`` (4096) ou ``strict`` (--ctx-strict).
        Opt-out nommé ``ACVRAM_CHAUFFE_CTX=0`` → ``ctx_tenu=non-verifie`` ; jamais sous ``ACVRAM_TYPE=service``."""
        n = int(self.max_model_len)
        if os.environ.get("ACVRAM_CHAUFFE_CTX") == "0":
            if os.environ.get("ACVRAM_TYPE") == "service":
                print("[acvram] ACVRAM_CHAUFFE_CTX=0 ignoré : un service prouve son contexte", flush=True)
            else:
                self.ctx_tenu = None
                return None
        self.reserve_chauffe: Optional[tuple[int, int]] = None           # (libre, seuil) de la dernière passe tenue
        # Ordre imposé (chef 21/09, GLM k48 : OOM à la CAPTURE, avant la chauffe — la capture alloue au ctx
        # demandé) : plan → clamp → capture → chauffe. Les graphes sont masqués pendant les essais : aucune
        # capture ne peut se faire à un contexte que la chauffe n a pas encore tenu ; la taille de capture prend
        # le contexte tenu (voir la fin).
        graphes, self.graphs = self.graphs, None

        essai = self._essai_de_chauffe

        t0 = time.time()
        tenu: Optional[int]
        if essai(n):
            tenu = n
        else:
            bas, haut = 0, n                                              # bas tenu (0 : rien), haut non tenu
            for _ in range(8):
                milieu = max(pas, ((bas + haut) // 2) // pas * pas)
                if milieu <= bas or milieu >= haut:
                    break
                if essai(milieu):
                    bas = milieu
                else:
                    haut = milieu
            tenu = bas
        self._oublier_la_chauffe()
        self.graphs = graphes
        self.ctx_demande = n
        self.ctx_tenu = tenu
        r = self.reserve_chauffe
        print(f"[acvram] chauffe du contexte : {tenu}/{n} jetons tenus en {time.time() - t0:.1f} s"
              + (f", {r[0] >> 20} Mio libres après la passe (réserve ≥ {r[1] >> 20})" if r else ""), flush=True)
        if tenu < n:
            if strict or tenu < CTX_TENU_MIN:
                raise ContexteNonTenu(n, tenu)
            self.max_model_len = tenu                                    # (c) clamp : 400 nommé au-delà, pas un 500
            if self.graphs is not None:
                self.graphs.max_model_len = tenu                         # la capture se dimensionne au tenu
            print(f"[acvram] contexte clampé à {tenu} (demandé {n}) : une invite au-delà reçoit un 400 nommé",
                  flush=True)
        return tenu

    def _recapturer(self, warm_max_len: int) -> int:
        """Graphes neufs au ``max_model_len`` courant (les captures précédentes sont rendues), puis capture d avance."""
        from .graphs import GraphRunner
        self.graphs = None
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        gr = GraphRunner(self.model, self.max_model_len, max_batch_size=self.max_batch_size)
        self.graphs = gr if gr.enabled else None
        return self.warm_graphs(warm_max_len) if self.graphs is not None else 0

    def demarrer_service(self, strict: bool = False, warm_max_len: int = 2048,
                         pas: int = 1024, pas_confirmation: int = 1024) -> tuple[Optional[int], int]:
        """La séquence de chargement d un service, dans l ordre qui ne peut pas mentir (chef 21/09) :
        plan → clamp → dichotomie SANS graphes → capture au tenu → passe de CONFIRMATION au tenu, graphes actifs
        (la requête réelle tourne avec leurs réserves : c est le contrôle qui peut rendre faux). Confirmation
        échouée → tenu − ``pas_confirmation``, recapture, deux fois au plus, puis refus nommé. Rend (ctx_tenu,
        captures). Test cassant : faux graphes qui coûtent 2 pas ⇒ tenu final = tenu − 2·pas, chargement réussi."""
        tenu = self.chauffer_contexte(pas=pas, strict=strict)
        captures = self.warm_graphs(warm_max_len) if self.graphs is not None else 0
        if tenu is None or self.graphs is None:
            return tenu, captures
        for baisse in range(3):
            if self._essai_de_chauffe(int(self.max_model_len), max_tokens=2):
                if baisse:
                    print(f"[acvram] confirmation avec graphes : {self.ctx_tenu} tenus après {baisse} baisse(s)", flush=True)
                self._oublier_la_chauffe()
                return self.ctx_tenu, captures
            nouveau = int(self.max_model_len) - pas_confirmation
            if baisse == 2 or strict or nouveau < CTX_TENU_MIN:
                raise ContexteNonTenu(self.ctx_demande, nouveau if baisse < 2 else int(self.max_model_len))
            print(f"[acvram] confirmation avec graphes échouée à {self.max_model_len} : contexte {nouveau}, recapture",
                  flush=True)
            self.max_model_len = self.ctx_tenu = nouveau
            captures = self._recapturer(warm_max_len)
        return self.ctx_tenu, captures                                   # jamais atteint

    def _apres_oom_de_chauffe(self) -> None:
        """Après un OOM de chauffe : séquences en cours abandonnées, blocs rendus, allocateur neuf, cache CUDA vidé."""
        for seq in list(self.running) + list(self.waiting):
            try:
                self._finish(seq, "chauffe")
            except Exception:                                            # noqa: BLE001
                pass
        self.running.clear(); self.waiting.clear()
        self._oublier_la_chauffe()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _oublier_la_chauffe(self) -> None:
        """Le cache de préfixe ne garde pas la séquence de chauffe (des « 1 » : un vrai préfixe de 1 y trouverait un
        KV) : l'allocateur est recréé à l'identique (mêmes blocs, même mode)."""
        self.allocator = BlockAllocator(self.allocator.num_blocks, self.allocator.enable_prefix_cache)
