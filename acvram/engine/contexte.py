"""Contexte servi : chauffe au chargement, clamp nommé, séquence de service (plan → clamp → dichotomie sans graphes →
capture au tenu → confirmation avec graphes). Déplacement PUR depuis `runner.py` (4 bis, 21/09) : `ChauffeContexte` est
un mixin d `Engine`, les corps sont ceux de runner.py au commit précédent, octet pour octet ; `ContexteNonTenu`,
`CTX_TENU_MIN` et `_ctx_texte` vivent ici et restent réexportés par `runner`."""

from __future__ import annotations

import os
import time
from typing import Optional

import torch

from ..memory.kvcache import BLOCK_SIZE, BlockAllocator
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
    txt = f" ctx_tenu={v}" if demande == v else f" ctx_tenu={v}(demandé {demande})"
    seuil = getattr(engine, "moe_seuil", None)
    return txt + (f" tranches>{seuil}" if seuil else "")            # d19 : au-delà, sortie non au bit du seul tenant


def _sur_carte(couche) -> bool:
    """La COUCHE vit sur la carte (``couche.device``, comme graphs.py:474). Pas les paramètres du bloc MoE : les poids
    quantifiés sont des attributs ordinaires de QuantLinear (layers.py:438) — un bloc MoE n a souvent AUCUN paramètre,
    et la 1re version de cette garde (premier paramètre du bloc) rendait faux partout : prise ya1 du 29/09, bras B
    identique au témoin à l octet près."""
    dev = getattr(couche, "device", None)
    return dev is not None and torch.device(dev).type == "cuda"


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

    def _essai_de_chauffe(self, L: int, max_tokens: int = 1, prefixe: bool = False) -> bool:
        """Une passe de chauffe de ``L`` jetons dans le régime courant : tenue ssi pas d OOM ET ≥ max(5 %, 64 Mio)
        libres après elle. ``max_tokens=2`` force un pas de décodage (les graphes, s ils sont actifs, rejouent).
        ``prefixe`` (cqy, 29/09) : deux blocs de la séquence passent d'abord et restent au cache de préfixe, puis la
        séquence entière les réutilise → préfill à q_offset > 0, le chemin d'une requête claude qui répète son
        début (8fx : masque dense de 2 Gio à 34 k, que la passe depuis 0 ne voyait pas)."""
        self._avant_essai_de_chauffe()                                   # (3) chaque pas part d un allocateur vide
        seq = self.sequence_de_chauffe(L - 2)
        en_cache = self.stats.cached_prompt_tokens
        # kv31b : le pic transitoire de la passe (alloué, pas réservé) — ce que la réserve de préfill doit couvrir
        dev = self.model.embed_tokens.device
        mesure = torch.cuda.is_available() and dev.type == "cuda"
        if mesure:
            torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
        try:
            if prefixe:
                for _ in self.generate(seq[:2 * BLOCK_SIZE], SamplingParams(max_tokens=1, temperature=0.0)):
                    pass
                self._avant_essai_de_chauffe()                           # la passe mesurée part d un allocateur vide
            for _ in self.generate(seq, SamplingParams(max_tokens=max_tokens, temperature=0.0)):
                pass
        except Exception as exc:                                         # noqa: BLE001
            oom = isinstance(exc, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(exc).lower() \
                or "blocs KV" in str(exc)
            if not oom:
                raise
            self._apres_oom_de_chauffe()
            return False
        finally:
            # le préfixe réutilisé par la chauffe ne compte pas comme un cache servi
            self.stats.cached_prompt_tokens = en_cache
        libre, total = self._libre_apres_chauffe()
        seuil = max(total * 5 // 100, 64 << 20)
        if mesure:
            self.pic_chauffe = (L, int(torch.cuda.max_memory_allocated(dev) - base))
        self._oublier_la_chauffe()
        if libre < seuil:                                                # (b) : tenu sans réserve = non tenu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        self.reserve_chauffe = (libre, seuil)
        return True

    def _a_des_tranches(self) -> bool:
        """Un bloc que la chauffe peut passer par tranches au-delà du seul tenant (d19) : MoE ou GDN, morceau non nul."""
        from . import attention as _att, gdn as _gdn, moe as _moe
        return (any((isinstance(m, _moe.MoEBlock) and _moe._MOE_MORCEAU > 0)
                    or (isinstance(m, _gdn.GatedDeltaNet) and _gdn._GDN_MORCEAU > 0) for m in self.model.modules())
                or (self._dense_pur() and _att._MLP_MORCEAU > 0))

    def _dense_pur(self) -> bool:
        """a5v : ni MoE ni GDN — seul cas où le MLP dense passe par tranches (un MoE ou un hybride garde d19 tel quel)."""
        from . import attention as _att, gdn as _gdn, moe as _moe
        mods = list(self.model.modules())
        return (not any(isinstance(m, (_moe.MoEBlock, _gdn.GatedDeltaNet)) for m in mods)
                and any(type(m) is _att.MLP for m in mods))

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
        def dichotomie(bas: int, haut: int, essai=essai) -> int:        # bas tenu (0 : rien), haut non tenu
            for _ in range(8):
                milieu = max(pas, ((bas + haut) // 2) // pas * pas)
                if milieu <= bas or milieu >= haut:
                    break
                if essai(milieu):
                    bas = milieu
                else:
                    haut = milieu
            return bas

        from . import attention as _att, gdn as _gdn, moe as _moe
        dense = self._dense_pur()

        def seuils(v):
            _moe.definir_seuil(v); _gdn.definir_seuil(v); _att.definir_seuil(v if dense else None)
        seuils(None)
        self.moe_seuil: Optional[int] = None
        tenu: Optional[int] = n if essai(n) else dichotomie(0, n)
        # d19 : au-delà du tenu d un seul tenant, cœur GDN et bloc MoE par tranches (tampons bornés). Engagés SEULEMENT
        # au-delà : toute invite qui tenait garde sa sortie au bit et son temps ; au-delà, elle recevait un 400. Pas au bit
        # du seul tenant (prise 5 de d19 : PPL égale, KL moyen 7e-3).
        if tenu < n and self._a_des_tranches():
            seuils(max(tenu, pas))
            tenu2 = n if essai(n) else dichotomie(tenu, n)
            if tenu2 > tenu:
                print(f"[acvram] préfill par tranches (GDN {_gdn._GDN_MORCEAU}, MoE {_moe._MOE_MORCEAU}"
                      f"{f', MLP {_att._MLP_MORCEAU}' if dense else ''}) au-delà de "
                      f"{tenu} jetons : {tenu2} tenus", flush=True)
                self.moe_seuil, tenu = tenu, tenu2
            else:
                seuils(None)
        # cqy : le tenu doit aussi tenir quand l invite réutilise un préfixe en cache (q_offset > 0) ; sinon on descend.
        if tenu and self.allocator.enable_prefix_cache:
            # sous deux blocs + 2, le « préfixe » serait l invite entière : rien de plus à prouver que la passe depuis 0
            essai_p = lambda L: L - 2 <= 2 * BLOCK_SIZE or self._essai_de_chauffe(L, prefixe=True)   # noqa: E731
            if not essai_p(tenu):
                avant, tenu = tenu, dichotomie(0, tenu, essai_p)
                print(f"[acvram] chauffe avec préfixe en cache : {tenu} jetons tenus (sans préfixe : {avant})", flush=True)
        self._oublier_la_chauffe()
        self.graphs = graphes
        self.ctx_demande = n
        self.ctx_tenu = tenu
        self._enregistrer_pic_de_chauffe(tenu)
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

    def _enregistrer_pic_de_chauffe(self, tenu: Optional[int]) -> None:
        """kv31b : compare le pic mesuré de la passe tenue à la formule et le dépose pour le prochain chargement
        (`loader.enregistrer_chauffe`) — la réserve de préfill se cale sur la mesure, plus seulement sur la formule."""
        pic = getattr(self, "pic_chauffe", None)
        if not tenu or not pic or pic[0] != tenu:
            return
        from .loader import enregistrer_chauffe
        spec = self.model.spec
        formule = int(spec.activations_prefill_bytes(tenu))
        manifest = getattr(self.loaded, "manifest", None) if hasattr(self, "loaded") else None
        d = enregistrer_chauffe(getattr(spec, "name", "") or "modele", tenu, pic[1], formule, manifest=manifest,
                                kv_format=self.kv_format_servi() if hasattr(self, "kv_format_servi") else "?",
                                plafond=getattr(spec, "mlp_prefill_plafond", None), max_model_len=int(self.max_model_len))
        print(f"[acvram] chauffe : pic transitoire du préfill {pic[1] / 2**30:.2f} Gio à {tenu} jetons "
              f"({pic[1] // tenu // 1024} Kio/jeton ; formule {formule / 2**30:.2f} Gio, {formule // tenu // 1024} Kio/jeton) — "
              f"excès {d['exces_par_jeton'] // 1024} Kio/jeton enregistré pour le prochain chargement", flush=True)

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
        self._construire_piles_avant_chauffe()
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

    def _construire_piles_avant_chauffe(self) -> int:
        """ya1 (29/09) : sans graphes (``--no-cuda-graphs``, passes rapides d edz), les piles d experts et leur repack
        Marlin se construisaient PARESSEUSEMENT dans le premier forward (moe.py, ``_stack_state == "?"``) — celui de la
        première passe de chauffe : leur transitoire (pile empilée puis repackée, 2 × 324 Mio par couche sur Coder-30B)
        s ajoutait aux activations du préfill plein, et la chauffe concluait 3 072 sur 4 096. Avec graphes,
        `GraphRunner._eligible` (graphs.py:473) les construit AVANT la chauffe ; même construction ici, pour que la
        chauffe mesure le régime servi. Mêmes piles, même sortie : seul le moment change. Rend le nombre de couches
        décidées."""
        from .moe import MoEBlock
        n = 0
        for couche in getattr(self.model, "layers", ()):
            if not _sur_carte(couche):
                continue
            for mod in couche.modules():
                if isinstance(mod, MoEBlock) and mod._stack_state == "?":
                    mod._stack_state = "oui" if mod._try_build_stacks() else "non"
                    n += 1
        if n:
            print(f"[acvram] piles d experts construites avant la chauffe : {n} couches", flush=True)
        return n

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
            torch.cuda.synchronize()
        from .layers import oublier_precharges                           # emplacements du pool dense en vol : rendus
        oublier_precharges(self.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _oublier_la_chauffe(self) -> None:
        """Le cache de préfixe ne garde pas la séquence de chauffe (des « 1 » : un vrai préfixe de 1 y trouverait un
        KV) : l'allocateur est recréé à l'identique (mêmes blocs, même mode)."""
        self.allocator = BlockAllocator(self.allocator.num_blocks, self.allocator.enable_prefix_cache)
