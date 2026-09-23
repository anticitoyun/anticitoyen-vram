"""Le bloc à mélange d'experts : routage, chemins de préfill (GEMM groupée,
Marlin, MMA FP4 native) et de décodage (GEMV groupée, MMA, fusionné), colle du
tri des paires (jeton, expert), portes de qualité W4A4/W4A8 et témoins de
routage.

Déplacement PUR depuis `engine/model.py` (scission 22/09, module 4) : pas une
ligne de logique changée, les noms restent importables depuis `engine.model`,
qui les réexporte — `model.MoEBlock is moe.MoEBlock`, et les réglages lus à
l'import sont les mêmes objets.

Deux conséquences à connaître, parce qu'elles sont silencieuses :
* un `monkeypatch.setattr(model, "_MOE_*", ...)` ne change plus ce que lit ce
  module-ci : un masquage doit viser `acvram.engine.moe` (les tests et les
  cibles `lu_a` de `regime.VARIABLES` ont été migrés dans le même commit) ;
* les CONTENEURS (`_QA_COMPTEURS`, `_ROUTAGES`, `_TEMOINS_ROUTAGE`,
  `_BORNES_EXPERTS`) restent les mêmes objets des deux côtés : les vider ou
  les remplir par `model.X` marche encore.

`_mla_glue` suit le bloc parce que `MoEBlock` l'appelle ; son corps est
inchangé (lecture différée de `mla._MLA_GLUE`) et `model` le réexporte pour
`DecoderLayerGDN` et `ACVRamModel`.
"""

from __future__ import annotations

from typing import Optional

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels
from ..memory import trace_routage as _trace_routage
from ..quant.calibrate import fwht_activations
from .attention import MLP
from .layers import QuantLinear

__all__ = [
    "MoEBlock", "_MOE_GROUPED_MAX", "_MOE_GEMM_MAX", "_MOE_MMA",
    "_PREFILL_GROUPED", "_PREFILL_A4", "_PREFILL_W8R", "_PREFILL_A8",
    "_PREFILL_A8_FMT", "_E2M1", "_E2M1_MILIEUX", "fausse_quant_a8",
    "fausse_quant_nvfp4", "_MOE_MMA_BT", "_MOE_MMA_ETAGES", "_MOE_MMA_KS",
    "_MOE_DECODE_MMA", "_MOE_DECODE_MMA_BT", "_MOE_AWQ_TEMOIN", "_QA_COMPTE",
    "_QA_COMPTEURS", "_qa_compteurs", "_qa_imprime", "_MOE_DECODE_MMA_MIN_T",
    "_ROUTE_PREP", "_MOE_DECODE_FUSED", "_MOE_FUSED_TN", "_MOE_FUSED_ATOMIQUE",
    "_MOE_FUSED_ETAGES", "_MOE_ROUTE_PACK", "_MOE_GEMV", "_GEMV_LAYOUT",
    "_MOE_DECODE_MMA_MARLIN", "_MARLIN_DISTINCT", "_MOE_W13", "_TRACE_ROUTAGE", "_ROUTAGES",
    "_ROUTAGE_TEMOIN", "_TEMOINS_ROUTAGE", "temoins_routage", "_DOUBLE_DIAG",
    "_mla_glue", "_COLLE_MOE", "_BORNES_EXPERTS", "_comptes_tries",
    "_colle_moe_triton", "MoEBlockGemma", "_DUMP_MOE",
]


class MoEBlock(nn.Module):
    """Mélange d'experts creux.

    Seuls les ``top_k`` experts vers lesquels un jeton a été routé sont évalués :
    le coût d'une couche est donc indépendant du nombre d'experts qu'elle
    possède. C'est ce qui permet à 128 experts de résider en mémoire vive pendant
    que le modèle décode à un rythme exploitable — par jeton, la machine en lit
    8, pas 128.
    """

    def __init__(self, router: QuantLinear, experts: list[MLP], top_k: int,
                 shared: Optional[MLP] = None,
                 norm_topk_prob: bool = True,
                 shared_gate: Optional[torch.Tensor] = None,
                 scoring: str = "softmax",
                 score_bias: Optional[torch.Tensor] = None,
                 routed_scale: float = 1.0) -> None:
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.shared = shared
        # porte sigmoïde de l'expert partagé (Qwen3-Next) : vecteur [1, d]
        self.shared_gate = shared_gate
        self.norm_topk_prob = norm_topk_prob
        # routage DeepSeek (kimi-linear) : scores sigmoïde, biais de sélection
        # (e_score_correction_bias) hors des poids, renormalisation, échelle
        self.scoring = scoring
        self.score_bias = score_bias
        self.routed_scale = routed_scale
        self._stack_state = "?"                # ? | oui | non
        self._stacks = None
        # Index de la couche, pose par le chargeur. -1 quand personne ne l'a
        # pose : la trace de routage l'ecrit tel quel plutot que d'inventer un
        # numero, et une trace pleine de -1 se voit tout de suite.
        self.index_couche = -1
        # activation des experts : le chemin groupé la reproduit (SiLU par
        # défaut, GELU-tanh pour Gemma 4)
        a = getattr(experts[0], "act", "silu") if experts else "silu"
        self.act = "gelu_tanh" if str(a).startswith("gelu") else "silu"
        # Histogramme de routage par expert : voir _compter_routage. None tant
        # qu'aucun pas ne l'a réservé (aucun forward encore, ou couche dense).
        self._usage_routage: Optional[torch.Tensor] = None
        # Placement par expert (bead pds, posé par loader.py) : None = pas de
        # placement par expert pour cette couche (comportement d'aujourd'hui,
        # `mlp_storage` seul décide). `_pin_experts` : les ids résidents,
        # lu par `Engine.__init__` pour peupler `_pin` (REPIN réel).
        # `_table_qw`/`_table_bscale` : `{"gate_proj"|"up_proj"|"down_proj":
        # tenseur [E] int64}` — le contrat de `memory/table_adresses.py`,
        # NVFP4 seulement (voir loader.py).
        self._pin_experts: Optional[set] = None
        self._table_qw: Optional[dict] = None
        self._table_bscale: Optional[dict] = None

    def _act(self, g: torch.Tensor) -> torch.Tensor:
        if self.act == "gelu_tanh":
            return F.gelu(g, approximate="tanh")
        return F.silu(g)

    def _compter_routage(self, topi: torch.Tensor) -> None:
        """Histogramme des experts routés, accumulé SUR DEVICE, jamais lu ici.

        Premier compteur du chantier colibrì (`revue/colibri-lecture-code.md`) :
        colibrì sépare `rt_count()` (comptage, toujours actif, un incrément par
        expert unique déjà testé au lookup) de `rt_trace()` (trace textuelle,
        opt-in, plus chère) — chez nous `_trace_routage` confondait les deux
        dans un seul mécanisme opt-in. Celui-ci tourne à CHAQUE pas, sans
        condition : un `torch.bincount` sur les indices que le routeur vient de
        produire, du même ordre de grandeur que ce qu'il a déjà lu, négligeable
        devant le GEMM du MLP qui suit dans la même couche.

        Capturable par un graphe CUDA : le tampon est réservé une seule fois,
        HORS capture, puis seulement modifié en place (`+=`, jamais réassigné)
        — même exigence que le cache RoPE (`RotaryEmbedding.reserver`, voir
        `test_rope_ne_se_realloue_pas_sous_capture`) : une (ré)allocation
        pendant une capture laisserait au graphe une adresse morte au replay.
        Refuse plutôt que d'en créer une.

        `torch.bincount` a été essayé d'abord et REFUSÉ : sa forme de sortie
        dépend de la plus grande valeur d'entrée, ce que CUDA ne peut décider
        sans lire la carte — « Cannot copy between CPU and CUDA tensors during
        CUDA graph capture », mesuré ici même. `scatter_add_` n'a pas ce
        défaut : la forme de sa sortie est celle du tampon, fixée d'avance,
        indépendante des VALEURS de `topi` — seule sa forme à lui compte, déjà
        connue avant le lancement du noyau.

        Un index hors domaine PAR LE HAUT (bogue amont — le routage ne doit
        jamais en produire) est vérifié sur CPU : `scatter_add_` y refuse
        tout index hors limites (`RuntimeError`, testé ci-dessous). Sur CUDA
        cette même situation est UN COMPORTEMENT NON DÉFINI de `scatter_add_`
        — ni exception garantie ni sécurité mémoire — donc PAS le filet de
        sécurité pour ce cas côté carte ; c'est le test CPU qui doit attraper
        une régression du routage avant qu'elle n'atteigne le GPU.

        `-1` PAR LE BAS est, lui, un cas NORMAL et attendu depuis le
        masquage des créneaux fantômes (`forward`, bead pds 14/09) : un
        jeton de remplissage qu'aucun compte ne doit voir. `clamp(min=0)`
        rend l'index valide pour `scatter_add_` (jamais -1 en pratique,
        adresse toujours dans les bornes) tandis que `poids` vaut 0 pour ces
        entrées — comptées zéro fois, pas comptées « à l'expert 0 ».
        """
        n = len(self.experts)
        if self._usage_routage is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "compteur de routage non réservé avant la capture du "
                    "graphe CUDA — un premier appel hors capture le réserve")
            self._usage_routage = torch.zeros(n, dtype=torch.int64,
                                              device=topi.device)
        idx = topi.reshape(-1).to(torch.int64)
        poids = (idx >= 0).to(torch.int64)
        self._usage_routage.scatter_add_(0, idx.clamp(min=0), poids)

    # ------------------------------------------------------------------
    # Pile d'experts pour le chemin groupé. Les qweight/échelles de tous les
    # experts d'une projection sont recopiés dans un tenseur [E, ...] contigu,
    # puis chaque expert reçoit une *vue* de la pile — la mémoire n'est pas
    # doublée, et la boucle par expert du prefill continue de marcher.
    # ------------------------------------------------------------------
    def _noms_experts(self) -> tuple:
        return ("gate_proj", "up_proj", "down_proj") if hasattr(self.experts[0], "gate_proj") \
            else ("up_proj", "down_proj")

    def _table_pile(self, nom: str, ws: list):
        """Pendant table (bead pds) du chemin empilé : chaque expert lu par
        adresse (`table_qw[e]`), résident ou épinglé — pas de pile contiguë,
        donc valable même si une partie des experts est streamed. Rend None
        si la table n'existe pas pour cette projection (format non NVFP4
        homogène à toutes les couches, ou construction refusée au
        chargement) : le chemin par expert reste alors le repli."""
        table_qw = (self._table_qw or {}).get(nom)
        table_bscale = (self._table_bscale or {}).get(nom)
        if table_qw is None or table_bscale is None:
            return None
        cache = self.__dict__.setdefault("_gs_table", {})
        gs = cache.get(nom)
        if gs is None:
            gs = torch.tensor([w.global_scale_float() for w in ws],
                              dtype=torch.float32, device=table_qw.device)
            cache[nom] = gs
        return ("nvfp4_table", table_qw, table_bscale, gs,
                ws[0].padded_in, ws[0].shape[0])

    def _try_build_stacks(self) -> bool:
        from ..quant.int4 import INT4Tensor
        from ..quant.nvfp4 import NVFP4Tensor

        # « pile d'experts hétérogène » recouvrait trois causes distinctes
        # sous un seul message (graphs.py) : impossible de savoir, sans
        # lire le code, si le repli lent vient d'une VRAIE hétérogénéité de
        # format/forme, d'une échelle AWQ posée sur certains experts
        # seulement, ou d'un manque transitoire de VRAM pendant la
        # construction (branche OOM plus bas, qui avait deja son propre
        # message). Trouve le 14/09 sur Qwen3-Coder-30B-A3B reconverti a
        # snr_floor=0 : le repli SEMBLAIT etre un bogue d'heterogeneite
        # (aucune trace du message OOM dans le journal) mais le manifeste
        # etait parfaitement uniforme sur format/forme/group_size — la
        # vraie cause etait l'echelle AWQ, posee sur certains experts
        # seulement par le convertisseur une fois l'echappatoire int8
        # retiree (snr_floor=0), invisible sans ce champ.
        self._raison_repli = ""

        awq = {}                               # nom -> table [E, K] bf16 (x / s[e]) ou None
        hadamard = {}                          # nom -> bloc de rotation (0 : aucune)

        def one(nom, projs):
            ws = [p.qweight for p in projs]
            # Échelle AWQ PAR EXPERT dans la pile (Sage, sage-glm-awq-pile-15-09) :
            # les lignes sont déjà rassemblées par expert (route+pack, eid) ;
            # x_ligne / s[e] avant la quantification / le GEMV = exactement ce que
            # fait la boucle par expert (QuantLinear.forward : scaler.apply).
            # Refus seulement pour la rotation Hadamard (pas par ligne).
            scs = [p.scaler for p in projs]
            # Rotation de Hadamard (sage-hadamard-16-09, `hadamard_block` du
            # manifeste, 512 sur GLM) : acceptée si tous les experts de la
            # projection portent le MÊME bloc — la pile tourne l'entrée une
            # fois pour tous (noyau nvfp4_quant_act sur le chemin MMA,
            # fwht_activations en torch sur GEMV / direct / _grouped_mm), la
            # boucle par expert le fait dans ChannelScaler.apply : mêmes bits.
            blocs = {int(sc.hadamard_block or 0) for sc in scs if sc is not None} or {0}
            if len(blocs) != 1:
                self._raison_repli = f"{nom} : blocs de Hadamard différents entre experts {sorted(blocs)}"
                return None
            hadamard[nom] = blocs.pop()
            if hadamard[nom] and any(sc is None for sc in scs):
                self._raison_repli = f"{nom} : rotation Hadamard sur une partie des experts seulement"
                return None
            if any(sc is not None and sc.scale is not None for sc in scs) or _MOE_AWQ_TEMOIN:
                # ACVRAM_MOE_AWQ_TEMOIN=1 : tables de 1 même sans échelle (le
                # chargeur les voit, la garde d'unité doit les sauter : coût 0
                # scellé par Sage § 8) ; =2 : produit forcé, témoin du coût du
                # chemin (+0,12 ms/pas mesuré le 15/09) ; =3 : tables ignorées
                # (sorties fausses, témoin de coût d'un convertisseur à échelles).
                K_in = getattr(ws[0], "padded_in", None) or ws[0].qweight.shape[1]
                dev = ws[0].qweight.device
                table = torch.ones(len(projs), K_in, dtype=torch.bfloat16, device=dev)
                for e, sc in enumerate(scs):
                    if sc is not None and sc.scale is not None:
                        v = sc.scale.to(dev, torch.bfloat16)
                        table[e, :v.numel()] = v
                # Table = unité (échelle absente écrite comme identité explicite,
                # Manon 2205709) : x / 1 ne change rien, on saute le produit.
                unite = bool(torch.all(table == 1).item())
                awq[nom] = None if (unite and _MOE_AWQ_TEMOIN != 2) or _MOE_AWQ_TEMOIN == 3 else table
            else:
                awq[nom] = None
            if all(isinstance(w, NVFP4Tensor) for w in ws):
                if len({(w.shape, w.padded_in) for w in ws}) != 1:
                    self._raison_repli = f"{nom} : forme ou padding differents entre experts"
                    return None
                if any(p.streamed is not None for p in projs):
                    return self._table_pile(nom, ws)
                qw = torch.stack([w.qweight for w in ws]).contiguous()
                bs = torch.stack([w.block_scale.view(torch.uint8)
                                  for w in ws]).contiguous()
                gs = torch.tensor([w.global_scale_float() for w in ws],
                                  dtype=torch.float32, device=qw.device)
                for e, w in enumerate(ws):     # vues : une seule mémoire
                    w.qweight = qw[e]
                    w.block_scale = bs[e].view(torch.float8_e4m3fn)
                return ("nvfp4", qw, bs, gs, ws[0].padded_in, ws[0].shape[0])
            if all(isinstance(w, INT4Tensor) for w in ws):
                if any(p.streamed is not None for p in projs):
                    self._raison_repli = f"{nom} : INT4 exilé, pas de table pour ce format"
                    return None             # pas de pendant table pour INT4
                if len({(w.shape, w.padded_in, w.group_size) for w in ws}) != 1:
                    self._raison_repli = f"{nom} : forme, padding ou group_size differents entre experts (INT4)"
                    return None
                qw = torch.stack([w.qweight for w in ws]).contiguous()
                sc = torch.stack([w.scales for w in ws]).contiguous()
                zr = torch.stack([w.zeros for w in ws]).contiguous()
                for e, w in enumerate(ws):
                    w.qweight, w.scales, w.zeros = qw[e], sc[e], zr[e]
                return ("int4", qw, sc, zr, ws[0].padded_in,
                        ws[0].group_size, ws[0].shape[0])
            # "ni tout NVFP4, ni tout INT4" NE PROUVE PAS un mélange : un
            # troisième format UNIFORME (bf16 clair, q3n...) tombe ici aussi
            # et n'a rien d'hétérogène. Distinguer les deux : le premier est
            # attendu (ce chemin ne sait construire une pile QUE pour NVFP4/
            # INT4 — aucun noyau groupé bf16 n'existe, `_forward_prefill_
            # grouped` refuse tout pile dont le tag n'est pas "nvfp4") ; le
            # second est une vraie anomalie de conversion à investiguer.
            # Trouvé le 15/09 sur un GLM-4.7-Flash converti `--format bf16` :
            # le message annonçait un mélange qui n'existait pas.
            types = {type(w).__name__ for w in ws}
            if len(types) == 1:
                self._raison_repli = (
                    f"{nom} : format {types.pop()} uniforme mais non pris en "
                    f"charge par ce chemin (ni NVFP4 ni INT4 — attendu pour "
                    f"un modèle non quantifié, pas une anomalie)")
            else:
                self._raison_repli = (
                    f"{nom} : formats de quantification réellement mélangés "
                    f"entre experts ({', '.join(sorted(types))})")
            return None

        piles = {}
        try:
            for nom in self._noms_experts():
                pile = one(nom, [getattr(e, nom) for e in self.experts])
                if pile is None:
                    print(f"[acvram] repli lent (pas de graphes, boucle par "
                          f"expert) sur cette couche : {self._raison_repli}",
                          flush=True)
                    return False
                piles[nom] = pile
        except torch.OutOfMemoryError:
            # la pile d'une projection double transitoirement sa mémoire ;
            # un modèle qui remplit la carte (80B) reste sur la boucle par
            # expert plutôt que de mourir ici
            self._raison_repli = "mémoire GPU insuffisante pendant la construction de la pile"
            torch.cuda.empty_cache()
            print("[acvram] piles d'experts : mémoire GPU insuffisante, boucle par expert",
                  flush=True)
            return False
        # gate et up lisent la même entrée : échelles égales -> un seul
        # rassemblement et une seule quantification (la table d'up EST celle
        # de gate) ; distinctes (GLM AWQ par expert, sage-glm-awq-pile) ->
        # seconde ligne xs2 / x_u, seconde quantification, gate+up non fusionnés.
        g, u = awq.get("gate_proj"), awq.get("up_proj")
        if g is not None and u is not None and torch.equal(g, u):
            awq["up_proj"] = g
        awq["up_distinct"] = not (awq.get("up_proj") is g)
        awq["hadamard"] = hadamard
        if hadamard.get("gate_proj", 0) != hadamard.get("up_proj", 0):
            self._raison_repli = "gate et up : blocs de Hadamard différents (même entrée)"
            return None
        self._stacks = piles
        self._stacks_awq = awq
        self._raison_marlin = ""
        self._stacks_marlin = self._construire_marlin(piles, awq, hadamard)
        self._tensor_refus = self._raison_tensor()              # pièce 65 : "" = chemin tensor pris en charge
        if self._stacks_marlin is not None:
            if _DOUBLE_DIAG:
                self.__dict__["experts_layout"] = "double(diag)"
            else:
                self._liberer_pile_naturelle()
        self._rendre_le_cache_apres_la_pile()
        return True

    @staticmethod
    def _rendre_le_cache_apres_la_pile() -> None:
        """Rend au pilote les blocs libérés par la construction de la pile.

        gemma-4-26B-A4B (chantier-gemma-capture-godet1-20-09) : la pile se
        construit PARESSEUSEMENT, dans `GraphRunner._eligible` du premier
        moteur (graphs.py) ou au premier `forward`. Chaque couche empile ses
        experts (copie), les repacke en Marlin (seconde copie) puis rend les
        sources et la pile naturelle — mais « rend » à l'allocateur PyTorch,
        qui garde les segments en cache : sur gemma, 3 840 experts de moins
        d'un Mio par projection (704 × 2 816 / 2 = 991 Kio) vivent dans le
        bassin des petits blocs (segments de 2 Mio), inutilisable pour toute
        autre demande. Au moment de la première capture, `mem_get_info` ne
        voit presque rien de libre alors que `reserved − allocated` porte
        des Gio (Ornith, journal du 19/09 : 376 Mio libres sur 31,4 Gio pour
        ~19 Gio de poids + KV d'après le manifeste — ≥ 11 Gio en cache). Or
        l'allocateur NE rend PAS son cache pendant une capture (garde
        `captures_underway` de CUDACachingAllocator::malloc) : le bassin
        privé du graphe exige des segments neufs, cudaMalloc échoue, la
        capture échoue — seul le PREMIER moteur, celui qui a construit les
        piles ; les suivants passent parce que capture-godets.py fait
        `empty_cache()` entre deux moteurs. Même cause pour Triton (Ornith :
        `Triton Error [CUDA]: out of memory` dans l'autotune de fla) : le
        pilote alloue hors de l'allocateur PyTorch, donc hors de son cache.
        Un `empty_cache()` ici, après chaque pile construite, coûte une
        synchronisation par couche au chargement et rien en service.

        `ACVRAM_PILE_SANS_RENDU=1` : témoin de mesure (bras A de la chaîne
        carte, sans le correctif) — nommé une fois au journal, jamais un
        défaut."""
        if os.environ.get("ACVRAM_PILE_SANS_RENDU") == "1":
            if not getattr(MoEBlock, "_sans_rendu_dit", False):
                MoEBlock._sans_rendu_dit = True
                print("[acvram] TÉMOIN ACVRAM_PILE_SANS_RENDU=1 : cache non rendu après "
                      "les piles (bras sans correctif)", flush=True)
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _liberer_pile_naturelle(self) -> None:
        """Disposition UNIQUE (sage-p1-disposition-unique-18-09) : la pile
        NVFP4 n'a servi que de source au repack ; ses codes et échelles sont
        rendus (préfill ET décodage lisent `_stacks_marlin`). Les experts
        gardent un GABARIT vide (même geste que runner._demote_expert :
        `[:0].cpu().clone()`, métadonnées intactes) — tout chemin qui
        relirait la pile naturelle casse au lieu de mesurer une double
        disposition. `_stacks[n]` garde son genre « nvfp4 », son K et son M
        (`pg[4]`, `pg[5]` servent aux formes) mais qw/bs valent None."""
        for nom in list(self._stacks):
            pile = self._stacks[nom]
            if pile[0] != "nvfp4":
                continue
            for e in self.experts:
                lin = getattr(e, nom)
                t = lin.qweight
                if t is None or t.qweight is None or t.qweight.numel() == 0:
                    continue
                gabarit = type(t)(t.qweight[:0].cpu().clone(), t.block_scale[:0].cpu().clone(),
                                  t.global_scale.detach().cpu().clone(), t.shape, t.padded_in)
                gs = t.__dict__.get("_gs_f")
                if gs is not None:
                    gabarit.__dict__["_gs_f"] = gs
                lin.qweight = gabarit
            _, _, _, gs, k, m = pile
            self._stacks[nom] = ("nvfp4", None, None, gs, k, m)
        self.__dict__["experts_layout"] = "marlin-w13" if "w13" in (getattr(self, "_stacks_marlin", None) or {}) else "marlin"

    def _raison_tensor(self) -> str:
        """Pièce 65 : raison statique du repli GEMV du chemin tensor ("" = pris en charge), depuis les piles
        et le port. Sans port compilé (à sec, CI sans carte), le défaut REPLIE au lieu de lever (Jerome, 23/09) :
        un défaut ne lève jamais là où le GEMV marchait."""
        marlin = getattr(self, "_stacks_marlin", None)
        awq = getattr(self, "_stacks_awq", {}) or {}
        if marlin is None:
            return f"pas de piles Marlin ({getattr(self, '_raison_marlin', '') or 'disposition naturelle'})"
        from ..kernels import marlin_port as MP
        if MP.charger(compiler=False) is None:
            return "port Marlin non compilé (à sec : python outils/banc-marlin-p1-18-09.py --compiler-seulement)"
        mg, md = marlin["gate_proj"], marlin["down_proj"]
        return forme_tensor_refus({"nvfp4"}, mg[3], mg[4], md[4], len(self.experts),
                                  awq.get("gate_proj") is None and awq.get("up_proj") is None and awq.get("down_proj") is None,
                                  awq.get("hadamard", {}).get("down_proj", 0))

    def _construire_marlin(self, piles, awq, hadamard):
        """P1 (sage-p1-porte-marlin-18-09, X = 152 TFLOPS au banc) : la
        DISPOSITION des experts, repackée pour la GEMM groupée classe Marlin
        (kernels/marlin_port, vLLM v0.29.0) — gate, up, down chacune avec son
        échelle globale par expert (gate et up ne partagent pas la leur :
        pas de w13 fusionné). Disposition UNIQUE depuis
        sage-p1-disposition-unique-18-09 : le décodage la lit aussi
        (`nvfp4_gemv_marlin`, forme (b)) et la pile NVFP4 est rendue après le
        repack (`_liberer_pile_naturelle`) : `experts_layout=marlin`, rien de
        plus à réserver au Plan.
        AWQ par expert et Hadamard (Coder classé 302025e : table [E, K])
        sont appliqués À L'ACTIVATION avant toute GEMM du préfill
        (`_forward_prefill_grouped` : `xs / awq_g[e_sorted]`, `xs_u`,
        `act / awq_d[e_sorted]`, `fwht_activations` hors mma) — les poids
        repackés sont les mêmes codes, l'échelle reste côté x : rien à
        refuser (une première version refusait toute table AWQ, à tort —
        Sage/Jérôme 18/09). Refusée, avec sa raison : pile non NVFP4, formes
        hors tuiles (K, N multiples de 64), extension non compilée à sec
        (REGLES § 6 : jamais de nvcc sous le verrou —
        `outils/banc-marlin-p1-18-09.py --compiler-seulement`)."""
        if _PREFILL_GROUPED != "marlin" and _GEMV_LAYOUT != "marlin" and not _DOUBLE_DIAG:
            return None
        raison = None
        if any(piles[n][0] != "nvfp4" for n in ("gate_proj", "up_proj", "down_proj") if n in piles) \
                or "gate_proj" not in piles:
            raison = "piles non NVFP4"
        elif awq.get("up_distinct") and _MARLIN_DISTINCT != "1":
            # la disposition unique ne rend la pile naturelle QUE si le GEMV
            # Marlin couvre la forme du MoE (Sage, 19/09, verdict-glm-b12) :
            # gate et up à entrées distinctes (tables AWQ séparées, GLM
            # k48-calibA) ne sont pas servies par le noyau fusionné — au premier
            # pas GLM, `_grouped` recevait une pile rendue (qw=None), serveur
            # mort. Refus nommé : pile gardée, chemins d'avant (prefill
            # « groupe », décodage `_grouped`) ; la forme distincte est le
            # chantier C10 (GEMV Marlin à une projection, gate puis up).
            raison = "gate/up à entrées distinctes (tables AWQ séparées) : GEMV Marlin fusionné inapplicable — chemin d'avant gardé (C10)"
        elif piles["gate_proj"][1].device.type != "cuda":
            # dimension « appareil » (MECANISMES) : gptq_marlin_repack n'a qu'un
            # noyau CUDA ; à sec (tests, CUDA_VISIBLE_DEVICES vide) la pile
            # naturelle reste le seul chemin, sans NotImplementedError.
            raison = "pile hors CUDA (à sec) : Marlin exige la carte"
        else:
            for n in ("gate_proj", "up_proj", "down_proj"):
                _, qw, bs, gs, k, m = piles[n]
                if k % 64 or qw.shape[1] % 64:
                    raison = f"{n} : K={k} ou N={qw.shape[1]} non multiple de 64"
        if raison is None:
            from ..kernels import marlin_port as MP
            if MP.charger(compiler=False) is None:
                raison = "extension Marlin non compilée à sec (banc-marlin-p1 --compiler-seulement)"
        if raison is not None:
            # La raison vit sur l INSTANCE (pièce 44 du 22/09) : le message
            # n est imprimé qu une fois par session, mais `regime_ligne` doit
            # pouvoir nommer LA couche refusée et POURQUOI — « marlin(47/48) »
            # sans la raison a coûté une mesure de carte pour rien.
            self._raison_marlin = raison
            if not getattr(MoEBlock, "_marlin_refus_dit", False):
                MoEBlock._marlin_refus_dit = True
                print(f"[acvram] disposition Marlin refusée : {raison} — pile naturelle gardée, prefill « groupe », décodage d'avant", flush=True)
            return None
        from ..kernels import marlin_port as MP
        out = {}
        for n in ("gate_proj", "up_proj", "down_proj"):
            _, qw, bs, gs, k, m = piles[n]
            w, sc, g = MP.preparer_pile(qw, bs.view(torch.float8_e4m3fn) if bs.dtype == torch.uint8 else bs,
                                        gs.reshape(-1).to(torch.float32))
            out[n] = (w, sc, g, k, m)
        if _MOE_W13 and not awq.get("up_distinct"):          # entrées distinctes : deux GEMV, pas de w13
            wg, sg, gg, k, m = out["gate_proj"]
            wu, su, gu, ku, mu_ = out["up_proj"]
            if (k, m) != (ku, mu_) or wg.shape != wu.shape or sg.shape != su.shape:
                raise ValueError(f"ACVRAM_MOE_W13 : gate {tuple(wg.shape)} et up {tuple(wu.shape)} de formes différentes")
            # La disposition Marlin est locale par tuile de 64 colonnes : chaque ligne de tuiles de w13 porte les
            # tuiles de gate puis celles d up — c est une pile Marlin valide de largeur 2N (pièce 75 : l inverse,
            # un w13 vLLM découpé, sert gate et up à 8·10⁻³).
            w13, s13 = torch.cat([wg, wu], dim=2).contiguous(), torch.cat([sg, su], dim=2).contiguous()
            del wg, wu, sg, su
            out["w13"] = (w13, s13, gg, gu, k, m, (gu / gg).contiguous(), torch.ones_like(gg))
            # gate/up rendues : les champs de forme restent (K, N, échelles globales) ; poids et échelles à None,
            # tout chemin qui les relirait casse au lieu de mesurer une double disposition.
            out["gate_proj"], out["up_proj"] = (None, None, gg, k, m), (None, None, gu, k, m)
        return out

    def _grouped(self, x32: torch.Tensor, pile, expert_ids, token_ids, tri=None):
        if pile[0] == "nvfp4":
            _, qw, bs, gs, k, m = pile
            if tri is not None:
                eid_s, ordre = tri
                return kernels.nvfp4_gemv_grouped_v2(x32, qw, bs, gs, eid_s, token_ids[ordre.long()].contiguous(),
                                                     ordre, k)[:, :m]
            return kernels.nvfp4_gemv_grouped(x32, qw, bs, gs, expert_ids,
                                              token_ids, k)[:, :m]
        if pile[0] == "nvfp4_table":
            _, tq, tb, gs, k, m = pile
            return kernels.nvfp4_gemv_grouped_table(x32, tq, tb, gs, expert_ids,
                                                    token_ids, k, m)[:, :m]
        _, qw, sc, zr, k, gsz, m = pile
        return kernels.int4_gemv_grouped(x32, qw, sc, zr, expert_ids,
                                         token_ids, k, gsz)[:, :m]

    # -- prefill : GEMM groupées sur les jetons triés par expert -------------
    def _pile_bf16(self, pile) -> Optional[torch.Tensor]:
        """La pile d'experts d'une projection déquantifiée en bf16 [E, M, K]
        (transitoire : ~850 Mo par projection pour 180 experts de 1024x2304)."""
        if pile[0] != "nvfp4":
            return None
        _, qw, bs, gs, k, m = pile
        E, M = qw.shape[0], qw.shape[1]
        from ..quant.formats import NVFP4Tensor
        plat = NVFP4Tensor.__new__(NVFP4Tensor)
        plat.qweight = qw.view(E * M, -1)
        plat.block_scale = bs.view(E * M, -1).view(torch.float8_e4m3fn)   # la pile garde des octets
        plat.global_scale = torch.ones((), dtype=torch.float32, device=qw.device)
        plat.padded_in = k
        plat.shape = (E * M, k)
        # échelle globale par expert appliquée dans le noyau (une passe de
        # moins sur ~30 Go de bf16 au prefill d'un 30B)
        w = kernels.nvfp4_dequant(plat, torch.bfloat16,
                                  gscale_rows=gs.reshape(-1).to(torch.float32),
                                  rows_per_group=M).view(E, M, k)
        if _PREFILL_W8R == "1":
            # porte W8r (sage-poursuite-chantiers-19-09) : les experts que ce chemin
            # déquantifie en bf16 sont re-arrondis en int8 par ligne — la perte d'un
            # format d'expert int8 par ligne, mesurée sans le stocker (chemins
            # grouped_mm/bmm seulement : ils sont les seuls à lire cette pile bf16) ;
            # EN PLACE, par blocs de lignes : aucun temporaire de la taille de la
            # pile (OOM 768 Mio, verdict-porte-w8r-19-09)
            from ..quant.fakequant_activation import fake_quantize_w8_row
            w = fake_quantize_w8_row(w)
            self._chemin('w8r')
        return w[:, :m, :]

    @staticmethod
    def _plan_bmm(cnt: torch.Tensor, pas: int = 32, etirement: float = 1.5):
        """Plan d'une GEMM groupée bf16 par ``torch.bmm`` sur seaux d'experts
        (sage-profil-verdict-17-09 § 2 A) : sur sm_120 / torch 2.14,
        ``torch._grouped_mm`` se déroule en un ``aten::mm`` par expert plus
        une copie DtoH des ``offs`` — 432 synchronisations par prefill de
        48 couches, et des GEMM de 128 lignes à ~26 % du pic.

        Les experts non vides, triés par compte décroissant, sont groupés en
        seaux : un seau reçoit les experts dont le compte est ≥ cap /
        ``etirement`` (cap = compte maximal du seau arrondi à ``pas``), les
        lignes de chaque expert sont rembourrées à ``cap`` — le travail
        inutile est borné par ``etirement`` — et un seul ``bmm`` sert le
        seau. UNE lecture des comptes sur l'hôte par couche (les trois
        projections partagent le plan), au lieu d'une par pile.

        Renvoie (seaux, src, dst, G) : ``seaux`` = [(experts LongTensor,
        cap, base)], ``src[i]`` = ligne d'entrée triée par expert, ``dst[i]``
        = son emplacement dans le tampon rembourré (tous seaux bout à bout,
        ``base`` = début du seau), ``G`` = nombre de lignes du tampon."""
        comptes = cnt.tolist()
        offs = [0]
        for c in comptes:
            offs.append(offs[-1] + c)
        ordre = sorted((e for e, c in enumerate(comptes) if c), key=lambda e: -comptes[e])
        seaux, base = [], 0
        i = 0
        while i < len(ordre):
            cap = -(-comptes[ordre[i]] // pas) * pas
            j = i + 1                                  # le premier expert entre toujours
            while j < len(ordre) and comptes[ordre[j]] * etirement >= cap:
                j += 1
            experts = ordre[i:j]
            seaux.append((experts, cap, base))
            base += cap * len(experts)
            i = j
        src, dst = [], []
        for experts, cap, b0 in seaux:
            for r, e in enumerate(experts):
                src.append(torch.arange(offs[e], offs[e + 1]))
                dst.append(torch.arange(b0 + r * cap, b0 + r * cap + comptes[e]))
        src = torch.cat(src).to(cnt.device) if src else torch.empty(0, dtype=torch.long, device=cnt.device)
        dst = torch.cat(dst).to(cnt.device) if dst else src
        seaux = [(torch.tensor(ex, dtype=torch.long, device=cnt.device), cap, b0)
                 for ex, cap, b0 in seaux]
        return seaux, src, dst, base

    @staticmethod
    def _grouped_bmm(xs: torch.Tensor, w: torch.Tensor, plan) -> torch.Tensor:
        """``xs`` [G, K] trié par expert, ``w`` [E, M, K] bf16 → [G, M] bf16 :
        même résultat que ``torch._grouped_mm(xs, w.transpose(1, 2), offs)``
        à l'ordre d'accumulation près (juge : tests/test_gemm_grouped_w4a16.py)."""
        seaux, src, dst, G = plan
        M = w.shape[1]
        tampon = torch.zeros(G, xs.shape[1], dtype=xs.dtype, device=xs.device)
        tampon[dst] = xs[src]
        sortie = torch.empty(G, M, dtype=xs.dtype, device=xs.device)
        for experts, cap, b0 in seaux:
            n = experts.numel()
            bloc = tampon[b0:b0 + n * cap].view(n, cap, -1)
            sortie[b0:b0 + n * cap] = torch.bmm(bloc, w[experts].transpose(1, 2)).view(n * cap, M)
        y = torch.empty(xs.shape[0], M, dtype=xs.dtype, device=xs.device)
        y[src] = sortie[dst]
        return y

    @staticmethod
    def _tuiles(cnt: torch.Tensor, bt: int = 16, t_max: Optional[int] = None):
        """Découpe chaque expert en tuiles de ``bt`` jetons consécutifs.

        Renvoie (expert, premier jeton, compte) par tuile — la grille du
        noyau de GEMM groupée, qui ne connaît qu'un expert par bloc.

        Grille de taille FIXE ``t_max`` (bead runner, 14/09 soir — prérequis
        (ii) du levier MoE MMA décodage, Sage : l'ancien ``int(ntiles.sum())``
        synchronisait l'hôte, incapturable dans un graphe CUDA où le rejeu
        est 98 % du pas). Les tuiles au-delà du compte réel sont neutralisées
        à ``n=0`` (jamais ``e<0``) : `nvfp4_gemm_grouped_kernel` et
        `nvfp4_gemm_grouped_mma_kernel` déréférencent `gscales[e]`/
        `table_qw[e]` AVANT de lire `n` (acvram_kernels.cu:1879-1882,
        2067-2072) — contrairement aux tuiles fantômes du routage
        (`bucket_batch`, bead pds 14/09), CES noyaux n'ont pas de garde
        `e<0` ; un index hors bornes y serait un accès mémoire invalide, pas
        une tuile ignorée. `e` reste donc toujours dans `[0, E)`, `n=0`
        suffit (déjà le cas normal d'une dernière tuile partielle — la
        garde `j < nt` du noyau, ligne ~1899/1940, s'applique identiquement).

        ``t_max`` : à fournir par l'appelant qui veut une grille capturable —
        il connaît ``t`` (jetons) et ``top_k`` statiquement, ``ceil(T/bt) +
        E`` couvre le pire cas (jetons maximalement dispersés sur les
        experts — chaque expert non vide coûte au moins une tuile, quel que
        soit son compte). Omis : taille EXACTE (``tot``, ancien
        comportement bit-à-bit — un appelant qui n'a pas explicitement
        demandé une grille rembourrée ne doit rien voir de différent, ni
        dans la forme des tenseurs rendus ni dans la mémoire allouée) ;
        ``.item()``, non capturable, pas le chemin de ce bead.
        """
        dev = cnt.device
        E = cnt.numel()
        if t_max is None:
            t_max = int((((cnt + bt - 1) // bt).sum()).item())
        starts = torch.cumsum(cnt, 0) - cnt
        ntiles = (cnt + bt - 1) // bt
        base = torch.cumsum(ntiles, 0) - ntiles
        slot = torch.arange(t_max, device=dev)
        tile_e = (torch.searchsorted(base, slot, right=True) - 1).clamp(
            min=0, max=max(E - 1, 0))
        idx = slot - base[tile_e]
        t0 = starts[tile_e] + idx * bt
        n = torch.clamp(cnt[tile_e] - idx * bt, min=0, max=bt)
        return (tile_e.to(torch.int32), t0.to(torch.int32), n.to(torch.int32))

    def _gemm(self, pile, xs, tiles, brut=False):
        """``brut`` : rend la sortie rembourrée [G, M_pile] sans la vue [:, :m]
        (la glue lit la foulée elle-même et évite une copie)."""
        _, qw, bs, gs, k, m = pile
        y = kernels.get_extension().nvfp4_gemm_grouped(
            qw, bs, gs, xs, tiles[0], tiles[1], tiles[2], k)
        return y if brut else y[:, :m]

    def _tables_adresses(self, pile):
        """Tables d'adresses identité (contrat bead pds) : chaque expert à sa
        tranche de la pile résidente. Le placement par expert les remplacera."""
        _, qw, bs, _, _, _ = pile
        cle = (qw.data_ptr(), bs.data_ptr())
        cache = self.__dict__.setdefault("_tables_mma", {})
        if cle not in cache:
            E = qw.shape[0]
            ar = torch.arange(E, dtype=torch.int64)
            cache[cle] = ((qw.data_ptr() + ar * qw.stride(0)).to(qw.device),
                          (bs.data_ptr() + ar * bs.stride(0)).to(qw.device))
        return cache[cle]

    def _gemm_mma(self, pile, xq, xsf, tiles, brut=False, bt=None, grow=None):
        """``grow`` [G] fp32 : l'échelle globale par ligne d'activation rendue
        par nvfp4_quant_act (sage-glm-pile-correctif § 7), multipliée par
        gscales[e] dans l'épilogue — sans elle les codes E2M1 sont faux."""
        _, qw, _, gs, k, m = pile
        tq, tb = self._tables_adresses(pile)
        y = kernels.get_extension().nvfp4_gemm_grouped_mma(
            tq, tb, gs, xq, xsf, tiles[0], tiles[1], tiles[2],
            qw.shape[1], k, bt or _MOE_MMA_BT, _MOE_MMA_ETAGES, _MOE_MMA_KS, grow)
        return y if brut else y[:, :m]

    def _decal_marlin(self, nom: str) -> int:
        """`decal` d'exposant des échelles Marlin de la pile `nom` : 15 + log2 facteur,
        facteur = g_naturel · 2¹¹⁹ / g_marlin (traiter_echelle_globale), une puissance
        de 2 par pile — calculé une fois."""
        cache = self.__dict__.setdefault("_decals_marlin", {})
        if nom not in cache:
            import math
            g_nat = self._stacks[nom][3].reshape(-1)[0].item()
            g_mar = self._stacks_marlin[nom][2].reshape(-1)[0].item()
            facteur = g_nat * (2.0 ** 119) / g_mar
            lf = int(round(math.log2(facteur)))
            if 2.0 ** lf != facteur:
                raise RuntimeError(f"C17 : facteur Marlin de {nom} n'est pas une puissance de 2 ({facteur})")
            cache[nom] = 15 + lf
        return cache[nom]

    def _gemm_mma_marlin(self, nom: str, xq, xsf, tiles, grow=None, bt=None):
        """C17 : la GEMM groupée MMA (mma2) sur la DISPOSITION MARLIN de la pile `nom`
        (w_marlin [E, K/16, 2N] int32, s_marlin [E, K/16, N]), échelle globale
        NATURELLE à l'épilogue ; rend y [G, N] bf16 (N rembourré de la pile)."""
        w, sc, _, k, m = self._stacks_marlin[nom]
        gs = self._stacks[nom][3]
        cle = ("marlin", w.data_ptr(), sc.data_ptr())
        cache = self.__dict__.setdefault("_tables_mma", {})
        if cle not in cache:
            ar = torch.arange(w.shape[0], dtype=torch.int64)
            cache[cle] = ((w.data_ptr() + ar * w.stride(0) * w.element_size()).to(w.device),
                          (sc.data_ptr() + ar * sc.stride(0)).to(w.device))
        tq, tb = cache[cle]
        N = w.shape[2] // 2
        return kernels.get_extension().nvfp4_gemm_grouped_mma(
            tq, tb, gs, xq, xsf, tiles[0], tiles[1], tiles[2],
            N, k, bt or 16, _MOE_MMA_ETAGES, _MOE_MMA_KS, grow, self._decal_marlin(nom))

    # Chemins de DÉCODAGE MoE, tels que `_chemin` les compte : la ligne de régime
    # imprime celui qui a été atteint (runner.chemin_moe_atteint), jamais l env.
    CHEMINS_DECODAGE = ("gemv_marlin", "gemv_v1", "decode_mma", "decode_mma_marlin", "bmm", "marlin_tensor")

    def _chemin(self, nom: str) -> None:
        """Compteur du chemin RÉELLEMENT pris au préfill (REGLES § 7 : « noyau
        atteint, pas fonction appelée » — trois tests d'équivalence ont
        comparé sans l'atteindre) ; tout test de régime l'asserte AVANT de
        comparer (`conftest.attendre_chemin`)."""
        c = self.__dict__.setdefault("chemins", {})
        c[nom] = c.get(nom, 0) + 1
        self.__dict__["dernier_chemin"] = nom

    def _forward_prefill_grouped(self, x, topw, topi) -> Optional[torch.Tensor]:
        if self._stacks is None or "gate_proj" not in self._stacks:
            return None                                # experts sans porte : boucle
        pg, pu, pd = (self._stacks[n] for n in ("gate_proj", "up_proj", "down_proj"))
        if any(p[0] != "nvfp4" for p in (pg, pu, pd)):
            return None
        ext = kernels.get_extension()
        unique = getattr(self, "_stacks_marlin", None) is not None and pg[1] is None
        # La GEMM groupée relit les poids d'un expert une fois par tuile de
        # 16 jetons ; la déquantification, elle, les écrit puis les relit en
        # bf16 une seule fois quel que soit le lot. Le premier gagne tant que
        # les experts reçoivent peu de jetons — croisement mesuré vers 50
        # jetons par expert, voir _MOE_GEMM_MAX.
        par_expert = topi.numel() / max(1, pg[3].shape[0])          # E : l'échelle globale [E] survit à la disposition unique (pg[1] rendu)
        # W4A4 sur la MMA FP4 native (revue/mma-fp4-native-sm120.md) : coupé
        # par défaut tant que la perte de qualité des activations en E2M1
        # n'est pas ramenée sous 1 % (Manon, 13/09 : +2,58 % sans lissage).
        # Une optimisation qui change la sortie est un bogue jusqu'à preuve.
        if _PREFILL_W8R == "1" and (unique or _MOE_MMA or _PREFILL_GROUPED not in ("grouped_mm", "bmm")):
            # la porte W8r ne vit que dans `_pile_bf16` : sur un autre chemin
            # (marlin/groupe/mma, disposition unique) elle serait INERTE et la PPL
            # mesurerait le défaut sous son nom (verdict-porte-w8r-19-09 : 0 chemin
            # w8r compté sous mma). Refus nommé plutôt qu'une mesure fausse.
            raise RuntimeError(
                "ACVRAM_PREFILL_W8R=1 inerte sur ce chemin : il faut ACVRAM_PREFILL_GROUPED=grouped_mm|bmm, "
                "ACVRAM_GEMV_LAYOUT=naturel et ACVRAM_MOE_MMA=0 (la MMA prime sur la pile bf16) — "
                f"ici PREFILL_GROUPED={_PREFILL_GROUPED!r}, disposition unique={unique}, MOE_MMA={_MOE_MMA}")
        mma = (_MOE_MMA and not unique and ext is not None and hasattr(ext, "nvfp4_gemm_grouped_mma")
               and not os.environ.get("ACVRAM_PREFILL_DEQUANT")
               and pg[4] % 64 == 0 and pd[4] % 64 == 0
               and ext.nvfp4_gemm_grouped_mma_disponible())
        direct = mma or (ext is not None and not unique and hasattr(ext, "nvfp4_gemm_grouped")
                         and not os.environ.get("ACVRAM_PREFILL_DEQUANT")
                         and par_expert <= _MOE_GEMM_MAX
                         and pg[4] % 64 == 0 and pd[4] % 64 == 0)
        if not direct and _PREFILL_GROUPED == "grouped_mm" and not hasattr(torch, "_grouped_mm"):
            return None
        t, k = topi.shape
        E = pg[3].shape[0]                             # [E] échelles globales : survit à pg[1] = None (disposition unique)
        permut = kernels.prefill_compact("permut")
        colle = None            # posé par la branche sans « permut » seulement ; lu plus bas par le chemin groupe
                                # (T4 20/09 : UnboundLocalError sous PREFILL_COMPACT=1 + témoin PREFILL_GROUPED=groupe)
        if permut:
            # C15-prefill : les MÊMES ordre / cnt / xs que le chemin d'avant, en
            # moins de lancements — tri stable sur les clés int32 de topi (4
            # passes radix au lieu de 8 sur leur copie int64), comptes par
            # `searchsorted` sur la liste triée (deux petits noyaux au lieu de
            # l'histogramme à atomiques), lignes de x par `ordre // k` (flat_t
            # [i] = i // k : ni arange, ni repeat_interleave, ni gather d'index).
            flat_e = topi.reshape(-1)
            ordre = torch.argsort(flat_e, stable=True)
            e_sorted = flat_e[ordre]
            cnt = _comptes_tries(e_sorted, E)
            if self.__dict__.pop("_compte_en_attente", False):
                self._usage_routage.add_(cnt)          # = scatter_add des uns de `_compter_routage`
            xs = x[torch.div(ordre, k, rounding_mode="floor")].to(torch.bfloat16).contiguous()
        else:
            flat_e = topi.reshape(-1).to(torch.int64)
            flat_t = torch.arange(t, device=x.device).repeat_interleave(k)
            colle = _colle_moe_triton(flat_e.numel(), E, x.device)
            if colle is not None:
                # P0 : tri + histogramme en UN lancement (colle_moe.trier_paires),
                # mêmes ordre/cnt qu'argsort stable + bincount (tests/test_colle_moe.py)
                ordre, _, cnt = colle.trier_paires(flat_e, E)
            else:
                ordre = torch.argsort(flat_e, stable=True)
                cnt = torch.bincount(flat_e, minlength=E)
            xs = x[flat_t[ordre]].to(torch.bfloat16).contiguous()        # [G, H]
            e_sorted = flat_e[ordre]
        # échelle AWQ par expert (sage-glm-awq-pile-15-09) : x_ligne / s[e]
        # comme ChannelScaler.apply en boucle — au prefill aussi, sinon la
        # sortie change en silence dès que les experts portent une échelle
        awq = getattr(self, "_stacks_awq", {})
        xs_u = xs
        awq_g = awq.get("gate_proj")
        awq_u = awq.get("up_proj") if awq.get("up_distinct") else awq_g
        awq_d = awq.get("down_proj")
        hd = awq.get("hadamard", {})
        hd_x, hd_d = hd.get("gate_proj", 0), hd.get("down_proj", 0)
        if not mma:
            # chemins bf16 (direct, _grouped_mm) : rotation et division en
            # torch ; la branche mma les fusionne dans nvfp4_quant_act
            if hd_x:
                xs = fwht_activations(xs.to(torch.bfloat16), hd_x).to(xs.dtype)
                xs_u = xs
            if awq.get("up_distinct") and awq_u is not None:
                xs_u = (xs / awq_u[e_sorted, :xs.shape[1]]).contiguous()
            if awq_g is not None:
                xs = (xs / awq_g[e_sorted, :xs.shape[1]]).contiguous()
                if not awq.get("up_distinct"):
                    xs_u = xs
        # Glue en deux noyaux (moe_act, moe_reduce_trie) : le profil du 13/09
        # donnait 25 % du pas aux conversions fp32, produit, rembourrage,
        # permutation inverse et somme faits en torch sur [G, M] entiers.
        glue = (ext is not None and hasattr(ext, "moe_act")
                and not os.environ.get("ACVRAM_MOE_GLUE_TORCH"))
        code_act = 1 if self.act == "gelu_tanh" else 0

        def _activation(g, u, m, kd, awq_d=awq_d, hd_d=hd_d):
            if glue and g.dtype == torch.bfloat16 and not hd_d:
                return ext.moe_act(g, u, m, kd, code_act, awq_d,
                                   e_sorted.to(torch.int32) if awq_d is not None else None)
            act = (self._act(g[:, :m].to(torch.float32))
                   * u[:, :m].to(torch.float32)).to(torch.bfloat16)
            if hd_d:
                act = fwht_activations(act, hd_d)
            if awq_d is not None:
                act = act / awq_d[e_sorted, :m]
            if act.shape[1] != kd:
                act = F.pad(act, (0, kd - act.shape[1]))
            return act.contiguous()

        if _PREFILL_A4 != "off" and not mma:
            # porte qualité W4A4 (torch) : entrée de gate/up arrondie en NVFP4 ; `both` arrondit
            # aussi l'entrée de down (via _activation ci-dessous)
            partage_a4 = xs_u is xs
            xs = fausse_quant_nvfp4(xs)
            xs_u = xs if partage_a4 else fausse_quant_nvfp4(xs_u)
            if _PREFILL_A4 == "both":
                _act_sans_a4 = _activation
                def _activation(g, u, m, kd, _f=_act_sans_a4):   # noqa: F811
                    return fausse_quant_nvfp4(_f(g, u, m, kd))
        if _PREFILL_A8 != "off" and not mma:
            # porte qualité W4A8 (sage-w4a4-clos-w4a8-porte-19-09) : même geste que la
            # porte A4, activations arrondies en int8 par jeton (le quantificateur du
            # chemin a8/cublas, au bit) ou en E4M3 bloc 16 (témoin `.kind::mxf8f6f4`)
            partage_a8 = xs_u is xs
            xs = fausse_quant_a8(xs, _PREFILL_A8_FMT)
            xs_u = xs if partage_a8 else fausse_quant_a8(xs_u, _PREFILL_A8_FMT)
            if _PREFILL_A8 == "both":
                _act_sans_a8 = _activation
                def _activation(g, u, m, kd, _f=_act_sans_a8):   # noqa: F811
                    return fausse_quant_a8(_f(g, u, m, kd), _PREFILL_A8_FMT)
        if mma:
            # poids ET activations en 4 bits : les activations sont quantifiées
            self._chemin('mma')
            # une fois par entrée (E2M1 bloc 16 + UE4M3) et servent à gate et up
            # Grille EXACTE (t_max omis) : ce chemin n'est jamais capturé dans
            # un graphe (prefill), la grille rembourrée du bead runner (14/09,
            # prérequis (ii) decodage) ne sert a rien ici et a provoque un
            # acces memoire illegal sous forte pression VRAM (cause non
            # identifiee plus loin — la grille exacte l'evite completement,
            # c'est le comportement d'avant ce bead, inchange). `_tuiles`
            # reste capable de rendre une grille fixe pour qui la demande
            # explicitement (decodage, pas ce chemin).
            tiles = self._tuiles(cnt, _MOE_MMA_BT)
            if xs.shape[1] != pg[4]:                   # entrée rembourrée
                xs = F.pad(xs, (0, pg[4] - xs.shape[1])).contiguous()
            es32 = e_sorted.to(torch.int32).contiguous() if (awq_g is not None or awq_u is not None or awq_d is not None) else None
            # x/s[e] et l'échelle globale par ligne dans le noyau ; la GEMM
            # multiplie grow[r] × gscales[e] dans son épilogue
            cpt = _qa_compteurs(xs.device)
            xq, xsf, gr = ext.nvfp4_quant_act(xs, awq_g, es32, cpt, hd_x)
            xq2, xsf2, gr2 = (xq, xsf, gr) if awq_u is awq_g else ext.nvfp4_quant_act(xs, awq_u, es32, cpt, hd_x)
            g = self._gemm_mma(pg, xq, xsf, tiles, brut=True, grow=gr)
            u = self._gemm_mma(pu, xq2, xsf2, tiles, brut=True, grow=gr2)
            act = _activation(g, u, pg[5], pd[4], awq_d=None, hd_d=0)
            aq, asf, gra = ext.nvfp4_quant_act(act, awq_d, es32, cpt, hd_d)
            d = self._gemm_mma(pd, aq, asf, tiles, brut=True, grow=gra)
        elif (_PREFILL_GROUPED == "marlin" or unique) and getattr(self, "_stacks_marlin", None) is not None:
            # AVANT `direct` (Laure, verdict-marlin-p1-situ-18-09 : à petit T par
            # expert, `direct` passait devant et un test n'atteignait jamais Marlin)
            self._chemin('marlin')
            # P1 : GEMM groupée classe Marlin (port vLLM) sur la seconde
            # disposition ; lignes déjà triées par expert (xs), blocs alignés
            # par expert depuis e_sorted, sortie dans l'ordre de xs — la
            # recombinaison ci-dessous ne change pas
            from ..kernels import marlin_port as MP
            G = xs.shape[0]
            mg, mu, md = (self._stacks_marlin[n] for n in ("gate_proj", "up_proj", "down_proj"))
            bloc = MP.choisir_block_size(t, k, E)
            if permut:
                # C15-prefill : e_sorted est déjà trié et cnt connu — les mêmes
                # sorted_ids / expert_ids sans second tri ni second histogramme
                s_ids, e_ids, n_post = MP.aligner_blocs_tries(e_sorted, cnt, bloc, E)
            else:
                s_ids, e_ids, n_post = MP.aligner_blocs(e_sorted.to(torch.int32).unsqueeze(1), bloc, E)
            ws = self._marlin_workspace(x.device)
            uns = self._marlin_uns(G, x.device)
            xs_m = xs if xs.shape[1] == mg[3] else F.pad(xs, (0, mg[3] - xs.shape[1]))
            w13 = self._stacks_marlin.get("w13")
            if w13 is not None:
                # Pièce 82 ter : au préfill, gate et up sont lues DANS w13 par deux GEMM de largeur N (vues de
                # colonnes, largeur stockée 2N). Le Marlin découpe K selon prob_n (82 bis) : une GEMM de 2N ne
                # serait pas au bit des deux GEMM de N, et la KL de fin de préfill le montrait (82 : 0,41 → 0,93).
                # Mêmes octets, même découpe : au bit du chemin séparé. Seul le décodage tensor garde la GEMM 2N.
                n = w13[1].shape[2] // 2
                bg = (w13[0][:, :, :2 * n], w13[1][:, :, :n], w13[2])
                bu = (w13[0][:, :, 2 * n:], w13[1][:, :, n:], w13[3])
            else:
                n = mg[1].shape[2]
                bg, bu = mg[:3], mu[:3]
            g = MP.gemm_moe(xs_m.contiguous(), *bg, s_ids, e_ids, n_post, uns, bloc, 1, G, n, mg[3], ws)
            xu_m = xs_m if xs_u is xs else (xs_u if xs_u.shape[1] == mu[3] else F.pad(xs_u, (0, mu[3] - xs_u.shape[1]))).contiguous()
            u = MP.gemm_moe(xu_m, *bu, s_ids, e_ids, n_post, uns, bloc, 1, G, n, mu[3], ws)
            act = _activation(g, u, pg[5], pd[4])
            d = MP.gemm_moe(act, md[0], md[1], md[2], s_ids, e_ids, n_post, uns, bloc, 1, G, md[1].shape[2], md[3], ws)
        elif direct:
            self._chemin('direct')
            # les poids restent en 4 bits : plus de pile bf16 intermédiaire
            # (trois passes de plusieurs Gio par couche en moins)
            # Grille EXACTE ici aussi -- meme raison que la branche `mma`
            # juste au-dessus.
            tiles = self._tuiles(cnt)
            if xs.shape[1] != pg[4]:                   # entrée rembourrée
                partage = xs_u is xs
                xs = F.pad(xs, (0, pg[4] - xs.shape[1])).contiguous()
                xs_u = xs if partage else F.pad(xs_u, (0, pg[4] - xs_u.shape[1])).contiguous()
            g = self._gemm(pg, xs, tiles, brut=True)
            u = self._gemm(pu, xs_u, tiles, brut=True)
            act = _activation(g, u, pg[5], pd[4])
            d = self._gemm(pd, act, tiles, brut=True)
        elif _PREFILL_GROUPED == "grouped_mm":
            self._chemin('grouped_mm')
            # défaut : `torch._grouped_mm` (déroulé sur sm_120 en un mm par
            # expert + une copie DtoH par pile — mais ses copies s'arrêtent là)
            offs = torch.cumsum(cnt, 0).to(torch.int32)
            wg = self._pile_bf16(pg); g = torch._grouped_mm(xs, wg.transpose(1, 2), offs=offs); del wg
            wu = self._pile_bf16(pu); u = torch._grouped_mm(xs_u, wu.transpose(1, 2), offs=offs); del wu
            act = _activation(g, u, pg[5], pd[4])
            wd = self._pile_bf16(pd); d = torch._grouped_mm(act, wd.transpose(1, 2), offs=offs); del wd
        elif _PREFILL_GROUPED == "w4a16":
            self._chemin('w4a16')
            # B1 : la grille de B0, les poids lus en NVFP4 dans la tuile —
            # ni pile bf16 (nvfp4_dequant 50,8 ms), ni relecture de 60 Go
            from ..kernels import gemm_groupe as gg
            G = xs.shape[0]
            tiles = self._tuiles(cnt, gg.BT, t_max=-(-G // gg.BT) + E)
            g = gg.gemm_groupe_nvfp4(xs, pg[1], pg[2], pg[3].reshape(-1).to(torch.float32), tiles, m=pg[5])
            u = gg.gemm_groupe_nvfp4(xs_u, pu[1], pu[2], pu[3].reshape(-1).to(torch.float32), tiles, m=pu[5])
            act = _activation(g, u, pg[5], pd[4])
            d = gg.gemm_groupe_nvfp4(act, pd[1], pd[2], pd[3].reshape(-1).to(torch.float32), tiles, m=pd[5])
        elif _PREFILL_GROUPED == "groupe" or _PREFILL_GROUPED == "marlin":
            self._chemin('groupe')
            # B0 : un lancement persistant pour les 128 experts, lignes lues
            # par index dans le noyau, grille de tuiles à taille fixe (aucun
            # offset relu sur l'hôte) ; P0 : la grille en un lancement
            from ..kernels import gemm_groupe as gg
            G = xs.shape[0]
            t_max = -(-G // gg.BT) + E
            tiles = colle.tuiles(cnt, gg.BT, t_max) if colle is not None else self._tuiles(cnt, gg.BT, t_max=t_max)
            wg = self._pile_bf16(pg); g = gg.gemm_groupe(xs, wg, tiles); del wg
            wu = self._pile_bf16(pu); u = gg.gemm_groupe(xs_u, wu, tiles); del wu
            act = _activation(g, u, pg[5], pd[4])
            wd = self._pile_bf16(pd); d = gg.gemm_groupe(act, wd, tiles); del wd
        else:
            # bmm par seaux (A, sage-profil-verdict-17-09) : réfuté −36 %, témoin
            self._chemin("bmm")
            plan = self._plan_bmm(cnt)
            wg = self._pile_bf16(pg); g = self._grouped_bmm(xs, wg, plan); del wg
            wu = self._pile_bf16(pu); u = self._grouped_bmm(xs_u, wu, plan); del wu
            act = _activation(g, u, pg[5], pd[4])
            wd = self._pile_bf16(pd); d = self._grouped_bmm(act, wd, plan); del wd
        m_out = pd[5]
        if permut and glue and d.dtype == torch.bfloat16 and x.dtype == torch.bfloat16:
            # C15-prefill : la permutation inverse écrite en int32 d'emblée (le
            # noyau la lit ainsi) — une conversion de moins ; mêmes valeurs
            inv = torch.empty(ordre.numel(), dtype=torch.int32, device=x.device)
            inv[ordre] = torch.arange(ordre.numel(), dtype=torch.int32, device=x.device)
            tw = topw.reshape(-1)
            tw = tw if tw.dtype == torch.float32 else tw.to(torch.float32)
            return ext.moe_reduce_trie(d, tw.contiguous(), inv, m_out, k)
        inv = torch.empty_like(ordre); inv[ordre] = torch.arange(ordre.numel(), device=x.device)
        if glue and d.dtype == torch.bfloat16 and x.dtype == torch.bfloat16:
            tw = topw.reshape(-1)
            tw = tw if tw.dtype == torch.float32 else tw.to(torch.float32)
            return ext.moe_reduce_trie(d, tw.contiguous(), inv.to(torch.int32).contiguous(), m_out, k)
        d = d[:, :m_out].to(torch.float32) * topw.reshape(-1)[ordre].to(torch.float32).unsqueeze(-1)
        return d[inv].view(t, k, -1).sum(dim=1).to(x.dtype)

    _marlin_ws: dict = {}

    def _marlin_workspace(self, device):
        cle = str(device)
        if cle not in MoEBlock._marlin_ws:
            from ..kernels import marlin_port as MP
            MoEBlock._marlin_ws[cle] = MP.espace_travail(device, 4)
        return MoEBlock._marlin_ws[cle]

    def _marlin_uns(self, G: int, device):
        cache = self.__dict__.setdefault("_marlin_uns_cache", {})
        if G not in cache:
            cache[G] = torch.ones(G, 1, dtype=torch.float32, device=device)
        return cache[G]

    def _forward_grouped_mma(self, x, topw, topi) -> Optional[torch.Tensor]:
        """Décodage (t petit) par la GEMM groupée MMA FP4 native — levier (3)
        de Sage (revue/sage-moe-mma-decodage-14-09.md) : à b=12 les GEMV
        dépensent 1,4-2,6 instructions par octet DRAM (71 % des instructions
        du pas, chaque noyau au plafond de 400 W) là où la MMA block-scaled
        en fait 0,13. Même code que la branche ``mma`` du prefill, mais
        CAPTURABLE : grille de tuiles FIXE (``_tuiles(cnt, bt, t_max)``,
        Océane c8096c0), aucun ``.item()``, formes constantes en t et top_k.
        Créneaux fantômes (``topi == -1``) : expert 0 avec poids 0 — une
        contribution nulle et FINIE (une ligne sans tuile lirait une sortie
        non initialisée, que 0 × NaN ne neutralise pas). Rend ``None`` si
        le chemin n'est pas disponible (piles non nvfp4, noyau absent)."""
        st = self._stacks
        if st is None or "gate_proj" not in st:
            return None
        pg, pu, pd = (st[n] for n in ("gate_proj", "up_proj", "down_proj"))
        if any(p[0] != "nvfp4" for p in (pg, pu, pd)):
            return None
        marlin_c17 = False
        if pg[1] is None:                              # disposition unique : la pile naturelle est rendue
            st_m = getattr(self, "_stacks_marlin", None)
            awq0 = getattr(self, "_stacks_awq", {})
            if not (_MOE_DECODE_MMA_MARLIN and st_m is not None
                    and all(n in st_m for n in ("gate_proj", "up_proj", "down_proj")) and "w13" not in st_m
                    and awq0.get("gate_proj") is None and awq0.get("down_proj") is None
                    and not awq0.get("hadamard", {}).get("gate_proj", 0) and not awq0.get("hadamard", {}).get("down_proj", 0)
                    and pd[1] is None):
                return None
            marlin_c17 = True                          # C17 : mma2 lit les tuiles Marlin
        ext = kernels.get_extension()
        if (ext is None or not hasattr(ext, "nvfp4_gemm_grouped_mma")
                or not hasattr(ext, "moe_act") or not hasattr(ext, "moe_reduce_trie")
                or pg[4] % 64 != 0 or pd[4] % 64 != 0
                or not ext.nvfp4_gemm_grouped_mma_disponible()):
            return None
        t, k = topi.shape
        E = pg[3].shape[0]                             # [E] échelles globales : survit à pg[1] = None (disposition unique)
        bt = _MOE_DECODE_MMA_BT
        t_max = -(-(t * k) // bt) + E
        if _MOE_ROUTE_PACK and hasattr(ext, "moe_route_pack") and t * k <= 1024:
            # Un lancement pour tout le frontend (sage-reprise-15-09-b § 2) :
            # ~40 lancements torch par couche (22 µs sous ncu) en un.
            awq = getattr(self, "_stacks_awq", {})
            # les échelles AWQ ne sont plus appliquées ici : nvfp4_quant_act
            # les fusionne (table + e_sorted), une passe [G, K] de moins
            xs, ordre, inv, tw, _cnt, te, t0, tn, e_sorted, xs2 = ext.moe_route_pack(
                topi.contiguous(), topw.contiguous(), x.contiguous(), E, bt, t_max, pg[4])
            tiles = (te, t0, tn)
            awq_g = awq.get("gate_proj")
            awq_u = awq.get("up_proj") if awq.get("up_distinct") else awq_g
            tourne_torch = False
        else:
            # Témoin torch (ACVRAM_MOE_ROUTE_PACK=0) : le contrat bit à bit du
            # noyau, tests/test_moe_route_pack.py.
            fantome = topi < 0
            flat_e = torch.where(fantome, torch.zeros_like(topi), topi).reshape(-1).to(torch.int64)
            tw = torch.where(fantome, torch.zeros_like(topw), topw).reshape(-1)
            tw = tw if tw.dtype == torch.float32 else tw.to(torch.float32)
            flat_t = torch.arange(t, device=x.device).repeat_interleave(k)
            ordre = torch.argsort(flat_e, stable=True)
            # scatter_add_, pas bincount : bincount lit le max sur l'hôte (forme de
            # sortie), incapturable dans un graphe CUDA.
            cnt = torch.zeros(E, dtype=torch.int64, device=x.device).scatter_add_(
                0, flat_e, torch.ones_like(flat_e))
            tiles = self._tuiles(cnt, bt, t_max=t_max)
            xs = x[flat_t[ordre]].to(torch.bfloat16)
            if xs.shape[1] != pg[4]:
                xs = F.pad(xs, (0, pg[4] - xs.shape[1]))
            awq = getattr(self, "_stacks_awq", {})
            e_sorted = flat_e[ordre].to(torch.int32)
            if awq.get("hadamard", {}).get("gate_proj", 0):
                # rotation AVANT la division (ordre de ChannelScaler.apply) ;
                # le noyau ne la refera pas : hd_x est remis à 0 ci-dessous
                xs = fwht_activations(xs, awq["hadamard"]["gate_proj"])
            xs2 = xs
            if awq.get("up_distinct"):
                xs2 = xs if awq.get("up_proj") is None else (xs / awq["up_proj"][e_sorted.long()]).contiguous()
            if awq.get("gate_proj") is not None:
                xs = xs / awq["gate_proj"][e_sorted.long()]
            xs = xs.contiguous()
            if not awq.get("up_distinct"):
                xs2 = xs
            awq_g = awq_u = None                    # déjà divisées en torch (témoin)
            tourne_torch = True
            inv = torch.empty_like(ordre); inv[ordre] = torch.arange(ordre.numel(), device=x.device)
            inv = inv.to(torch.int32)
        awq_d = awq.get("down_proj")
        hd = awq.get("hadamard", {})
        hd_x, hd_d = (0 if tourne_torch else hd.get("gate_proj", 0)), hd.get("down_proj", 0)
        es32 = e_sorted if (awq_g is not None or awq_u is not None or awq_d is not None) else None
        cpt = _qa_compteurs(xs.device)
        xq, xsf, gr = ext.nvfp4_quant_act(xs, awq_g, es32, cpt, hd_x)
        xq2, xsf2, gr2 = ((xq, xsf, gr) if (xs2 is xs and awq_u is awq_g)
                          else ext.nvfp4_quant_act(xs2, awq_u, es32, cpt, hd_x))
        code_act = 1 if self.act == "gelu_tanh" else 0
        if (_MOE_DECODE_FUSED and not marlin_c17 and hasattr(ext, "nvfp4_moe_fused") and bt == 16 and awq_d is None
                and not hd_x and not hd_d and xs2 is xs and awq_u is awq_g
                and pg[4] % 128 == 0 and pg[5] == pd[4] and pd[4] % _MOE_FUSED_TN == 0 and pd[5] % 128 == 0):
            # Port de b12x (sage-reprise-15-09-b § 3-4) : gate+up+act+quant en
            # shared, down par tranches, split-K sériel (bit-reproductible),
            # sortie d [G, M] réduite par moe_reduce_trie comme le chemin B.
            if _MOE_FUSED_ATOMIQUE:
                # témoin atomiques (non reproductible au bit) : rend y directement
                y = self._moe_fused(pg, pu, pd, xq, xsf, tiles, code_act, ordre=ordre, tw=tw, k=k, t=t, grow=gr)
                return y if x.dtype == torch.bfloat16 else y.to(x.dtype)
            d = self._moe_fused(pg, pu, pd, xq, xsf, tiles, code_act, grow=gr)
        elif marlin_c17:
            self._chemin("decode_mma_marlin")
            g = self._gemm_mma_marlin("gate_proj", xq, xsf, tiles, grow=gr, bt=bt)
            u = self._gemm_mma_marlin("up_proj", xq2, xsf2, tiles, grow=gr2, bt=bt)
            act = ext.moe_act(g, u, pg[5], pd[4], code_act)
            aq, asf, gra = ext.nvfp4_quant_act(act, awq_d, es32, cpt, hd_d)
            d = self._gemm_mma_marlin("down_proj", aq, asf, tiles, grow=gra, bt=bt)
        else:
            self._chemin("decode_mma")             # le chemin mma-a4 du décodage, compté comme les autres
            g = self._gemm_mma(pg, xq, xsf, tiles, brut=True, bt=bt, grow=gr)
            u = self._gemm_mma(pu, xq2, xsf2, tiles, brut=True, bt=bt, grow=gr2)
            act = ext.moe_act(g, u, pg[5], pd[4], code_act)
            aq, asf, gra = ext.nvfp4_quant_act(act, awq_d, es32, cpt, hd_d)
            d = self._gemm_mma(pd, aq, asf, tiles, brut=True, bt=bt, grow=gra)
        y = ext.moe_reduce_trie(d, tw.contiguous(), inv.contiguous(), pd[5], k)
        return y if x.dtype == torch.bfloat16 else y.to(x.dtype)

    _fused_ws: dict = {}                 # (device, T, NT2, NS) -> (ws, compteurs), partagé par toutes les couches

    def _moe_fused(self, pg, pu, pd, xq, xsf, tiles, code_act, ordre=None, tw=None, k=0, t=0, grow=None):
        ext = kernels.get_extension()
        tq_g, tb_g = self._tables_adresses(pg)
        tq_u, tb_u = self._tables_adresses(pu)
        tq_d, tb_d = self._tables_adresses(pd)
        K, I, M_out, tn = pg[4], pd[4], pd[1].shape[1], _MOE_FUSED_TN
        T, NT2, NS = tiles[0].numel(), M_out // 128, I // tn
        cle = (xq.device, T, NT2, NS)
        buf = MoEBlock._fused_ws.get(cle)
        if buf is None:
            # partiels [T][NT2][NS][16 x 128] fp32 + compteurs (remis a zero par le
            # noyau) : alloués une fois, réutilisés par toutes les couches
            ws = torch.empty(T * NS * 16 * M_out, dtype=torch.float32, device=xq.device)
            cpt = torch.zeros(T, dtype=torch.int32, device=xq.device)
            buf = MoEBlock._fused_ws[cle] = (ws, cpt)
        ws, cpt = buf
        atom = ordre is not None
        # gate/up : grow[r] × gscales[e] dans l'épilogue FC1 ; l'activation est
        # requantifiée DANS le noyau fusionné sans échelle globale (plancher
        # E4M3 non corrigé sur ce chemin témoin, ACVRAM_MOE_DECODE_FUSED=0)
        return ext.nvfp4_moe_fused(tq_g, tb_g, pg[3], tq_u, tb_u, pu[3], tq_d, tb_d, pd[3],
                                   xq, xsf, tiles[0], tiles[1], tiles[2], ws, cpt,
                                   K, I, M_out, code_act, tn,
                                   ordre.contiguous() if atom else tiles[0], tw.contiguous() if atom else pg[3],
                                   k, t, atom, _MOE_FUSED_ETAGES, grow)

    def _forward_grouped(self, x, topw, topi, eid=None):
        t = x.shape[0]
        ext = kernels.get_extension()
        if eid is not None:                            # route_prep : eid déjà prêt, index par godet
            from ..kernels import route_prep as _rp
            tok, _seq = _rp.index_jetons(t, self.top_k, x.device)
        else:
            eid = topi.reshape(-1).to(torch.int32)
            tok = torch.arange(t, device=x.device,
                               dtype=torch.int32).repeat_interleave(self.top_k)
            _seq = None
        if _TRACE_ROUTAGE and not torch.cuda.is_current_stream_capturing():
            # routages réels pour les rejouer au banc (banc-marlin-decode --routages)
            _ROUTAGES.append(eid.view(-1, self.top_k).cpu())
        if "gate_proj" not in self._stacks:            # experts sans porte (ReLU²)
            u = self._grouped(x.to(torch.float32), self._stacks["up_proj"], eid, tok)
            act = F.relu(u); act = act * act
            seq = _seq if _seq is not None else torch.arange(eid.shape[0], device=x.device, dtype=torch.int32)
            d = self._grouped(act, self._stacks["down_proj"], eid, seq)
            d = d * topw.reshape(-1, 1).to(d.dtype)
            return d.view(t, self.top_k, -1).sum(dim=1).to(x.dtype)
        pg, pu = self._stacks["gate_proj"], self._stacks["up_proj"]
        awq = getattr(self, "_stacks_awq", {})
        distinct = bool(awq.get("up_distinct"))
        hd = awq.get("hadamard", {})
        hd_x, hd_d = hd.get("gate_proj", 0), hd.get("down_proj", 0)
        if hd_x:
            # rotation de Hadamard (poids tournés) : x·H par bloc, la même
            # arithmétique que ChannelScaler.apply — une fois par jeton
            x = fwht_activations(x.to(torch.bfloat16), hd_x).to(x.dtype)
        glue = _mla_glue() >= 1
        eid64 = None                                   # C15 : eid int64 une fois par couche (glue)
        # Pièce 47 : le GEMV Marlin porte lui-même `x / s[e]` (paramètre
        # `xscale`, au bit) — on saute alors le gather, la division et le cast
        # en torch. Condition : chemin Marlin fusionné (gate et up partagent
        # leur table, sinon le noyau n'a qu'un x), extension à jour, et x en
        # bf16 comme la table.
        fuse_ech = (getattr(self, "_stacks_marlin", None) is not None
                    and _GEMV_LAYOUT == "marlin" and ext is not None
                    and hasattr(ext, "nvfp4_gemv_marlin_gateup")
                    and _gemv_marlin_porte_echelle(ext) and not distinct
                    and x.dtype == torch.bfloat16
                    and (awq.get("gate_proj") is not None or awq.get("down_proj") is not None))
        # Lu par la ligne de régime (`echelle_awq=`) : prouver dans le processus
        # qui mesure que la fusion a pris, au lieu de la déduire d'un compte de
        # lancements après coup (REGLES § 3).
        self._echelle_awq = ("gemv" if fuse_ech else
                             "torch" if (awq.get("gate_proj") is not None
                                         or awq.get("down_proj") is not None or distinct)
                             else "aucune")
        if not fuse_ech and (awq.get("gate_proj") is not None or distinct):
            # échelle AWQ par expert : la ligne (jeton, expert) est divisée par
            # s[e] avant les projections, comme ChannelScaler.apply en boucle
            if glue:
                # C15 (chantier-c15-19-09 § Reste) : les mêmes gathers et
                # divisions, mais `tok` int64 servi d'avance (index_jetons),
                # `eid` converti UNE fois par couche au lieu de trois, la ligne
                # x[tok] rassemblée une fois pour gate et up (entrées
                # distinctes), et `tok_g` = `seq` déjà réservé (même arange
                # int32) : −3 lancements par couche Marlin, −6 par couche à
                # tables distinctes, aucun bit changé (des index).
                eid64 = eid.long()
                tok64 = _rp.index_jetons_long(t, self.top_k, x.device) if _seq is not None else tok.long()
                xe = x[tok64].to(torch.bfloat16)
                tg, tu = awq.get("gate_proj"), awq.get("up_proj")
                x_g = (xe / tg[eid64, :x.shape[1]] if tg is not None else xe).to(x.dtype)
                if distinct:
                    x_u = (xe / tu[eid64, :x.shape[1]] if tu is not None else xe).to(x.dtype)
                else:
                    x_u = x_g
                tok_g = _seq if _seq is not None else torch.arange(eid.shape[0], device=x.device, dtype=torch.int32)
            else:
                def _ligne(table):
                    xe = x[tok.long()].to(torch.bfloat16)
                    if table is not None:
                        xe = xe / table[eid.long(), :x.shape[1]]
                    return xe.to(x.dtype)
                x_g = _ligne(awq.get("gate_proj"))
                x_u = _ligne(awq.get("up_proj")) if distinct else x_g
                tok_g = torch.arange(eid.shape[0], device=x.device, dtype=torch.int32)
        else:
            x_g, x_u, tok_g = x, x, tok
        ech_gu = awq.get("gate_proj") if fuse_ech else None
        # v2 : paires triées par expert (argsort stable : déterministe, sous
        # graphe) ; le noyau écrit chaque paire à sa place d'origine
        tri = None
        act = None
        marlin = getattr(self, "_stacks_marlin", None)
        if marlin is not None and _GEMV_LAYOUT != "marlin" and pg[1] is not None:
            marlin = None                              # piles Marlin présentes mais témoin naturel demandé
        # Pièce 65 : défaut aux godets ≥ 2 (T ≥ 2 jetons) ; repli statique nommé (`_tensor_refus`, ligne de régime),
        # repli dynamique dit une fois par taille de godet (règle 6 : jamais un repli muet).
        if "_tensor_refus" not in self.__dict__:               # piles posées sans _try_build_stacks (tests, outils)
            self.__dict__["_tensor_refus"] = self._raison_tensor()
        statique_ok = _MOE_TENSOR and marlin is not None and not self.__dict__["_tensor_refus"]
        tensor_ok = (statique_ok and not distinct and x_g.shape[0] >= _MOE_TENSOR_MIN_T
                     and x_g.dtype == torch.bfloat16 and ech_gu is None and awq.get("down_proj") is None
                     and x_g.shape[0] * self.top_k == eid.shape[0])   # paires en ordre jeton-majeur (index_jetons)
        if statique_ok and not tensor_ok and x_g.shape[0] >= _MOE_TENSOR_MIN_T and eid.shape[0] not in self.__dict__.setdefault("_dit_tensor_refus", set()):
            self.__dict__["_dit_tensor_refus"].add(eid.shape[0])
            print(f"[moe] chemin tensor non pris sur ce godet (GEMV gardé) : distinct={distinct} "
                  f"G={eid.shape[0]} T={x_g.shape[0]} k={self.top_k} dtype={x_g.dtype} ech_gu={ech_gu is not None} "
                  f"awq_down={awq.get('down_proj') is not None}", flush=True)
        if _MOE_TENSOR and not _MOE_TENSOR_FUSION and not self.__dict__.get("_dit_glue_a4"):
            self.__dict__["_dit_glue_a4"] = True
            print("[moe] ACVRAM_MOE_TENSOR_FUSION=0 : glue A4 (aligneur vLLM par atomiques) — TÉMOIN, non reproductible au bit", flush=True)
        if tensor_ok:
            from ..kernels import marlin_port as MP                # port présent : vérifié par _raison_tensor (repli nommé sinon)
            self._chemin("marlin_tensor")
            pd_ = self._stacks["down_proj"]
            d = gemm_experts_tensor(MP, ext, x_g, eid, marlin, self.top_k, pg[5], pd_[4],
                                    1 if self.act == "gelu_tanh" else 0,
                                    self._marlin_workspace(x.device), self._marlin_uns(eid.shape[0], x.device),
                                    self.__dict__.setdefault("_tensor_tampons", {}), self.__dict__.setdefault("_tensor_sorties", {}),
                                    fusion=_MOE_TENSOR_FUSION, w13_fusionne=not _EN_PREFILL[0])
            tw = topw.reshape(-1)
            if tw.dtype != torch.float32:
                tw = tw.to(torch.float32)
            if ext is not None and hasattr(ext, "moe_reduce"):
                return ext.moe_reduce(d.contiguous(), tw.contiguous(), self.top_k)
            d = d.float() * tw.reshape(-1, 1)
            return d.view(t, self.top_k, -1).sum(dim=1).to(x.dtype)
        if marlin is not None and ext is not None and hasattr(ext, "nvfp4_gemv_marlin_gateup") and not distinct:
            # forme (b), sage-p1-disposition-unique-18-09 : le GEMV lit la
            # disposition Marlin (tuiles 16 k × 64 n) — une seule disposition
            # des experts ; juge fp32 par ligne (tests/test_gemv_marlin.py)
            mg, mu = marlin["gate_proj"], marlin["up_proj"]
            self._chemin("gemv_marlin")
            w13 = marlin.get("w13")
            if w13 is not None:                      # pièce 82 : gate et up lues dans w13, au bit du chemin séparé
                act = ext.nvfp4_gemv_marlin_w13(
                    w13[0], w13[1], w13[2], w13[3], eid, tok_g, x_g.contiguous(),
                    w13[4], w13[5], 1 if self.act == "gelu_tanh" else 0, ech_gu)[:, :pg[5]]
            else:
                act = ext.nvfp4_gemv_marlin_gateup(
                    mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid, tok_g, x_g.contiguous(),
                    mg[3], mg[4], 1 if self.act == "gelu_tanh" else 0, ech_gu)[:, :pg[5]]
        elif marlin is not None and ext is not None and hasattr(ext, "nvfp4_gemv_marlin") and distinct:
            # gate et up ont des ENTRÉES distinctes (tables AWQ séparées : GLM
            # k48-calibA, verdict-glm-b12-19-09) : le noyau fusionné n'a qu'un
            # x, donc gate puis up par le GEMV Marlin à une projection, et
            # l'activation en torch — la pile naturelle est rendue (disposition
            # unique), ce chemin ne peut plus y retomber (Manon : qw=None au
            # premier pas de GLM, serveur mort)
            mg, mu = marlin["gate_proj"], marlin["up_proj"]
            self._chemin("gemv_marlin")
            g = ext.nvfp4_gemv_marlin(mg[0], mg[1], mg[2], eid, tok_g, x_g.contiguous(), mg[3], mg[4])[:, :pg[5]]
            u = ext.nvfp4_gemv_marlin(mu[0], mu[1], mu[2], eid, tok_g, x_u.contiguous(), mu[3], mu[4])[:, :pu[5]]
            act = self._act(g) * u
        elif (_MOE_GEMV == "v2" and pg[0] == "nvfp4" and pu[0] == "nvfp4" and ext is not None
                and hasattr(ext, "nvfp4_gemv_grouped_gateup_v2") and not distinct
                and self._stacks.get("down_proj", ("",))[0] == "nvfp4"
                and pg[4] * 4 <= 48 * 1024):
            ordre = torch.argsort(eid, stable=True).to(torch.int32)
            tri = (eid[ordre.long()].contiguous(), ordre)
        if marlin is not None and act is not None:
            pass                                       # gemv_marlin pris ci-dessus
        elif tri is not None:
            act = ext.nvfp4_gemv_grouped_gateup_v2(
                pg[1], pg[2], pg[3], pu[1], pu[2], pu[3], tri[0], tok_g[ordre.long()].contiguous(), ordre,
                x_g.contiguous(), pg[4], 1 if self.act == "gelu_tanh" else 0)[:, :pg[5]]
        elif (pg[0] == "nvfp4" and pu[0] == "nvfp4" and ext is not None
                and hasattr(ext, "nvfp4_gemv_grouped_gateup")
                and not distinct                       # un seul x pour gate et up
                and pg[4] * 4 <= 48 * 1024):
            # gate, up et SiLU·up en un lancement, activation bf16 lue telle quelle
            self._chemin("gemv_v1")
            act = ext.nvfp4_gemv_grouped_gateup(
                pg[1], pg[2], pg[3], pu[1], pu[2], pu[3], eid, tok_g,
                x_g.contiguous(), pg[4],
                1 if self.act == "gelu_tanh" else 0)[:, :pg[5]]
        elif (pg[0] == "nvfp4_table" and pu[0] == "nvfp4_table" and ext is not None
                and hasattr(ext, "nvfp4_gemv_grouped_gateup_table")
                and not distinct
                and pg[4] * 4 <= 48 * 1024):
            # pendant table (bead pds) : couche au placement hétérogène,
            # chaque expert lu par adresse plutôt que par une pile contiguë
            act = ext.nvfp4_gemv_grouped_gateup_table(
                pg[1], pg[2], pg[3], pu[1], pu[2], pu[3], eid, tok_g,
                x_g.contiguous(), pg[5], pg[4],
                1 if self.act == "gelu_tanh" else 0)[:, :pg[5]]
        else:
            if pg[0] == "nvfp4" and pg[1] is None:
                # dimension « pile rendue » (MECANISMES) : sous la disposition
                # unique, aucun repli ne relit la pile naturelle — une erreur
                # nommée plutôt qu'un None dans un noyau (GLM, 19/09)
                raise RuntimeError(
                    "MoE décodage : pile NVFP4 naturelle rendue (disposition unique Marlin) et aucun "
                    f"chemin Marlin applicable (distinct={distinct}, ext={ext is not None}, "
                    f"marlin={marlin is not None}) — REGLES § 4, verdict-glm-b12-19-09")
            x32 = x_g.to(torch.float32)
            g = self._grouped(x32, pg, eid, tok_g)
            u = self._grouped(x32 if x_u is x_g else x_u.to(torch.float32), pu, eid, tok_g)
            act = self._act(g) * u              # [G, I] fp32
        if fuse_ech and self.dernier_chemin != "gemv_marlin":
            # La division `x / s[e]` a été sautée parce que le GEMV Marlin devait
            # la porter : un autre chemin ici rendrait des sorties fausses en
            # silence. Ce contrôle peut rendre « faux » — c'est son objet.
            raise RuntimeError(
                "echelle AWQ confiee au GEMV Marlin mais chemin "
                f"'{self.dernier_chemin}' pris : sorties non echelonnees")
        seq = _seq if _seq is not None else torch.arange(eid.shape[0], device=x.device, dtype=torch.int32)
        if hd_d:
            act = fwht_activations(act.to(torch.bfloat16), hd_d).to(act.dtype)
        porte_down = (fuse_ech and marlin is not None and self.dernier_chemin == "gemv_marlin"
                      and ext is not None and hasattr(ext, "nvfp4_gemv_marlin"))
        if awq.get("down_proj") is not None and not porte_down:
            if eid64 is None:
                eid64 = eid.long()
            act = (act.to(torch.bfloat16) / awq["down_proj"][eid64, :act.shape[1]]).to(act.dtype)
        if marlin is not None and self.dernier_chemin == "gemv_marlin" and ext is not None \
                and hasattr(ext, "nvfp4_gemv_marlin"):
            md = marlin["down_proj"]
            d = ext.nvfp4_gemv_marlin(md[0], md[1], md[2], eid, seq, act.contiguous(), md[3], md[4],
                                      awq.get("down_proj") if porte_down else None)
        else:
            d = self._grouped(act, self._stacks["down_proj"], eid, seq, tri=tri)
        # Chaque jeton possède exactement top_k lignes contiguës : une somme
        # sur cet axe remplace l'index_add_ atomique — déterministe, plus
        # rapide, et rejouable dans un graphe CUDA sans écart d'un rejeu à
        # l'autre. Pondération, somme et conversion tiennent en un lancement.
        if (ext is not None and hasattr(ext, "moe_reduce")
                and d.dtype == torch.float32 and x.dtype == torch.bfloat16):
            tw = topw.reshape(-1)
            if tw.dtype != torch.float32:
                tw = tw.to(torch.float32)
            return ext.moe_reduce(d.contiguous(), tw.contiguous(), self.top_k)
        d = d * topw.reshape(-1, 1).to(d.dtype)
        return d.view(t, self.top_k, -1).sum(dim=1).to(x.dtype)

    def _router_logits(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 SEULEMENT quand il sert : sigmoid+biais (GLM) départage des
        # experts à égalité proche où un arrondi bf16 de la SORTIE du
        # F.linear, même avec accumulation cuBLAS fp32, suffit à faire
        # basculer le top-k (mesuré : 4/16 jetons de l'équivalence
        # GLM-4.7-Flash, revue/prediction-routeur-fp32-14-09.md). Un modèle
        # softmax sans biais (Coder-30B) n'a pas cette égalité à départager
        # — le fp32 y coûtait un cast + un F.linear fp32 de plus par couche
        # sans corriger quoi que ce soit, mesuré par Laure (bissection ABAB,
        # 15/09) : +0,187 ms/pas à b=1, 3,9 µs/couche sur 48 couches.
        dtype_voulu = (torch.float32
                      if self.scoring == "sigmoid" and self.score_bias is not None
                      else x.dtype)
        cache = getattr(self, "_router_w", None)
        if cache is None:
            cache = {}
            self._router_w = cache
        w = cache.get(dtype_voulu)
        if w is None and hasattr(self.router.qweight, "weight"):
            w = self.router.qweight.weight.to(dtype_voulu)
            cache[dtype_voulu] = w
        if w is not None:
            return F.linear(x.to(dtype_voulu), w)
        return self.router(x).to(dtype_voulu)

    def _routeur_compact(self, x: torch.Tensor):
        """C15-3c : (poids du routeur AU DTYPE DU TÉMOIN, arrondi bf16 ?) pour
        `route_logits_fusee` — le même tenseur `_router_w[dtype_voulu]` que
        `_router_logits`, donc le même appel cuBLAS et les mêmes logits au bit ;
        None si le routeur n'est pas un poids plein (le témoin reste)."""
        if not hasattr(self.router.qweight, "weight"):
            return None
        dtype_voulu = (torch.float32
                      if self.scoring == "sigmoid" and self.score_bias is not None
                      else x.dtype)
        cache = getattr(self, "_router_w", None)
        if cache is None:
            cache = {}
            self._router_w = cache
        w = cache.get(dtype_voulu)
        if w is None:
            w = self.router.qweight.weight.to(dtype_voulu)
            cache[dtype_voulu] = w
        if w.dim() != 2 or w.shape[1] != x.shape[1] or w.shape[0] > 1024:
            return None
        return w, dtype_voulu == torch.bfloat16

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Poids et indices des top-k experts par jeton."""
        logits = self._router_logits(x)
        ext = kernels.get_extension() if x.is_cuda else None
        if (ext is not None and hasattr(ext, "moe_route")
                and logits.shape[-1] <= 1024 and self.top_k <= 32):
            # un lancement : scores, biais, top-k, renormalisation, échelle
            bias = self.score_bias if self.score_bias is not None \
                else torch.empty(0, device=x.device)
            topw, topi = ext.moe_route(logits, bias, self.top_k,
                                       self.scoring == "sigmoid",
                                       bool(self.norm_topk_prob),
                                       float(self.routed_scale))
        else:
            if self.scoring == "sigmoid":
                scores = torch.sigmoid(logits)
                sel = scores if self.score_bias is None else scores + self.score_bias
                _, topi = torch.topk(sel, self.top_k, dim=-1)
                topw = scores.gather(-1, topi)
            else:
                weights = F.softmax(logits, dim=-1)
                topw, topi = torch.topk(weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                topw = topw / topw.sum(dim=-1, keepdim=True)
            if self.routed_scale != 1.0:
                topw = topw * self.routed_scale
        # Trace de routage : un test de booleen quand elle est eteinte, et le
        # module ne touche a rien de plus. Sous trace, elle synchronise — c'est
        # le prix d'une mesure d'ordre, et elle n'est jamais active en service.
        if _trace_routage.actif():
            _trace_routage.noter(self.index_couche, topi, topw)
        return topw, topi

    def forward(self, x: torch.Tensor,
               valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        t, h = x.shape
        # topw reste en fp32 : il sort du routage ainsi et y retourne pour la
        # réduction pondérée ; l'aller-retour en bf16 coûtait deux copies par
        # couche pour rien
        # Poste F, fusion (2) : logits du routeur (cuBLAS) puis `moe_route` +
        # route_prep en UN noyau Triton (kernels/route_prep.route_fusee) —
        # même arithmétique que moe_route (fp32, égalités vers l'indice bas).
        fusee = (_ROUTE_PREP == 2 and x.is_cuda and getattr(self, "_usage_routage", None) is not None
                 and self._stack_state == "oui" and t <= _MOE_GROUPED_MAX
                 and self.top_k <= 32)
        if fusee:
            from ..kernels import route_prep as _rp
            fusee = _rp.disponible()
        compact = None
        if fusee and kernels.glue_compact("routeur"):
            # C15-3c : mêmes logits que _router_logits (même F.linear cuBLAS sur
            # le même poids), sélection `_route_fusee_kernel` à num_warps=1
            # (réductions intra-warp) : routage identique au témoin, sélection
            # plus courte ; même nombre de lancements. Témoin : GLUE_COMPACT=0.
            compact = self._routeur_compact(x)
        if fusee and compact is None:
            logits = self._router_logits(x)
            if logits.shape[-1] > 1024:
                fusee = False
        if fusee:
            # C15-3b : les deux routeurs alimentent le MÊME chemin d'experts —
            # le compact n'est plus une branche à part qui retombait dans
            # `_route` (cuBLAS + moe_route + route_prep rejoués, verdict-c15-
            # niveau3-coder-19-09 : 96 lancements/pas de trop, Triton perdu)
            if compact is not None:
                w, arrondi = compact
                topw, topi, eid = _rp.route_logits_fusee(
                    x, w, self.score_bias if self.scoring == "sigmoid" else None, self.top_k,
                    self.scoring == "sigmoid", bool(self.norm_topk_prob), float(self.routed_scale),
                    valid, self._usage_routage, arrondi_bf16=arrondi)
            else:
                topw, topi, eid = _rp.route_fusee(
                    logits, self.score_bias if self.scoring == "sigmoid" else None, self.top_k,
                    self.scoring == "sigmoid", bool(self.norm_topk_prob), float(self.routed_scale),
                    valid, self._usage_routage)
            if _trace_routage.actif():
                _trace_routage.noter(self.index_couche, topi, topw)
            if valid is not None:
                topi = eid.view(t, self.top_k)
            if _ROUTAGE_TEMOIN:
                tm = self.__dict__.get("_temoin_topi")
                if tm is None or tm.shape != topi.shape:
                    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("témoin de routage alloué pendant une capture de graphe")
                    tm = self.__dict__["_temoin_topi"] = torch.empty_like(topi)
                    _TEMOINS_ROUTAGE[id(self)] = tm       # préfill [t·b, k] puis godet [16, k] : le dernier seul
                tm.copy_(topi)
            mma_ok = _MOE_DECODE_MMA and t >= _MOE_DECODE_MMA_MIN_T
            y = (self._forward_grouped_mma(x, topw, topi) if mma_ok else None)
            if y is None:
                y = self._forward_grouped(x, topw, topi, eid=eid)
            if y is not None:
                if self.shared is not None:
                    y = y + self._shared_out(x)
                return y
        topw, topi = self._route(x)
        # Poste F, fusion (1) : masque des fantômes + compteur d'usage + eid en
        # UN lancement (kernels/route_prep), `tok`/`seq` réservés par godet —
        # à la place de ~10 petits noyaux torch par couche (verdict-lancements-
        # b1-17-09). Même eid, même compteur, au bit ; chemin groupé seul.
        if (_ROUTE_PREP >= 1 and x.is_cuda and topi.dtype == torch.int32
                and getattr(self, "_usage_routage", None) is not None and self._stack_state == "oui"
                and t <= _MOE_GROUPED_MAX):
            from ..kernels import route_prep as _rp
            if _rp.disponible():
                eid = _rp.route_prep(topi, valid, self._usage_routage)
                if valid is not None:
                    topi = eid.view(t, self.top_k)
                mma_ok = _MOE_DECODE_MMA and t >= _MOE_DECODE_MMA_MIN_T
                y = (self._forward_grouped_mma(x, topw, topi) if mma_ok else None)
                if y is None:
                    y = self._forward_grouped(x, topw, topi, eid=eid)
                if y is not None:
                    if self.shared is not None:
                        y = y + self._shared_out(x)
                    return y
        if valid is not None:
            # Créneaux fantômes du remplissage godet (`bucket_batch`,
            # graphs.py) : `x` y est nul, mais x=0 route quand même —
            # DÉTERMINISTE (logits nuls, `topk` départage par index
            # croissant) — vers [0..top_k-1], et RIEN ne garantit que ces
            # experts sont résidents (mesuré le 14/09 : sur les 48 couches
            # de Coder-30B exilé par expert, TOUTES en avaient au moins un
            # froid). -1 les retire du routage réel : ni comptés
            # (`_compter_routage`), ni dispatchés (les noyaux groupés
            # rendent zéro sans lire aucun poids pour un expert < 0 — la
            # boucle par expert les ignore de même, plus bas).
            topi = topi.masked_fill(~valid.unsqueeze(-1), -1)
        # Ici, pas dans _route : `_route` est surchargée (MoEBlockGemma) sans
        # appeler super(), alors que `forward` est le seul point que tous les
        # chemins de routage traversent une fois topi connu.
        # C15-prefill (« permut ») : au préfill groupé le compte est fait par
        # `_forward_prefill_grouped` depuis les comptes par expert qu'il calcule
        # déjà (un `add_` au lieu de cinq petits noyaux : copie int64, ≥ 0, to,
        # clamp, scatter_add) — les mêmes entiers, aucun -1 sans `valid`. Si ce
        # chemin décline (None), le compte est fait ici après coup.
        compte_differe = (kernels.prefill_compact("permut") and valid is None and x.is_cuda
                          and self._stack_state == "oui" and t > _MOE_GROUPED_MAX
                          and self._usage_routage is not None)
        if compte_differe:
            self.__dict__["_compte_en_attente"] = True
        else:
            self._compter_routage(topi)

        # Chemin groupé : trois lancements pour toute la couche, quel que soit
        # le nombre d'experts touchés. La boucle par expert reste le chemin des
        # grands lots de prefill (le regroupement par expert y redevient
        # rentable) et le repli des piles hétérogènes.
        if x.is_cuda and self._stack_state != "non":
            if self._stack_state == "?":
                self._stack_state = "oui" if self._try_build_stacks() else "non"
            if self._stack_state == "oui":
                if t <= _MOE_GROUPED_MAX:
                    # Garde de lot (Sage, revue/sage-mma-lot-15-09.md) : à b=1 la
                    # MMA perd 36 % de débit et 14 % de J (Laure : 4,48 → 7,00 ms,
                    # 1,51 → 1,72 J) — une tuile m16 pour un jeton. En eager `t`
                    # est le lot réel ; sous graphes c'est le GODET (le chemin est
                    # figé à la capture, un lot réel de 5 dans un godet de 8 prend
                    # le chemin du godet) — pas de `.item()` sur `valid` : une
                    # synchronisation par couche, et rien de capturable.
                    mma_ok = _MOE_DECODE_MMA and t >= _MOE_DECODE_MMA_MIN_T
                    y = (self._forward_grouped_mma(x, topw, topi) if mma_ok else None)
                    if y is None:
                        y = self._forward_grouped(x, topw, topi)
                else:
                    y = self._forward_prefill_grouped(x, topw, topi)
                if y is not None:
                    if self.shared is not None:
                        y = y + self._shared_out(x)
                    return y
        if self.__dict__.pop("_compte_en_attente", False):
            self._compter_routage(topi)                # le chemin groupé a décliné

        if t == 1 and not os.environ.get("ACVRAM_MOE_DECODE_MASQUES"):
            # Décodage, un jeton : les masques par expert (nonzero, index)
            # coûtaient une trentaine de synchronisations hôte par couche —
            # le profil du 8/09 y voyait 2,7 ms de processeur pour 1,2 ms de
            # carte. Une seule synchronisation (la liste des experts routés),
            # toutes les copies lancées d'abord, puis les GEMV.
            ids = topi.reshape(-1).tolist()          # unique synchronisation
            poids = topw.reshape(-1).to(x.dtype)
            for e in ids:
                if e < 0:                            # créneau fantôme
                    continue
                exp = self.experts[e]
                for lin in (getattr(exp, "gate_proj", None), exp.up_proj, exp.down_proj):
                    if lin is not None:
                        lin.precharger()
            # Accumulateur en float32. Dix termes sommes en bf16 laissent
            # 5,4e-3 d'ecart relatif rien qu'en changeant leur ordre — mesure
            # du 8/09/2026, a poids et ponderations identiques ; en float32 le
            # meme changement d'ordre donne zero exact. C'est ce bruit-la qui
            # faisait diverger ce chemin de celui par masques, qui somme dans
            # l'ordre trie de `unique()`. Le cout est un tenseur [t, cache] par
            # couche, la ou chaque expert en produit deja un.
            out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
            for j, e in enumerate(ids):
                if e < 0:                            # créneau fantôme : contribution nulle
                    continue
                out += (self.experts[e](x) * poids[j]).to(torch.float32)
            out = out.to(x.dtype)
            if self.shared is not None:
                out = out + self._shared_out(x)
            return out

        # Meme accumulateur float32 que le chemin direct, et pour la meme
        # raison : sans lui les deux chemins ne rendent pas le meme vecteur.
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        # On regroupe les jetons par expert, pour que chaque expert fasse un
        # seul produit matriciel par lot au lieu d'un par jeton.
        flat_expert = topi.reshape(-1)
        flat_weight = topw.reshape(-1).to(x.dtype)   # topw est en fp32
        flat_token = torch.arange(t, device=x.device).repeat_interleave(self.top_k)
        for e in flat_expert.unique().tolist():
            if e < 0:                                # créneau fantôme : rien à faire
                continue
            sel = flat_expert == e
            tok = flat_token[sel]
            y = self.experts[e](x[tok])
            out.index_add_(0, tok, (y * flat_weight[sel].unsqueeze(-1)).to(torch.float32))
        out = out.to(x.dtype)
        if self.shared is not None:
            out = out + self._shared_out(x)
        return out

    def _shared_out(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shared(x)
        if self.shared_gate is not None:
            y = y * torch.sigmoid(x @ self.shared_gate.t())
        return y

    def prefetch(self) -> None:
        # Seul l'expert partagé sert à chaque jeton et peut être préchargé.
        # Les experts routés ne se connaissent qu'après le routage : leur
        # QuantLinear, sur un pool, ignore de toute façon le préchargement.
        if self.shared is not None and not os.environ.get("ACVRAM_SANS_PRECHARGE"):
            for lin in self.shared.modules():
                if isinstance(lin, QuantLinear):
                    lin.prefetch()
# Jetons au-delà desquels le MoE repasse de la GEMV groupée (une paire
# (jeton, expert) par tranche de grille) à la boucle par expert : la boucle
# coûte ~0,6 ms par expert visité, la GEMV groupée relit les poids de
# l'expert pour chaque jeton — croisement mesuré vers quelques milliers.
_MOE_GROUPED_MAX = int(os.environ.get("ACVRAM_MOE_GROUPED_MAX", "32"))
# Jetons par expert au-delà desquels le prefill repasse de la GEMM groupée
# NVFP4 à la déquantification en bf16 suivie de torch._grouped_mm.
# Mesuré le 13/09/2026, Qwen3-Coder-30B-A3B-nvfp4 (8 actifs / 128), RTX 5090,
# moteur chaud, cache de préfixe coupé, 7 rép, médian (revue/banc-prefill-
# moe-12-09.md) — GEMM groupée contre déquant, en jetons/s :
#   jetons/expert   32     48     64     96    128
#   GEMM          3913   4260   4511   4749   4864
#   déquant       3022   4173   5075   6592   7592
#   écart         +29 %   +2 %  -11 %  -28 %  -36 %
# L'ancien défaut 64 était du mauvais côté du croisement.
_MOE_GEMM_MAX = float(os.environ.get("ACVRAM_MOE_GEMM_MAX", "48"))
# GEMM groupée W4A4 sur la MMA FP4 native de sm_120 (ACVRAM_MOE_MMA=0 pour
# revenir au chemin déquant+GEMM bf16) et jetons par tuile (16, 32 ou 64) ;
# voir _forward_prefill_grouped. Activée par défaut le 13/09/2026 : PPL
# Qwen3-Coder-30B-A3B-nvfp4, experts A4 (gate/up/down), routeur+attention
# A16, +0,919 % contre le seuil scellé +1 % (revue/verdict-moe-mma-reel-
# qwen3-coder.md) — dans la fourchette prédite avant mesure.
_MOE_MMA = os.environ.get("ACVRAM_MOE_MMA", "1") == "1"
# GEMM groupée bf16 du prefill au-delà de _MOE_GEMM_MAX jetons par expert :
# "groupe" (B0, kernels/gemm_groupe, DÉFAUT depuis sage-b0-et-cause-lm4-17-09 :
# un lancement Triton persistant pour tous les experts ; Coder 9 913 j/s contre
# 8 633 (+14,8 %), GLM 4 661 contre 4 462 (+4,5 %, sous la prédiction 5 000
# mais aucune régression, PPL ±0,002 tenue sur les deux) | "grouped_mm"
# (torch._grouped_mm, ancien défaut, gardé comme témoin) | "bmm" (seaux
# d'experts, RÉFUTÉ : Laure 0edc3b9, 5 486 j/s contre 8 614 — tuiles d'un
# petit M inchangées et ~58 Go de copies w[experts] par prefill ; gardé comme
# témoin d'une fausse piste, jamais comme défaut)
# | "marlin" (P1, GEMM groupée classe Marlin sur la disposition unique des
# experts : DÉFAUT depuis l'adoption utilisateur du 18/09 — Coder prefill
# 15 987 j/s (+63 %), GLM 5 502 (+18 %), b=12 1 239 t/s / 0,303 J, au prix de
# −1,2 % à b=1 ; sage-p1-situ-verdict-18-09 ; « groupe » reste le témoin)
_PREFILL_GROUPED = os.environ.get("ACVRAM_PREFILL_GROUPED", "marlin")
# ACVRAM_PREFILL_A4=off|gateup|both : porte qualité W4A4 du prefill MoE (sage-lecture-profils-coder-17-09) —
# fausse quantification NVFP4 des ACTIVATIONS en torch (E2M1 bloc 16, échelle de bloc UE4M3, échelle
# globale par ligne, comme nvfp4_quant_act), sur l'entrée de gate/up (gateup) et aussi sur celle de
# down (both) ; aucun noyau : mesure la perte de qualité qu'un GEMM W4A4 imposerait, pas sa vitesse.
_PREFILL_A4 = os.environ.get("ACVRAM_PREFILL_A4", "off")
if _PREFILL_A4 not in ("off", "gateup", "both"):
    raise ValueError(f"ACVRAM_PREFILL_A4={_PREFILL_A4!r} : off | gateup | both")
# ACVRAM_PREFILL_A8=off|gateup|both : porte qualité W4A8 (sage-w4a4-clos-w4a8-porte-19-09) — la
# fausse quant des activations du prefill MoE en int8 par jeton (ACVRAM_PREFILL_A8_FMT=int8,
# défaut : l'arrondi de `quantifier_a8`, celui des chemins a8/cublas) ou en E4M3 bloc 16
# (=e4m3, témoin : le format d'activation de la MMA mxf8f6f4). Aucun noyau : la perte, pas la
# vitesse. Exclusive de PREFILL_A4 (deux arrondis empilés ne mesureraient rien).
# ACVRAM_PREFILL_W8R=1 : porte qualité W8r — les piles d'experts déquantifiées par
# `_pile_bf16` (chemins PREFILL_GROUPED=grouped_mm|bmm, pile naturelle donc
# GEMV_LAYOUT=naturel) re-arrondies en int8 symétrique par ligne ; aucun noyau.
_PREFILL_W8R = os.environ.get("ACVRAM_PREFILL_W8R", "0")
if _PREFILL_W8R not in ("0", "1"):
    raise ValueError(f"ACVRAM_PREFILL_W8R={_PREFILL_W8R!r} : 0 | 1")
_PREFILL_A8 = os.environ.get("ACVRAM_PREFILL_A8", "off")
_PREFILL_A8_FMT = os.environ.get("ACVRAM_PREFILL_A8_FMT", "int8")
if _PREFILL_A8 not in ("off", "gateup", "both"):
    raise ValueError(f"ACVRAM_PREFILL_A8={_PREFILL_A8!r} : off | gateup | both")
if _PREFILL_A8_FMT not in ("int8", "e4m3"):
    raise ValueError(f"ACVRAM_PREFILL_A8_FMT={_PREFILL_A8_FMT!r} : int8 | e4m3")
if _PREFILL_A8 != "off" and _PREFILL_A4 != "off":
    raise ValueError("ACVRAM_PREFILL_A8 et ACVRAM_PREFILL_A4 ne se cumulent pas : une porte à la fois")
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MILIEUX = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def fausse_quant_a8(x: torch.Tensor, fmt: str = "int8") -> torch.Tensor:
    """x [G, K] → x arrondi comme le ferait un GEMM W4A8 (`int8` par jeton au
    bit du noyau a8, ou `e4m3` bloc 16) — `quant.fakequant_activation.
    fake_quantize_a8`, partagée avec la porte FP8-MLA (engine/mla.py)."""
    from ..quant.fakequant_activation import fake_quantize_a8
    return fake_quantize_a8(x, fmt)


def fausse_quant_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """x [G, K] (K multiple de 16) → x arrondi comme le ferait nvfp4_quant_act :
    échelle globale par ligne s = amax_ligne / (448 × 6), échelle de bloc (16) en
    UE4M3 = amax_bloc / (6 s), valeur E2M1 au plus proche (milieux des paliers),
    puis déquantifié. Sortie dans le dtype de x. Torch pur (porte qualité)."""
    G, K = x.shape
    assert K % 16 == 0, K
    xf = x.to(torch.float32)
    s_row = (xf.abs().amax(dim=1, keepdim=True) / (448.0 * 6.0)).clamp_min(1e-12)   # [G,1]
    xb = xf.view(G, K // 16, 16)
    bs = (xb.abs().amax(dim=2, keepdim=True) / (6.0 * s_row.unsqueeze(2)))           # [G,K/16,1]
    bs = bs.to(torch.float8_e4m3fn).to(torch.float32)                                # UE4M3 (arrondi fp8)
    ech = (bs * s_row.unsqueeze(2)).clamp_min(1e-12)
    v = (xb / ech).clamp(-6.0, 6.0)
    idx = torch.bucketize(v.abs(), _E2M1_MILIEUX.to(x.device))
    q = _E2M1.to(x.device)[idx] * torch.sign(v)
    return (q * ech).view(G, K).to(x.dtype)
if _PREFILL_GROUPED not in ("bmm", "grouped_mm", "groupe", "w4a16", "marlin"):
    raise ValueError(f"ACVRAM_PREFILL_GROUPED={_PREFILL_GROUPED!r} : attendu groupe, w4a16, grouped_mm ou bmm")
_MOE_MMA_BT = int(os.environ.get("ACVRAM_MOE_MMA_BT", "64"))
# Étages du pipeline cp.async du noyau MMA (0 = chargements directs).
# Mesuré le 14/09/2026, Coder-30B, prefill chaud L=2048 : 0 → 10 411 j/s,
# 2 → 16 655, 3 → 16 851, 4 → 16 938 (revue/mma-fp4-native-sm120.md).
_MOE_MMA_ETAGES = int(os.environ.get("ACVRAM_MOE_MMA_ETAGES", "4"))
# Profondeur d'un étage du pipeline (64 ou 128) ; 128 = deux MMA par
# synchronisation. Mesuré le 14/09/2026 (bead 0si), Coder-30B prefill chaud :
# L=2048 16 911 → 19 148 j/s, L=512 8 761 → 9 853 (revue/mma-fp4-native-sm120.md).
_MOE_MMA_KS = int(os.environ.get("ACVRAM_MOE_MMA_KS", "128"))
# Décodage MoE par la MMA groupée (levier (3) de Sage, 14/09) : OUVERT par
# défaut depuis le 15/09 (revue/sage-moe-mma-qualite-15-09.md § 2) — PPL par
# le chemin de décodage MMA=1 / MMA=0 = 0,9995 (Manon, teacher forcing,
# 8191/8191) ; b=12 Coder-30B mode moyen (Laure, rondes ctx 2048, c6377d5) :
# 17,37 → 16,25 ms et 0,620 → 0,538 J/jeton (−6,4 % / −13,2 %) ; régime
# court (revue/moe-mma-decodage-pas-complet-14-09.md) : −13,6 % / −19 %.
# ACVRAM_MOE_DECODE_MMA=0 reste le témoin (bancs, A/B). Tuile de 16 : à b=12
# un expert reçoit au plus 12 jetons (bras `experts`).
_MOE_DECODE_MMA = os.environ.get("ACVRAM_MOE_DECODE_MMA", "1") == "1"
_MOE_DECODE_MMA_BT = int(os.environ.get("ACVRAM_MOE_DECODE_MMA_BT", "16"))
_MOE_AWQ_TEMOIN = int(os.environ.get("ACVRAM_MOE_AWQ_TEMOIN", "0"))
# ACVRAM_QA_COMPTE=1 : compteurs (blocs non nuls, flushés à zéro, saturés)
# cumulés sur le processus par nvfp4_quant_act (échelle globale PAR LIGNE,
# sage-glm-pile-correctif § 7 : saturation impossible par construction, flush
# sous ~2,2e-6 × amax de ligne), imprimés à la sortie — la preuve demandée par
# Sage sur la passe de PPL réelle (attendu 0 saturé, flush ≤ 0,01 %).
_QA_COMPTE = os.environ.get("ACVRAM_QA_COMPTE", "0") == "1"
_QA_COMPTEURS: dict = {}


def _qa_compteurs(device):
    if not _QA_COMPTE:
        return None
    c = _QA_COMPTEURS.get(device)
    if c is None:
        c = _QA_COMPTEURS[device] = torch.zeros(3, dtype=torch.int64, device=device)
        if len(_QA_COMPTEURS) == 1:
            import atexit
            atexit.register(_qa_imprime)
    return c


def _qa_imprime():
    for dev, c in _QA_COMPTEURS.items():
        n, z, sat = (int(v) for v in c.tolist())
        print(f"[acvram] quant_act {dev} : blocs non nuls {n}, flushés {z} "
              f"({z / max(n, 1):.6%}), saturés {sat} ({sat / max(n, 1):.6%})", file=sys.stderr, flush=True)
# Lot minimal pour le chemin MMA : 5 (godets 8, 12 et 16 ; 2 et 4 en GEMV).
# Courbe de Laure, 15/09, MMA/GEMV : b=2 +36 % ms / −3,5 % J ; b=3 +27 / −6,3 ;
# b=4 +24 / +2,2 ; b=12 −6,4 / −13,2. Le 9 (v0.6.3) laissait le godet 8 au
# GEMV ; après route+pack la cellule b=5 (Laure, 16/09,
# revue/verdict-cellule-b5-16-09.md) donne ms −0,3 % (égalité) et J −4,1 % :
# la GEMM sort du plafond 400 W (SM 2 937 MHz au lieu de 2 727) → 5.
_MOE_DECODE_MMA_MIN_T = int(os.environ.get("ACVRAM_MOE_DECODE_MMA_MIN_T", "5"))
# Poste F, fusion (1) : préparation du routage en un lancement (kernels/route_prep)
# — défaut depuis verdict-f1-route-prep-17-09 (tenu : bit-à-bit b=1/b=12,
# 1275→795 lancements, b=1 313,4 t/s pur)
# 0 : chemin torch | 1 (F1 tenu f912f90) : route_prep après moe_route |
# 2 (défaut depuis verdict-f2-topk-17-09, critère plancher tenu) :
# moe_route + route_prep fusionnés (route_fusee)
_ROUTE_PREP = int(os.environ.get("ACVRAM_ROUTE_PREP", "2"))
if _ROUTE_PREP not in (0, 1, 2):
    raise ValueError(f"ACVRAM_ROUTE_PREP={_ROUTE_PREP!r} : attendu 0, 1 ou 2")
# MoE fusionné au décodage (port b12x, 15/09) : coupé tant que les seuils de
# Sage ne sont pas tenus (≤ 75 µs/couche, pas b=12 ≤ 11,3 ms, J ≤ 0,38).
_MOE_DECODE_FUSED = os.environ.get("ACVRAM_MOE_DECODE_FUSED", "0") == "1"
_MOE_FUSED_TN = int(os.environ.get("ACVRAM_MOE_FUSED_TN", "64"))
_MOE_FUSED_ATOMIQUE = os.environ.get("ACVRAM_MOE_FUSED_ATOMIQUE", "0") == "1"   # témoin, non reproductible au bit
_MOE_FUSED_ETAGES = int(os.environ.get("ACVRAM_MOE_FUSED_ETAGES", "3"))          # 2 : shared plus petite, 2-3 CTA par SM
# Frontend route+pack en un noyau (15/09, Sage) ; "0" = témoin torch.
_MOE_ROUTE_PACK = os.environ.get("ACVRAM_MOE_ROUTE_PACK", "1") == "1"
# GEMV groupée du décodage MoE (18/09, sage-lecture-profils-coder-17-09 § 2) :
# v1 (défaut jusqu'au scellé : une passe de poids PAR PAIRE (expert, jeton)) |
# v2 (paires triées par expert, poids lus une fois pour ≤ 4 jetons du même
# expert, sortie identique au bit). Scellé : experts 6,6 → ≤ 5,3 ms, Coder
# b=12 nu ≥ 1 300 t/s (faux < 1 200).
_MOE_GEMV = os.environ.get("ACVRAM_MOE_GEMV", "v1")
# Pièce 62 (23/09, A4) : au décodage, godet ≥ 2, les experts par la GEMM groupée Marlin du port
# (tensor cores, `marlin_port.gemm_moe`, chemin `marlin_tensor`) au lieu du GEMV par paire
# (cœurs CUDA, plancher de calcul à 96 paires : revue/gaelle-creneaux-experts-23-09). Opt-in
# jusqu à la cellule ABBA de Manon (A5) ; le godet 1 reste au GEMV (Marlin 1,40 contre 0,82 ms/pas).
# Pas au bit : arithmétique tensor-core (biais 1-2·10⁻⁴, 0 ligne hors 2⁻⁷) — jugé par ulp et KL.
# Pièce 65 (23/09) : chemin tensor-core SERVI PAR DÉFAUT aux godets ≥ MOE_TENSOR_MIN_T (A5 Manon : +12,2 % t/s,
# −30 % J/jeton à b=12) ; =0 témoin GEMV scalaire partout.
_MOE_TENSOR = os.environ.get("ACVRAM_MOE_TENSOR", "1") == "1"
# Pièce 65 mesuré (chaine-p65, Coder b=2/4/8/12, frontière 200 pas) : tensor +12,7 % à b=2, +9,5 % à b=4,
# −4,3 % à b=8, −10,4 % à b=12 — le GEMV par paire gagne tant que les experts distincts sont peu nombreux
# (bande passante), la GEMM groupée à partir de 8 jetons. Seuil par godet comme MOE_DECODE_MMA_MIN_T.
_MOE_TENSOR_MIN_T = int(os.environ.get("ACVRAM_MOE_TENSOR_MIN_T", "8"))
# Pièce 65 : la glue A4 (aligneur vLLM par atomiques) n est PAS reproductible — jamais servie par défaut.
TENSOR_E_MAX = 4096                                    # mémoire partagée de moe_aligner_petit (2·E + G entiers)


def forme_tensor_refus(formats, K: int, N_gu: int, N_d: int, E: int, awq_unite: bool, hadamard_down: int = 0) -> str:
    """Pièce 65 — règle STATIQUE d acceptation d une couche MoE par le chemin tensor-core, partagée par le
    moteur (`MoEBlock._raison_tensor`) et le contrôle à sec (`outils/controle-moe-tensor-alias.py`) : rend ""
    si la forme est prise en charge, sinon la raison du repli (nommée sur la ligne de régime)."""
    formats = set(formats)
    if formats != {"nvfp4"}:
        return f"experts {'/'.join(sorted(formats)) or 'sans format'} (le port Marlin est NVFP4 seul)"
    if K % 64 or N_gu % 64 or N_d % 64:
        return f"formes K={K} N={N_gu}/{N_d} non multiples de 64 (tuiles Marlin)"
    if E > TENSOR_E_MAX:
        return f"E={E} > {TENSOR_E_MAX} (aligneur)"
    if not awq_unite:
        return "tables AWQ d activation par expert non unité (pièce 47 : gate/up distinctes → pas de piles Marlin ; sinon le chemin tensor ne les applique pas)"
    if hadamard_down:
        return f"rotation Hadamard ({hadamard_down}) de l entrée de down_proj (le chemin tensor ne la tourne pas)"
    return ""


# Pièce 63 (23/09) : glue du chemin tensor fusionnée (aligneur en un lancement qui absorbe le clamp,
# reduce lisant le bf16 sans cast) — témoin ACVRAM_MOE_TENSOR_FUSION=0 : la glue A4 (clamp + aligneur
# vLLM + d.float()), au bit contre la fusion (tests/test_moe_tensor_glue_fusee.py).
_MOE_TENSOR_FUSION = os.environ.get("ACVRAM_MOE_TENSOR_FUSION", "1") == "1"
# Pièce 82 (23/09, revue/oceane-piece82-w13-chemin-tensor-23-09.md) : gate·up en UNE pile Marlin w13 (disposition
# unique : gate et up sont rendues, même nombre d octets — la 71 bis ajoutait w13 à côté, +8,9 Go, hors mémoire).
# Décodage tensor : une GEMM w13 à l échelle globale de gate, l échelle d up corrigée dans moe_act
# (g_up/g_gate) — un arrondi bf16 de plus sur up et une autre découpe de K : sortie au 2⁻⁷, PAS au bit. GEMV (godets
# < MIN_T, b = 1) : `nvfp4_gemv_marlin_w13` lit gate et up dans w13 avec leurs échelles propres — au bit du chemin
# séparé. Préfill (pièce 82 ter) : deux GEMM de largeur N sur des vues de w13 (largeur stockée `ldn` du port) — au
# bit du chemin séparé ; la 82 y perdait la KL de fin de préfill.
# Défaut depuis la pièce 82 ter (23/09, ordre Jerome) : préfill au bit du chemin séparé, GEMV au bit, décodage
# tensor au 2⁻⁷ ; KL Coder b=12 ≤ témoin + 0,025 et b=1 identique au témoin, jouées dans la même prise
# (revue/oceane-piece82ter-w13-decodage-23-09.md). Témoin : ACVRAM_MOE_W13=0.
_MOE_W13 = os.environ.get("ACVRAM_MOE_W13", "1") == "1"
# Pièce 82 ter : phase de la passe en cours, posée par le modèle (model.forward : batch.is_prefill ; decode_fixed :
# False). Un préfill court (T ≤ _MOE_GROUPED_MAX) passe par le même `_forward_grouped` qu un pas de décodage : sans
# ce drapeau, il prendrait la GEMM w13 de largeur 2N et changerait la fin du préfill (KL b=1 de la 82 ter, invite 3 :
# 0,068 → 0,166). Lu en Python : sous graphe, c est la valeur à la capture (décodage seulement, graphs.py:632).
_EN_PREFILL = [False]


def gemm_experts_tensor(MP, ext, x: torch.Tensor, eid: torch.Tensor, marlin: dict, top_k: int, m_gate: int,
                        k_down: int, code_act: int, ws, uns, tampons: dict, sorties: dict,
                        fusion: bool = True, w13_fusionne: bool = True) -> torch.Tensor:
    """Pièce 62 A4 — experts au décodage par la GEMM groupée Marlin du port (tensor cores).
    ``x`` [T, K] bf16 (lignes du godet), ``eid`` [G = T·top_k] int32 (expert de chaque paire, ordre
    jeton-majeur, −1 = fantôme), ``marlin`` = piles servies (disposition du port : (w, s, g, K, N)).
    Rend d [G, N] par paire, dans l ordre des paires, pour `moe_reduce` : bf16 si ``fusion`` (reduce lit le
    bf16, pièce 63), fp32 sinon (glue A4). Capturable : aligneur en un lancement, tampons (``tampons``) et
    sorties (``sorties``) à adresses fixes."""
    w13 = marlin.get("w13")
    E = (w13[0] if w13 is not None else marlin["gate_proj"][0]).shape[0]
    T, G = x.shape[0], eid.shape[0]
    assert G == T * top_k, (G, T, top_k)
    bloc = MP.choisir_block_size(T, top_k, E)
    cle = (G, bloc)
    if cle not in tampons:
        P = -(-(G + E * (bloc - 1)) // bloc) * bloc
        tampons[cle] = (torch.empty(P, dtype=torch.int32, device=x.device),
                        torch.empty(P // bloc, dtype=torch.int32, device=x.device),
                        torch.empty(1, dtype=torch.int32, device=x.device))
    fusion = fusion and ext is not None and hasattr(ext, "moe_aligner_petit")
    if fusion:                                    # un lancement : clamp des fantômes + comptage + préfixe + dispersion
        s_ids, e_ids, n_post = tampons[cle]
        ext.moe_aligner_petit(eid.contiguous(), E, bloc, s_ids, e_ids, n_post)
    else:
        eid_al = eid.clamp(min=0)                 # fantômes → expert 0 sur une ligne nulle ; poids 0 au reduce
        aligner = MP.aligner_blocs_cuda if hasattr(MP.charger(compiler=False), "moe_align_block_size") else MP.aligner_blocs_capturable
        s_ids, e_ids, n_post = aligner(eid_al, bloc, E, tampons[cle])
    mg, mu, md = marlin["gate_proj"], marlin["up_proj"], marlin["down_proj"]
    n_gu, n_d = (w13[1].shape[2] // 2 if w13 is not None else mg[1].shape[2]), md[1].shape[2]

    def sortie(nom, forme):
        k = (nom, forme)
        if k not in sorties:                      # zéros à la création : les lignes jamais écrites restent finies
            sorties[k] = torch.zeros(*forme, dtype=torch.bfloat16, device=x.device)
        return sorties[k]
    xm = x.contiguous()
    if w13 is not None and not w13_fusionne:
        # Pièce 82 ter : préfill court servi ici — gate et up lues dans w13 par des vues de largeur N, au bit du
        # chemin séparé (même découpe de K), comme `_forward_prefill_grouped`.
        mg = (w13[0][:, :, :2 * n_gu], w13[1][:, :, :n_gu], w13[2], w13[4], mg[4])
        mu = (w13[0][:, :, 2 * n_gu:], w13[1][:, :, n_gu:], w13[3], w13[4], mu[4])
        w13 = None
    if w13 is not None:
        # Pièce 82 : UNE GEMM gate‖up (N = 2·n_gu) à l échelle globale de gate ; moe_act corrige up par g_up/g_gate
        # (gs_gate = 1, gs_up = rapport ; e_sorted = l expert de chaque paire, fantômes −1 → expert 0, poids 0).
        c13 = MP.gemm_moe(xm, w13[0], w13[1], w13[2], s_ids, e_ids, n_post, uns, bloc, top_k, T, 2 * n_gu, w13[4], ws,
                          c=sortie("w13", (G, 2 * n_gu)))
        act = ext.moe_act(c13[:, :n_gu], c13[:, n_gu:], m_gate, k_down, code_act, None, eid.contiguous(), w13[7], w13[6])
        d = MP.gemm_moe(act.contiguous(), md[0], md[1], md[2], s_ids, e_ids, n_post, uns, bloc, 1, G, n_d, md[3], ws,
                        c=sortie("d", (G, n_d)))
        return d if fusion else d.float()
    g = MP.gemm_moe(xm, mg[0], mg[1], mg[2], s_ids, e_ids, n_post, uns, bloc, top_k, T, n_gu, mg[3], ws, c=sortie("g", (G, n_gu)))
    u = MP.gemm_moe(xm, mu[0], mu[1], mu[2], s_ids, e_ids, n_post, uns, bloc, top_k, T, n_gu, mu[3], ws, c=sortie("u", (G, n_gu)))
    if ext is not None and hasattr(ext, "moe_act"):
        act = ext.moe_act(g, u, m_gate, k_down, code_act)
    else:
        f = F.gelu(g[:, :m_gate].float(), approximate="tanh") if code_act == 1 else F.silu(g[:, :m_gate].float())
        act = (f * u[:, :m_gate].float()).to(torch.bfloat16)
        if act.shape[1] != k_down:
            act = F.pad(act, (0, k_down - act.shape[1]))
    d = MP.gemm_moe(act.contiguous(), md[0], md[1], md[2], s_ids, e_ids, n_post, uns, bloc, 1, G, n_d, md[3], ws, c=sortie("d", (G, n_d)))
    return d if fusion else d.float()
# P1 disposition unique (sage-p1-disposition-unique-18-09, forme (b)) : le GEMV
# du décodage lit la disposition Marlin (« marlin », exige
# ACVRAM_PREFILL_GROUPED=marlin) ou la pile NVFP4 naturelle (« naturel », témoin).
# Défaut « marlin » avec PREFILL_GROUPED (adoption du 18/09) ; témoin : les deux à naturel/groupe.
_GEMV_LAYOUT = os.environ.get("ACVRAM_GEMV_LAYOUT", "marlin")
# C17 (chantier-c17-mma2-lit-marlin-19-09, scellé sage-c17-scelle-mesure1-ter) : sous la
# disposition unique Marlin (pile naturelle rendue), le décodage MoE par la MMA groupée
# (`_forward_grouped_mma`, MOE_DECODE_MMA, t ≥ MIN_T) lit les TUILES MARLIN au lieu de rendre
# None — même noyau que le préfill naturel, aucune copie. Mesure 1-ter sous 2 700 : ×0,893 du
# temps de Marlin à 45 distincts, ×0,771 à 27 ; ×1,23 à u=8 et ×1,40 à u=16 : sous MIN_T le
# GEMV Marlin reste. 0 (défaut jusqu'au scellé : capture 5/5, ppl-decode-kv ± 0,002, pas b=12
# ≤ défaut − 0,35 ms) = jamais, la GEMV Marlin sert.
_MOE_DECODE_MMA_MARLIN = os.environ.get("ACVRAM_MOE_DECODE_MMA_MARLIN", "0") == "1"
# C10 (sage-c9-119b-cache-experts-19-09 § Ordre (3)) : ACVRAM_MARLIN_DISTINCT=1 lève le
# refus « gate/up à entrées distinctes » de la disposition unique — le décodage prend alors
# le GEMV Marlin à UNE projection deux fois (gate puis up, activation torch), le préfill
# la GEMM Marlin ; scellé GLM b=12 ≥ chemin d'avant × 1,05, sinon le refus reste le défaut.
_MARLIN_DISTINCT = os.environ.get("ACVRAM_MARLIN_DISTINCT", "0")
if _MARLIN_DISTINCT not in ("0", "1"):
    raise ValueError(f"ACVRAM_MARLIN_DISTINCT={_MARLIN_DISTINCT!r} : 0 | 1")
# ACVRAM_TRACE_ROUTAGE_PT=<fichier.pt> : les routages (expert par paire, [B, top_k])
# de chaque appel décodé hors graphe, sauvés à la sortie (torch.save d'une liste)
# — à rejouer par outils/banc-marlin-decode-18-09.py --routages. Nom DISTINCT
# de ACVRAM_TRACE_ROUTAGE (journal texte par couche, memory/trace_routage.py,
# le seul que lit `taux_de_succes`) : depuis le 18/09 les deux lisaient le même
# nom et le torch.save d'ici, exécuté en dernier à la sortie, ÉCRASAIT le
# journal texte d'une trace M1 (deux mécanismes sur un nom, MECANISMES ; 19/09).
_TRACE_ROUTAGE = os.environ.get("ACVRAM_TRACE_ROUTAGE_PT", "")
_ROUTAGES: list = []
# ACVRAM_ROUTAGE_TEMOIN=1 (C15-3d, contrôle « experts égaux » SOUS GRAPHES) : chaque couche
# MoE copie topi dans un tampon persistant (alloué hors capture ; sous capture la copie est
# dans le graphe) — lu après chaque pas par equiv-b12.py, comparé au bit entre bras.
_ROUTAGE_TEMOIN = os.environ.get("ACVRAM_ROUTAGE_TEMOIN", "0") == "1"
_TEMOINS_ROUTAGE: dict = {}      # id(couche) → tampon [t, top_k] int32 (remplacé si la forme change)


def temoins_routage() -> list:
    """Les tampons témoins, dans l'ordre du premier appel des couches."""
    return list(_TEMOINS_ROUTAGE.values())
if _TRACE_ROUTAGE:
    import atexit as _atexit
    _atexit.register(lambda: torch.save(_ROUTAGES, _TRACE_ROUTAGE) if _ROUTAGES else None)
if _GEMV_LAYOUT not in ("naturel", "marlin"):
    raise ValueError(f"ACVRAM_GEMV_LAYOUT={_GEMV_LAYOUT!r} : naturel | marlin")
# Diagnostic seulement (bissection du biais, sage-p1-situ-verdict-18-09) : les
# DEUX dispositions gardées — préfill {groupe|marlin} × décodage {v1|(b)} sur les
# mêmes piles ; jamais un régime servi (régime : experts_layout=double(diag)).
_DOUBLE_DIAG = os.environ.get("ACVRAM_DOUBLE_DISPOSITION_DIAG", "0") == "1"
if not _DOUBLE_DIAG and (_GEMV_LAYOUT == "marlin") != (_PREFILL_GROUPED == "marlin"):
    # la disposition est UNIQUE : préfill et décodage lisent la même pile ;
    # « Marlin au préfill, naturelle au décodage » (double disposition,
    # experts_layout=double) n'existe plus (sage-p1-disposition-unique-18-09)
    raise ValueError(f"ACVRAM_GEMV_LAYOUT={_GEMV_LAYOUT!r} et ACVRAM_PREFILL_GROUPED={_PREFILL_GROUPED!r} : "
                     "les deux à marlin (disposition unique) ou aucun (témoin naturel)")


def _gemv_marlin_porte_echelle(ext) -> bool:
    """Vrai si l'extension compilée accepte `xscale` sur les GEMV Marlin.

    Une extension d'avant la pièce 47 refuse l'argument : on garde alors la
    division en torch plutôt que de servir des sorties non échelonnées."""
    v = getattr(ext, "_acvram_gemv_xscale", None)
    if v is None:
        doc = (getattr(ext.nvfp4_gemv_marlin, "__doc__", "") or "") + \
              (getattr(ext.nvfp4_gemv_marlin_gateup, "__doc__", "") or "")
        v = "xscale" in doc
        try:
            ext._acvram_gemv_xscale = v
        except AttributeError:
            pass
    return bool(v)


def _mla_glue() -> int:
    """C15 : niveau de glue MLA (engine/mla.py `_MLA_GLUE`, variable
    ACVRAM_MLA_GLUE) lu au moment de l'appel — le module mla n'est importé que
    par le chargeur, et un test masque l'attribut sans réimporter."""
    from . import mla as _mla
    return _mla._MLA_GLUE
# P0 (sage-profil-verdict-18-09) : colle du préfill MoE (tri des paires par
# expert + grille de tuiles) en deux lancements Triton au lieu d'argsort
# (radix, 8 lancements) + bincount + ~10 lancements de _tuiles ; mêmes
# tenseurs (tests/test_colle_moe.py). torch = témoin.
_COLLE_MOE = os.environ.get("ACVRAM_COLLE_MOE", "torch")
if _COLLE_MOE not in ("torch", "triton"):
    raise ValueError(f"ACVRAM_COLLE_MOE={_COLLE_MOE!r} : torch | triton")
_BORNES_EXPERTS: dict = {}


def _comptes_tries(e_sorted: torch.Tensor, E: int) -> torch.Tensor:
    """C15-prefill : ``bincount(e_sorted, minlength=E)`` (int64 [E]) pour une
    liste d'experts déjà TRIÉE — les bornes de chaque expert par recherche
    dichotomique (`searchsorted` sur 0..E) puis leur différence : deux petits
    noyaux au lieu de l'histogramme à atomiques (11,6 µs à G = 16 376). Les
    mêmes comptes ; `tests/test_prefill_compact.py` le tient contre bincount
    et casse si la liste n'est pas triée."""
    cle = (str(e_sorted.device), E, e_sorted.dtype)
    bornes = _BORNES_EXPERTS.get(cle)
    if bornes is None:
        bornes = torch.arange(E + 1, dtype=e_sorted.dtype, device=e_sorted.device)
        _BORNES_EXPERTS[cle] = bornes
    return torch.diff(torch.searchsorted(e_sorted, bornes))


def _colle_moe_triton(G: int, E: int, device):
    if _COLLE_MOE != "triton" or E & (E - 1):
        return None
    from ..kernels import colle_moe
    if not colle_moe.disponible() or G > colle_moe.G_MAX:
        return None
    if device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        return None
    return colle_moe


class MoEBlockGemma(MoEBlock):
    """MoE de Gemma 4 (26B-A4B) : routeur sur x normalisé (RMS sans poids)
    × échelle × h^-½, softmax, top-k sans renormalisation, poids × échelle
    par expert ; experts GELU-tanh, reproduite par le chemin groupé."""

    def __init__(self, router: QuantLinear, experts: list, top_k: int,
                 router_scale: torch.Tensor, per_expert_scale: torch.Tensor,
                 eps: float) -> None:
        super().__init__(router, experts, top_k, None, norm_topk_prob=False)
        self.router_scale = router_scale.to(torch.float32)
        self.per_expert_scale = per_expert_scale.to(torch.float32).reshape(-1)
        self.eps = eps
        self.root = float(router_scale.numel()) ** -0.5

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x32 = x.to(torch.float32)
        xr = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        xr = xr * self.router_scale * self.root
        probs = F.softmax(self._router_logits(xr), dim=-1)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        return topw * self.per_expert_scale[topi], topi
# ---------------------------------------------------------------------------
# Diagnostic (sage-p1-situ-verdict-18-09, piste « premier pas divergent ») :
# ACVRAM_DUMP_MOE=<dossier> — la sortie de chaque MoEBlock est recopiée dans un
# tampon STATIQUE par forme (alloué au premier passage eager, donc à adresse
# fixe : un graphe capturé y écrit à chaque rejeu) ; le moteur (runner.step)
# sauve tous les tampons après chaque pas. Comparaison : outils/comparer-dump-
# moe-18-09.py <A> <B> — (b) capturé contre (b) eager nomme le premier
# (pas, couche) divergent. Jamais actif en service.
_DUMP_MOE = os.environ.get("ACVRAM_DUMP_MOE", "")
if _DUMP_MOE:
    os.makedirs(_DUMP_MOE, exist_ok=True)
    _moe_forward_orig = MoEBlock.forward

    def _moe_forward_dump(self, x, valid=None):
        y = _moe_forward_orig(self, x, valid)
        bufs = self.__dict__.setdefault("_dump_bufs", {})
        cle = (tuple(y.shape), y.dtype)
        buf = bufs.get(cle)
        if buf is None:
            buf = bufs[cle] = torch.empty_like(y)
        buf.copy_(y)
        return y

    MoEBlock.forward = _moe_forward_dump
