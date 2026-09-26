"""Le transformeur lui-même : famille llama, dense et à mélange d'experts,
réparti sur les étages de mémoire.

Le périmètre est délibéré. Il couvre l'architecture qu'emploie à peu près tout
modèle à poids ouverts de la gamme de tailles qui vaut la peine d'être exécutée
sur cette machine — RMSNorm, RoPE, attention à requêtes groupées, SwiGLU, et
optionnellement un bloc à mélange d'experts creux — plutôt que d'essayer d'être
une ménagerie universelle. Llama, Mistral, Qwen2/3, Mixtral et DeepSeek y
entrent tous.

Ce qui est propre à ce projet, c'est qu'une couche sait sur quel appareil elle
s'exécute et si ses poids sont résidents ou transférés, et que le bloc à mélange
d'experts ne touche que les experts choisis par le routeur — ce qui fait de la
mémoire vive un endroit raisonnable pour garder les 120 autres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels
from ..memory import trace_routage as _trace_routage
from ..memory.kvcache import PagedKVCache, bucket_blocks
from ..memory import kv_lm4
from ..quant.calibrate import fwht_activations
from .config import ModelSpec
from .lot import ForwardBatch   # noqa: F401  (scission 21/09 : réexport, le lot vit dans engine/lot.py)
from .deepstack import ajouter_deepstack, disperser_images, niveaux_deepstack   # noqa: F401  (scission 21/09, module 2 : réexport)
from .attention import (Attention, MLP, MLP2, SEUIL_FUSION, _ROPE_KV,   # noqa: F401  (scission 21/09, module 3 : réexport)
                        _multi_projection, _multi_utilisable)
from .layers import (QuantLinear, RMSNorm, RotaryEmbedding, add_norm, apply_rope,
                     rope_fusee,
                     attention, batched_decode_attention,
                     decode_attention_fixed, repeat_kv)

__all__ = ["Attention", "MLP", "MoEBlock", "DecoderLayer", "ACVRamModel",
           "ForwardBatch"]


_SYNC_COUCHES = bool(os.environ.get("ACVRAM_SYNC_COUCHES"))

# ACVRAM_TRACE_COUCHES=1 : une ligne par couche, après synchronisation, avec
# l'horodatage — pour situer un pas qui ne rend jamais la main (70B en exil,
# 17/09 : 33 min sans une ligne, GPU 0 %, pile Python illisible sans ptrace).
# Lent (une synchronisation par couche) : diagnostic seulement.
_TRACE_COUCHES = bool(os.environ.get("ACVRAM_TRACE_COUCHES"))
# Pièce 156 F5 (défaut depuis le verdict 156 c, au bit ; 0 = témoin) : résidu différé aussi pour les couches Gated DeltaNet
# non MLA (Qwen3.5/3.8) — voir `_res_differe`.
_GDN_RES_DIFFERE = os.environ.get("ACVRAM_GDN_RES_DIFFERE", "1") == "1"


def _trace_couche(quoi: str, i: int, layer) -> None:
    if not _TRACE_COUCHES:
        return
    if layer.device.type == "cuda":
        torch.cuda.synchronize(layer.device)
    exile = any(getattr(m, "streamed", None) is not None for m in layer.modules())
    print(f"[acvram] {quoi} couche {i:3d} {'flux' if exile else 'résidente'} "
          f"t={time.monotonic():.3f}", file=sys.stderr, flush=True)

# Scission 22/09, module 4 : le bloc MoE vit dans engine/moe.py — réexport
# intégral, mêmes objets. Un masquage de réglage vise acvram.engine.moe.
from . import moe as _moe_mod
from .moe import (   # noqa: F401
    MoEBlock, _MOE_GROUPED_MAX, _MOE_GEMM_MAX,
    _MOE_MMA, _PREFILL_GROUPED, _PREFILL_A4,
    _PREFILL_W8R, _PREFILL_A8, _PREFILL_A8_FMT,
    _E2M1, _E2M1_MILIEUX, fausse_quant_a8,
    fausse_quant_nvfp4, _MOE_MMA_BT, _MOE_MMA_ETAGES,
    _MOE_MMA_KS, _MOE_DECODE_MMA, _MOE_DECODE_MMA_BT,
    _MOE_AWQ_TEMOIN, _QA_COMPTE, _QA_COMPTEURS,
    _qa_compteurs, _qa_imprime, _MOE_DECODE_MMA_MIN_T,
    _ROUTE_PREP, _MOE_DECODE_FUSED, _MOE_FUSED_TN,
    _MOE_FUSED_ATOMIQUE, _MOE_FUSED_ETAGES, _MOE_ROUTE_PACK,
    _MOE_GEMV, _GEMV_LAYOUT, _MOE_DECODE_MMA_MARLIN,
    _MARLIN_DISTINCT, _TRACE_ROUTAGE, _ROUTAGES,
    _ROUTAGE_TEMOIN, _TEMOINS_ROUTAGE, temoins_routage,
    _DOUBLE_DIAG, _mla_glue, _COLLE_MOE,
    _BORNES_EXPERTS, _comptes_tries, _colle_moe_triton,
    MoEBlockGemma, _DUMP_MOE,
)

# Scission 22/09, module 5 : les couches vivent dans engine/couches.py —
# réexport intégral (mêmes objets). Masquer un réglage : acvram.engine.couches.
from .couches import (   # noqa: F401
    _NORME_FUSEE, _MLA_BATCH, _STATIC,
    _sid_fantome, DecoderLayerGDN, DecoderLayer,
    DecoderLayerParallel, DecoderLayerGemma,
)


class ACVRamModel(nn.Module):
    """Le modèle assemblé, ses couches réparties sur plusieurs appareils."""

    def __init__(self, spec: ModelSpec, embed: torch.Tensor,
                 layers: list[DecoderLayer], norm: RMSNorm,
                 lm_head: QuantLinear, caches: dict[int, PagedKVCache],
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.spec = spec
        self.embed_tokens = embed          # kept as a plain tensor: it is a gather
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head
        self.caches = caches
        self.dtype = dtype
        # Tête de prédiction multi-jetons, quand le modèle en porte une : le
        # chargeur la pose ici. Le brouillon spéculatif a besoin de l'état
        # caché normalisé du dernier jeton ; on le recopie dans un tampon
        # statique pour que la capture du graphe de décodage l'emporte avec
        # elle — une affectation Python, elle, ne serait pas rejouée.
        self.mtp = None
        self._mtp_hidden: Optional[torch.Tensor] = None
        self._mtp_hidden_n: int = 0
        self._mtp_prefill: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, return_hidden: bool = False,
                logits_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits du dernier jeton de chaque séquence.

        Avec ``return_hidden``, ce sont les états cachés normalisés qui sont
        rendus, pour tous les jetons et non seulement le dernier — c'est sur eux
        que /v1/embeddings fait sa moyenne. Sauter lm_head évite aussi le produit
        matriciel le plus coûteux du modèle : plonger un document coûte donc
        nettement moins que d'engendrer à partir de lui.
        """
        _moe_mod._EN_PREFILL[0] = bool(batch.is_prefill)
        if _trace_routage.actif():
            # Pièce 27(a) : masque de modalité de CETTE passe (image/texte/spécial),
            # aligné sur la séquence réellement fournie ; lu par `noter`.
            _trace_routage.poser_modalites(_trace_routage.modalites_du_lot(
                batch.tokens, getattr(self.spec, "image_token_id", None),
                [p for i in range(len(batch.seq_lens)) for p in (batch.images_de(i) or [])]))
        idx = batch.tokens.to(self.embed_tokens.device)
        x = F.embedding(idx, self.embed_tokens).to(self.dtype)
        if self.spec.embedding_multiplier != 1.0:
            x = x * self.spec.embedding_multiplier
        if batch.images is not None:
            # Après le multiplicateur : HF Gemma disperse les traits d'image
            # dans des embeddings déjà mis à l'échelle (masked_scatter), les
            # traits eux-mêmes ne sont pas multipliés.
            x = disperser_images(x, batch)

        current = None
        # C15-prefill : résidu différé au préfill eager — (x, delta) d'une couche
        # à la suivante, la somme faite par add_norm de la couche suivante (et
        # par celle de la norme finale) ; réservé aux blocs ordinaires à
        # multiplicateur 1,0, sinon `forward` (le chemin d'avant, au bit).
        differe = kernels.prefill_compact("residu") and all(
            type(l) is DecoderLayer and l.self_attn is not None and l.mlp is not None
            and l.mlp_device == l.device and l.residual_multiplier == 1.0 for l in self.layers)
        delta = None
        # Deepstack (Qwen3-VL) : n niveaux à ajouter après les couches 0..n−1
        # aux lignes image ; 0 sans image, un entier comparé par couche.
        n_deep = niveaux_deepstack(batch) if batch.deepstack is not None else 0
        for i, layer in enumerate(self.layers):
            if layer.device != current:
                if delta is not None:
                    x, delta = x + delta, None
                x = x.to(layer.device, non_blocking=True)
                current = layer.device
            # On lance le transfert de la couche suivante avant d'exécuter
            # celle-ci, pour que la copie PCIe d'une couche résidant en mémoire
            # vive se cache derrière du vrai travail.
            if i + 1 < len(self.layers):
                self.layers[i + 1].prefetch()
            if differe:
                x, delta = layer.forward_res(x, delta, batch, self.caches.get(i))
            else:
                x = layer(x, batch, self.caches.get(i))
            if i < n_deep:
                x, delta = ajouter_deepstack(x, delta, batch, i)
            _trace_couche("forward", i, layer)
            if _SYNC_COUCHES:
                # Diagnostic : une faute CUDA asynchrone remonte au premier
                # point de synchronisation, loin de son origine. Synchroniser
                # après chaque couche la fait remonter avec le bon index.
                try:
                    torch.cuda.synchronize(layer.device)
                except Exception as exc:
                    raise RuntimeError(f"faute CUDA après la couche {i} "
                                       f"({type(layer).__name__} sur {layer.device}) : {exc}") from exc

        if delta is None:
            brut = x
            x = self.norm(x.to(self.norm.weight.device))
        elif x.device == self.norm.weight.device:
            # résidu différé : la dernière somme dans la norme finale
            brut, x = add_norm(x, delta, self.norm, 1.0)
        else:
            brut = x + delta
            x = self.norm(brut.to(self.norm.weight.device))
        if return_hidden:
            return x
        if self.mtp is not None:
            brut = brut.to(x.device)     # DeepSeek : la tête MTP lit l'état AVANT la norme finale
            etat = x if self._mtp_normalise() else brut      # Qwen3.5 (vLLM) : APRÈS (pièce 105)
            if batch.is_prefill:
                # Le brouillon MTP a besoin du contexte entier pour amorcer son
                # propre cache : au prefill on garde tous les etats, pas
                # seulement celui du dernier jeton.
                self._mtp_prefill = etat.detach()
            self._garder_hidden(etat[(batch.last_token_indices() if logits_positions
                                      is None else logits_positions).to(etat.device)])
        # La vérification spéculative et la perplexité ont toutes deux besoin
        # de logits ailleurs qu'à la position finale : les lignes qui atteignent
        # lm_head sont donc un paramètre. Cela compte, car lm_head est le plus
        # gros produit matriciel du modèle, et l'exécuter sur chaque jeton de
        # prefill au lieu d'un seul coûte du temps réel.
        idx = (batch.last_token_indices() if logits_positions is None
               else logits_positions)
        x = x[idx.to(x.device)]
        # LES LOGITS SE PRODUISENT EN FP32, ET C'EST L'ENTREE QU'ON CONVERTIT.
        # Le dtype de sortie des noyaux suit celui de x — nvfp4_gemv fait
        # `out = torch::empty({N, M}, xc.options())` — donc passer x en float
        # bascule le produit sur le chemin float de bout en bout. Convertir la
        # SORTIE ne restaurerait rien : la perte est dans l'accumulation, pas
        # dans un arrondi final.
        #
        # Ce que l'arrondi bf16 coutait, mesure sur quatre pas de GLM-4.7 :
        # logits centres sur 95,6 avec une amplitude de 23, donc dans
        # l'intervalle [64, 128) ou le pas bf16 vaut 0,5 — QUARANTE-CINQ
        # niveaux distincts pour 151 936 jetons, et une marge top1-top2 de un
        # a deux ULP.
        #
        # Et l'erreur n'est pas seulement du bruit. logsumexp est convexe,
        # donc par Jensen l'arrondi introduit un BIAIS de 1/2 sigma^2
        # (1 - somme p^2) qui NE DECROIT PAS avec la longueur du corpus :
        # verifie par simulation, 0,009875 nat mesure contre 0,010410 predit,
        # soit +1,05 % sur la perplexite. Le bruit, lui, decroit en 1/racine(N)
        # et vaut 0,09 % sur nos etalons de 148 920 jetons.
        #
        # Le biais depend du PAS, donc de la magnitude des logits, donc du
        # modele : 1,05 % sur GLM-4.7 contre 0,001 % sur un modele centre sur
        # zero, un rapport de 1024. C'est donc un biais SYSTEMATIQUE ENTRE
        # MODELES, qui fausse exactement les comparaisons de perplexite que
        # nous faisons. En fp32 pres de 116 le pas tombe a 7,6e-06 et le biais
        # a 2,4e-12 nat : la question ne se pose plus.
        #
        # Le cout est nul : l'etat cache fait quelques milliers d'elements, et
        # le vecteur de sortie 151 936 flottants, soit 594 Kio par jeton.
        # ACVRAM_LOGITS_BF16=1 retablit l'ancien comportement : il rend le
        # correctif MESURABLE par A/B sans recompiler, et sert de repli si le
        # cout en temps s'averait sensible. Un correctif qu'on ne peut pas
        # comparer a son absence n'est pas evaluable.
        logits = self._tete(x)
        if logits.dtype != torch.float32:
            # Un noyau qui rend autre chose que ce qu'on lui a donne annule le
            # correctif en silence. On le dit une fois plutot que de le taire.
            if not getattr(self, "_dit_logits_dtype", False):
                self._dit_logits_dtype = True
                print(f"[acvram] les logits sortent en {logits.dtype} malgre "
                      "une entree fp32 : le noyau de la tete impose son type, "
                      "et le gain de resolution n'est pas acquis.", flush=True)
        return self._logits_finaux(logits)

    def _tete(self, x: torch.Tensor) -> torch.Tensor:
        """L'entrée de ``lm_head`` en fp32 sur l'appareil de la tête — LE MÊME
        texte pour le chemin eager et le chemin à formes fixes (graphes,
        ACVRAM_GRAPHS_EAGER). Le fixe gardait la tête en bf16 (a5fac1c n'avait
        porté le fp32 qu'en eager) : l'argmax basculait dès le 2e pas de
        décodage, 64-68 % de jetons justes à l'arbitre prefill = décodage
        (poste7-duel-verdict § 13, REGLES §7). ACVRAM_LOGITS_BF16=1 : témoin."""
        head_dev = getattr(self.lm_head.qweight, "qweight", None)
        target = head_dev.device if head_dev is not None else x.device
        if os.environ.get("ACVRAM_LOGITS_BF16") == "1":
            return self.lm_head(x.to(target))
        # Tete INT8 au decodage (poste7-duel-verdict par. 14 (ii)) : x reste en
        # bf16, le GEMV accumule et SORT en fp32 — memes produits, meme ordre
        # de sommes que x.to(float32) : logits egaux au bit, sans la conversion
        # de h ni le double trafic de x en fp32 (0,85 -> ~0,2 ms attendu).
        # ACVRAM_TETE_FP32_ENTREE=1 : temoin (l ancienne conversion).
        w = self.lm_head.qweight
        lin = self.lm_head
        if (x.dtype == torch.bfloat16 and getattr(w, "format", "") == "int8"
                and head_dev is not None and head_dev.is_cuda
                and getattr(lin, "scaler", None) is None and getattr(lin, "streamed", None) is None
                and getattr(lin, "bias", None) is None
                and kernels.tete_int8_entree_bf16(x.shape[0])
                and os.environ.get("ACVRAM_TETE_FP32_ENTREE") != "1"):
            return kernels.int8_matmul(x.to(target), w, sortie_fp32=True)
        # Tête NVFP4 à b ≥ DENSE_NVFP4_MIN_M (poste7-gemm-dense-palier2-non-
        # ouvert-17-09 § 2) : la même GEMM dense étroite que les projections
        # (poids lus une fois par pas), logits accumulés en fp32 ; la GEMV fp32
        # relisait 0,6 Go par séquence — 18 % du pas Qwen3.8 b=12 (poste3 0690bd4).
        if (x.dtype == torch.bfloat16 and getattr(w, "format", "") == "nvfp4"
                and head_dev is not None and head_dev.is_cuda
                and getattr(lin, "streamed", None) is None and getattr(lin, "bias", None) is None
                and kernels._DENSE_NVFP4 == "triton" and kernels._DENSE_NVFP4_MIN_M <= x.shape[0] <= 32
                and os.environ.get("ACVRAM_TETE_FP32_ENTREE") != "1"):
            from ..kernels import gemm_dense_etroit as _gde
            if _gde.disponible():
                xt = x.to(target)
                sc = getattr(lin, "scaler", None)
                if sc is not None and not sc.is_identity:
                    xt = sc.apply(xt)
                return _gde.gemm_dense_etroit(xt, w, sortie_fp32=True)[:, : w.shape[0]]
        return self.lm_head(x.to(target, dtype=torch.float32))

    def _logits_finaux(self, logits: torch.Tensor) -> torch.Tensor:
        # La tête de sortie est rembourrée à un multiple de 64 lignes pour ses
        # noyaux ; ces colonnes n'ont pas de jeton et ne doivent jamais gagner
        # l'argmax — sur muse-glimmer-30b (202 048 jetons, tête de 202 112)
        # l'une d'elles sortait vers le 250e jeton et faisait tomber le
        # plongement suivant sur un indice hors table.
        v = self.spec.vocab_size
        if v and logits.shape[-1] > v:
            logits = logits[..., :v]
        if self.spec.logits_scaling != 1.0:
            logits = logits / self.spec.logits_scaling
        c = self.spec.final_logit_softcapping
        if c:
            logits = torch.tanh(logits.to(torch.float32) / c) * c
        return logits

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     q_len: int = 1) -> torch.Tensor:
        """Logits d'un pas de décodage pur, à formes fixes.

        Le plongement est déjà fait — ``x`` est l'état caché d'entrée sur le
        périphérique des couches : l'indexation de la table de plongement vit
        hors du graphe, sur l'appareil où elle réside.
        """
        _moe_mod._EN_PREFILL[0] = False
        if self._res_differe():
            delta = None
            # C15 niveau 3 : `slots >= 0` (fantômes du godet) une fois par pas
            # au lieu d'une fois par couche (48 nœuds → 1) ; témoin : None,
            # chaque couche le recalcule.
            # (sans couche MoE — jouet dense ou MLA seul — ce `ge` serait un nœud de plus,
            # pas de moins : test_mla_glue_c15 le comptait, 20/09)
            if "_a_des_moe" not in self.__dict__:
                self.__dict__["_a_des_moe"] = any(isinstance(l.mlp, MoEBlock) for l in self.layers)
            valid = (slots >= 0) if kernels.glue_compact("kv") and self.__dict__["_a_des_moe"] else None
            for i, layer in enumerate(self.layers):
                if valid is not None and type(layer) is DecoderLayer:
                    x, delta = layer.decode_fixed_res(
                        x, delta, positions, slots, block_tables, seq_lens,
                        max_pos, self.caches.get(i), q_len, valid=valid)
                    continue
                x, delta = layer.decode_fixed_res(
                    x, delta, positions, slots, block_tables, seq_lens,
                    max_pos, self.caches.get(i), q_len)
            x, h = add_norm(x, delta, self.norm,
                            getattr(self.layers[-1], "residual_multiplier", 1.0))
            if self.mtp is not None:
                self._garder_hidden(h if self._mtp_normalise() else x)
            return self._logits_finaux(self._tete(h))
        for i, layer in enumerate(self.layers):
            x = layer.decode_fixed(x, positions, slots, block_tables,
                                   seq_lens, max_pos, self.caches.get(i), q_len)
            _trace_couche("décodage", i, layer)
        xn = self.norm(x)
        if self.mtp is not None:
            self._garder_hidden(xn if self._mtp_normalise() else x)
        return self._logits_finaux(self._tete(xn))

    # Lignes réservées d'avance pour l'état caché que lit la tête MTP : un lot
    # de vérification spéculative en pose k+1 par séquence.
    MTP_HIDDEN_LIGNES = 16

    def reserver_hidden(self, lignes: int, h: Optional[torch.Tensor] = None) -> None:
        """Réserve, hors de toute capture, le tampon où ``_garder_hidden``
        recopie l'état caché. Le tampon ne bouge plus ensuite.

        Il a été alloué à la volée, par ``clone()``, à la forme du pas courant :
        capturé dans un graphe de vérification (5 lignes), puis réalloué par le
        pas suivant (1 ligne), il laissait au graphe l'adresse d'un tenseur
        rendu au pool — le rejeu écrivait dans une page morte (``memcpy32_post``,
        Warp MMU Fault, reproduit sur Qwen3.8-27B le 5/09/2026), et le brouillon
        MTP lisait entre-temps un état périmé."""
        ref = h if h is not None else self._mtp_hidden
        if ref is None:
            return
        cap = max(lignes, self.MTP_HIDDEN_LIGNES)
        b = self._mtp_hidden
        if (b is not None and b.shape[0] >= cap and b.shape[1:] == ref.shape[1:]
                and b.dtype == ref.dtype and b.device == ref.device):
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("tampon MTP réservé pendant une capture de graphe : "
                               "appeler reserver_hidden() avant la capture")
        self._mtp_hidden = torch.zeros((cap,) + tuple(ref.shape[1:]),
                                       dtype=ref.dtype, device=ref.device)

    def _mtp_normalise(self) -> bool:
        """Pièce 105 : l'état caché que lit la tête MTP. DeepSeek-V3 passe l'état NON normalisé (sa tête le normalise,
        `hnorm`) ; Qwen3.5 dans vLLM passe l'état APRÈS la norme finale (qwen3_next.py:726, puis
        `pre_fc_norm_hidden`). ``ACVRAM_MTP_ETAT`` : brut (défaut d'avant la 105) | norme | auto (selon la convention
        de la tête posée par le chargeur, `mtp.convention`). Lu côté hôte, donc figé à la capture d'un graphe."""
        mode = os.environ.get("ACVRAM_MTP_ETAT", "brut")
        if mode == "norme":
            return True
        if mode == "auto":
            return getattr(self.mtp, "convention", "deepseek") == "qwen35"
        return False

    def _garder_hidden(self, h: torch.Tensor) -> None:
        """Recopie l'état caché normalisé dans les ``n`` premières lignes du
        tampon réservé. Une copie, et non une référence : sous graphe CUDA le
        tenseur source est réécrit à chaque rejeu, et une affectation Python ne
        serait jouée qu'à la capture. Le nombre de lignes valides est posé par
        le moteur (``_mtp_hidden_n``), lui aussi hors graphe."""
        h = h.detach()
        n = h.shape[0]
        b = self._mtp_hidden
        if b is None or b.shape[0] < n or b.shape[1:] != h.shape[1:] \
                or b.dtype != h.dtype or b.device != h.device:
            self.reserver_hidden(n, h)
            b = self._mtp_hidden
        b[:n].copy_(h)
        self._mtp_hidden_n = n

    def _res_differe(self) -> bool:
        """Le chemin à résidu différé n'est pris que si toutes les couches
        sont des blocs attention+MLP ordinaires — les hybrides et Gemma 4 ont
        leurs propres enchaînements de normes."""
        v = getattr(self, "_res_ok", None)
        if v is None:
            def ordinaire(l) -> bool:
                return (type(l) is DecoderLayer and l.self_attn is not None
                        and l.mlp is not None and l.mlp_device == l.device
                        and hasattr(l, "decode_fixed_res"))

            def mla_glue(l) -> bool:
                # C15 : couche MLA (GLM, DeepSeek) sous ACVRAM_MLA_GLUE ≥ 1 —
                # jamais GDN/Mamba ni les blocs parallèles, qui gardent leurs
                # enchaînements de normes
                return (type(l) is DecoderLayerGDN and hasattr(l.linear_attn, "rank")
                        and l.mlp is not None and l.mlp_device == l.device)
            def gdn_glue(l) -> bool:
                # Pièce 156 F5 : couche Gated DeltaNet (pas MLA, pas KDA ni
                # Mamba) à deux RMSNorm — `decode_fixed_res` y est générique,
                # et `add_norm` rend l'addition bf16 de torch au bit
                # (couches.py, docstring de `decode_fixed_res`)
                return (_GDN_RES_DIFFERE and type(l) is DecoderLayerGDN
                        and type(l.linear_attn).__name__ == "GatedDeltaNet"
                        and l.mlp is not None and l.mlp_device == l.device
                        and type(l.input_layernorm).__name__ == "RMSNorm"
                        and type(l.post_attention_layernorm).__name__ == "RMSNorm")
            glue = _mla_glue() >= 1
            v = all(ordinaire(l) or (glue and mla_glue(l)) or gdn_glue(l) for l in self.layers) \
                and type(self.norm).__name__ == "RMSNorm"
            self._res_ok = v
        return v

    @property
    def nbytes(self) -> int:
        total = self.embed_tokens.numel() * self.embed_tokens.element_size()
        for m in self.modules():
            if isinstance(m, QuantLinear):
                total += m.nbytes
        return total

    def usage_routage(self) -> dict:
        """Histogramme de routage par couche — À LA DEMANDE seulement.

        `MoEBlock._compter_routage` tourne à chaque pas ; ceci ne l'est PAS :
        c'est le point de lecture (dump fin de requête, endpoint /routage),
        jamais appelé depuis le chemin de décodage. Rend `{index_couche:
        tenseur}` pour chaque couche MoE qui a déjà tourné au moins une fois ;
        les couches denses ou pas encore sollicitées n'y figurent pas. Les
        tenseurs restent sur leur device — `.cpu()` (et donc la synchronisation
        qu'il implique) est décidé par l'appelant, pas ici."""
        return {m.index_couche: m._usage_routage for m in self.modules()
                if isinstance(m, MoEBlock) and m._usage_routage is not None}

    def nbytes_detail(self) -> dict:
        """Decompose `nbytes` pour que l'ecart a la prevision se NOMME.

        Le 10/09/2026, la prevision statique tiree des manifestes (embedding au
        dtype de chargement + somme des `QuantLinear.nbytes`) a donne 15,9994
        bits/poids pour Llama-2-7b-fp16pur quand le releve d'evaluation en
        portait 26,987 — un facteur 1,687. Meme forme pour les deux dossiers
        quantifies, mais a 10,3 % seulement (8,3391 prevu contre 9,198 releve ;
        4,7235 contre 5,215). Un ecart qui n'est pas proportionnel aux octets
        stockes : ce n'est donc pas une erreur de densite.

        Cette decomposition teste l'hypothese de tete : un meme objet de poids
        compte plusieurs fois, parce que `modules()` le rencontre sous
        plusieurs `QuantLinear` (projections fusionnees exposees aussi comme
        tranches, tete liee a l'embedding). `vus_plusieurs_fois` rend le compte
        exact des octets comptes en double ; s'il est nul, l'hypothese tombe et
        `par_classe` dit ou les octets sont reellement.
        """
        vus: dict[int, dict] = {}
        par_classe: dict[str, int] = {}
        n_ql = 0
        for nom, m in self.named_modules():
            if not isinstance(m, QuantLinear):
                continue
            n_ql += 1
            q = m.qweight
            o = getattr(q, "nbytes", 0)
            fmt = getattr(q, "format", type(q).__name__)
            par_classe[fmt] = par_classe.get(fmt, 0) + o
            e = vus.setdefault(id(q), {"octets": o, "fmt": fmt, "noms": []})
            e["noms"].append(nom)
        double = sum(e["octets"] * (len(e["noms"]) - 1) for e in vus.values())
        # Le compte par IDENTITE D'OBJET a rendu 0 a la premiere mesure
        # (230 QuantLinear, 230 objets distincts) : l'hypothese « un meme objet
        # rencontre plusieurs fois » est morte. Il reste 461 455 360 octets
        # au-dessus de la somme des tenseurs du DISQUE (7 223 386 112 mesures
        # contre 6 761 930 752 attendus pour l'int8 de Llama-2-7B), et le
        # detecteur ne pouvait pas les voir : deux objets DISTINCTS portant des
        # octets equivalents — une projection fusionnee materialisee a cote de
        # ses tranches — s'accordent avec « 0 objet vu deux fois ».
        #
        # D'ou ce second compte, par NOM et par forme : il attribue les octets
        # a des tenseurs nommes, donc il peut nommer les 461 Mo au lieu de les
        # constater. Note au passage : 230 QuantLinear pour 225 tenseurs
        # stockes au manifeste — cinq de plus, et 461 455 360 / 5 = 92 291 072,
        # soit exactement la taille d'un gate_up fusionne en int8. C'est une
        # PISTE, pas une explication : elle attend ce releve pour etre nommee.
        # OCTETS REELLEMENT ALLOUES, par stockage unique. `nbytes` somme des
        # `numel()` : une VUE y compte ses elements comme si elle possedait ses
        # octets. Or les quatre empileurs remplacent les originaux par des
        # tranches de la pile — c'est le but — donc `nbytes` compte deux fois
        # tout ce qui est fusionne.
        #
        # C'est ce qui explique ENTIEREMENT l'ecart que je cherchais depuis ce
        # matin. Llama-2-7b-fp16pur, tout PlainTensor en bf16 :
        #
        #   base (embed + 257 denses)          13 476 302 848
        #   vues q, k, v                        3 221 225 472
        #   vues gate, up                       5 771 362 304
        #   embed compte une seconde fois         262 144 000
        #   PREVU                              22 731 034 624
        #   MESURE                             22 731 030 528   ecart 4 096 o
        #
        # Les 4 096 octets restants sont les 2 048 parametres que le manifeste
        # compte en trop (6 738 417 664 contre 6 738 415 616 reels). Il n'y a
        # donc plus rien d'inexplique dans le 26,987 bits/poids : ce n'etait ni
        # un cache, ni un tampon, ni un doublon accidentel — c'etait l'unite de
        # mesure.
        vus_stockage: dict[int, int] = {}
        for nom, m in self.named_modules():
            if not isinstance(m, QuantLinear):
                continue
            q = m.qweight
            for champ in ("qweight", "weight", "scales", "zeros",
                          "block_scale", "global_scale_rows"):
                t = getattr(q, champ, None)
                if t is None or not hasattr(t, "untyped_storage"):
                    continue
                st = t.untyped_storage()
                vus_stockage[st.data_ptr()] = st.nbytes()
        st_emb = self.embed_tokens.untyped_storage()
        vus_stockage[st_emb.data_ptr()] = st_emb.nbytes()
        octets_stockage = sum(vus_stockage.values())

        par_forme: dict[str, int] = {}
        for e in vus.values():
            for nom in e["noms"]:
                par_forme[nom] = e["octets"]
        gros = sorted(par_forme.items(), key=lambda kv: -kv[1])[:12]
        emb = self.embed_tokens.numel() * self.embed_tokens.element_size()
        return {
            "total": self.nbytes,
            "embed_tokens": emb,
            "embed_dtype": str(self.embed_tokens.dtype),
            "quantlinear": n_ql,
            "objets_distincts": len(vus),
            "octets_comptes_en_double": double,
            "total_sans_doubles": emb + sum(e["octets"] for e in vus.values()),
            "par_format": par_classe,
            "vus_plusieurs_fois": [
                {"octets": e["octets"], "fmt": e["fmt"], "noms": e["noms"]}
                for e in vus.values() if len(e["noms"]) > 1],
            "octets_stockage_uniques": octets_stockage,
            "stockages_distincts": len(vus_stockage),
            "octets_dupliques_par_les_vues": self.nbytes - octets_stockage,
            "douze_plus_gros": [{"nom": n, "octets": o} for n, o in gros],
            "octets_par_nom_total": sum(par_forme.values()),
        }
