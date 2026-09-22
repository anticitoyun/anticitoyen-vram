"""MLA — Multi-head Latent Attention (couches d'attention pleine de
« kimi-linear »), en forme absorbée, fidèle au graphe llama.cpp.

Particularités de Kimi-Linear : **aucun RoPE** (la position ne passe que par
la récurrence KDA des autres couches) ; le cache d'une séquence est le latent
compressé ``[t, kv_lora_rank + rope_dim]`` (576 octets ×2 par jeton et par
couche) — pas de K/V par tête, pas de cache paginé.

Forme absorbée : ``q_eff = [k_b(q_nope) | q_pe]`` s'apparie au latent caché
``[c | k_pe]`` ; la valeur est relue dans l'espace latent puis décompressée
par ``v_b``. Échelle 1/√(dim_qk_complet), comme llama.cpp (kq_scale_mla).
"""

from __future__ import annotations

from typing import Optional

import os

import torch
import torch.nn as nn

__all__ = ["MLAttention", "MLA_BUCKET", "godet_mla"]

# Godet de longueur du cache latent : partagé avec le chemin des graphes.
# L'attention MLA balaie tout le godet, pas seulement les positions écrites :
# à 1024, une séquence de 264 jetons en paie quatre fois trop. Un godet plus
# fin coûte davantage de graphes capturés (un par palier), mais chaque pas
# lit moins. Réglable pour mesurer l'arbitrage.
MLA_BUCKET = int(os.environ.get("ACVRAM_MLA_BUCKET", "128"))
# Noyau MLA à une passe (sage-duel-verdict-16-09 § 3) : le cache latent lu une
# fois pour les H têtes au lieu de 2 × H fois (mla_scores + mla_reduce) ;
# =0 rejoue les deux noyaux d'avant (témoin d'équivalence).
_MLA_UNE_PASSE = os.environ.get("ACVRAM_MLA_UNE_PASSE", "1") == "1"
# ACVRAM_MLA_A8=off|e4m3|int8 : porte qualité FP8-MLA (sage-cloture-23h59-19-09) — les
# ENTRÉES de q_b, kv_a et o (les trois projections que garde bf16 la règle « jamais NVFP4
# sur MLA », REGLES § 9) arrondies en E4M3 bloc 16 (ou int8 par jeton, témoin) avant la
# projection, torch pur ; aucun noyau : la perte qu'un GEMM FP8 imposerait, pas sa vitesse.
# ACVRAM_MLA_CORE=fp32|tf32|bf16 (C13, sage-c7-clos-c13-attention-glm-19-09, sage-c13a-defaut-19-09
# § 1 et addendum 20 h 05, sage-m2-mma2-budgets-prefill-19-09 § 3) : précision des DEUX premiers
# produits du cœur d'attention MLA au PRÉFILL — scores q·C, o_lat = probs·V (einsum chunké/non
# chunké) ; le softmax, le masque et le noyau CUDA mla_1p restent en fp32.
#   fp32 = SIMT fp32 plein (GLM prefill : 220 ms/pas, 67 %) ;
#   tf32 = DÉFAUT depuis ce commit (C13-a tenu : 7 191 ≥ 6 200 j/s, ΔPPL géo +0,00066 ≤ 0,001,
#          Manon 3bcc173 / main a7397e2) : mêmes einsum fp32 sous tensor cores TF32 (entrées
#          10 bits, acc fp32) ;
#   bf16 = C13-b : entrées castées en bf16, accumulation fp32 des tensor cores, sortie bf16
#          (prédiction cœur 220 → 80-95 ms ; scellé contre fp32 : prefill ≥ 9 000 j/s ET ΔPPL
#          géo ≤ +0,002 → défaut).
#   flash = C13-c forme 1 (chantier-c13c-19-09 § Correction Sage ; kernels/attn_mla_causal.py) :
#          les deux produits, le masque et le softmax FUSIONNÉS en un noyau Triton causal, en
#          fp32 plein (tl.dot IEEE, softmax fp32) — un lancement par couche au lieu de 8 morceaux
#          × 5 ops, les scores ne passent plus par la HBM (≈ 70 Go/pas), la moitié masquée n'est
#          pas calculée ; sortie = fp32 ± 8 ulp de l'amplitude par ligne, donc aucune porte de
#          PPL et valable à TOUTES les longueurs (la règle des 2 048 clés ne le concerne pas).
#          Scellé Sage : cœur 220 → ≤ 110 ms, prefill GLM ≥ 9 000 j/s. Le décodage n'est pas
#          touché. Sans Triton ni carte : repli fp32 einsum NOMMÉ (`_FLASH_REPLI`, ligne de
#          régime `mla_core=flash(repli fp32: …)`), jamais silencieux.
# Deux portées nommées à part, opt-in jusqu'à leur scellé (sage-c13a-defaut § 2) :
#   ACVRAM_MLA_CORE_VB=1 : le 3e produit du préfill, y = v_b·o_lat (8ae21997 ; scellé prefill
#          ≥ 7 450 j/s ET ΔPPL géo ≤ +0,001 contre 2 produits) suit MLA_CORE ; 0 (défaut) = fp32 ;
#   ACVRAM_MLA_CORE_DECODE=fp32|tf32|bf16 : le cœur du DÉCODAGE (y = v_b·o_lat, sgemm fp32 de
#          1,5 ms/pas à b=12 ; niveau 2, scellé ≤ 0,6 ms ET ppl-decode-kv ± 0,001 ET capture 5/5) ;
#          fp32 (défaut) quelle que soit la valeur de MLA_CORE.
_MLA_CORE = os.environ.get("ACVRAM_MLA_CORE", "tf32")
_MLA_CORE_DECODE = os.environ.get("ACVRAM_MLA_CORE_DECODE", "fp32")
_MLA_CORE_VB = os.environ.get("ACVRAM_MLA_CORE_VB", "0") == "1"
if _MLA_CORE not in ("fp32", "tf32", "bf16", "flash"):
    raise ValueError(f"ACVRAM_MLA_CORE={_MLA_CORE!r} : fp32 | tf32 | bf16 | flash")
if _MLA_CORE_DECODE not in ("fp32", "tf32", "bf16"):          # flash = préfill seul
    raise ValueError(f"ACVRAM_MLA_CORE_DECODE={_MLA_CORE_DECODE!r} : fp32 | tf32 | bf16")
_FLASH_REPLI: str | None = None      # raison du repli fp32 sous MLA_CORE=flash, posée au premier préfill


def regime_prep_texte() -> str:
    """`mla_prep=grille` (défaut 0.6.31 : mla_prep_batch regrillé, 492 blocs à b=12, au bit) | `mla_prep=temoin`
    (grille d'avant, 172 blocs) — nommé défaut compris (REGLES § 4)."""
    return "mla_prep=grille" if _MLA_PREP_GRILLE else "mla_prep=temoin"


def regime_glue_texte() -> str:
    """`mla_glue=2` (défaut 0.6.32 : b=1 par decode_static_batch_complet, M3 tenu : −611 nœuds, pas b=1
    −0,819 ms, J −4,4 %, PPL non établi pire) | `mla_glue=1(temoin)` (glue torch retirée, chemin =1 d'avant)
    | `mla_glue=0` — nommé défaut compris (REGLES § 4)."""
    return {2: "mla_glue=2", 1: "mla_glue=1(temoin)"}.get(_MLA_GLUE, "mla_glue=0")


def regime_coeur_texte() -> str:
    """Le mot `mla_core=…` de la ligne de régime : rien sous fp32 ; `tf32(≤2048 clés)` /
    `bf16(≤2048 clés)` (règle des clés vues) ; `flash(fp32)` — ou `flash(repli fp32: raison)`
    quand le noyau n'a pas pu être lancé (REGLES § 4 : le régime effectif, pas le demandé)."""
    if _MLA_CORE == "fp32":
        return ""
    if _MLA_CORE == "flash":
        if _FLASH_REPLI is not None:
            return f"mla_core=flash(repli fp32: {_FLASH_REPLI})"
        try:                                                  # la tuile est nommée : elle dépend de la carte
            from ..kernels.attn_mla_causal import tuile_par_carte
            bm, bn, w, st = tuile_par_carte("cuda:0" if torch.cuda.is_available() else "cpu")
            return f"mla_core=flash({_FLASH_OPERANDES},{bm}x{bn}w{w}s{st})"
        except Exception:                                     # noqa: BLE001
            return "mla_core=flash(fp32)"
    if _MLA_CORE_MAX_CLES >= (1 << 62):
        return f"mla_core={_MLA_CORE}(sans règle des clés)"
    return f"mla_core={_MLA_CORE}(≤{_MLA_CORE_MAX_CLES} clés)"


# sage-c14-defaut-tf32-8k-20-09, addendum 02 h 25 (verdict-tf32-8k, Manon c2c1e87 : max |Δ| 6,23 %
# à 8 k > 2 %, géo −0,22 %) : le régime réduit (tf32, bf16) ne vaut que quand le morceau de
# préfill VOIT ≤ 2 048 clés (pas sa taille) ; au-delà, fp32 pour ce morceau. Même règle pour F
# (bf16) et C13-c. Ligne de régime : `mla_core=tf32(≤2048 clés)`.
# ACVRAM_MLA_CORE_MAX_CLES (diagnostic, Manon 20/09 08 h : bras « règle neutralisée » de la prise à 36
# tranches) : 0 = aucune règle (tf32/bf16 à toutes longueurs, comme d01eb2cb) ; défaut 2048 = la règle.
_MLA_CORE_MAX_CLES = int(os.environ.get("ACVRAM_MLA_CORE_MAX_CLES", "2048"))
if _MLA_CORE_MAX_CLES <= 0:
    _MLA_CORE_MAX_CLES = 1 << 62                     # « ≤ ∞ clés » : la règle ne bascule jamais en fp32


def _regime_coeur(decode: bool = False, vb: bool = False, cles: int | None = None) -> str:
    """Le régime qui s'applique à un produit du cœur : décodage → MLA_CORE_DECODE ;
    3e produit du préfill → MLA_CORE si MLA_CORE_VB, sinon fp32 ; sinon MLA_CORE —
    et fp32 dès que le morceau voit plus de _MLA_CORE_MAX_CLES clés (`cles`)."""
    if decode:
        return _MLA_CORE_DECODE
    if vb and not _MLA_CORE_VB:
        return "fp32"
    if _MLA_CORE == "flash":
        return "flash"          # forme 1 = fp32 plein : la règle des 2 048 clés ne s'applique pas
    if cles is not None and cles > _MLA_CORE_MAX_CLES:
        return "fp32"
    return _MLA_CORE


def _dt_coeur(decode: bool = False, vb: bool = False, cles: int | None = None) -> torch.dtype:
    """dtype des opérandes du produit : bf16 sous le régime bf16, fp32 sinon."""
    return torch.bfloat16 if _regime_coeur(decode, vb, cles) == "bf16" else torch.float32


class _tf32_coeur:
    """Contexte : TF32 pour les matmuls fp32 pendant le bloc quand le régime du
    produit (décodage / 3e produit / préfill) est tf32, drapeau restauré après —
    la portée est le produit du cœur, pas le processus. Sous fp32 et bf16 : inerte
    (bf16 n'a pas besoin du drapeau)."""
    def __init__(self, decode: bool = False, vb: bool = False, cles: int | None = None):
        self._decode, self._vb, self._cles = decode, vb, cles

    def __enter__(self):
        self._actif = _regime_coeur(self._decode, self._vb, self._cles) == "tf32" and torch.cuda.is_available()
        if self._actif:
            self._avant = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
        return self

    def __exit__(self, *exc):
        if self._actif:
            torch.backends.cuda.matmul.allow_tf32 = self._avant
        return False


_MLA_A8 = os.environ.get("ACVRAM_MLA_A8", "off")
if _MLA_A8 not in ("off", "e4m3", "int8"):
    raise ValueError(f"ACVRAM_MLA_A8={_MLA_A8!r} : off | e4m3 | int8")


def _a8(x: torch.Tensor) -> torch.Tensor:
    """Entrée d'une projection MLA sous la porte : identité au défaut."""
    if _MLA_A8 == "off":
        return x
    from ..quant.fakequant_activation import fake_quantize_a8
    forme = x.shape
    return fake_quantize_a8(x.reshape(-1, forme[-1]), _MLA_A8).reshape(forme)
# Préparation du lot (einsum k_b, RoPE, norme kv_a, cat, conversion fp32) en un
# lancement `mla_prep_batch` au lieu d'une quinzaine (sage-duel-verdict § 6.2,
# marche « RoPE + cat ») ; =0 rejoue les opérations torch (témoin).
_MLA_PREP_NOYAU = os.environ.get("ACVRAM_MLA_PREP_NOYAU", "1") == "1"
# Sonde niveau 2 (REGLES § 6, 20/09) : sous ACVRAM_MLA_PREP_TEMOIN=1, chaque appel du chemin
# de lot copie DANS LE GRAPHE (clones capturés vers des tampons persistants alloués à
# l'échauffement, hors capture) les entrées de `mla_prep_batch`, ses sorties à la sortie du
# noyau, et q_eff une seconde fois juste avant l'attention. Lus après un rejeu, ils disent
# lequel des trois diverge du chemin torch recalculé sur les MÊMES entrées : les entrées
# (ce que le graphe donne), le noyau (tables, k_b), ou le bassin (q_eff recouvert avant
# l'attention). Diagnostic seulement : coût d'un clone par tenseur et par couche.
_MLA_PREP_TEMOIN = os.environ.get("ACVRAM_MLA_PREP_TEMOIN") == "1"
_TEMOINS: list = []          # un dict par couche, dans l'ordre du premier appel


def temoins_prep() -> list:
    """Les tampons témoins (voir `_MLA_PREP_TEMOIN`), une entrée par couche MLA."""
    return _TEMOINS
# C15 (sage-glm-decode-budget-c14-c15-19-09, chantier-c15-19-09) : glue torch
# du décodage GLM à b=1 — copies et concaténations qui ne calculent rien,
# retirées SANS changer un bit (mêmes opérations, mêmes arrondis) :
#  1 = _v_b32 réutilisé dans _sortie_decode (une conversion fp32 de v_b par
#      couche et par pas en moins), cat kvp évitée dans _prep_decode (elle
#      était redécoupée aussitôt), RoPE écrite en place (torch.stack en moins),
#      tables cos/sin demi indexées en un lancement, et dans model.py le
#      résidu différé des couches MLA (add + rmsnorm → add_norm, deux
#      lancements de moins par couche) ; 6 lancements de moins par couche ;
#      côté MoE (model.py `_forward_grouped`) : tok int64 servi d'avance, eid
#      converti une fois, x[tok] rassemblé une fois, tok_g = seq — −3 par
#      couche Marlin, −6 par couche à tables AWQ distinctes, des index (au bit) ;
#  2 = en plus, b=1 passe par decode_static_batch_complet (mla_prep_batch,
#      la numérique servie à b=12 : q_abs à ≤ 1 ulp bf16 de cuBLAS, PAS au bit
#      avec decode_static) ; non vérifié à sec (noyau CUDA) ;
#  0 = chemin d'avant (témoin).
#  DÉFAUT 1 depuis le 20/09 (verdict-c15-19-09, Manon, sage-c15-niveaux-20-09 § Ordre
#  04 h 20) : 2 622 → 2 163 lancements/pas, jetons identiques 256 pas, GLM b=1 servi
#  116,4 → 122,5 t/s (−0,43 ms/pas), capture 5/5 ; le niveau 2 est FAUX (1 552
#  lancements, PPL +3,3 % sur 1 tranche) et reste opt-in.
_MLA_GLUE = int(os.environ.get("ACVRAM_MLA_GLUE", "2"))   # 0.6.32 : =2 servi (M3 2a-bis tenu 4/4, Manon 9ee0c00a) ; 1 = témoin
if _MLA_GLUE not in (0, 1, 2):
    raise ValueError(f"ACVRAM_MLA_GLUE={_MLA_GLUE!r} : 0 | 1 | 2")
# Cache latent des créneaux en fp8 E4M3 par ligne (sage-avis-exterieur-16-09
# § 6, commit 3 du chantier MLA) : une ligne = W codes + 16 octets (échelle
# fp32 s = amax/448) ; la moitié des octets lus par l'attention. Lu par
# mla_decode_1p seulement (les deux noyaux d'avant restent bf16). Réfuté si
# la PPL sort de ± 0,004 → bf16 gardé ; défaut 0 tant que ce n'est pas mesuré.
_MLA_LATENT_FP8 = os.environ.get("ACVRAM_MLA_LATENT_FP8", "0") == "1"
# Sonde (β) du niveau 2 (verdict-c15 addendum 05 h 08, Manon) : sous
# ACVRAM_MLA_QABS_DEUX_MOITIES=1 le chemin =1 (decode_static, au bit avec torch) calcule
# q_abs en deux moitiés de nope accumulées en fp32 puis arrondies une fois — un autre
# ordre de somme, ≤ 1 ulp bf16, sans noyau. Si la PPL 8 192 + 512 bouge de +1 à +3 %
# comme le niveau 2, c'est le modèle (GLM nvfp4) qui est instable à la marge, pas le
# noyau ; sinon le niveau 2 porte autre chose qu'un ordre de somme. Diagnostic seul.
_MLA_QABS_DEUX_MOITIES = os.environ.get("ACVRAM_MLA_QABS_DEUX_MOITIES", "0") == "1"
# C14-b (chantier-c14b-19-09, sage-fiches-c5b-c13c-c14b-20-09 § 3) : sous
# ACVRAM_MLA_BATCH_FUSION=1, le chemin par lot (decode_static_batch_complet)
# demande à mla_decode_1p de rendre y = v_b·o_lat en bf16 depuis son combine
# (mla_1p_combine_vb_kernel) : l'einsum 'hvr,bhr->bhv' fp32 — que cuBLAS sert
# à M=12 par gemmSN_TN, 11,5 µs de latence par couche — et la conversion bf16
# de sa sortie disparaissent. Même arithmétique (v_b bf16 → fp32 exact, produits
# et sommes fp32, un arrondi bf16 final) à l'ordre des sommes près : ± quelques
# ulp fp32, jamais au bit — juge ppl-decode-kv au lot de 12 ± 0,002 (REGLES
# § 4 bis). Pris seulement sous MLA_CORE_DECODE=fp32 (le noyau ne sait pas
# tf32/bf16) et sortie bf16. Défaut 0 tant que ce n'est pas mesuré.
_MLA_BATCH_FUSION = os.environ.get("ACVRAM_MLA_BATCH_FUSION", "0") == "1"
# C14-b, geste (3) (Sage 09 h 00 : variable séparée, jugée seule par le nsys (a)) : sous
# ACVRAM_MLA_PREP_GRILLE=1, mla_prep_batch prend la grille regrillée (tuile k_b en shared,
# groupes de 4 créneaux : 492 blocs à b=12, .cu mla_prep_batch_kernel) ; 0 = la grille
# d'avant (temoin=True, nh·NR + B blocs). Les deux sont AU BIT (prep_faux 0 sur 799
# couches-pas, verdict Manon 09 h 00) : la variable ne porte que le temps — défaut 1 après
# (a) ≤ 8 µs/couche ET pas b=12 non perdu. Indépendante de MLA_BATCH_FUSION (le combine).
_MLA_PREP_GRILLE = os.environ.get("ACVRAM_MLA_PREP_GRILLE", "1") == "1"   # DÉFAUT 1 (0.6.31) si M1 bis tient : (a) ≤ 8 µs/couche, pas b=12 non perdu
FP8_PAD = 16


def _fp8_quant_rows(x: torch.Tensor) -> torch.Tensor:
    """[n, W] bf16 -> [n, W+16] uint8 : codes E4M3(x/s) puis s fp32 en octets
    (même arithmétique que mla_ecrit_latent : amax par ligne, s = amax/448
    divisé par un TENSEUR — tensor/scalaire multiplie par l'inverse)."""
    n, W = x.shape
    xf = x.to(torch.float32)
    amax = xf.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / torch.full_like(amax, 448.0), torch.ones_like(amax))
    out = torch.zeros(n, W + FP8_PAD, dtype=torch.uint8, device=x.device)
    out[:, :W] = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    out[:, W:W + 4] = s.contiguous().view(torch.uint8).reshape(n, 4)
    return out


def _fp8_dequant_rows(c: torch.Tensor, W: int, dtype=torch.bfloat16) -> torch.Tensor:
    """[n, W+16] uint8 -> [n, W] : code × échelle de ligne."""
    s = c[:, W:W + 4].contiguous().view(torch.float32)              # [n, 1]
    return (c[:, :W].contiguous().view(torch.float8_e4m3fn).to(torch.float32) * s).to(dtype)


def _est_fp8(st: dict) -> bool:
    return st["cache"].dtype == torch.uint8


def _mla_decode(ext, q_eff, cache, len_t, scores, bucket, rank, scale):
    """Une séquence : q_eff [nh, W] fp32, cache [>= bucket, W] bf16 (ou fp8
    [>= bucket, W+16] uint8) -> o_lat [nh, rank]."""
    fp8 = cache.dtype == torch.uint8
    if (_MLA_UNE_PASSE or fp8) and hasattr(ext, "mla_decode_1p"):
        return ext.mla_decode_1p(q_eff.unsqueeze(0), None, cache, len_t.reshape(1),
                                 bucket, rank, scale, fp8)[0]
    return ext.mla_decode(q_eff, cache, len_t, scores, bucket, rank, scale)


def _mla_decode_batch(ext, q, cache_ptrs, lens, scores, bucket, rank, scale, fp8=False):
    """B créneaux : q [B, nh, W] fp32, table d'adresses [B] -> o_lat [B, nh, rank]."""
    if (_MLA_UNE_PASSE or fp8) and hasattr(ext, "mla_decode_1p"):
        return ext.mla_decode_1p(q, cache_ptrs, None, lens, bucket, rank, scale, fp8)
    return ext.mla_decode_batch(q, cache_ptrs, lens, scores, bucket, rank, scale)


def _fusion_vb_possible(ext, x: torch.Tensor, v_b: torch.Tensor, fp8: bool) -> bool:
    """C14-b : le combine fusionné (v_b dans mla_decode_1p) s'applique quand il est demandé
    (ACVRAM_MLA_BATCH_FUSION=1), que le noyau à une passe est le chemin pris, que v_b est bf16
    (le noyau le convertit exactement en fp32 : ce que lisait _v_b32), que la sortie est bf16
    et que le régime du cœur au décodage est fp32 (le noyau n'a ni tf32 ni bf16)."""
    return (_MLA_BATCH_FUSION and (_MLA_UNE_PASSE or fp8) and ext is not None
            and hasattr(ext, "mla_decode_1p") and v_b.dtype == torch.bfloat16
            and x.dtype == torch.bfloat16 and _regime_coeur(decode=True) == "fp32")


def _refuser_en_capture(quoi: str) -> None:
    """REGLES § 6 : un tenseur alloué pendant une capture de graphe vit dans le bassin du
    graphe et meurt avec la capture suivante — silencieusement. Tout cache paresseux du
    chemin MLA passe par ici ; `chauffer()` les matérialise avant `warm_graphs`."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"MLA : {quoi} alloué pendant une capture de graphe — "
                           "appeler MLAttention.chauffer() avant la capture (REGLES § 6)")


def godet_mla(longueur: int) -> int:
    """Palier de cache latent couvrant ``longueur``, en puissances de deux.

    L'attention MLA balaie tout le godet, pas seulement les positions écrites :
    à godet fixe de 1024, une séquence de 264 jetons payait quatre fois trop.
    Un godet fixe et fin coûterait en revanche un graphe par palier — 256
    paliers pour un contexte de 32 768, bien au-delà de ce qu'on capture. Des
    paliers doublants tiennent les deux bouts : une courte séquence ne lit que
    ce qu'il lui faut, et le contexte entier ne demande que neuf paliers.
    """
    n = MLA_BUCKET
    while n < longueur:
        n *= 2
    return n


def _extension():
    import os
    if os.environ.get("ACVRAM_HYBRID_KERNELS", "1") == "0":
        return None
    from .. import kernels
    ext = kernels.get_extension()
    return ext if ext is not None and hasattr(ext, "mla_decode") else None


def _flash_prefill(q_eff: torch.Tensor, cache: torch.Tensor, passe: int, scale: float,
                   rank: int):
    """C13-c forme 2 (`sage-c13c-forme1-faux-forme2-tf32-20-09` § 2) : sous `MLA_CORE=flash`, le cœur
    causal fusionné en **TF32** pour les requêtes qui voient ≤ `_MLA_CORE_MAX_CLES` clés (règle des
    2 048 clés : la requête i voit passe + i + 1 clés), le reste des requêtes reste au chemin fp32
    par morceaux. Rend (o_lat [t_flash, nh, rank] fp32, t_flash) — t_flash = 0 quand rien n'est
    éligible (passe ≥ 2 048) ; None quand le régime n'est pas `flash` ou après un repli nommé
    (`_FLASH_REPLI`, ligne de régime `mla_core=flash(repli fp32: …)`), jamais silencieux."""
    global _FLASH_REPLI
    if _MLA_CORE != "flash" or _FLASH_REPLI is not None:
        return None
    t = q_eff.shape[0]
    t_flash = max(0, min(t, _MLA_CORE_MAX_CLES - passe))
    if t_flash == 0:
        return torch.empty(0, q_eff.shape[1], rank, dtype=torch.float32, device=q_eff.device), 0
    try:
        from ..kernels import attn_mla_causal
        if not attn_mla_causal.disponible():
            raise RuntimeError("Triton absent ou carte sans noyau flash")
        o = attn_mla_causal.attention_mla_causale(q_eff[:t_flash].to(torch.float32).contiguous(),
                                                  cache[:passe + t_flash], passe, scale, rank,
                                                  operandes=_FLASH_OPERANDES)
        return o, t_flash
    except Exception as ex:                                   # noqa: BLE001
        _FLASH_REPLI = f"{type(ex).__name__}: {ex}"[:120]
        print(f"[mla] MLA_CORE=flash : noyau indisponible, repli fp32 par morceaux pour tout le préfill "
              f"({_FLASH_REPLI})", flush=True)
        return None


_FLASH_OPERANDES = os.environ.get("ACVRAM_MLA_FLASH_OPERANDES", "tf32")   # tf32 (forme 2) | fp32 (forme 1, faux sur sm_120) | tf32x3 (sonde)


class MLAttention(nn.Module):
    def __init__(self, q_proj: nn.Module, kv_a_proj: nn.Module,
                 o_proj: nn.Module,
                 kv_a_norm: torch.Tensor,          # [kv_lora_rank]
                 k_b: torch.Tensor,                # [heads, kv_lora_rank, qk_nope]
                 v_b: torch.Tensor,                # [heads, v_dim, kv_lora_rank]
                 num_heads: int, qk_nope: int, qk_rope: int,
                 kv_lora_rank: int, v_dim: int,
                 eps: float = 1e-6,
                 q_a_proj: Optional[nn.Module] = None,
                 q_a_norm: Optional[torch.Tensor] = None,
                 q_b_proj: Optional[nn.Module] = None,
                 rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.q_proj, self.kv_a_proj, self.o_proj = q_proj, kv_a_proj, o_proj
        # DeepSeek-V2/GLM : q en bas rang (q_b(norm(q_a(x)))) et RoPE sur la
        # partie pe de q et de k (Kimi : ni l'un ni l'autre)
        self.q_a_proj, self.q_b_proj = q_a_proj, q_b_proj

        self.q_a_norm = nn.Parameter(q_a_norm, requires_grad=False) if q_a_norm is not None else None
        self.rope_emb = rope
        self.kv_a_norm = nn.Parameter(kv_a_norm, requires_grad=False)
        self.k_b = nn.Parameter(k_b, requires_grad=False)
        self.v_b = nn.Parameter(v_b, requires_grad=False)
        self.nh = num_heads
        self.nope, self.rope = qk_nope, qk_rope
        self.rank, self.dv = kv_lora_rank, v_dim
        self.scale = (qk_nope + qk_rope) ** -0.5
        self.eps = eps
        self.q_kv = None            # q_proj et kv_a_proj empilées
        self.qa_kv = None           # q_a_proj et kv_a_proj empilées (bas rang)
        self.q_a_taille = 0


    # les trois projections sous la porte FP8-MLA (_MLA_A8) : les modules gardent
    # leur nom dans l'arbre (exil, chargeur), seul l'appel passe par _a8
    def _o(self, y):
        return self.o_proj(_a8(y))

    def _kv_a(self, x):
        return self.kv_a_proj(_a8(x))

    def _q_b(self, x):
        return self.q_b_proj(_a8(x))

    def _proj_entree(self, x: torch.Tensor):
        """Première projection de q et latent kv : une GEMV quand les deux
        lisent la même entrée, deux sinon.

        C'est le cas de toutes les variantes DeepSeek-V2 et GLM : ``q_a_proj``
        et ``kv_a_proj`` partent l'une comme l'autre de l'état caché.
        """
        if self.qa_kv is not None:
            g = self.qa_kv(x)
            return g[:, :self.q_a_taille], g[:, self.q_a_taille:]
        if self.q_kv is not None:
            g = self.q_kv(x)
            nq = self.nh * (self.nope + self.rope)
            return g[:, :nq], g[:, nq:]
        prem = self.q_proj(x) if self.q_a_proj is None else self.q_a_proj(x)
        return prem, self._kv_a(x)

    def _norme(self, t: torch.Tensor, poids: torch.Tensor) -> torch.Tensor:
        """RMSNorm : un lancement quand le noyau est là, sept sinon.

        La formulation PyTorch — conversion, carré, moyenne, racine inverse,
        deux multiplications, reconversion — coûte sept noyaux pour une poignée
        de milliers d'éléments. Sur 47 couches et deux normalisations par
        couche, c'est plus de la moitié des noyaux élémentaires d'un jeton.
        """
        ext = _extension() if t.is_cuda else None
        if os.environ.get("ACVRAM_MLA_NORME_NOYAU") == "0":   # témoin de mesure
            ext = None
        if (ext is not None and hasattr(ext, "rmsnorm_bf16")
                and t.dtype == torch.bfloat16 and poids.dtype == torch.bfloat16):
            return ext.rmsnorm_bf16(t, poids, self.eps)[0]
        t32 = t.to(torch.float32)
        return (t32 * torch.rsqrt(t32.pow(2).mean(-1, keepdim=True) + self.eps)
                ).to(t.dtype) * poids

    def _q_depuis(self, prem: torch.Tensor) -> torch.Tensor:
        """De la sortie de la première projection au q complet."""
        if self.q_a_proj is None:
            return prem
        return self._q_b(self._norme(prem, self.q_a_norm))

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return self._q_depuis(self._proj_entree(x)[0])

    def _rope(self, q_pe: torch.Tensor, k_pe: torch.Tensor,
              positions: torch.Tensor, max_pos: int):
        """RoPE (demi-rotation) sur les parties pe : q_pe [t, nh, r], k_pe [t, r]."""
        if self.rope_emb is None:
            return q_pe, k_pe
        # DeepSeek-V2/GLM : RoPE de type « norm » (paires adjacentes 2i, 2i+1),
        # pas la demi-rotation NEOX — llama.cpp le range hors LLAMA_ROPE_TYPE_NEOX
        if _MLA_GLUE and hasattr(self.rope_emb, "tables_demi"):
            # C15 : les demi-tables cos/sin empilées, UNE indexation par pas
            # au lieu de deux (mêmes valeurs bf16, découpées après coup)
            cs = self.rope_emb.tables_demi(positions, q_pe.device, q_pe.dtype, max_pos)
            half = cs.shape[-1]
            c = cs[:, 0].unsqueeze(1)                          # [t, 1, r/2]
            s = cs[:, 1].unsqueeze(1)
        else:
            cos, sin = self.rope_emb(positions, q_pe.device, q_pe.dtype, max_pos=max_pos)
            half = cos.shape[-1] // 2
            c = cos[..., :half].unsqueeze(1)                   # [t, 1, r/2]
            s = sin[..., :half].unsqueeze(1)

        def tourner(x: torch.Tensor) -> torch.Tensor:
            x2 = x.reshape(*x.shape[:-1], half, 2)
            x0, x1 = x2[..., 0], x2[..., 1]
            if _MLA_GLUE:
                # C15 : y0 et y1 écrits en place dans le tampon entrelacé —
                # les quatre produits et la somme/différence bf16 sont les
                # mêmes lancements, torch.stack (une copie) disparaît
                y2 = torch.empty_like(x2)
                torch.sub(x0 * c, x1 * s, out=y2[..., 0])
                torch.add(x0 * s, x1 * c, out=y2[..., 1])
                return y2.reshape(x.shape)
            y0 = x0 * c - x1 * s
            y1 = x0 * s + x1 * c
            return torch.stack((y0, y1), dim=-1).reshape(x.shape)
        # q et k tournent ensemble : la rotation est point à point et ne
        # dépend pas du nombre de têtes, si bien que les traiter d'un bloc
        # rend exactement les mêmes bits pour moitié moins de lancements.
        ensemble = tourner(torch.cat([q_pe, k_pe.unsqueeze(1)], dim=1))
        return ensemble[:, :-1], ensemble[:, -1]

    def forward(self, x: torch.Tensor,
                cache: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, cache latent)."""
        t = x.shape[0]
        prem, kvp = self._proj_entree(x)                      # kvp [t, rank+rope]
        q = self._q_depuis(prem).reshape(t, self.nh, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        if self.rope_emb is not None:
            passe0 = 0 if cache is None else cache.shape[0]
            pos = torch.arange(passe0, passe0 + t, device=x.device)
            q_pe, k_pe = self._rope(q_pe, k_pe, pos, passe0 + t + 1)
        c = self._norme(c, self.kv_a_norm)

        # q absorbé : [t, nh, rank] = k_b [nh, rank, nope] @ q_nope [t, nh, nope]
        q_abs = torch.einsum('hrn,thn->thr', self.k_b.to(x.dtype), q_nope)
        q_eff = torch.cat([q_abs, q_pe], dim=-1)              # [t, nh, rank+rope]
        k_new = torch.cat([c, k_pe], dim=-1)                  # [t, rank+rope]

        cache = k_new if cache is None else torch.cat([cache, k_new], dim=0)
        total = cache.shape[0]

        if t == 1:
            # Décodage : même formulation en godet que ``decode_static`` (le
            # chemin des graphes), pour que les deux arrondissent pareil —
            # une GEMM sur N colonnes n'accumule pas comme sur 1024.
            bucket = godet_mla(total)
            C = torch.zeros(bucket, cache.shape[1], dtype=cache.dtype,
                            device=cache.device)
            C[:total] = cache
            # Hors créneau (b > ACVRAM_HYBRID_SLOTS), ce chemin servait tout
            # le decodage en torch pur alors que `decode_static` a le meme
            # godet et emprunte deja le noyau fusionne — verifie equivalent
            # au bruit d'arrondi pres (test unitaire, total=1..5, len inclusif).
            ext = _extension() if x.is_cuda else None
            if os.environ.get("ACVRAM_MLA_EAGER_TORCH") == "1":
                ext = None
            if ext is not None:
                len_t = torch.tensor(total - 1, dtype=torch.long, device=x.device)
                scores_buf = torch.zeros(self.nh, bucket, dtype=torch.float32,
                                         device=x.device)
                o_lat = _mla_decode(ext, q_eff.to(torch.float32)[0].contiguous(),
                                    C, len_t, scores_buf, bucket,
                                    self.rank, self.scale)             # [nh, rank]
                if os.environ.get("ACVRAM_MLA_DEBUG_ECART"):
                    pos_dbg = torch.arange(bucket, device=x.device)
                    masque_dbg = pos_dbg > (total - 1)
                    sc_dbg = torch.einsum('thr,sr->ths', q_eff.to(torch.float32),
                                          C.to(torch.float32)) * self.scale
                    sc_dbg = sc_dbg.masked_fill(masque_dbg, float('-inf'))
                    probs_dbg = sc_dbg.softmax(dim=-1)
                    o_lat_torch = torch.einsum('ths,sr->thr', probs_dbg,
                                               C[:, :self.rank].to(torch.float32))[0]
                    ecart = (o_lat - o_lat_torch).abs()
                    print(f"[debug-ecart] total={total} bucket={bucket} "
                          f"abs_max={ecart.max().item():.4e} "
                          f"abs_med={ecart.median().item():.4e} "
                          f"o_lat_norm={o_lat.norm().item():.4e}", flush=True)
                with _tf32_coeur(decode=True):
                    y = torch.einsum('hvr,hr->hv', self.v_b.to(_dt_coeur(decode=True)), o_lat.to(_dt_coeur(decode=True)))
                y = y.reshape(t, self.nh * self.dv).to(x.dtype)
                return self._o(y), cache
            pos = torch.arange(bucket, device=x.device)
            masque = pos > (total - 1)
        else:
            # prefill : par tranches de requêtes, sinon les scores
            # [t, nh, total] fp32 pèsent des gigaoctets (2 Go à 4k jetons)
            passe = total - t
            flash = _flash_prefill(q_eff, cache, passe, self.scale, self.rank)   # C13-c : None hors MLA_CORE=flash / repli
            o_flash, t_flash = flash if flash is not None else (None, 0)
            pos_k = torch.arange(total, device=x.device)
            morceaux = [] if o_flash is None or t_flash == 0 else [o_flash]
            C_dt = {}                                  # cache converti par dtype, une fois
            for d0 in range(t_flash, t, 256):          # les requêtes au-delà de 2 048 clés vues : fp32 par morceaux
                d1 = min(t, d0 + 256)
                cles = passe + d1                      # clés VUES par ce morceau (causal) : la règle ≤ 2 048
                dt = _dt_coeur(cles=cles)
                if dt not in C_dt:
                    C_dt[dt] = cache.to(dt)
                C32 = C_dt[dt]; V32 = C32[:, :self.rank]
                with _tf32_coeur(cles=cles):
                    sc = torch.einsum('thr,sr->ths', q_eff[d0:d1].to(dt), C32).to(torch.float32) * self.scale
                pos_q = torch.arange(d0, d1, device=x.device).unsqueeze(-1) + passe
                sc = sc.masked_fill(pos_k > pos_q.unsqueeze(1), float('-inf'))
                with _tf32_coeur(cles=cles):
                    morceaux.append(torch.einsum('ths,sr->thr', sc.softmax(dim=-1).to(dt), V32))
            o_lat = torch.cat(morceaux) if len(morceaux) != 1 else morceaux[0]
            dtv = _dt_coeur(vb=True, cles=total)
            with _tf32_coeur(vb=True, cles=total):  # 3e produit du cœur au préfill (8ae21997, opt-in MLA_CORE_VB)
                y = torch.einsum('hvr,thr->thv', self.v_b.to(dtv), o_lat.to(dtv))
            y = y.reshape(t, self.nh * self.dv).to(x.dtype)
            return self._o(y), cache
        cles = int(total)                              # clés vues (causal jusqu'à total)
        dt = _dt_coeur(cles=cles)
        with _tf32_coeur(cles=cles):
            scores = torch.einsum('thr,sr->ths', q_eff.to(dt), C.to(dt)).to(torch.float32) * self.scale
        scores = scores.masked_fill(masque, float('-inf'))
        probs = scores.softmax(dim=-1)

        with _tf32_coeur(cles=cles):
            o_lat = torch.einsum('ths,sr->thr', probs.to(dt), C[:, :self.rank].to(dt))
        dtv = _dt_coeur(vb=True, cles=cles)
        with _tf32_coeur(vb=True, cles=cles):      # 3e produit (8ae21997, opt-in MLA_CORE_VB)
            y = torch.einsum('hvr,thr->thv', self.v_b.to(dtv), o_lat.to(dtv))
        y = y.reshape(t, self.nh * self.dv).to(x.dtype)
        return self._o(y), cache

    def fuse_projections(self) -> bool:
        """Une GEMV au lieu de deux à l'entrée de l'attention.

        Les modèles à q de bas rang (DeepSeek-V2, GLM-4.x) étaient exclus :
        seule la paire ``q_proj``/``kv_a_proj`` était traitée, et en INT8
        seulement. Or sur GLM-4.7-Flash, ``q_a_proj`` [768, 2048] et
        ``kv_a_proj`` [576, 2048] lisent toutes deux l'état caché — 47 couches
        qui lançaient chacune une GEMV de trop.
        """
        from .layers import stack_int8_linears, stack_nvfp4_linears

        def empiler(lins):
            return stack_int8_linears(lins) or stack_nvfp4_linears(lins)

        self.q_kv = self.qa_kv = None
        if self.q_a_proj is not None:
            self.qa_kv = empiler([self.q_a_proj, self.kv_a_proj])
            if self.qa_kv is not None:
                self.q_a_taille = int(self.q_a_proj.qweight.shape[0])
            self._marquer_etroit()
            return self.qa_kv is not None
        self.q_kv = empiler([self.q_proj, self.kv_a_proj])
        self._marquer_etroit()
        return self.q_kv is not None

    def _marquer_etroit(self) -> None:
        """Projections int8 du décodage (qa_kv / q_kv, q_b, o) sur le GEMM
        étroit à tensor cores (narrow_gemm, M ≤ 16) au lieu du GEMV int8 —
        sage-duel-verdict § 14 (i) : int8_gemv<4,12> coûtait 4,4 ms/pas à
        b=12 (31 µs par lancement pour ~3 Mo : borné par le calcul, pas par
        les octets), attendu −2,0 ms, même arithmétique à l'ordre des sommes
        près (arbitre ≥ 80/84). ACVRAM_NARROW_MLA=0 : témoin (GEMV)."""
        if os.environ.get("ACVRAM_NARROW_MLA", "1") != "1":
            return
        for w in (getattr(self.qa_kv, "qweight", None), getattr(self.q_kv, "qweight", None),
                  getattr(getattr(self, "q_b_proj", None), "qweight", None),
                  getattr(getattr(self, "o_proj", None), "qweight", None)):
            if w is not None and getattr(w, "format", "") == "int8":
                try:
                    w.etroit = True
                except (AttributeError, TypeError):
                    pass

    # -- chemin à formes fixes (graphes CUDA) --------------------------------
    def new_static(self, device: torch.device, max_len: int,
                   dtype: torch.dtype) -> dict:
        W = self.rank + self.rope
        cache = (torch.zeros(max_len, W + FP8_PAD, dtype=torch.uint8, device=device)
                 if _MLA_LATENT_FP8 and device.type == "cuda" else
                 torch.zeros(max_len, W, dtype=dtype, device=device))
        return {"cache": cache,
                "len": torch.zeros((), dtype=torch.long, device=device),
                # scores de travail du noyau fusionné [nh, max_len]
                "scores": torch.zeros(self.nh, max_len, dtype=torch.float32,
                                      device=device)}

    @staticmethod
    def static_load(st: dict, etat) -> None:
        if etat is None:
            st["len"].zero_()
            return
        n = etat.shape[0]
        if _est_fp8(st):
            st["cache"][:n].copy_(_fp8_quant_rows(etat))
        else:
            st["cache"][:n].copy_(etat)
        st["len"].fill_(n)

    @staticmethod
    def static_export(st: dict):
        n = int(st["len"].item())
        if _est_fp8(st):
            return _fp8_dequant_rows(st["cache"][:n], st["cache"].shape[1] - FP8_PAD)
        return st["cache"][:n].clone()

    @staticmethod
    def _ecrit_ligne(st: dict, k_new: torch.Tensor) -> None:
        """Une ligne [1, W] à la position ``len`` du cache du créneau (bf16 ou fp8)."""
        if _est_fp8(st):
            st["cache"].index_copy_(0, st["len"].view(1), _fp8_quant_rows(k_new))
        else:
            st["cache"].index_copy_(0, st["len"].view(1), k_new)

    def _prep_decode(self, x: torch.Tensor, st: dict, bucket: int):
        """Préparation d'un jeton pour un créneau : projections, RoPE, norme,
        écriture du latent à la ligne ``len``. Rend ``q_eff`` [1, nh, rank+rope].
        Partagée par ``decode_static`` et ``decode_static_batch`` — un seul
        texte, donc les deux chemins arrondissent pareil."""
        prem, kvp = self._proj_entree(x)
        q = self._q_depuis(prem).reshape(1, self.nh, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        if self.rope_emb is not None:
            q_pe, k_pe_r = self._rope(q_pe, k_pe, st["len"].view(1), bucket + 1)
            if _MLA_GLUE:
                # C15 : la cat [c, k_pe tourné] était redécoupée à la ligne
                # suivante — même tenseurs, une copie de moins par couche
                k_pe = k_pe_r
            else:
                kvp = torch.cat([c, k_pe_r], dim=-1)
                c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        c = self._norme(c, self.kv_a_norm)
        if _MLA_QABS_DEUX_MOITIES:
            m = self.nope // 2
            kb = self.k_b.to(torch.float32); qn = q_nope.to(torch.float32)
            q_abs = (torch.einsum('hrn,thn->thr', kb[..., :m], qn[..., :m])
                     + torch.einsum('hrn,thn->thr', kb[..., m:], qn[..., m:])).to(x.dtype)
        else:
            q_abs = torch.einsum('hrn,thn->thr', self.k_b.to(x.dtype), q_nope)
        q_eff = torch.cat([q_abs, q_pe], dim=-1)             # [1, nh, rank+rope]
        k_new = torch.cat([c, k_pe], dim=-1)                 # [1, rank+rope]
        self._ecrit_ligne(st, k_new)
        return q_eff

    def _sortie_decode(self, x: torch.Tensor, st: dict, o_lat: torch.Tensor) -> torch.Tensor:
        # C15 : v_b converti une fois (_v_b32) au lieu d'à chaque couche et
        # chaque pas — la conversion bf16 → fp32 est exacte, mêmes bits ;
        # C13 : le produit lit son régime (_dt_coeur), .to() inerte sous fp32/tf32
        v_b32 = self._v_b32() if _MLA_GLUE else self.v_b.to(torch.float32)
        with _tf32_coeur(decode=True):
            y = torch.einsum('hvr,hr->hv', v_b32.to(_dt_coeur(decode=True)), o_lat.to(_dt_coeur(decode=True)))
        st["len"].add_(1)
        return self._o(y.reshape(1, self.nh * self.dv).to(x.dtype))

    def decode_static_batch(self, x: torch.Tensor, sts: list, bucket: int,
                            cache_ptrs: torch.Tensor, scores_batch: torch.Tensor
                            ) -> torch.Tensor:
        """``decode_static`` pour B créneaux avec UN lancement du noyau
        d'attention (bead 6wa) ; préparation et sortie restent par créneau,
        donc la sortie est bit-identique à la boucle. ``cache_ptrs`` [B] int64
        = adresses des caches (stables, construites une fois par lot)."""
        ext = _extension()
        B = len(sts)
        q = torch.stack([self._prep_decode(x[i:i + 1], sts[i], bucket).to(torch.float32)[0]
                         for i in range(B)]).contiguous()          # [B, nh, W]
        lens = torch.stack([st["len"] for st in sts])
        o_lat = _mla_decode_batch(ext, q, cache_ptrs, lens, scores_batch, bucket,
                                  self.rank, self.scale, _est_fp8(sts[0]))   # [B, nh, rank]
        return torch.cat([self._sortie_decode(x[i:i + 1], sts[i], o_lat[i]) for i in range(B)], dim=0)

    def chauffer(self, device, max_pos: int) -> None:
        """Remède C15 niveau 2 (REGLES § 6) : matérialise hors capture tout ce que le chemin de
        décodage crée paresseusement — k_b contigu, v_b fp32, tables RoPE fp32 et demi-tables.
        Idempotent ; à appeler avant toute capture (graphs.py le fait avec `reserver`)."""
        if self.k_b.device.type == "cuda":
            self._k_b_c(); self._v_b32(); self._v_b_c()
        if self.rope_emb is not None:
            self.rope_emb.reserver(max_pos, device, self.k_b.dtype)

    def _temoin_prep(self, tenseurs: dict) -> None:
        """Copie `tenseurs` dans les tampons témoins de cette couche (alloués au premier
        appel, hors capture ; sous capture le clone est enregistré dans le graphe)."""
        t = self.__dict__.get("_temoin")
        if t is None:
            t = self.__dict__["_temoin"] = {"couche": len(_TEMOINS), "appels": 0}
            _TEMOINS.append(t)
        for nom, x in tenseurs.items():
            if nom not in t:
                _refuser_en_capture(f"témoin {nom}")
                t[nom] = torch.empty_like(x)
            t[nom].copy_(x)
        if "q" in tenseurs:
            t["appels"] += 1

    def _k_b_c(self) -> torch.Tensor:
        """``k_b`` [nh, rank, nope] contigu, une fois (le noyau de préparation le lit tel quel)."""
        kb = self.__dict__.get("_k_b_c_cache")
        if kb is None or kb.device != self.k_b.device:
            _refuser_en_capture("k_b contigu")
            kb = self.__dict__["_k_b_c_cache"] = self.k_b.detach().contiguous()
        return kb

    def _v_b32(self) -> torch.Tensor:
        """``v_b`` en fp32, converti une fois (la conversion à chaque appel
        coûtait un lancement et une passe mémoire par couche et par pas)."""
        vb = self.__dict__.get("_v_b32_cache")
        if vb is None or vb.device != self.v_b.device:
            _refuser_en_capture("v_b fp32")
            vb = self.__dict__["_v_b32_cache"] = self.v_b.detach().to(torch.float32).contiguous()
        return vb

    def _v_b_c(self) -> torch.Tensor:
        """``v_b`` [nh, dv, rank] bf16 contigu, une fois (C14-b : le combine fusionné le lit tel quel)."""
        vb = self.__dict__.get("_v_b_c_cache")
        if vb is None or vb.device != self.v_b.device:
            _refuser_en_capture("v_b contigu")
            vb = self.__dict__["_v_b_c_cache"] = self.v_b.detach().contiguous()
        return vb

    def decode_static_batch_complet(self, x: torch.Tensor, sts: list, bucket: int,
                                    cache_ptrs: torch.Tensor, scores_batch: torch.Tensor,
                                    len_ptrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Tout ``decode_static`` batché sur les B créneaux : projections, RoPE,
        norme, ``q_eff`` en une passe sur ``x`` [B, D] (comme ``forward_batch``),
        UN lancement d'attention (``mla_decode_batch``), einsum et ``o_proj``
        sur le lot. L'écriture du latent et l'avance des longueurs passent
        par ``mla_ecrit_latent`` (un lancement, tables d'adresses ``[B]`` des
        caches et des longueurs) quand ``len_ptrs`` est fourni, par créneau
        sinon. Numérique : celle de ``forward_batch`` (chemin
        eager), pas celle de la boucle — les GEMV à M=B et M=1 n'arrondissent
        pas forcément pareil ; le test d'équivalence le mesure."""
        ext = _extension()
        B = len(sts)
        lens = torch.stack([st["len"] for st in sts])                      # [B]
        prem, kvp = self._proj_entree(x)
        q = self._q_depuis(prem).reshape(B, self.nh, self.nope + self.rope)
        if (_MLA_PREP_NOYAU and hasattr(ext, "mla_prep_batch") and q.dtype == torch.bfloat16
                and kvp.dtype == torch.bfloat16 and self.k_b.dtype == torch.bfloat16
                and self.kv_a_norm.dtype == torch.bfloat16):
            if self.rope_emb is not None:
                cos32, sin32 = self.rope_emb.tables32(bucket + 1, x.device)
                # le noyau lit cos32[pos·rope + j] sans garde et pos < bucket : la table
                # doit couvrir le godet (Manon 03:36 : une lecture hors table ne lève rien)
                if cos32.shape[0] <= bucket:
                    raise RuntimeError(f"tables RoPE fp32 de {cos32.shape[0]} lignes pour un godet de {bucket}")
            else:
                cos32 = sin32 = None
            q_c, kvp_c = q.contiguous(), kvp.contiguous()
            q_eff, k_new = ext.mla_prep_batch(q_c, kvp_c, lens, cos32, sin32,
                                              self._k_b_c(), self.kv_a_norm, self.nope, self.rope,
                                              self.rank, self.eps,
                                              not _MLA_PREP_GRILLE)          # fp32 [B, nh, W], bf16 [B, W] ; temoin = grille d'avant
            if _MLA_PREP_TEMOIN:
                self._temoin_prep({"q": q_c, "kvp": kvp_c, "lens": lens, "q_eff": q_eff, "k_new": k_new})
        else:
            q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
            if self.rope_emb is not None:
                c0, k_pe0 = kvp.split([self.rank, self.rope], dim=-1)
                q_pe, k_pe0 = self._rope(q_pe, k_pe0, lens, bucket + 1)
                kvp = torch.cat([c0, k_pe0], dim=-1)
            c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
            c = self._norme(c, self.kv_a_norm)
            q_abs = torch.einsum('hrn,bhn->bhr', self.k_b.to(x.dtype), q_nope)
            q_eff = torch.cat([q_abs, q_pe], dim=-1).to(torch.float32)       # [B, nh, W]
            k_new = torch.cat([c, k_pe], dim=-1).contiguous()              # [B, W]
        fp8 = _est_fp8(sts[0])
        un_lancement = len_ptrs is not None and hasattr(ext, "mla_ecrit_latent")
        if un_lancement:
            ext.mla_ecrit_latent(k_new, cache_ptrs, len_ptrs, fp8)         # cache_b[len_b] = k_new[b] ; len_b += 1
        else:
            for i, st in enumerate(sts):
                self._ecrit_ligne(st, k_new[i:i + 1])
        if _MLA_PREP_TEMOIN and "_temoin" in self.__dict__:
            self._temoin_prep({"q_eff_avant_attn": q_eff, "lens_avant_attn": lens})
        if _fusion_vb_possible(ext, x, self.v_b, fp8):
            # C14-b : attention + combine + v_b·o_lat en deux lancements, y bf16 [B, nh, dv]
            y = ext.mla_decode_1p(q_eff.contiguous(), cache_ptrs, None, lens, bucket, self.rank,
                                  self.scale, fp8, self._v_b_c())
            if not un_lancement:
                for st in sts:
                    st["len"].add_(1)
            return self._o(y.reshape(B, self.nh * self.dv))
        o_lat = _mla_decode_batch(ext, q_eff.contiguous(), cache_ptrs, lens,
                                  scores_batch, bucket, self.rank, self.scale, fp8)  # [B, nh, rank]
        if _MLA_PREP_TEMOIN and "_temoin" in self.__dict__:
            self._temoin_prep({"o_lat": o_lat})
        with _tf32_coeur(decode=True):
            y = torch.einsum('hvr,bhr->bhv', self._v_b32().to(_dt_coeur(decode=True)), o_lat.to(_dt_coeur(decode=True)))
        if not un_lancement:
            for st in sts:
                st["len"].add_(1)
        return self._o(y.reshape(B, self.nh * self.dv).to(x.dtype))

    def decode_static(self, x: torch.Tensor, st: dict, bucket: int
                      ) -> torch.Tensor:
        """Un jeton, une séquence ; attention sur ``cache[:bucket]`` masquée
        au-delà de ``len`` ; écrit le latent à la ligne ``len`` puis avance."""
        # mêmes formulations que ``forward`` (t = 1), pour arrondir pareil
        q_eff = self._prep_decode(x, st, bucket)
        cache = st["cache"]
        ext = _extension() if x.is_cuda else None
        if ext is not None:
            o_lat = _mla_decode(ext, q_eff.to(torch.float32)[0].contiguous(),
                                cache, st["len"], st["scores"], bucket,
                                self.rank, self.scale)           # [nh, rank]
            return self._sortie_decode(x, st, o_lat)
        C = _fp8_dequant_rows(cache[:bucket], self.rank + self.rope) if _est_fp8(st) else cache[:bucket]
        scores = torch.einsum('thr,sr->ths', q_eff.to(torch.float32),
                              C.to(torch.float32)) * self.scale
        pos = torch.arange(bucket, device=x.device)
        scores = scores.masked_fill(pos > st["len"], float('-inf'))
        probs = scores.softmax(dim=-1)
        o_lat = torch.einsum('ths,sr->thr', probs,
                             C[:, :self.rank].to(torch.float32))
        # même geste C15 que _sortie_decode : le repli à sec le prouve
        v_b32 = self._v_b32() if _MLA_GLUE else self.v_b.to(torch.float32)
        y = torch.einsum('hvr,thr->thv', v_b32, o_lat)
        st["len"].add_(1)
        return self._o(y.reshape(1, self.nh * self.dv).to(x.dtype))

    # -- chemin eager, lot de séquences hors créneau -------------------------
    def peut_batcher_decode(self, x: torch.Tensor) -> bool:
        """Vrai si le noyau fusionné est disponible pour batcher le décodage
        eager : seul chemin qui vaille la peine d'être batché, le repli torch
        de ``forward`` reste par séquence."""
        if os.environ.get("ACVRAM_MLA_BATCH") == "0":
            return False
        return x.is_cuda and _extension() is not None

    def forward_batch(self, x: torch.Tensor, caches: list[Optional[torch.Tensor]]
                      ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Décodage MLA d'un lot de ``b`` séquences hors créneau : projections,
        RoPE et normes en une seule passe sur le lot (elles ne dépendent que de
        ``x``), une boucle uniquement pour ``ext.mla_decode`` — seule étape qui
        a besoin du cache, de longueur différente par séquence. Les caches
        restent une liste plutôt qu'un tenseur empilé : le jour d'un cache
        latent paginé, seule cette boucle change, pas la signature."""
        b = x.shape[0]
        prem, kvp = self._proj_entree(x)                      # kvp [b, rank+rope]
        q = self._q_depuis(prem).reshape(b, self.nh, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        if self.rope_emb is not None:
            passe0 = torch.tensor([0 if ca is None else ca.shape[0] for ca in caches],
                                  device=x.device)
            max_pos = int(passe0.max().item()) + 2
            q_pe, k_pe = self._rope(q_pe, k_pe, passe0, max_pos)
        c = self._norme(c, self.kv_a_norm)

        q_abs = torch.einsum('hrn,bhn->bhr', self.k_b.to(x.dtype), q_nope)
        q_eff = torch.cat([q_abs, q_pe], dim=-1)              # [b, nh, rank+rope]
        k_new = torch.cat([c, k_pe], dim=-1)                  # [b, rank+rope]

        ext = _extension()
        o_lats = []
        nouvelles_caches = []
        for i in range(b):
            cache_i = (k_new[i:i + 1] if caches[i] is None
                      else torch.cat([caches[i], k_new[i:i + 1]], dim=0))
            total = cache_i.shape[0]
            bucket = godet_mla(total)
            C = torch.zeros(bucket, cache_i.shape[1], dtype=cache_i.dtype,
                            device=cache_i.device)
            C[:total] = cache_i
            len_t = torch.tensor(total - 1, dtype=torch.long, device=x.device)
            scores_buf = torch.zeros(self.nh, bucket, dtype=torch.float32,
                                     device=x.device)
            # même noyau (une passe) et même godet que les créneaux : le chemin
            # hors créneau et decode_static_batch_complet arrondissent pareil
            o_lat = _mla_decode(ext, q_eff[i].to(torch.float32).contiguous(),
                                C, len_t, scores_buf, bucket,
                                self.rank, self.scale)         # [nh, rank]
            o_lats.append(o_lat)
            nouvelles_caches.append(cache_i)

        o_lat_batch = torch.stack(o_lats, dim=0)              # [b, nh, rank]
        with _tf32_coeur(decode=True):
            y = torch.einsum('hvr,bhr->bhv', self.v_b.to(_dt_coeur(decode=True)), o_lat_batch.to(_dt_coeur(decode=True)))
        y = y.reshape(b, self.nh * self.dv).to(x.dtype)
        return self._o(y), nouvelles_caches
