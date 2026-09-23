"""C13-c forme 1 (chantier-c13c-19-09 § Correction Sage) : le cœur d'attention MLA
causal fusionné en Triton (kernels/attn_mla_causal.py), à sec sous
``TRITON_INTERPRET=1``, contre le chemin fp32 actuel de `mla.py` (les einsum
chunkés, MLA_CORE=fp32) : par LIGNE de sortie, max|Δ| ≤ 8 ulp fp32 de
l'amplitude de la ligne (le juge de C14). Formes réduites (t 128-256, nh 4,
rank 64, rope 16, passe ∈ {0, 100}) ; le cache latent en bf16 comme au service.

Témoins qui doivent casser (REGLES § 7) : masque décalé d'une position, passe
faux, tuile de clés non diagonale sautée. Plus : le noyau est en fp32 plein
(`input_precision="ieee"` ≡ `allow_tf32=False`, jamais tf32 implicite), la forme 2
est refusée, `flash` est dans la table des régimes et la ligne de régime le nomme.

Ce que ce fichier ne prouve PAS : tout temps ; l'efficacité de `tl.dot` fp32
sur sm_120 ; l'interpréteur ne dit rien des registres (acc 8 × [64, 64] fp32).
Un juge Triton vert sous l'interpréteur n'est pas une preuve du noyau (REGLES
§ 7) : la chaîne carte (scratchpad/c13c-carte-19-09/chaine.sh) le rejoue aux
formes réelles.
"""
import importlib
import os
import pathlib
import re

import pytest
import torch
import torch.nn as nn

from acvram.engine import mla as M_mla

RACINE = pathlib.Path(__file__).resolve().parent.parent
ULP = 2 ** -23
JUGE_ULP = 8


def _noyau():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    k = importlib.import_module("acvram.kernels.attn_mla_causal")
    if not k.disponible():
        pytest.skip("Triton indisponible")
    return k


def _ulp_lignes(o: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """par ligne : |Δ| / (amplitude de la ligne de référence × ulp fp32), max sur la ligne."""
    o2 = o.reshape(-1, o.shape[-1]).double()
    r2 = ref.reshape(-1, ref.shape[-1]).double()
    amp = r2.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return ((o2 - r2).abs() / (amp * ULP)).amax(1)


def _ulp_ligne(o: torch.Tensor, ref: torch.Tensor) -> float:
    return float(_ulp_lignes(o, ref).amax())


def _fp64(q, C, passe, scale, rank, decalage=0):
    """L'arbitre : la même attention causale en float64, par morceaux de 256 lignes."""
    t, total = q.shape[0], C.shape[0]
    C64 = C.double()
    pos_k = torch.arange(total, device=C.device)
    out = []
    for d0 in range(0, t, 256):
        d1 = min(t, d0 + 256)
        sc = torch.einsum('thr,sr->ths', q[d0:d1].double(), C64) * scale
        pos_q = torch.arange(d0, d1, device=C.device).unsqueeze(-1) + passe + decalage
        sc = sc.masked_fill(pos_k > pos_q.unsqueeze(1), float('-inf'))
        out.append(torch.einsum('ths,sr->thr', sc.softmax(dim=-1), C64[:, :rank]))
    return torch.cat(out)


JUGE_EXCES_LIGNE = 6.0        # ulp fp32 (Sage § 1 bis) : 2 × le témoin à sec (excès max 3,0 sur 15/512 lignes), publié à côté du seuil


def _juge(o, ref, v64):
    """Le juge (Sage, `sage-c13c-juge-fp64-ordre-fin-nuit-20-09` § 1, précisé par la mesure à sec) :
    la référence fp32 s'écarte ELLE-MÊME du float64 (jusqu'à 7,4 ulp : sommes de 256 clés en un
    autre ordre) — « ± 8 ulp contre fp32 » se prend à la référence (REGLES § 7). Distance au
    float64 PAR LIGNE ; mais « ≤ d(référence) + 1 ulp par ligne » est réfuté à sec par deux ordres
    de somme légitimes : 3 % des lignes ont le noyau plus loin que la référence de 1,3-3,0 ulp
    (t=128 : 15/512, excès max 3,0 ; t=256 : 3/1024, 1,3) alors qu'il est PLUS PROCHE en médiane
    (1,40 contre 1,64 ; 1,53 contre 2,44) et au max (3,9 contre 7,4). Juge retenu : par ligne
    d(noyau, fp64) ≤ d(référence, fp64) + 6 ulp (Sage § 1 bis : 2 × le témoin), ET en distribution médiane(noyau) ≤ médiane(réf)
    ET max(noyau) ≤ max(réf) + 1 — le noyau n'est jamais pire que le chemin qu'il remplace.
    Rend (tenu, d_ref max, d_fp64 max, lignes en excès > 4 ulp)."""
    d_ref, d_n64, d_r64 = _ulp_lignes(o, ref), _ulp_lignes(o, v64), _ulp_lignes(ref, v64)
    pires = d_n64 > d_r64 + JUGE_EXCES_LIGNE
    tenu = bool((~pires).all()) and float(d_n64.median()) <= float(d_r64.median()) \
        and float(d_n64.amax()) <= float(d_r64.amax()) + 1.0
    print(f"[juge] seuil +{JUGE_EXCES_LIGNE:.0f} ulp par ligne ; témoin excès max noyau−réf = {float((d_n64 - d_r64).amax()):.2f} ulp")
    return tenu, float(d_ref.amax()), float(d_n64.amax()), int(pires.sum())


def _montage(t, passe, nh=4, rank=64, rope=16, graine=0, cdt=torch.bfloat16):
    torch.manual_seed(graine)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = rank + rope
    q = (torch.randn(t, nh, W, device=dev) * 0.3)
    C = (torch.randn(t + passe, W, device=dev) * 0.5).to(cdt)
    return q, C, W ** -0.5


def _reference(q, C, passe, scale, rank, decalage=0, tuile_sautee=None, BN=64):
    """Le chemin einsum fp32 de mla.py (morceaux de 256, masque `pos_k > pos_q`),
    avec deux fautes injectables : masque décalé de `decalage` positions ; tuile
    de clés [k0, k0+BN) retirée (`tuile_sautee` = k0)."""
    t, total = q.shape[0], C.shape[0]
    C32 = C.float()
    pos_k = torch.arange(total, device=C.device)
    out = []
    for d0 in range(0, t, 256):
        d1 = min(t, d0 + 256)
        sc = torch.einsum('thr,sr->ths', q[d0:d1].float(), C32) * scale
        pos_q = torch.arange(d0, d1, device=C.device).unsqueeze(-1) + passe + decalage
        masque = pos_k > pos_q.unsqueeze(1)
        if tuile_sautee is not None:
            masque = masque | ((pos_k >= tuile_sautee) & (pos_k < tuile_sautee + BN))
        sc = sc.masked_fill(masque, float('-inf'))
        out.append(torch.einsum('ths,sr->thr', sc.softmax(dim=-1), C32[:, :rank]))
    return torch.cat(out)


XFAIL_CARTE = pytest.mark.xfail(
    torch.cuda.is_available(), strict=True,
    reason="sage-t4-tri-69-20-09 (20/09) : C13-c flash causal opt-in CLOS ; juge de 8 ulp calibré à sec "
           "(interpréteur, excès max 3,0), sur carte 6,5-9,5 ulp par ligne — attendu rouge tant que le noyau est clos")

@pytest.mark.parametrize("t,passe", [(128, 0), (128, 100), (256, 0), (256, 100), (200, 100)])
@XFAIL_CARTE
def test_flash_causal_egal_fp32_a_8_ulp_par_ligne(t, passe):
    """Sortie = chemin fp32 actuel ± 8 ulp de l'amplitude par ligne, passe 0 et 100,
    t multiple de la tuile (128, 256) et non (200 : lignes hors t masquées au store)."""
    k = _noyau()
    q, C, scale = _montage(t, passe, graine=t + passe)
    o = k.attention_mla_causale(q, C, passe, scale, rank=64)
    ref = _reference(q, C, passe, scale, 64)
    assert o.shape == (t, 4, 64) and o.dtype is torch.float32 and not o.isnan().any()
    tenu, d_ref, d_64, n_arb = _juge(o, ref, _fp64(q, C, passe, scale, 64))
    assert tenu, f"t={t} passe={passe} : {d_ref:.2f} ulp contre fp32, {d_64:.2f} contre fp64, {n_arb} lignes plus loin que la référence de > 6 ulp (témoin à sec : excès max 3,0)"
    # la référence du module est bien ce chemin (c'est elle que la chaîne carte rejoue)
    assert torch.equal(k.reference_fp32(q, C, passe, scale, 64), ref)


@XFAIL_CARTE
def test_temoins_cassants_masque_passe_tuile():
    """Trois fautes que le juge DOIT voir : (1) masque décalé d'UNE position (une
    clé de plus ou de moins par ligne) ; (2) passe faux d'une unité (le noyau lancé
    tel quel, en contournant la garde du lanceur) ; (3) une tuile de clés NON
    diagonale retirée. Chacune sort de 8 ulp ; la garde du lanceur refuse un passe
    qui ne vaut pas total − t."""
    k = _noyau()
    t, passe = 256, 100
    q, C, scale = _montage(t, passe, graine=3)
    o = k.attention_mla_causale(q, C, passe, scale, rank=64)
    ref = _reference(q, C, passe, scale, 64)
    assert _juge(o, ref, _fp64(q, C, passe, scale, 64))[0]
    # (1) masque décalé : +1 (une clé future visible) et −1 (la clé diagonale perdue)
    for dec in (+1, -1):
        d = _ulp_ligne(o, _reference(q, C, passe, scale, 64, decalage=dec))
        assert d > 100 * JUGE_ULP, f"décalage {dec:+d} : {d:.1f} ulp, le juge ne voit pas le masque"
    # (2) passe faux : le noyau lui-même, lancé avec passe − 1 (garde contournée)
    o_faux = torch.empty_like(o)
    W = q.shape[-1]
    import triton
    k._flash_causal_kernel[(triton.cdiv(t, k.BM_DEFAUT), 4)](
        q, C, o_faux, t, C.shape[0], passe - 1, scale * k.LOG2E,
        q.stride(0), q.stride(1), C.stride(0), o_faux.stride(0), o_faux.stride(1),
        W=W, BM=k.BM_DEFAUT, BN=k.BN_DEFAUT, BK=k.BK, W_TUILES=triton.cdiv(W, k.BK),
        RANK_TUILES=1, PRECISION="ieee", num_warps=8, num_stages=2)
    d = _ulp_ligne(o_faux, ref)
    assert d > 100 * JUGE_ULP, f"passe − 1 : {d:.1f} ulp"
    assert _juge(o_faux, _reference(q, C, passe, scale, 64, decalage=-1),
                 _fp64(q, C, passe, scale, 64, decalage=-1))[0]                                  # = masque décalé de −1
    with pytest.raises(AssertionError):
        k.attention_mla_causale(q, C, passe + 1, scale, rank=64)
    # (3) tuile pleine [64, 128) sautée : visible par toutes les lignes (passe = 100 ≥ 128 − 1 ? non :
    # les lignes 0..27 la voient en partie) — on prend [0, 64), vue entièrement par toute ligne
    d = _ulp_ligne(o, _reference(q, C, passe, scale, 64, tuile_sautee=0))
    assert d > 100 * JUGE_ULP, f"tuile [0, 64) sautée : {d:.1f} ulp"


@XFAIL_CARTE
def test_cache_fp32_et_amplitude_des_scores():
    """Cache latent fp32 (même valeurs que bf16 converti) : même juge. Et la limite
    du juge, mesurée contre un arbitre float64 : à |scores| ≈ 5 le noyau et la
    référence fp32 sont chacun à < 8 ulp du fp64 ; à |scores| ≈ 19 la référence
    fp32 s'écarte ELLE-MÊME de > 8 ulp du fp64 (rounding de exp sur des exposants
    de 19 : |x|·ε) — le seuil de 8 ulp ne juge donc que des scores modérés, et
    sur carte le juge à trois bras s'impose (REGLES § 7 : vérifier un seuil contre
    la référence elle-même). Contrôle : le noyau n'est jamais plus loin du fp64
    que 1,25 × la référence + 1 ulp."""
    k = _noyau()
    t, passe, rank = 200, 100, 64
    q, C, scale = _montage(t, passe, graine=5, cdt=torch.float32)
    o = k.attention_mla_causale(q, C, passe, scale, rank)
    assert _juge(o, _reference(q, C, passe, scale, rank), _fp64(q, C, passe, scale, rank))[0]

    for ech, borne_ref in ((1.0, JUGE_ULP), (2.0, None)):
        q2, C2 = q * ech / 0.3, (C * ech / 0.5).float()
        o2 = k.attention_mla_causale(q2, C2, passe, scale, rank)
        ref32 = _reference(q2, C2, passe, scale, rank)
        v = _fp64(q2, C2, passe, scale, rank)
        d_noyau, d_ref = _ulp_ligne(o2, v), _ulp_ligne(ref32, v)
        assert d_noyau <= 1.25 * d_ref + 1, (ech, d_noyau, d_ref)
        if borne_ref is not None:
            assert d_ref <= borne_ref and d_noyau <= borne_ref, (ech, d_noyau, d_ref)
        else:
            assert d_ref > JUGE_ULP, (ech, d_ref)          # la référence fp32 elle-même sort du seuil


def test_forme_1_fp32_plein_et_forme_2_reservee():
    """Le .py ne fait jamais de TF32 : les `tl.dot` portent `input_precision=PRECISION`
    et la forme 1 vaut "ieee" (≡ `allow_tf32=False`, mot-clé déprécié en Triton 3.8 qui,
    implicite, cède à TRITON_F32_DEFAULT) ; aucun `allow_tf32=True`, aucun "tf32"
    littéral hors la liste de la forme 2 ; `operandes="tf32"|"bf16"` lève
    NotImplementedError (signature prête, pas de chemin silencieux)."""
    k = _noyau()
    src = (RACINE / "acvram" / "kernels" / "attn_mla_causal.py").read_text()
    assert k._PRECISION["fp32"] == "ieee" and k._PRECISION.get("tf32") == "tf32"     # forme 2 : tf32 (Manon f2dec8cf : 0,92 x cuBLAS tf32)
    dots = re.findall(r"tl\.dot\(([^\n]*)\)", src)
    assert len(dots) == 9 and all("input_precision=PRECISION" in d for d in dots), dots
    assert "allow_tf32=True" not in src                                    # la précision passe par input_precision, jamais par le drapeau global
    assert '_FORME2 = ("bf16",)' in src                                   # forme 2 = tf32 ecrite ; bf16 (F) non
    q, C, scale = _montage(128, 0)
    for op in ("bf16",):                                               # tf32 = forme 2 écrite
        with pytest.raises(NotImplementedError):
            k.attention_mla_causale(q, C, 0, scale, 64, operandes=op)
    with pytest.raises(ValueError):
        k.attention_mla_causale(q, C, 0, scale, 64, operandes="fp16")


def test_regime_flash_dans_la_table_et_la_ligne(monkeypatch):
    """`ACVRAM_MLA_CORE=flash` : 4e valeur acceptée par mla.py (décodage : non, préfill
    seul), notée dans regime.VARIABLES, ligne de régime `mla_core=flash(fp32)` ; le repli
    est nommé `flash(repli fp32: …)` ; sous flash la règle des 2 048 clés ne réduit rien
    (fp32 plein) et le 3e produit / le décodage restent gouvernés par leurs variables."""
    from acvram import regime
    src = (RACINE / "acvram" / "engine" / "mla.py").read_text()
    assert '("fp32", "tf32", "bf16", "flash")' in src and '_MLA_CORE_DECODE not in ("fp32", "tf32", "bf16")' in src
    assert "flash" in {v.nom: v for v in regime.VARIABLES}["MLA_CORE"].note
    monkeypatch.setattr(M_mla, "_MLA_CORE", "flash")
    monkeypatch.setattr(M_mla, "_FLASH_REPLI", None)
    monkeypatch.setattr(M_mla, "_MLA_CORE_VB", False)
    monkeypatch.setattr(M_mla, "_MLA_CORE_DECODE", "fp32")
    assert M_mla.regime_coeur_texte().startswith("mla_core=flash(")     # la tuile est nommée derrière
    assert "mla_core=flash(" in regime.regime_ligne()
    assert M_mla._regime_coeur(cles=8192) == "flash" and M_mla._dt_coeur(cles=8192) is torch.float32
    assert M_mla._regime_coeur(vb=True, cles=2047) == "fp32" and M_mla._regime_coeur(decode=True) == "fp32"
    with M_mla._tf32_coeur(cles=256) as c:
        assert not c._actif
    monkeypatch.setattr(M_mla, "_FLASH_REPLI", "RuntimeError: Triton absent")
    assert regime.regime_ligne().count("mla_core=flash(repli fp32: RuntimeError: Triton absent)") == 1
    from acvram.engine import runner
    assert runner._mla_core_texte().startswith(" mla_core=flash(repli fp32: RuntimeError: Triton absent)")   # puis mla_prep=… (C14-b)
    monkeypatch.setattr(M_mla, "_MLA_CORE", "fp32")
    assert M_mla.regime_coeur_texte() == "" and "mla_core=" not in regime.regime_ligne()


@XFAIL_CARTE
def test_mla_forward_prefill_flash_egal_fp32(monkeypatch):
    """Le branchement dans `MLAttention.forward` : préfill jouet (t = 200, cache de
    100 jetons, nh 4, rank 64, rope 16, bf16 comme au service) sous MLA_CORE=flash
    contre MLA_CORE=fp32 — le noyau est bien appelé (une fois par forward), o_lat
    identique à 8 ulp par ligne, y de sortie identique à 1 ulp bf16 par ligne ; le
    cache rendu est le même au bit (le noyau ne l'écrit pas). Repli : sans noyau,
    la sortie est celle du fp32 et le régime le dit."""
    _noyau()
    from acvram.engine.layers import RotaryEmbedding
    H, NH, NOPE, ROPE, RANK, DV, QA = 64, 4, 32, 16, 64, 32, 48
    DT = torch.bfloat16
    torch.manual_seed(11)
    lin = lambda o, i: nn.Linear(i, o, bias=False).to(DT)                       # noqa: E731
    mod = M_mla.MLAttention(lin(NH * (NOPE + ROPE), H), lin(RANK + ROPE, H), lin(H, NH * DV),
                            (torch.rand(RANK) + 0.5).to(DT), (torch.randn(NH, RANK, NOPE) * 0.1).to(DT),
                            (torch.randn(NH, DV, RANK) * 0.1).to(DT), NH, NOPE, ROPE, RANK, DV, eps=1e-5,
                            q_a_proj=lin(QA, H), q_a_norm=(torch.rand(QA) + 0.5).to(DT),
                            q_b_proj=lin(NH * (NOPE + ROPE), QA), rope=RotaryEmbedding(ROPE, 4096, dtype=DT))
    x = (torch.randn(200, H) * 0.5).to(DT)
    x0 = (torch.randn(100, H) * 0.5).to(DT)
    monkeypatch.setattr(M_mla, "_MLA_CORE_VB", False)
    monkeypatch.setattr(M_mla, "_MLA_CORE_DECODE", "fp32")
    monkeypatch.setattr(M_mla, "_FLASH_REPLI", None)

    o_lats = {}
    vrai = M_mla._flash_prefill

    def espion(q_eff, cache, passe, scale, rank):
        o = vrai(q_eff, cache, passe, scale, rank)
        o_lats.setdefault(M_mla._MLA_CORE, []).append((o, q_eff.clone(), cache.clone(), passe))
        return o
    monkeypatch.setattr(M_mla, "_flash_prefill", espion)

    sorties = {}
    torch.set_grad_enabled(False)
    for regime in ("fp32", "flash"):
        monkeypatch.setattr(M_mla, "_MLA_CORE", regime)
        _, cache0 = mod.forward(x0, None)
        y, cache = mod.forward(x, cache0)
        sorties[regime] = (y, cache)
    assert torch.equal(sorties["fp32"][1], sorties["flash"][1])                  # cache au bit
    assert [o is None for o, *_ in o_lats["fp32"]] == [True, True]              # fp32 : jamais le noyau
    assert [o is not None for o, *_ in o_lats["flash"]] == [True, True] and M_mla._FLASH_REPLI is None
    o, q_eff, cache, passe = o_lats["flash"][1]
    o = o[0] if isinstance(o, tuple) else o                                # forme 2 : (o_lat[:t_flash], t_flash)
    assert passe == 100 and o.shape == (200, NH, RANK)
    assert _juge(o, _reference(q_eff.float(), cache, passe, mod.scale, RANK), _fp64(q_eff.float(), cache, passe, mod.scale, RANK))[0]
    y32, y16 = sorties["fp32"][0].float(), sorties["flash"][0].float()
    amp = y32.abs().amax(1, keepdim=True).clamp_min(1e-6)
    assert float(((y16 - y32).abs() / (amp * 2 ** -8)).amax()) <= 1.0            # 1 ulp bf16 par ligne
    # repli nommé : le noyau « indisponible » → sortie fp32, régime `flash(repli fp32: …)`
    monkeypatch.setattr(M_mla, "_flash_prefill", vrai)
    monkeypatch.setattr(M_mla, "_FLASH_REPLI", None)
    import acvram.kernels.attn_mla_causal as K
    monkeypatch.setattr(K, "disponible", lambda: False)
    y_repli, _ = mod.forward(x, cache0)
    torch.set_grad_enabled(True)
    assert torch.equal(y_repli, sorties["fp32"][0]) and M_mla._FLASH_REPLI.startswith("RuntimeError: Triton absent")
    assert M_mla.regime_coeur_texte().startswith("mla_core=flash(repli fp32: RuntimeError")


def test_tuile_choisie_par_la_shared_de_la_carte(monkeypatch):
    """Manon 20/09 : 64-64-w8-s2 demande 319 488 o de shared contre 101 376 sur sm_120 (OutOfResources) ;
    la tuile se choisit par `shared_memory_per_block_optin`, jamais par un défaut aveugle."""
    from acvram.kernels import attn_mla_causal as k
    k._CHOIX.clear()
    monkeypatch.delenv("ACVRAM_MLA_FLASH_TUILE", raising=False)
    class P:
        def __init__(self, v): self.shared_memory_per_block_optin = v
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: P(101376))     # sm_120
    assert k.tuile_par_carte("cuda:0") == (32, 64, 4, 1)
    k._CHOIX.clear(); monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: P(232448))   # H100
    assert k.tuile_par_carte("cuda:0") == (64, 64, 8, 2)
    k._CHOIX.clear(); monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: P(48 * 1024))
    assert k.tuile_par_carte("cuda:0") == (16, 32, 4, 1)
    monkeypatch.setenv("ACVRAM_MLA_FLASH_TUILE", "32,32,4,1")
    assert k.tuile_par_carte("cuda:0") == (32, 32, 4, 1)
    k._CHOIX.clear()
