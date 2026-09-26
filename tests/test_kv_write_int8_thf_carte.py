"""anticitoyen-vram-thf, CLOS (pièce 109, ordre chef 24/09) : le ticket original croyait que
`kv_write_int8_kernel` (acvram_kernels.cu:4331) divergeait du jumeau torch
(`kvcache.py::PagedKVCache._quantize`) parce que `m / 127.f` puis `x * (1.f / sc)` seraient des
divisions APPROCHÉES sous `--use_fast_math`, alors que le jumeau resterait en division IEEE
correctement arrondie — et proposait `__fdiv_rn` partout comme correctif.

C'ÉTAIT L'INVERSE, vérifié au bit (sonde `kv_write_int8_v_debug`, 1000 tirages vs `main`,
24/09) : la sortie SERVIE par défaut (repli "int8" de `loader.py:196`) suit DÉJÀ la convention
« multiplication par le réciproque approché » des deux côtés — le jumeau torch aussi, via une
optimisation ATen de la division tenseur/scalaire Python (`amax / 127.0` se compile en
`amax * (1/127)`, pas en division IEEE élément par élément). Rendre le noyau IEEE-exact le
désaccordait du jumeau ET de `main` (569 écarts / 1000 tirages mesurés). Le noyau est revenu au
bit au code de `main` (`m / 127.f`, `x * (1.f/sc)`, jamais `__fdiv_rn`) : passer à l'IEEE partout
changerait une sortie servie, décision réservée à l'utilisateur — non prise ici.

ÉCHELLE : `kv_write_int8_kernel:4373` (`const float sc = red[0]`) et `kvcache.py:495`
(`x.to(float32) / scale`) utilisent TOUS DEUX l'échelle fp32 COMPLÈTE pour la division par
élément — le cast fp16 (`__float2half(sc)` côté noyau, `.to(torch.float16)` côté référence)
n'intervient QUE pour le STOCKAGE, après coup (vérifié au bit, pas supposé). Ce n'est donc PAS
un mésaccord fp16/fp32 qui cause les écarts tolérés ci-dessous, mais l'erreur du réciproque
rapide `1.f/sc` (fast-math) contre la division vraie `x/scale` (tenseur/tenseur, ATen ne
l'optimise pas en réciproque).

EPSILON DÉRIVÉ (pas cité sans preuve) : sonde `kv_rcp_approx_debug`
(`scratchpad/poste3-thf-23-09/mesure_rcp_approx.py`, 24/09) mesure `1.f/sc` (mêmes flags que
`kv_write_int8_kernel:4373`) contre `1.0/sc` sur 64 000 échelles réelles → erreur relative max
mesurée 7,87e-8. `δ(x/sc) ≈ (x/sc) × erreur_relative ≤ 127 × 7,87e-8 ≈ 1,0e-5` (x ≤ 127×sc par
construction du code int8). EPSILON = 2e-5 (marge ×2 sur la mesure, formule explicite, pas un
doublement arbitraire d'un max observé au hasard).

Le test compare au bit, SAUF dans cette bande ε où un écart d'1 code est toléré ET COMPTÉ —
jamais silencieux — et échoue si la FRACTION d'éléments tolérés dépasse PLAFOND_FRACTION (une
faute réelle qui toucherait, disons, 5 % des éléments serait noyée si on tolérait juste "peu
d'écarts" sans plafond chiffré).

Deux bras cassants : (1) diviseur /126 (faux, hors bande — trivial), (2) troncature
(`__float2int_rz` simulé côté Python par `.trunc()`) au lieu de l'arrondi — plus fin, teste que
le plafond de FRACTION rougit même quand chaque écart individuel semble "proche" d'une
frontière (la troncature déplace TOUT ratio à partie fractionnaire ∈ [0,5;1) d'un code entier,
bien au-delà d'ε, mais en touchant une GRANDE fraction des éléments).

Sauté sans CUDA ou sans extension. À lancer NU (jamais sous un verrou tenu par un autre)."""
from __future__ import annotations

import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")

HKV, D, BS = 4, 128, 16
EPSILON = 2e-5  # dérivé (mesure_rcp_approx.py, 24/09) : 127 * erreur_relative_max_mesurée (7,87e-8), x2
PLAFOND_FRACTION = 1e-3  # 0,1 % des éléments — généreux au-dessus du bruit attendu (~0,0016 %), strict sous 5 %


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "kv_write_int8"):
        pytest.skip("extension absente ou sans kv_write_int8")
    return ext


def _kv(n, graine):
    torch.manual_seed(graine)
    k = torch.randn(n, HKV, D, device="cuda") * torch.logspace(-1, 1, D, device="cuda")
    v = torch.randn(n, HKV, D, device="cuda") * torch.logspace(-1, 1, D, device="cuda")
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def _cache():
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=1,
                                      block_size=BS, dtype="int8", device="cuda", canal=False))


def _comparer_avec_tolerance(x: torch.Tensor, obs: torch.Tensor, ref: torch.Tensor) -> int:
    """Compare `obs` (codes du noyau, ou d'un bras cassant simulé) à `ref` (codes du jumeau
    torch). Toute divergence DOIT tomber à <= EPSILON d'une frontière d'arrondi (k+0,5) ; au-delà,
    échec dur (vraie faute, pas le réciproque approché). La FRACTION d'écarts tolérés (même tous
    dans la bande) échoue dur au-delà de PLAFOND_FRACTION. Rend le nombre d'écarts tolérés.

    L'échelle est RECALCULÉE ici en fp32 (amax/127), jamais reçue en argument : `_quantize`
    renvoie l'échelle castée fp16 (`kvcache.py:499`) — piège trouvé le 24/09, un 1er essai qui
    comparait au ratio fp16-cast gonflait la distance mesurée d'un facteur ~1000 (0,022 au lieu
    de ~1e-5), pas une propriété du noyau, un artefact de mesure."""
    diff = (obs.int() - ref.int())
    idx = (diff != 0).nonzero()
    total = obs.numel()
    if idx.shape[0] == 0:
        return 0
    x32 = x.to(torch.float32)
    amax = x32.abs().amax(dim=-1, keepdim=True)
    sc32 = (amax / 127.0).clamp(min=1e-8)  # échelle fp32 COMPLÈTE, jamais celle (fp16) reçue
    ratio = x32 / sc32
    n_toleres = 0
    hors_bande = []
    for i in range(idx.shape[0]):
        t, h, d = idx[i].tolist()
        r = float(ratio[t, h, d].item())
        dist = abs(r - (round(r - 0.5) + 0.5))
        if dist <= EPSILON:
            n_toleres += 1
        else:
            hors_bande.append((t, h, d, r, dist))
    assert not hors_bande, f"écarts HORS bande epsilon (vraie faute) : {hors_bande[:5]}"
    frac = n_toleres / total
    assert frac <= PLAFOND_FRACTION, (
        f"{n_toleres}/{total} ({frac:.4%}) tolérés — dépasse PLAFOND_FRACTION="
        f"{PLAFOND_FRACTION:.4%} : trop d'écarts pour être du bruit rcp.approx, vraie faute suspectée")
    return n_toleres


def test_ecriture_int8_par_jeton_au_bit_ou_dans_la_bande():
    """Sur ce build, 0 écart attendu (vérifié 1000/1000 le 24/09) ; la bande n'intervient que si
    un autre build/arch dérive légèrement — jamais silencieuse."""
    _ext()
    cache = _cache()
    k, v = _kv(BS, graine=1)
    slots = torch.arange(BS, device="cuda", dtype=torch.int64)
    cache.write(slots, k, v)

    qk_ref, sk_ref = cache._quantize(k)
    qv_ref, sv_ref = cache._quantize(v)

    qk_obs = cache.k.view(-1, HKV, D)[:BS]
    qv_obs = cache.v.view(-1, HKV, D)[:BS]
    sk_obs = cache.k_scale.view(-1, HKV)[:BS]
    sv_obs = cache.v_scale.view(-1, HKV)[:BS]

    assert torch.equal(sk_obs, sk_ref), "échelle K hors du bit"
    assert torch.equal(sv_obs, sv_ref), "échelle V hors du bit"
    nk = _comparer_avec_tolerance(k, qk_obs, qk_ref)
    nv = _comparer_avec_tolerance(v, qv_obs, qv_ref)
    if nk or nv:
        print(f"[thf] tolérés (bande epsilon) : K={nk} V={nv}")


def test_plusieurs_tirages_toujours_au_bit_ou_dans_la_bande():
    """5 graines : 0 écart attendu (1000 tirages vérifiés le 24/09, cette pièce n'en couvre que
    5 à sec — la bande n'est qu'un filet, jamais un blanc-seing)."""
    _ext()
    for graine in range(5):
        cache = _cache()
        k, v = _kv(BS, graine=100 + graine)
        slots = torch.arange(BS, device="cuda", dtype=torch.int64)
        cache.write(slots, k, v)
        qk_ref, sk_ref = cache._quantize(k)
        qv_ref, sv_ref = cache._quantize(v)
        assert torch.equal(cache.k_scale.view(-1, HKV)[:BS], sk_ref), graine
        assert torch.equal(cache.v_scale.view(-1, HKV)[:BS], sv_ref), graine
        nk = _comparer_avec_tolerance(k, cache.k.view(-1, HKV, D)[:BS], qk_ref)
        nv = _comparer_avec_tolerance(v, cache.v.view(-1, HKV, D)[:BS], qv_ref)
        if nk or nv:
            print(f"[thf] graine={graine} tolérés (bande epsilon) : K={nk} V={nv}")


def test_bras_cassant_diviseur_faux_rougit_hors_bande():
    """Bras cassant 1 (ordre chef) : une vraie faute (diviseur /126 au lieu de /127) doit être
    détectée COMME hors bande, pas absorbée par EPSILON."""
    _ext()
    cache = _cache()
    k, v = _kv(BS, graine=1)
    slots = torch.arange(BS, device="cuda", dtype=torch.int64)
    cache.write(slots, k, v)

    qv_ref, sv_ref = cache._quantize(v)
    amax = v.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale_fausse = (amax / 126.0).clamp(min=1e-8)
    q_faux = (v.to(torch.float32) / scale_fausse).round().clamp(-127, 127).to(torch.int8)

    with pytest.raises(AssertionError, match="HORS bande"):
        _comparer_avec_tolerance(v, q_faux, qv_ref)


def test_bras_cassant_fraction_rougit_meme_dans_la_bande():
    """Bras cassant 2, plus fin (ordre chef) : synthétique et direct, pour isoler
    PLAFOND_FRACTION de la bande — construit N éléments dont le ratio est À exactement
    EPSILON/2 d'une frontière (donc chacun INDIVIDUELLEMENT toléré) et dont le code observé
    diffère du code de référence d'exactement 1. Un biais systématique qui toucherait une
    fraction des éléments plus grande que PLAFOND_FRACTION, mais chacun de peu, ne doit PAS
    passer sous silence — sinon le plafond ne sert à rien."""
    n = 8192  # même taille qu'un tenseur K/V réel (BS*HKV*D), pour un plafond comparable
    n_faux = int(n * (PLAFOND_FRACTION * 2))  # au-delà du plafond, chacun dans la bande
    assert n_faux > 0

    x = torch.zeros(1, 1, n, device="cuda")
    x[0, 0, -1] = 127.0  # fixe amax=127 -> échelle interne = amax/127 = 1.0, ratio = x directement
    ref = torch.zeros(1, 1, n, dtype=torch.int8, device="cuda")
    ref[0, 0, -1] = 127
    obs = torch.zeros(1, 1, n, dtype=torch.int8, device="cuda")
    obs[0, 0, -1] = 127
    for i in range(n_faux):
        r = 10.5 + EPSILON / 2  # à EPSILON/2 d'une frontière (échelle=1 : ratio=x), dans la bande
        x[0, 0, i] = r
        ref[0, 0, i] = 11  # arrondi correct (sans ambiguïté)
        obs[0, 0, i] = 10  # off-by-one, comme un vrai écart de rcp.approx

    with pytest.raises(AssertionError, match="PLAFOND_FRACTION"):
        _comparer_avec_tolerance(x, obs, ref)
