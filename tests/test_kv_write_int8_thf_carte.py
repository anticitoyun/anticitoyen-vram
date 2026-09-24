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

Le test compare au bit, SAUF dans une bande epsilon autour des frontières d'arrondi
(`|x/échelle - (k+0,5)| <= EPSILON`), où un écart d'UN SEUL code est toléré et compté (mesure
empirique : la magnitude du réciproque approché fait dériver le ratio de jusqu'à ~0,0255 d'une
frontière exacte sur ce jeu de données — `scratchpad/poste3-thf-23-09/calibre_epsilon.py`, 24/09 ;
EPSILON = 2x cette marge mesurée, pas une constante citée sans preuve).

Sauté sans CUDA ou sans extension. À lancer NU (jamais sous un verrou tenu par un autre)."""
from __future__ import annotations

import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")

HKV, D, BS = 4, 128, 16
EPSILON = 5.1e-2  # marge empirique (calibre_epsilon.py, 24/09) : max mesuré 0,02545, x2


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


def _comparer_avec_tolerance(x: torch.Tensor, sc: torch.Tensor,
                              obs: torch.Tensor, ref: torch.Tensor) -> int:
    """Compare `obs` (codes du noyau) à `ref` (codes du jumeau torch). Toute divergence DOIT
    tomber à <= EPSILON d'une frontière d'arrondi (k+0,5) ; au-delà, échec dur (c'est une vraie
    faute, pas le réciproque approché). Rend le nombre d'écarts tolérés (à consigner, jamais
    silencieux)."""
    diff = (obs.int() - ref.int())
    idx = (diff != 0).nonzero()
    if idx.shape[0] == 0:
        return 0
    x32 = x.to(torch.float32)
    ratio = x32 / sc.unsqueeze(-1).to(torch.float32)
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
    nk = _comparer_avec_tolerance(k, sk_ref, qk_obs, qk_ref)
    nv = _comparer_avec_tolerance(v, sv_ref, qv_obs, qv_ref)
    assert nk == 0, f"{nk} code(s) K tolérés (bande epsilon) — à consigner si >0"
    assert nv == 0, f"{nv} code(s) V tolérés (bande epsilon) — à consigner si >0"


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
        nk = _comparer_avec_tolerance(k, sk_ref, cache.k.view(-1, HKV, D)[:BS], qk_ref)
        nv = _comparer_avec_tolerance(v, sv_ref, cache.v.view(-1, HKV, D)[:BS], qv_ref)
        assert nk == 0, (graine, nk)
        assert nv == 0, (graine, nv)


def test_bras_cassant_diviseur_faux_rougit_hors_bande():
    """Bras cassant (ordre chef) : une vraie faute (diviseur /126 au lieu de /127) doit être
    détectée par `_comparer_avec_tolerance` COMME hors bande, pas absorbée par EPSILON — sinon le
    filet ne filtre rien et les deux tests ci-dessus ne prouveraient rien."""
    _ext()
    cache = _cache()
    k, v = _kv(BS, graine=1)
    slots = torch.arange(BS, device="cuda", dtype=torch.int64)
    cache.write(slots, k, v)

    qv_ref, sv_ref = cache._quantize(v)
    # Référence FAUSSE délibérée : diviseur 126 au lieu de 127 (vraie faute, pas un aléa de rcp).
    amax = v.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale_fausse = (amax / 126.0).clamp(min=1e-8)
    q_faux = (v.to(torch.float32) / scale_fausse).round().clamp(-127, 127).to(torch.int8)

    with pytest.raises(AssertionError, match="HORS bande"):
        _comparer_avec_tolerance(v, sv_ref, q_faux, qv_ref)
