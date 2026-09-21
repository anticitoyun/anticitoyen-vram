"""Rejeu sous graphe CUDA du chemin MLA à créneaux == eager, pas après pas
(Sage § 11, verdict-mla-seuil-creneaux-16-09 : sous graphes le 2e pas de
décodage est faux, l'eager est juste). Ce test isole le MODULE : un graphe
capturé une fois sur `decode_static` (b = 1) et sur
`decode_static_batch_complet` (b = 4), rejoué 4 pas avec des entrées
différentes, contre le même module en eager sur des créneaux jumeaux — len,
lignes du cache, sortie. S'il passe, le module relit bien le créneau vivant
et le défaut est dans le runner (liaison, remplissage, clé) ; s'il casse, il
est dans le module et le pas fautif est nommé."""
import pytest
import torch

from tests.test_mla_decode_batch import _ext, _module

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
NH, NOPE, ROPE, RANK, DV, HIDDEN, L = 4, 32, 16, 64, 32, 128, 128


def _module_rope(graine, rope):
    """Le module du test de lot, avec ou sans RoPE (GLM en a une : tables
    réservées avant la capture comme graphs._capture le fait)."""
    la = _module(NH, NOPE, ROPE, RANK, DV, HIDDEN, graine)
    if rope:
        from acvram.engine.layers import RotaryEmbedding
        la.rope_emb = RotaryEmbedding(ROPE, 4096, 10000.0, None, torch.device("cuda"), torch.bfloat16)
        la.rope_emb.reserver(L + 2, torch.device("cuda"), torch.bfloat16)
    return la


def _creneaux(la, lens, graine):
    torch.manual_seed(graine)
    sts = []
    for n in lens:
        st = la.new_static(torch.device("cuda"), L + 16, torch.bfloat16)
        etat = (torch.randn(n, RANK + ROPE, device="cuda") * 0.3).to(torch.bfloat16)
        la.static_load(st, etat if n else None)
        sts.append(st)
    return sts


def _rejeu(la, sts, appel, B, pas=4):
    """Capture ``appel(x)`` une fois (échauffement sur flux annexe, instantané
    des créneaux restauré comme graphs._capture), rejoue ``pas`` fois avec des
    entrées différentes ; rend les sorties et l'état final des créneaux."""
    x_buf = torch.zeros(B, HIDDEN, dtype=torch.bfloat16, device="cuda")
    torch.manual_seed(99)
    entrees = [torch.randn(B, HIDDEN, device="cuda").to(torch.bfloat16) for _ in range(pas)]
    inst = [la.static_export(st) for st in sts]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        for _ in range(2):
            appel(x_buf)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        out = appel(x_buf)
    for st, e in zip(sts, inst):
        la.static_load(st, e)
    torch.cuda.synchronize()
    sorties = []
    for k in range(pas):
        x_buf.copy_(entrees[k])
        g.replay()
        torch.cuda.synchronize()
        sorties.append(out.clone())
    return entrees, sorties, (g, x_buf, out)


def _eager(la, sts, appel, entrees):
    sorties = []
    with torch.inference_mode():
        for x in entrees:
            sorties.append(appel(x).clone())
            torch.cuda.synchronize()
    return sorties


def _compare(sorties_g, sorties_e, sts_g, sts_e, lens):
    for k, (a, b) in enumerate(zip(sorties_g, sorties_e)):
        assert torch.equal(a, b), f"pas {k + 1} : sortie graphe ≠ eager (max {(a.float() - b.float()).abs().max().item():.3e})"
    for i, (sg, se, n) in enumerate(zip(sts_g, sts_e, lens)):
        assert int(sg["len"]) == int(se["len"]) == n + len(sorties_g), f"créneau {i} : len {int(sg['len'])} / {int(se['len'])}"
        assert torch.equal(sg["cache"][:n + len(sorties_g)], se["cache"][:n + len(sorties_g)]), f"créneau {i} : lignes du cache différentes"


@pytest.mark.parametrize("rope", [False, True])
def test_decode_static_b1_rejoue_comme_eager(rope):
    _ext()
    la = _module_rope(21, rope)
    lens = [100]
    sts_g, sts_e = _creneaux(la, lens, 5), _creneaux(la, lens, 5)
    appel_g = lambda x: la.decode_static(x, sts_g[0], L)
    appel_e = lambda x: la.decode_static(x, sts_e[0], L)
    entrees, sorties_g, _ = _rejeu(la, sts_g, appel_g, 1)
    sorties_e = _eager(la, sts_e, appel_e, entrees)
    _compare(sorties_g, sorties_e, sts_g, sts_e, lens)


@pytest.mark.parametrize("rope", [False, True])
def test_decode_static_batch_complet_b4_rejoue_comme_eager(rope):
    _ext()
    la = _module_rope(22, rope)
    lens = [100, 3, 0, 60]
    sts_g, sts_e = _creneaux(la, lens, 6), _creneaux(la, lens, 6)

    def lot(sts):
        ptrs = torch.tensor([st["cache"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
        lptrs = torch.tensor([st["len"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
        scores = torch.zeros(4, NH, L + 16, device="cuda")
        return ptrs, scores, lptrs
    pg, sg, lg = lot(sts_g)
    pe, se, le = lot(sts_e)
    appel_g = lambda x: la.decode_static_batch_complet(x, sts_g, L, pg, sg, lg)
    appel_e = lambda x: la.decode_static_batch_complet(x, sts_e, L, pe, se, le)
    entrees, sorties_g, _ = _rejeu(la, sts_g, appel_g, 4)
    sorties_e = _eager(la, sts_e, appel_e, entrees)
    _compare(sorties_g, sorties_e, sts_g, sts_e, lens)


@pytest.mark.parametrize("rope", [False, True])
def test_static_load_apres_capture_puis_rejeu(rope):
    """Le motif de graphs._capture : capture, restauration par static_load,
    rejeu — puis un NOUVEAU static_load (autre séquence, autre longueur) sur le
    même créneau et rejeu : le graphe doit lire la nouvelle longueur et les
    nouvelles lignes, pas celles vues à la capture."""
    _ext()
    la = _module_rope(23, rope)
    sts_g, sts_e = _creneaux(la, [100], 7), _creneaux(la, [100], 7)
    appel_g = lambda x: la.decode_static(x, sts_g[0], L)
    appel_e = lambda x: la.decode_static(x, sts_e[0], L)
    entrees, _, (g, x_buf, out) = _rejeu(la, sts_g, appel_g, 1, pas=1)
    _eager(la, sts_e, appel_e, entrees)
    # nouvelle séquence dans le même créneau, MÊME graphe
    torch.manual_seed(31)
    etat = (torch.randn(37, RANK + ROPE, device="cuda") * 0.3).to(torch.bfloat16)
    la.static_load(sts_g[0], etat); la.static_load(sts_e[0], etat)
    torch.cuda.synchronize()
    torch.manual_seed(32)
    entrees2 = [torch.randn(1, HIDDEN, device="cuda").to(torch.bfloat16) for _ in range(3)]
    sorties_g = []
    for x in entrees2:
        x_buf.copy_(x); g.replay(); torch.cuda.synchronize(); sorties_g.append(out.clone())
    sorties_e = _eager(la, sts_e, appel_e, entrees2)
    _compare(sorties_g, sorties_e, sts_g, sts_e, [37])


@pytest.mark.parametrize("puits", [False, True], ids=["ordinaire", "puits-pos0"])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("B", [1, 4])
def test_creneaux_egalent_la_reference_hors_creneau(rope, B, puits):
    """Ce que le test de rejeu ne disait pas : le chemin à créneaux
    (static_load d'un cache de prefill, puis decode_static /
    decode_static_batch_complet, 4 pas) contre la RÉFÉRENCE `forward`
    hors créneau (mla.py, chemin eager juste de l'arbitre de Laure), même
    séquence, mêmes entrées — sortie et lignes du cache. Tolérance : le
    godet diffère (L contre godet_mla(total)) donc l'ordre des sommes fp32
    du noyau aussi ; 2^-6 relatif, pas bit-identique."""
    _ext()
    la = _module_rope(41 + B, rope)
    torch.manual_seed(50 + B)
    n0 = 100
    prompt = torch.randn(n0, HIDDEN, device="cuda").to(torch.bfloat16)
    entrees = [torch.randn(B, HIDDEN, device="cuda").to(torch.bfloat16) for _ in range(4)]
    with torch.inference_mode():
        # référence : prefill puis 4 pas hors créneau, une séquence par ligne du lot
        refs, caches = [], []
        for i in range(B):
            _, cache = la.forward(prompt if i == 0 else torch.roll(prompt, i, 0), None)
            if puits:
                # Sage § 12 : puits d'attention en position 0 ([gMASK]<sop>) —
                # la ligne 0 du latent domine le softmax de toutes les têtes
                cache[0] = (cache[0].float() * 40).to(cache.dtype)
            caches.append(cache)
        for k in range(4):
            ys = []
            for i in range(B):
                y, caches[i] = la.forward(entrees[k][i:i + 1], caches[i])
                ys.append(y)
            refs.append(torch.cat(ys, 0))
        # créneaux : static_load du cache de prefill (les mêmes n0 lignes)
        sts = []
        for i in range(B):
            st = la.new_static(torch.device("cuda"), L + 16, torch.bfloat16)
            _, c0 = la.forward(prompt if i == 0 else torch.roll(prompt, i, 0), None)
            if puits:
                c0[0] = (c0[0].float() * 40).to(c0.dtype)
            la.static_load(st, c0)
            sts.append(st)
        if B == 1:
            appel = lambda x: la.decode_static(x, sts[0], L)
        else:
            ptrs = torch.tensor([st["cache"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
            lptrs = torch.tensor([st["len"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
            scores = torch.zeros(B, NH, L + 16, device="cuda")
            appel = lambda x: la.decode_static_batch_complet(x, sts, L, ptrs, scores, lptrs)
        outs = [appel(entrees[k]).clone() for k in range(4)]
    torch.cuda.synchronize()
    for k in range(4):
        a, b = outs[k].float(), refs[k].float()
        tol = b.abs() * 2 ** -6 + 1e-2 * b.abs().max()
        hors = int(((a - b).abs() > tol).sum())
        assert hors == 0, f"pas {k + 1} : {hors} valeurs hors tolérance vs référence (max {(a - b).abs().max().item():.3e})"
    for i in range(B):
        assert int(sts[i]["len"]) == n0 + 4
        assert torch.equal(sts[i]["cache"][:n0 + 4], caches[i]), f"créneau {i} : lignes du cache ≠ référence"
