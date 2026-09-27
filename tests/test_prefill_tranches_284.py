"""Pièce 284 b : le préfill « une par une » d'un hybride au-delà de la frontière d'instantané (cache de préfixe ON), réordonné
couche par couche (`Engine._prefill_tranches`, `ACVRAMModel.forward_tranches`), rend AU BIT la boucle d'avant
(`ACVRAM_PREFILL_TRANCHES=0`) : logits du préfill et jetons décodés, à b=1 et à b=3. Preuve de prise : le compteur
`prefill_tranches`. Cassant : tranches croisées (l'état caché d'une séquence passé avec le lot d'une autre) → la
comparaison DOIT voir l'écart. Cache : un 2e tour qui partage l'amorce reprend à l'instantané pris à la frontière."""
import importlib.util
import pathlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="hybride GDN : carte requise")

_ICI = pathlib.Path(__file__).parent
_s = importlib.util.spec_from_file_location("t167", _ICI / "test_gdn_int8_canal_conversion_reelle_167.py")
T167 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T167)
PAS = 32                                    # frontière d'instantané (multiple de BLOCK_SIZE = 16)


@pytest.fixture(scope="module")
def modele(tmp_path_factory):
    """Le petit hybride de la 167 (une couche GDN, une couche d'attention pleine), complété de la norme finale qu'il
    n'avait pas (il ne servait qu'à la conversion) pour être CHARGEABLE ; même conversion que `T167._convertir`."""
    from safetensors.torch import load_file, save_file
    from acvram.engine.config import load_model_spec
    from acvram.hardware.profiles import load_profile
    from acvram.memory.tiering import PlannerOptions, auto_plan
    d = tmp_path_factory.mktemp("hyb284")
    ckpt = T167._checkpoint_167(d)
    f = pathlib.Path(ckpt) / "model.safetensors"
    sd = load_file(str(f))
    sd.setdefault("model.norm.weight", torch.ones(T167.H, dtype=next(iter(sd.values())).dtype))
    save_file(sd, str(f))
    spec = load_model_spec(ckpt, "tiny-gdn-284")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"), PlannerOptions(max_model_len=256, max_concurrent_seqs=4))
    out = d / "out"
    T167.convert_checkpoint(ckpt, plan, T167.ConversionOptions(
        out_dir=str(out), attn_qkvo_int8_canal=True, gdn_int8_canal=True, snr_floor=999.0,
        promotion_classes=T167._CLASSES_ATTN_GDN, max_promotions=1.0), spec=spec)
    return out


def _invites(n, graine=284):
    g = torch.Generator().manual_seed(graine)
    longueurs = [50, 70, 45, 66][:n]
    return [torch.randint(10, T167.V - 1, (k,), generator=g).tolist() for k in longueurs]


def _tourner(modele, monkeypatch, tranches, invites, croiser=False):
    from acvram.engine import runner as R
    from acvram.engine.loader import load_model
    from acvram.engine.sampler import SamplingParams
    monkeypatch.setenv("ACVRAM_INSTA_PAS", str(PAS))
    monkeypatch.setattr(R, "_PREFILL_TRANCHES", tranches)
    l = load_model(str(modele), dtype=torch.bfloat16, max_model_len=256, max_concurrent_seqs=4)
    eng = R.Engine(l, None, max_batch_size=4, max_model_len=256, enable_cuda_graphs=False)
    assert eng.est_hybride and eng.allocator.enable_prefix_cache
    eng._eos = set()
    if croiser:
        vrai_ft = eng.model.forward_tranches

        def croise(batches):
            return vrai_ft(batches[1:] + batches[:1]) if len(batches) > 1 else vrai_ft(batches)
        eng.model.forward_tranches = croise
    logits, vrai = {}, eng._emit

    def emit(lg, seqs):
        for i, s in enumerate(seqs):
            logits.setdefault(s.request_id, []).append(lg[i].float().cpu().clone())
        return vrai(lg, seqs)
    eng._emit = emit
    for k, p in enumerate(invites):
        eng.add_request(p, SamplingParams(temperature=0.0, max_tokens=4), request_id=f"r{k}")
    jetons = {}
    while eng.running or eng.waiting:
        for o in eng.step():
            jetons.setdefault(o.request_id, []).extend(o.token_ids)
    return logits, jetons, eng


def _egaux(a, b):
    return a.keys() == b.keys() and all(len(a[k]) == len(b[k]) and all(torch.equal(x, y) for x, y in zip(a[k], b[k]))
                                        for k in a)


@pytest.mark.parametrize("n", [1, 3])
def test_au_bit_de_la_boucle_une_par_une(modele, monkeypatch, n):
    inv = _invites(n)
    assert all(len(p) > PAS for p in inv)
    lr, jr, er = _tourner(modele, monkeypatch, False, inv)
    ln, jn, en = _tourner(modele, monkeypatch, True, inv)
    assert er.stats.prefill_tranches == 0
    assert en.stats.prefill_tranches == (1 if n > 1 else 0), "le chemin par tranches n'a pas été pris"
    assert _egaux(lr, ln), "logits du préfill différents"
    assert jr == jn, (jr, jn)


def test_cassant_tranches_croisees(modele, monkeypatch):
    inv = _invites(3)
    lr, _, _ = _tourner(modele, monkeypatch, False, inv)
    lc, _, ec = _tourner(modele, monkeypatch, True, inv, croiser=True)
    assert ec.stats.prefill_tranches == 1
    assert not _egaux(lr, lc), "la comparaison ne voit pas des tranches croisées : elle ne juge rien"


def test_second_tour_reprend_a_l_instantane(modele, monkeypatch):
    from acvram.engine.sampler import SamplingParams
    inv = _invites(3)
    _, _, eng = _tourner(modele, monkeypatch, True, inv)
    suite = inv[0][:PAS + 8] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    seq = eng.add_request(suite, SamplingParams(temperature=0.0, max_tokens=2), request_id="tour2")
    eng.step()
    assert seq.cached_len >= PAS, f"2e tour : cached_len {seq.cached_len} < frontière {PAS} (instantané non repris)"
