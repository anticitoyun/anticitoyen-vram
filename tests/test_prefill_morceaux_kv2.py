"""kv31b levier 2, étape 1 (poste6, 30/09, ordre chef) : préfill de l'ATTENTION par morceaux, derrière
`ACVRAM_PREFILL_MORCEAU` (jetons, 0 = OFF, défaut), déclaré au régime (« prefill=…(morceaux@N) »), via `forward_tranches`
(284 b, couche par couche). Au bit sur processeur : les logits du dernier jeton d'invite sont IDENTIQUES à ceux d'un
seul tenant, pour plusieurs longueurs (multiples et non multiples du morceau, une invite plus courte que le morceau qui
ne se découpe pas), modèle bf16 et cache KV bf16 (K/V rendus exacts ; en fp32 + cache bf16 l'arrondi du cache donne 7e-4 :
mesuré, c'est le cache qui arrondit, pas les morceaux) — et le même contrôle en int8 (celui servi), dont l'écart est
rapporté, pas caché. Le repli séquentiel (le modèle refuse les tranches : images, deepstack, MTP) découpe quand même et
reste au bit. Les bornes de morceaux évitent les images (`_eviter_coupe_image`) : une image plus longue qu'un morceau
va d'un seul tenant jusqu'à sa fin. OFF → aucun morceau, `prefill_morceaux` = 0, régime sans mention.
Plancher : un morceau (le dernier compris) fait ≥ max(128, SEUIL_FUSION + 1) lignes — sous 128 le jouet n'est plus au bit
(chemins à petit M : 32-96 lignes ≠ 128-160, mesuré 30/09) et sous le seuil de fusion le MLP change de chemin (1,8e-4) ;
un réglage plus bas est ignoré et dit. Cassure : `_prefill_morceaux` rendant None → compteur à 0 (test 1 rouge) ;
`_limites_morceaux` sans `_eviter_coupe_image` → test images rouge ; plancher retiré → réglage 32 accepté, écarts."""
import contextlib
import types

import pytest
import torch

import acvram.engine.runner as R
from acvram.engine.sampler import SamplingParams

LONGUEURS = (17, 257, 300, 400, 511)    # < morceau ; 2 morceaux ; reste de 44 (fondu) ; 3 + reste 144 ; 3 + reste 127 (fondu)
MORCEAU = 128                             # ≥ plancher 128 : au bit sous un chemin MLP unique


def _moteur(converted, kv_format, monkeypatch):
    import acvram.engine.attention as A
    import acvram.memory.tiering as T
    # UN SEUL chemin MLP pour les deux (fusion gate/up imposée à toute longueur) : le jouet (512 jetons au plus) ne peut
    # pas mettre un seul tenant ET tous ses morceaux au-dessus du seuil de fusion de service (256) ; à 64 (les deux non
    # fusionnés) le jouet n'est PAS au bit (4,5e-3 : le chemin non fusionné du jouet dépend des lignes — non élucidé, dit)
    monkeypatch.setattr(A, "SEUIL_FUSION", 10 ** 6)
    from acvram.engine.loader import load_model
    monkeypatch.setattr(T, "_KV_FORMAT", kv_format)
    # bf16 : les K/V calculés en bf16 sont rendus EXACTS par un cache bf16 (fp32 + cache bf16 : 7e-4 d'écart, mesuré)
    loaded = load_model(converted, dtype=torch.bfloat16, device_override="cpu", max_concurrent_seqs=2)
    eng = R.Engine(loaded, None, max_batch_size=2, max_model_len=512, enable_prefix_cache=False,
                   enable_cuda_graphs=False)
    eng._eos = set()
    return eng


def _logits_invite(eng, n):
    """Logits du dernier jeton d'invite (ce que `_emit` reçoit), invite déterministe de n jetons."""
    vus = []
    emit = eng._emit

    def _capture(logits, seqs):
        vus.append(logits.detach().clone())
        return emit(logits, seqs)
    eng._emit = _capture
    ids = [(7 + i * 13) % 200 + 3 for i in range(n)]
    for _ in eng.generate(ids, SamplingParams(temperature=0.0, max_tokens=1)):
        pass
    eng._emit = emit
    assert vus, "aucun logit émis"
    return vus[0]


@pytest.mark.parametrize("kv_format", ["bf16", None])
def test_morceaux_au_bit_du_seul_tenant(converted, monkeypatch, kv_format):
    monkeypatch.setattr(R, "_PREFILL_MORCEAU", 0)
    eng = _moteur(converted, kv_format, monkeypatch)
    seul = {n: _logits_invite(eng, n) for n in LONGUEURS}
    assert eng.stats.prefill_morceaux == 0 and "morceaux@" not in eng.regime_ligne()
    monkeypatch.setattr(R, "_PREFILL_MORCEAU", MORCEAU)
    eng = _moteur(converted, kv_format, monkeypatch)
    assert f"(morceaux@{MORCEAU})" in eng.regime_ligne()
    ecarts = {}
    for n in LONGUEURS:
        m = _logits_invite(eng, n)
        assert m.shape == seul[n].shape
        ecarts[n] = float((m.float() - seul[n].float()).abs().max())
    assert eng.stats.prefill_morceaux == sum(1 for n in LONGUEURS if n > MORCEAU), eng.stats.prefill_morceaux
    if kv_format == "bf16":
        assert all(ecarts[n] == 0.0 for n in LONGUEURS), ecarts
    else:
        # int8 : les morceaux relisent des K/V quantifiés (comme une reprise après le cache de préfixe) — l'écart est
        # RAPPORTÉ ; au bit ici aussi sur ce jouet si le cache CPU rend l'exact, sinon petit (< 1e-2 sur des logits fp32)
        assert max(ecarts.values()) < 5e-2, ecarts
        print("écart int8 (morceaux − seul tenant), logits bf16 :", ecarts)


def test_repli_sequentiel_sans_tranches_reste_au_bit(converted, monkeypatch):
    monkeypatch.setattr(R, "_PREFILL_MORCEAU", 0)
    seul = _logits_invite(_moteur(converted, "bf16", monkeypatch), 300)
    monkeypatch.setattr(R, "_PREFILL_MORCEAU", MORCEAU)
    eng = _moteur(converted, "bf16", monkeypatch)
    monkeypatch.setattr(eng.model, "tranches_possibles", lambda *a, **k: False)
    assert torch.equal(_logits_invite(eng, 300), seul) and eng.stats.prefill_morceaux == 1


def test_bornes_evitent_les_images(monkeypatch):
    import acvram.engine.attention as A
    monkeypatch.setattr(A, "SEUIL_FUSION", 64)                       # plancher de lignes = max(128, 65) = 128
    Im = lambda d, f: types.SimpleNamespace(debut=d, fin=f)
    S = lambda pl, ims, n: types.SimpleNamespace(prefill_len=pl, images=ims, prompt_ids=[0] * n)
    assert R.Engine._limites_morceaux(S(0, [], 600), 600, 128) == [128, 256, 384, 600]      # reste 88 < 128 fondu
    assert R.Engine._limites_morceaux(S(0, [], 640), 640, 128) == [128, 256, 384, 512, 640]
    assert R.Engine._limites_morceaux(S(0, [Im(200, 300)], 600), 600, 128) == [128, 200, 328, 456, 600]   # 256 dans l'image → 200
    assert R.Engine._limites_morceaux(S(0, [Im(100, 500)], 600), 600, 128) == [100, 600]             # image > morceau, reste fondu
    assert R.Engine._limites_morceaux(S(300, [], 600), 600, 128) == [428, 600]                       # reprise à 300
    assert R.Engine._limites_morceaux(S(0, [], 257), 257, 128) == [128, 257]


def test_reglage_sous_le_plancher_ignore_et_dit(converted, monkeypatch, capsys):
    monkeypatch.setattr(R, "_PREFILL_MORCEAU", 32)
    monkeypatch.setattr(R, "_MORCEAU_LIGNES_MIN", 128)
    eng = _moteur(converted, "bf16", monkeypatch)                    # plancher 128 (fusion imposée : min(128, 10**6+1) = 128)
    _logits_invite(eng, 300)
    assert eng.stats.prefill_morceaux == 0 and "morceaux@" not in eng.regime_ligne()
    assert "ACVRAM_PREFILL_MORCEAU=32 ignoré" in capsys.readouterr().out


def test_frontiere_d_instantane_passe_aussi_par_morceaux(converted, monkeypatch):
    """Modèle à couches typées (`est_hybride`, gemma-4) + cache de préfixe : le préfill est coupé à la frontière
    d'instantané et la première passe (presque toute l'invite : 7 936 sur 7 953 en service) partait d'un seul tenant —
    preuve S1 du 01/10 : `prefill_morceaux` restait à 0 avec morceaux@4096. Ici : frontière 256 sur 511 → la passe
    jusqu'à la frontière compte UN morceaux (2 × 128) ; le reste (255 = 128 + 127, le dernier fondu) passe d'un seul
    tenant — 0 avant le correctif, 1 après ; au bit du seul tenant coupé pareil."""
    def moteur(morceau):
        monkeypatch.setattr(R, "_PREFILL_MORCEAU", morceau)
        monkeypatch.setenv("ACVRAM_INSTA_PAS", "256")
        eng = _moteur(converted, "bf16", monkeypatch)
        eng.allocator.enable_prefix_cache = True
        eng.est_hybride = True
        return eng
    seul = _logits_invite(moteur(0), 511)
    eng = moteur(MORCEAU)
    m = _logits_invite(eng, 511)
    assert eng.stats.prefill_morceaux == 1, eng.stats.prefill_morceaux
    assert float((m.float() - seul.float()).abs().max()) == 0.0
