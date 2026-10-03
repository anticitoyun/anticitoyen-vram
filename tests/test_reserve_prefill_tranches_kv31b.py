"""kv31b, levier 1 (poste6, 30/09, ordre chef) : la réserve de préfill suit la tranche réellement utilisée.
gemma-4-31B à 32 768 (edz définitif, 15 refus) : le plancher KV d'une séquence (15,1 Gio) exilait déjà les 60 MLP, si bien
qu'a5v (`_plafonner_mlp_prefill`) ne voyait « pas moins d'exil » avec le plafond et gardait la réserve d'un seul tenant
(10,4 Gio) — le plancher ne tenait plus. Désormais, si la réserve plafonnée fait TENIR le plancher là où la pleine échoue,
le plus grand plafond qui tient est posé ; la réserve (`activations_prefill_bytes`) le suit au bit, comme le seuil MLP
posé à l'exécution (`attention.definir_seuil`).
PRÉDICTION (écrite avant le test, réplique de gemma-4-31B, CUDA simulé) : réserve 10,7 → 4,2-5,0 Gio à plafond 4 096 ;
à 28,1 Gio libres (edz) le refus disparaît ; MLP exilés inchangés (60/60) ; à 33,6 (carte seule) tient aussi.
RENDU : à 28,1 la borne plafonnée vaut 14,62 Gio pour un plancher de 15,12 — il manque 0,5 Gio, le refus RESTE
(prédiction réfutée sur ce point : la réserve seule ne suffit pas à 28,1, le llama-server hors verrou de l'edz pesait
5,5 Gio) ; à 33,6 le plafond 4 096 tient ET ne laisse que 33 MLP exilés sur 60 (le plus grand plafond qui tient sans
exiler plus : 8 192 en exilerait 38, 16 384 : 47). La fenêtre qui tient annoncée à 28,1 passe de 23 552 à 31 744 (réserve
plafonnée), ce que claude (≥ 29 120) accepte.
TROUVÉ EN ROUTE (balayage libre × fenêtre à sec) : à 29,5 Gio libres, 32 768 REFUSAIT alors que 29,0 et 30,0 servaient —
`_reajuster_plan` prenait sa marge sur min(capacité, libre) (5 % de 29,5) et `_borner_kv_par_la_vram` sur la carte entière
(5 % de 34) : la remontée en VRAM rendait un MLP que la borne refusait 230 Mio plus loin, exil/remontée en boucle,
« 1 Mio manquants » après 4 tours. Même base de marge des deux côtés désormais : test 3.
Cassure : remettre `if plancher >= plein: return False` → les relances (28,0 / 25 600…) refusent (test 4) ; remettre la marge
sur min(capacité, libre) → test 3 rouge. Chiffres re-dérivés le 30/09 soir après le terme d'attention de la réserve."""
import contextlib
import io
import sys

import pytest
import torch

import acvram.engine.attention as A
from acvram.engine import loader as LD

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_fenetre_qui_tient_kv31b import G, _manifest, _plan, _spec   # noqa: E402  (réplique de gemma-4-31B)


def _preparer(monkeypatch, libre, ctx=32768, cap=34 * G):
    monkeypatch.setattr(A, "_MLP_MORCEAU", 4096)
    monkeypatch.delenv("ACVRAM_MTP", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (libre, cap))
    spec, man = _spec(), _manifest()

    def planifier():
        p = _plan(cap)
        p.kv_bytes_per_token = spec.kv_bytes_per_token(8)
        p.kv_budget = {"cuda:0": p.kv_bytes_per_token * ctx}
        LD._reajuster_plan(p, man, top_k=8, reserve=LD._reserve_prefill(spec, ctx, man, p),
                           kv_min={"cuda:0": LD._kv_plancher(p, spec, ctx, "cuda:0")})
        return p
    return spec, man, planifier


def _refuse(spec, man, p, ctx):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            LD._borner_kv_avec_exil(p, man, lambda n: n, spec, max_model_len=ctx, reserve=LD._reserve_prefill(spec, ctx, man, p))
    except RuntimeError as e:
        return str(e)
    return None


def test_a_28_gio_le_plafond_fait_tenir_le_plancher_en_exilant(monkeypatch, capsys):
    """Avant le levier 2 : refus à 28,1 Gio, fenêtre annoncée 25 600 (d19 : 27 648). Levier 2 (clés d'une couche à fenêtre TRANCHÉES
    à fenêtre + M pour les morceaux : scores 4,0 → 0,5 Gio à 32 768) : la réserve plafonnée (3,97 Gio) fait tenir le plancher KV de
    15,1 Gio au prix de 58/60 MLP exilés — servi DÉGRADÉ au lieu de refusé ; la chauffe décide du tenu réel."""
    ctx = 32768
    spec, man, planifier = _preparer(monkeypatch, libre=int(28.1 * G))
    assert _refuse(spec, man, planifier(), ctx), "témoin : la réserve d'un seul tenant devait refuser"
    assert LD._plafonner_mlp_prefill(spec, ctx, planifier(), planifier, man) and spec.mlp_prefill_plafond == 4096
    assert _refuse(spec, man, planifier(), ctx) is None
    assert 55 <= LD._mlp_exiles(planifier()) <= 60


def test_a_33_gio_le_plafond_fait_tenir_le_plancher_kv(monkeypatch, capsys):
    ctx = 32768
    spec, man, planifier = _preparer(monkeypatch, libre=int(33.6 * G))
    p0 = planifier()
    pleine = LD._reserve_prefill(spec, ctx, man, p0)
    assert _refuse(spec, man, planifier(), ctx), "témoin : la réserve d'un seul tenant devait refuser"
    assert LD._plafonner_mlp_prefill(spec, ctx, p0, planifier, man), "aucun plafond posé"
    c = spec.mlp_prefill_plafond
    assert c == 4096, c                                               # le plus grand qui tient SANS exiler plus
    plafonnee = LD._reserve_prefill(spec, ctx, man, planifier())
    # 8,5-9,5 avant d19 (résiduel borné : 7,5), 3,97 avec le levier 2 (clés de fenêtre tranchées : scores 4,0 → 0,5 Gio)
    assert 3.5 * G <= plafonnee <= 4.5 * G and pleine >= 14 * G, (plafonnee / G, pleine / G)
    assert _refuse(spec, man, planifier(), ctx) is None, "le refus devait disparaître"
    assert 25 <= LD._mlp_exiles(planifier()) <= 58                    # 55 avant d19, 49 résiduel borné, 28 clés tranchées (levier 2)
    spec.mlp_prefill_plafond = c
    assert f"MLP dense par tranches et attention par morceaux au-delà de {c} jetons" in capsys.readouterr().out   # d19


def test_sans_gain_ni_besoin_rien_ne_change(monkeypatch):
    """Carte large : la réserve pleine tient et n'exile rien de plus que la plafonnée → pas de plafond (a5v inchangé)."""
    spec, man, planifier = _preparer(monkeypatch, libre=60 * G, cap=64 * G)
    assert not LD._plafonner_mlp_prefill(spec, 32768, planifier(), planifier, man)
    assert spec.mlp_prefill_plafond is None


def test_a_29_5_gio_plus_d_oscillation_exil_remontee(monkeypatch):
    """29,0 et 30,0 servaient, 29,5 refusait pour « 1 Mio manquants » : les deux marges (exil, borne) sont sur la même base.
    Sous la formule avec terme d'attention, 29,5 Gio sert jusqu'à 26 624 (balayage monotone : 28,1 → 25 600, 29,5 → 26 624,
    30,7 → 28 672, 32,0 → 30 720, 33,6 → 32 768)."""
    spec, man, planifier = _preparer(monkeypatch, libre=int(29.5 * G), ctx=26624)
    with contextlib.redirect_stdout(io.StringIO()):
        LD._plafonner_mlp_prefill(spec, 26624, planifier(), planifier, man)
    p = planifier()
    assert _refuse(spec, man, p, 26624) is None
    assert 25 <= LD._mlp_exiles(p) <= 60                              # d19 : résiduel borné ; levier 2 : clés tranchées (34)


def test_la_relance_a_la_fenetre_annoncee_sert_grace_a_la_branche_plancher_tenu(monkeypatch):
    """Les cases que la nouvelle branche change (balayage avec/sans, pas de 1 024) sont exactement les RELANCES à la fenêtre
    annoncée : 28,0 Gio / 25 600, 30,0 / 28 672, 32,0 / 31 744 — a5v seul n'exile pas moins (60 = 60) et refuse ; la réserve
    plafonnée fait tenir le plancher → plafond posé, sert (60/60 exilés, le KV occupe la place). Cassure : `if plancher >= plein:
    return False` remis → refus."""
    for libre, ctx in ((28.0, 25600), (30.0, 28672), (32.0, 31744)):
        spec, man, planifier = _preparer(monkeypatch, libre=int(libre * G), ctx=ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            assert LD._plafonner_mlp_prefill(spec, ctx, planifier(), planifier, man), (libre, ctx)
        assert spec.mlp_prefill_plafond == 4096
        assert _refuse(spec, man, planifier(), ctx) is None, (libre, ctx)
