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
Cassure : remettre `if plancher >= plein: return False` → la case 28,6 Gio / 32 768 refuse (test 4) ; remettre la marge sur
min(capacité, libre) → test 3 rouge."""
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


def test_a_28_gio_le_refus_reste_mais_la_fenetre_annoncee_monte(monkeypatch, capsys):
    ctx = 32768
    spec, man, planifier = _preparer(monkeypatch, libre=int(28.1 * G))
    assert not LD._plafonner_mlp_prefill(spec, ctx, planifier(), planifier, man) and spec.mlp_prefill_plafond is None
    msg = _refuse(spec, man, planifier(), ctx)
    assert msg and "fenêtre qui tient : 31744 jetons" in msg, msg


def test_a_33_gio_le_plafond_fait_tenir_le_plancher_kv(monkeypatch, capsys):
    ctx = 32768
    spec, man, planifier = _preparer(monkeypatch, libre=int(33.6 * G))
    p0 = planifier()
    pleine = LD._reserve_prefill(spec, ctx, man, p0)
    assert _refuse(spec, man, planifier(), ctx), "témoin : la réserve d'un seul tenant devait refuser"
    assert LD._plafonner_mlp_prefill(spec, ctx, p0, planifier, man), "aucun plafond posé"
    c = spec.mlp_prefill_plafond
    assert c == 4096, c                                               # le plus grand qui tient SANS exiler plus (33)
    plafonnee = LD._reserve_prefill(spec, ctx, man, planifier())
    assert 4.2 * G <= plafonnee <= 5.0 * G and pleine >= 10.4 * G, (plafonnee / G, pleine / G)
    assert _refuse(spec, man, planifier(), ctx) is None, "le refus devait disparaître"
    assert LD._mlp_exiles(planifier()) == 33
    spec.mlp_prefill_plafond = 8192
    assert LD._mlp_exiles(planifier()) > 33 and _refuse(spec, man, planifier(), ctx) is None
    spec.mlp_prefill_plafond = c
    assert f"MLP dense par tranches au-delà de {c} jetons" in capsys.readouterr().out


def test_sans_gain_ni_besoin_rien_ne_change(monkeypatch):
    """Carte large : la réserve pleine tient et n'exile rien de plus que la plafonnée → pas de plafond (a5v inchangé)."""
    spec, man, planifier = _preparer(monkeypatch, libre=60 * G, cap=64 * G)
    assert not LD._plafonner_mlp_prefill(spec, 32768, planifier(), planifier, man)
    assert spec.mlp_prefill_plafond is None


def test_a_29_5_gio_plus_d_oscillation_exil_remontee(monkeypatch):
    """29,0 et 30,0 servaient, 29,5 refusait pour « 1 Mio manquants » : les deux marges (exil, borne) sont sur la même base."""
    spec, man, planifier = _preparer(monkeypatch, libre=int(29.5 * G))
    with contextlib.redirect_stdout(io.StringIO()):
        LD._plafonner_mlp_prefill(spec, 32768, planifier(), planifier, man)
    p = planifier()
    assert _refuse(spec, man, p, 32768) is None
    assert 50 <= LD._mlp_exiles(p) <= 60


def test_a_28_6_gio_la_branche_plancher_tenu_sert(monkeypatch):
    """La seule case du balayage que la nouvelle branche change : 28,6 Gio, 32 768 — a5v seul n'exile pas moins (60 = 60)
    et refuse ; la réserve plafonnée fait tenir le plancher → plafond posé, sert (60/60 exilés, le KV occupe la place)."""
    spec, man, planifier = _preparer(monkeypatch, libre=int(28.6 * G))
    with contextlib.redirect_stdout(io.StringIO()):
        assert LD._plafonner_mlp_prefill(spec, 32768, planifier(), planifier, man)
    assert spec.mlp_prefill_plafond == 4096
    assert _refuse(spec, man, planifier(), 32768) is None
