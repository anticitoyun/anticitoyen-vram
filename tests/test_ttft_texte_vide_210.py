"""Pièce 210 : `ttft-service-p145` compte le premier jeton même quand son texte est vide. Réel (mixte servi,
invite(512002, 512), max_tokens=1) : un seul fragment SSE `{"text": "", "finish_reason": "length"}`, usage
`completion_tokens` = 1 — le service a généré le jeton ; l'outil exigeait un texte non vide et levait « aucun jeton
reçu ». Faux flux SSE (httpx.MockTransport) : aucun serveur, aucune carte."""
import importlib.util
import json
import pathlib

import httpx
import pytest

OUTIL = pathlib.Path(__file__).resolve().parent.parent / "outils" / "gpu" / "mesure" / "ttft-service-p145.py"


@pytest.fixture
def outil(monkeypatch):
    monkeypatch.setenv("TTFT_URL", "http://faux:1")
    spec = importlib.util.spec_from_file_location("ttft_service_p145", OUTIL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _client(fragments):
    corps = "".join(f"data: {json.dumps(f)}\n\n" for f in fragments) + "data: [DONE]\n\n"
    return httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, text=corps, headers={"content-type": "text/event-stream"})))


VIDE = {"choices": [{"index": 0, "text": "", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 512, "completion_tokens": 1, "total_tokens": 513}}


def test_un_jeton_au_texte_vide_est_un_premier_jeton(outil):
    outil.VIDES[0] = 0
    t, err = outil.ttft(_client([VIDE]), list(range(10, 522)))
    assert t > 0 and err == 0
    assert outil.VIDES[0] == 1, "le jeton au texte vide doit être compté dans jetons_texte_vide"


def test_temoin_l_ancien_critere_levait(outil, monkeypatch):
    """Le critère d'avant (texte non vide) sur le même flux : « aucun jeton reçu ». Sans ce témoin, le test ci-dessus
    passerait aussi sur un outil qui ne regarde pas le texte pour une autre raison."""
    monkeypatch.setattr(outil, "premier_fragment",
                        lambda d: bool(d.get("choices")) and bool(d["choices"][0].get("text")))
    with pytest.raises(RuntimeError, match="aucun jeton reçu"):
        outil.ttft(_client([VIDE]), list(range(10, 522)))


def test_un_flux_sans_choix_reste_une_erreur(outil):
    with pytest.raises(RuntimeError, match="aucun jeton reçu"):
        outil.ttft(_client([{"usage": {"prompt_tokens": 512, "completion_tokens": 0}}]), [1, 2, 3])
