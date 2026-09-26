"""Ajout GUI n°3 (poste7-gui-ajouts-18-09 § 3), corrigé le 18/09
(poste7-metrics-energie-fenetre-18-09) : `AnneauEnergie` intègre le J/jeton
sur une fenêtre glissante alimentée par un fil de fond, et `j_par_jeton()`
ne CONSOMME JAMAIS ce qu'il lit — la première version faisait diverger
deux lecteurs concurrents de `/metrics` sur la même fenêtre réelle (trouvé
par poste3 au contrôle GUI). Recette : à vide « — » (None), jamais 0."""

import json
import os

import pytest
import torch

from acvram.server import capteurs

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402


def _anneau(mj_lecteur):
    return capteurs.AnneauEnergie(fenetre_s=10.0, tic_s=999.0, lecteur=mj_lecteur)


def test_vide_sans_assez_d_echantillons():
    a = _anneau(lambda: {0: 1_000_000})
    assert a.j_par_jeton() is None
    a.tic(100)
    assert a.j_par_jeton() is None   # un seul echantillon : pas de fenetre


def test_ratio_calcule_sur_deux_tics():
    valeurs = iter([{0: 1_000_000}, {0: 1_050_000}])
    a = _anneau(lambda: next(valeurs))
    a.tic(100)
    a.tic(150)
    # delta 50 000 mJ = 50 J, delta 50 jetons -> 1,0 J/jeton
    assert a.j_par_jeton() == 1.0


def test_lecture_ne_consomme_pas():
    """Le bras cassant : deux lecteurs concurrents de la MEME fenetre
    doivent rendre la MEME valeur, pas des valeurs qui divergent parce
    que le premier aurait « mangé » l'échantillon du second."""
    valeurs = iter([{0: 2_000_000}, {0: 2_100_000}])
    a = _anneau(lambda: next(valeurs))
    a.tic(200)
    a.tic(250)
    premiere_lecture = a.j_par_jeton()
    deuxieme_lecture = a.j_par_jeton()
    assert premiere_lecture is not None
    assert premiere_lecture == deuxieme_lecture == 2.0


def test_fenetre_purge_les_echantillons_trop_vieux(monkeypatch):
    # fenetre 10 s : au 3e tic (t=10,9), limite = 0,9 -> purge t=0, garde t=1
    horloge = iter([0.0, 1.0, 10.9])
    monkeypatch.setattr(capteurs.time, "time", lambda: next(horloge))
    valeurs = iter([{0: 1_000_000}, {0: 1_010_000}, {0: 1_040_000}])
    a = _anneau(lambda: next(valeurs))
    a.tic(10)    # t=0
    a.tic(20)    # t=1  -> delta 10 000 mJ / 10 jetons = 1,0 J/jeton
    assert a.j_par_jeton() == 1.0
    a.tic(50)    # t=10,9 -> purge t=0, garde t=1 et t=10,9
    # delta 30 000 mJ / 30 jetons = 1,0 J/jeton, sur la fenetre 1..10,9
    assert a.j_par_jeton() == 1.0


def test_absent_sans_nvml_ni_jetons():
    a = _anneau(lambda: {})   # NVML absent
    a.tic(10)
    a.tic(20)
    assert a.j_par_jeton() is None
    a2 = _anneau(lambda: {0: 1_000_000})
    a2.tic(10)
    a2.tic(10)   # aucun jeton decode entre les deux tics
    assert a2.j_par_jeton() is None


def test_jetons_fenetre_accompagne_toujours_j_par_jeton():
    """poste7 (poste7-metrics-energie-fenetre-18-09) : `jetons_fenetre()` = 0
    est la SEULE raison publiee de None — les deux se lisent ensemble."""
    a = _anneau(lambda: {0: 1_000_000})
    assert a.jetons_fenetre() == 0   # pas assez d'echantillons
    a.tic(10)
    assert a.jetons_fenetre() == 0   # un seul echantillon
    a.tic(10)   # meme mj, meme jetons
    assert a.jetons_fenetre() == 0 and a.j_par_jeton() is None


def test_jetons_fenetre_egale_le_delta():
    valeurs = iter([{0: 5_000_000}, {0: 5_010_000}])
    a = _anneau(lambda: next(valeurs))
    a.tic(1000)
    a.tic(1042)
    assert a.jetons_fenetre() == 42
    assert a.j_par_jeton() == round(10_000 / 1000.0 / 42, 4)


def test_sous_charge_continue_jamais_none():
    """Deuxieme test de la decision : un lecteur qui tombe toutes les
    100 ms sous une charge continue ne doit jamais voir None une fois la
    fenetre amorcee — meme si son passage tombe juste apres un tic."""
    mj = [0]
    a = capteurs.AnneauEnergie(fenetre_s=10.0, tic_s=1.0,
                                lecteur=lambda: {0: mj[0]})
    tokens = 0
    a.tic(tokens)   # amorce
    for _ in range(20):
        mj[0] += 10_000        # 10 J par tic
        tokens += 5
        a.tic(tokens)
        assert a.j_par_jeton() is not None   # jamais None une fois amorce
        assert a.jetons_fenetre() > 0


@pytest.fixture(scope="module")
def client(converted):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.server.app import create_app
    from acvram.server.chat import load_tokenizer

    vocab = {f"tok{i}": i for i in range(1024)}
    for i, w in enumerate(["<|im_start|>", "<|im_end|>", "hello", "world",
                           "the", "a", "of", "and", "</s>"]):
        vocab[w] = 1000 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(converted, "tokenizer.json"))
    json.dump({"eos_token": "</s>", "bos_token": "<|im_start|>",
               "chat_template": "{% for m in messages %}<|im_start|>{{m['role']}}\n"
                                "{{m['content']}}<|im_end|>\n{% endfor %}"
                                "{% if add_generation_prompt %}"
                                "<|im_start|>assistant\n{% endif %}"},
              open(os.path.join(converted, "tokenizer_config.json"), "w"))

    loaded = load_model(converted, dtype=torch.float32, device_override="cpu",
                        max_concurrent_seqs=4)
    loaded.plan.kv_planned_seqs = 4   # cf. test_server.py : pas de GPU ici, le kwarg est ignoré
    tokenizer = load_tokenizer(converted)
    engine = Engine(loaded, tokenizer, max_batch_size=4, max_model_len=256)
    with TestClient(create_app(engine, tokenizer, "tiny")) as c:
        yield c, engine


def test_metrics_expose_j_par_jeton_10s(client, monkeypatch):
    """Integration : /metrics rend le champ que la console lit, calcule a
    partir de l'anneau reel du serveur (fil de fond neutralise par un
    lecteur controle, pas de materiel requis)."""
    c, engine = client
    anneau = c.app.state.anneau_energie
    anneau.stop()   # le fil de fond reel tique toutes les 1 s : l'arreter
    valeurs = iter([{0: 3_000_000}, {0: 3_020_000}])
    monkeypatch.setattr(anneau, "_lecteur", lambda: next(valeurs))
    anneau._echantillons.clear()
    anneau.tic(300)
    anneau.tic(320)
    m = c.get("/metrics").json()
    assert m["energie"]["j_par_jeton_10s"] == 1.0
    assert m["energie"]["fenetre_s"] == 10.0
    assert m["energie"]["jetons_fenetre"] == 20
    assert "cartes" in m["energie"]
