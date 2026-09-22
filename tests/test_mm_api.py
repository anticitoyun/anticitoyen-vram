"""Pièce (b) P2 API multimodale, à sec (sage-go-multimodal-organisation-20-09 § 2).

Trois états pour un ``image_url``, jamais un silence :
1. alias texte (manifeste sans ``vision: oui``) : 400 nommé « sans tour de vision » ;
2. alias vision : accepté, la requête interne porte ``images=[ImageFragment]``
   avec la plage [debut, fin) des N jetons image, comptés dans max_model_len ;
3. source hors périmètre (file:// hors ACVRAM_IMAGES_DIR, http:// non local) :
   refus nommé.
Témoin : une requête texte pure rend exactement les jetons d'avant et ne
touche jamais le processeur. Les processeurs sont factices : ni transformers
ni Pillow ne sont requis.
"""

import base64
import json
import os
import shutil

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402

from acvram.engine.images import ImageFragment, sha256_pixel_values   # noqa: E402

N_IMG = 8                   # jetons image par image, processeur factice
OCTETS_A = b"\x89PNG-image-A" + bytes(range(40))
OCTETS_B = b"\x89PNG-image-B" + bytes(range(40, 80))


def _data_url(octets: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(octets).decode()


def _ecrire_tokenizer(dossier: str) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    # 1023 entrées, puis <image> ajouté comme jeton spécial : id 1023, dans le
    # vocabulaire du modèle tiny (1024) — le moteur sans tour de vision
    # (pièce (c) à venir) doit pouvoir plonger ce jeton comme un autre.
    vocab = {f"tok{i}": i for i in range(1014)}
    for i, w in enumerate(["<|im_start|>", "<|im_end|>", "hello", "world",
                           "the", "a", "of", "and", "</s>"]):
        vocab[w] = 1014 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.add_special_tokens(["<image>"])
    tok.save(os.path.join(dossier, "tokenizer.json"))
    gabarit = ("{% for m in messages %}<|im_start|>{{m['role']}}\n"
               "{% if m['content'] is string %}{{m['content']}}{% else %}"
               "{% for p in m['content'] %}{% if p['type'] == 'image' %}<image>"
               "{% else %}{{p['text']}}{% endif %}{% endfor %}{% endif %}"
               "<|im_end|>\n{% endfor %}"
               "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    json.dump({"eos_token": "</s>", "bos_token": "<|im_start|>", "chat_template": gabarit},
              open(os.path.join(dossier, "tokenizer_config.json"), "w"))


class FauxProcesseur:
    """Ce qu'un AutoProcessor rend : input_ids avec chaque <image> expansé en
    N jetons image, pixel_values [n_images, 3, 2, 2] dérivés des octets."""

    def __init__(self, tokenizer, n: int, image_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.n = n
        self.image_token_id = image_token_id
        self.appels = 0

    def __call__(self, text, images, return_tensors="pt", add_special_tokens=False):
        self.appels += 1
        ids, img = [], self.image_token_id
        for t in self.tokenizer.encode(text):
            ids.extend([img] * self.n if t == img else [t])
        pv = torch.stack([torch.tensor([b / 255.0 for b in o[:12]]).reshape(3, 2, 2)
                          for o in images])
        # annexes façon Gemma4Processor (Manon 14 h 40) : image_position_ids [n_images, patches, 2] que
        # get_image_features exige, mm_token_type_ids [1, T] qui n'est PAS une entrée de la tour
        pos = torch.stack([torch.tensor([[i, p] for p in range(4)]) for i in range(len(images))])
        return {"input_ids": torch.tensor([ids]), "pixel_values": pv, "image_position_ids": pos,
                "mm_token_type_ids": torch.tensor([[1 if t == img else 0 for t in ids]])}


class _Banc:
    def __init__(self, client, engine, tokenizer, service, processeur, appels, img):
        self.client, self.engine, self.tokenizer = client, engine, tokenizer
        self.service, self.processeur, self.appels = service, processeur, appels
        self.img = img                               # id du jeton <image> tel qu'encodé


def _monter(converted, dossier: str, vision: bool, n_img: int = N_IMG):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.server.app import create_app
    from acvram.server.chat import ProcesseurVision, load_tokenizer

    shutil.copytree(converted, dossier)
    _ecrire_tokenizer(dossier)
    man_path = os.path.join(dossier, "acvram_manifest.json")
    man = json.load(open(man_path, encoding="utf-8"))
    if vision:
        man["vision"] = "oui"
    else:
        man.pop("vision", None)
    json.dump(man, open(man_path, "w", encoding="utf-8"))

    loaded = load_model(dossier, dtype=torch.float32, device_override="cpu",
                        max_concurrent_seqs=4)
    loaded.plan.kv_planned_seqs = 4
    tokenizer = load_tokenizer(dossier)
    (img,) = tokenizer.encode("<image>")             # un seul jeton, spécial
    assert img == 1023
    # Pièce (c) fusionnée (19e4d525) : Engine construit la tour (TourVision.depuis_dossier) dès que le
    # manifeste dit vision: oui — sans transformers à sec, une tour factice qui rend n_img traits
    from acvram.engine.vision import TourVision
    depuis_dossier = TourVision.__dict__["depuis_dossier"]
    h = loaded.spec.hidden_size
    if vision:
        TourVision.depuis_dossier = classmethod(
            lambda cls, path, man, dev: cls(lambda pv: torch.zeros(1, n_img, h), dev, nom="factice"))
    if vision:   # famille du masque des plages image (20/09, masque par famille) : le jouet Llama n en a pas
        loaded.spec.raw = {**(getattr(loaded.spec, "raw", None) or {}), "architectures": ["Gemma4ForConditionalGeneration"]}
    try:
        engine = Engine(loaded, tokenizer, max_batch_size=4, max_model_len=256)
    finally:
        TourVision.depuis_dossier = depuis_dossier
    appels = []                                  # (prompt_ids, images) reçus par le moteur
    original = engine.add_request

    def espion(prompt_ids, params, request_id="", images=None):
        seq = (original(prompt_ids, params, request_id, images=images) if images is not None
               else original(prompt_ids, params, request_id))
        appels.append((list(prompt_ids), seq.images))
        return seq
    engine.add_request = espion

    app = create_app(engine, tokenizer, "tiny-vl" if vision else "tiny")
    service = app.state.service
    processeur = FauxProcesseur(tokenizer, n_img, img)
    if vision:
        service._processeur_vision = ProcesseurVision(processeur, img, ouvrir=lambda b: b)
    else:
        def _jamais():
            raise ValueError("processeur chargé sur un alias texte")   # -> 400, jamais 500
        service.processeur_vision = _jamais
    client = TestClient(app)
    client.__enter__()
    return _Banc(client, engine, tokenizer, service, processeur, appels, img)


@pytest.fixture(scope="module")
def texte(converted, tmp_path_factory):
    b = _monter(converted, str(tmp_path_factory.mktemp("mm") / "texte"), vision=False)
    yield b
    b.client.__exit__(None, None, None)


@pytest.fixture(scope="module")
def vision(converted, tmp_path_factory):
    b = _monter(converted, str(tmp_path_factory.mktemp("mm") / "vision"), vision=True)
    yield b
    b.client.__exit__(None, None, None)


def _requete(*contenu, max_tokens=2):
    return {"model": "tiny", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": list(contenu)}]}


def _texte(s):
    return {"type": "text", "text": s}


def _image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def _message(r) -> str:
    corps = r.json()
    return corps.get("detail") or corps.get("error", {}).get("message") or r.text


# -- 1. alias texte ------------------------------------------------------------
def test_image_sur_alias_texte_400_nomme(texte):
    avant = len(texte.appels)
    r = texte.client.post("/v1/chat/completions",
                          json=_requete(_texte("hello"), _image(_data_url(OCTETS_A))))
    assert r.status_code == 400, r.text
    assert "sans tour de vision" in _message(r)
    assert "tiny" in _message(r)
    assert len(texte.appels) == avant, "rien ne doit atteindre le moteur"


def test_le_test_sait_dire_faux(texte, monkeypatch):
    """Si la garde du manifeste disait oui sur l'alias texte, le 400 « sans
    tour de vision » disparaîtrait : c'est bien elle qui refuse."""
    monkeypatch.setattr(texte.service, "vision_servie", lambda: True)
    r = texte.client.post("/v1/chat/completions",
                          json=_requete(_texte("hello"), _image(_data_url(OCTETS_A))))
    assert "sans tour de vision" not in _message(r)


# -- 2. alias vision -----------------------------------------------------------
def test_image_sur_alias_vision_acceptee(vision):
    from acvram.server.chat import render_chat

    avant = len(vision.appels)
    r = vision.client.post("/v1/chat/completions",
                           json=_requete(_texte("hello"), _image(_data_url(OCTETS_A)),
                                         _texte("world")))
    assert r.status_code == 200, r.text
    assert len(vision.appels) == avant + 1
    prompt_ids, images = vision.appels[-1]
    assert images is not None and len(images) == 1
    frag = images[0]
    # le moteur (pièce c) reçoit l'ImageFragment et le porte en ImageRequete : mêmes attributs
    assert all(hasattr(frag, a) for a in ("debut", "fin", "pixel_values", "sha256"))
    assert frag.fin - frag.debut == N_IMG
    assert prompt_ids[frag.debut:frag.fin] == [vision.img] * N_IMG
    assert prompt_ids[frag.fin] != vision.img and prompt_ids[frag.debut - 1] != vision.img
    assert frag.sha256 == sha256_pixel_values(frag.pixel_values)
    # les annexes du processeur suivent l'image (supplement), découpées par image, sans les hors-tour
    assert set(frag.supplement) == {"image_position_ids"}, frag.supplement.keys()
    assert tuple(frag.supplement["image_position_ids"].shape) == (1, 4, 2)
    assert tuple(frag.pixel_values.shape) == (1, 3, 2, 2)
    # N jetons comptés : l'invite rendue porte un <image>, expansé en N
    gabarit = render_chat(vision.tokenizer, [{"role": "user", "content": [
        _texte("hello"), {"type": "image"}, _texte("world")]}])
    sans = vision.tokenizer.encode(gabarit)
    assert sans.count(vision.img) == 1
    assert len(prompt_ids) == len(sans) - 1 + N_IMG
    assert r.json()["usage"]["prompt_tokens"] == len(prompt_ids)


def test_deux_images_deux_fragments_deux_sha(vision):
    r = vision.client.post("/v1/chat/completions",
                           json=_requete(_image(_data_url(OCTETS_A)), _texte("and"),
                                         _image(_data_url(OCTETS_B))))
    assert r.status_code == 200, r.text
    prompt_ids, images = vision.appels[-1]
    assert [f.fin - f.debut for f in images] == [N_IMG, N_IMG]
    assert [int(f.supplement["image_position_ids"][0, 0, 0]) for f in images] == [0, 1]   # chaque image SES positions
    assert images[0].fin <= images[1].debut
    assert images[0].sha256 != images[1].sha256
    assert prompt_ids.count(vision.img) == 2 * N_IMG


def test_n_jetons_image_au_dela_de_max_model_len(vision):
    vision.processeur.n = 300                    # > max_model_len 256
    try:
        avant = len(vision.appels)
        r = vision.client.post("/v1/chat/completions",
                               json=_requete(_texte("hello"), _image(_data_url(OCTETS_A))))
        assert r.status_code == 400, r.text
        assert "jetons image" in _message(r) and "max_model_len" in _message(r)
        assert len(vision.appels) == avant
    finally:
        vision.processeur.n = N_IMG


# -- 3. sources ----------------------------------------------------------------
def test_file_hors_dossier_refuse(vision, tmp_path, monkeypatch):
    dossier = tmp_path / "images"
    dossier.mkdir()
    (dossier / "ok.png").write_bytes(OCTETS_A)
    (tmp_path / "dehors.png").write_bytes(OCTETS_B)
    monkeypatch.setenv("ACVRAM_IMAGES_DIR", str(dossier))

    r = vision.client.post("/v1/chat/completions",
                           json=_requete(_image(f"file://{tmp_path}/dehors.png")))
    assert r.status_code == 400 and "hors du dossier" in _message(r), r.text
    r = vision.client.post("/v1/chat/completions",
                           json=_requete(_image(f"file://{dossier}/../dehors.png")))
    assert r.status_code == 400 and ".." in _message(r), r.text
    r = vision.client.post("/v1/chat/completions",
                           json=_requete(_image(f"file://{dossier}/ok.png")))
    assert r.status_code == 200, r.text
    assert vision.appels[-1][1][0].sha256 == sha256_pixel_values(
        torch.tensor([b / 255.0 for b in OCTETS_A[:12]]).reshape(1, 3, 2, 2))

    monkeypatch.delenv("ACVRAM_IMAGES_DIR")
    r = vision.client.post("/v1/chat/completions",
                           json=_requete(_image(f"file://{dossier}/ok.png")))
    assert r.status_code == 400 and "ACVRAM_IMAGES_DIR" in _message(r), r.text


def test_http_non_local_refuse(vision):
    for url in ("http://example.org/a.png", "https://10.0.0.7/a.png",
                "http://127.0.0.1.evil.net/a.png"):
        r = vision.client.post("/v1/chat/completions", json=_requete(_image(url)))
        assert r.status_code == 400, r.text
        assert "127.0.0.1" in _message(r) and "refusée" in _message(r), r.text


def test_schema_inconnu_et_data_non_image_refuses(vision):
    r = vision.client.post("/v1/chat/completions", json=_requete(_image("ftp://x/a.png")))
    assert r.status_code == 400 and "ftp" in _message(r)
    r = vision.client.post("/v1/chat/completions",
                           json=_requete(_image("data:text/plain;base64,aGVsbG8=")))
    assert r.status_code == 400 and "image/*" in _message(r)


# -- 4. témoin texte -----------------------------------------------------------
@pytest.mark.parametrize("banc", ["texte", "vision"])
def test_texte_pur_jetons_inchanges(banc, request):
    from acvram.server.chat import render_chat

    b = request.getfixturevalue(banc)
    appels_proc = b.processeur.appels
    for contenu in ("hello world", [_texte("hello "), _texte("world")]):
        r = b.client.post("/v1/chat/completions", json={
            "model": "tiny", "max_tokens": 2, "temperature": 0,
            "messages": [{"role": "user", "content": contenu}]})
        assert r.status_code == 200, r.text
        prompt_ids, images = b.appels[-1]
        assert not images                      # None (API) ou [] (Sequence, pièce c) : aucun fragment
        attendu = b.tokenizer.encode(render_chat(
            b.tokenizer, [{"role": "user", "content": "hello world"}]))
        assert prompt_ids == attendu
        assert b.img not in prompt_ids
    assert b.processeur.appels == appels_proc, "le chemin texte ne touche pas le processeur"
