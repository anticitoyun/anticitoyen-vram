"""Multimodal P1, pièce (c) moteur (sage-go-multimodal-organisation-20-09 § 2),
à sec, formes réduites, sans transformers réel.

(1) dispersion des traits d'image au seul site d'embedding, au bit ;
(2) masque bidirectionnel sur la plage image, causal ailleurs, y compris une
    plage qui chevauche une frontière de morceau de prefill ;
(3) clé du cache de préfixe = jetons + sha256 des images ;
(4) une image sur un modèle sans tour → refus nommé ;
(5) le contrôle sait dire FAUX : une dispersion décalée d'un jeton est vue.
"""

import math
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from acvram.engine import layers as L
from acvram.engine.config import ModelSpec
from acvram.engine.model import ACVRamModel, ForwardBatch, disperser_images
from acvram.engine.runner import Engine, Sequence
from acvram.engine.vision import (ImageRequete, SansTourVision, TourVision,
                                  regime_texte)
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
from acvram.engine.sampler import SamplingParams

H, V = 8, 32


# --------------------------------------------------------------------------
# (1) et (5) dispersion
# --------------------------------------------------------------------------
class _Identite(nn.Module):
    """Une norme finale qui rend x tel quel : `forward(return_hidden=True)`
    livre alors exactement les embeddings (dispersés ou non)."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H), requires_grad=False)

    def forward(self, x):
        return x


def _modele_sans_couche(multiplier=1.0):
    spec = ModelSpec(name="t", architecture="gemma", hidden_size=H,
                     intermediate_size=H, num_layers=0, num_attention_heads=1,
                     num_key_value_heads=1, vocab_size=V, max_position_embeddings=64,
                     embedding_multiplier=multiplier)
    g = torch.Generator().manual_seed(0)
    embed = torch.randn(V, H, generator=g)
    return ACVRamModel(spec, embed, [], _Identite(), None, {}, dtype=torch.float32)


def _lot(jetons, images=None, offset=0):
    n = len(jetons)
    return ForwardBatch(tokens=torch.tensor(jetons), positions=torch.arange(offset, offset + n),
                        seq_lens=[offset + n], query_lens=[n],
                        block_tables=[torch.zeros(4, dtype=torch.long)],
                        slot_mapping=torch.arange(n), is_prefill=True, images=images)


def test_dispersion_au_bit_et_texte_inchange():
    m = _modele_sans_couche(multiplier=3.0)
    jetons = [1, 2, 3, 4, 5, 6, 7, 8]
    debut, fin = 2, 5
    embeds = torch.full((fin - debut, H), 7.5).to(torch.bfloat16)

    sans = m(_lot(jetons), return_hidden=True)
    attendu_texte = F.embedding(torch.tensor(jetons), m.embed_tokens) * 3.0
    assert torch.equal(sans, attendu_texte)                      # texte : l'ancien résultat
    assert torch.equal(m(_lot(jetons), return_hidden=True), sans)  # images=None : au bit

    avec = m(_lot(jetons, images=[[(debut, fin, embeds)]]), return_hidden=True)
    attendu = attendu_texte.clone()
    attendu[debut:fin] = embeds.float()                          # pas de multiplicateur sur l'image
    assert torch.equal(avec, attendu)
    assert torch.equal(avec[:debut], sans[:debut]) and torch.equal(avec[fin:], sans[fin:])

    # (5) faute construite : une dispersion décalée d'un jeton doit être vue
    faux = attendu_texte.clone()
    faux[debut + 1:fin + 1] = embeds.float()
    assert not torch.equal(avec, faux)


def test_dispersion_sur_un_morceau_ne_prend_que_l_intersection():
    """Prefill par morceaux : le morceau [4, 10) d'une invite dont l'image est
    [2, 7) reçoit les lignes 4..6 des traits, rien d'autre."""
    embeds = torch.arange(5 * H, dtype=torch.float32).reshape(5, H).to(torch.bfloat16)
    x = torch.zeros(6, H)
    b = ForwardBatch(tokens=torch.zeros(6, dtype=torch.long), positions=torch.arange(4, 10),
                     seq_lens=[10], query_lens=[6], block_tables=[], slot_mapping=torch.arange(6),
                     is_prefill=True, images=[[(2, 7, embeds)]])
    y = disperser_images(x, b)
    assert torch.equal(y[:3], embeds[2:5].float())
    assert torch.equal(y[3:], torch.zeros(3, H))
    with pytest.raises(ValueError, match="forme"):
        disperser_images(torch.zeros(6, H), ForwardBatch(
            tokens=torch.zeros(6, dtype=torch.long), positions=torch.arange(6), seq_lens=[6],
            query_lens=[6], block_tables=[], slot_mapping=torch.arange(6), is_prefill=True,
            images=[[(0, 3, torch.zeros(4, H))]]))


def test_dispersion_deux_sequences_dans_le_lot():
    embeds_a = torch.full((2, H), 1.0)
    embeds_b = torch.full((3, H), 2.0)
    x = torch.zeros(9, H)
    b = ForwardBatch(tokens=torch.zeros(9, dtype=torch.long), positions=torch.zeros(9, dtype=torch.long),
                     seq_lens=[4, 5], query_lens=[4, 5], block_tables=[], slot_mapping=torch.arange(9),
                     is_prefill=True, images=[[(1, 3, embeds_a)], [(0, 3, embeds_b)]])
    y = disperser_images(x, b)
    assert torch.equal(y[1:3], embeds_a) and torch.equal(y[4:7], embeds_b)
    assert y[0].abs().sum() == 0 and y[3].abs().sum() == 0 and y[7:].abs().sum() == 0


# --------------------------------------------------------------------------
# (2) masque
# --------------------------------------------------------------------------
def _masque_reference(n, plages, window=0):
    """Construit à la main : causal (et fenêtre), plus tout-à-tout DANS une
    même plage image."""
    ok = torch.zeros(n, n, dtype=torch.bool)
    for r in range(n):
        for c in range(n):
            if c <= r and (window == 0 or c > r - window):
                ok[r, c] = True
            for d, f in plages:
                if d <= r < f and d <= c < f:
                    ok[r, c] = True
    return ok


def _attention_reference(q, k, v, ok, scale):
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    s = s.masked_fill(~ok[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v.float())


@pytest.mark.parametrize("window", [0, 4])
def test_masque_plage_image_bidirectionnelle_reste_causal(window):
    n, heads, d = 12, 2, 4
    plages = [(3, 7), (9, 11)]
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(n, heads, d, generator=g) for _ in range(3))
    scale = 1 / math.sqrt(d)
    ref = _attention_reference(q, k, v, _masque_reference(n, plages, window), scale)
    out = L.attention(q, k, v, True, scale, window=window, images=plages)
    assert torch.allclose(out, ref, atol=1e-5)
    # sans image : causal pur, et la référence causale seule DIFFÈRE de la
    # bidirectionnelle (le masque image change bien quelque chose)
    sans = L.attention(q, k, v, True, scale, window=window)
    assert torch.allclose(sans, _attention_reference(q, k, v, _masque_reference(n, [], window), scale), atol=1e-5)
    assert not torch.allclose(sans, ref, atol=1e-3)
    # masque booléen lui-même contre la référence
    ouvert = L.masque_images(n, n, 0, plages, q.device)
    rows = torch.arange(n).unsqueeze(1); cols = torch.arange(n).unsqueeze(0)
    assert torch.equal((cols <= rows) | ouvert, _masque_reference(n, plages))


def test_masque_plage_qui_chevauche_une_frontiere_de_morceau():
    """Invite de 12 jetons, image [5, 9), budget de morceau qui couperait à 7.
    Le moteur déplace la coupe (`_eviter_coupe_image`) ; les deux morceaux
    recomposent exactement l'attention d'un seul passage. Une coupe laissée
    DANS l'image est refusée, nommément — jamais un masque faux en silence."""
    n, heads, d = 12, 1, 4
    plages = [(5, 9)]
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(n, heads, d, generator=g) for _ in range(3))
    scale = 1 / math.sqrt(d)
    ref = _attention_reference(q, k, v, _masque_reference(n, plages), scale)

    seq = Sequence(list(range(n)), SamplingParams())
    seq.images = [ImageRequete(5, 9, None, "a")]
    seq.prefill_len = 0
    coupe = Engine._eviter_coupe_image(seq, 7)
    assert coupe == 5                          # reculée au début de l'image
    seq.prefill_len = 5
    assert Engine._eviter_coupe_image(seq, 7) == 9   # image déjà entamée : jusqu'à sa fin
    assert Engine._eviter_coupe_image(seq, 3) == 3   # hors image : inchangé
    assert Engine._eviter_coupe_image(seq, None) is None

    morceaux = [(0, coupe), (coupe, n)]
    out = torch.cat([L.attention(q[a:b], k[:b], v[:b], True, scale, q_offset=a, images=plages)
                     for a, b in morceaux])
    assert torch.allclose(out, ref, atol=1e-5)

    with pytest.raises(RuntimeError, match=r"plage image \[5, 9\) coupée"):
        L.attention(q[:7], k[:7], v[:7], True, scale, q_offset=0, images=plages)
    # le second morceau d'une coupe à 7 serait lui aussi faux : refusé
    with pytest.raises(RuntimeError, match="coupée"):
        L.masque_images(2, 7, 5, plages, q.device)
    # une image entièrement AVANT le morceau ne gêne pas (préfixe servi)
    assert L.masque_images(3, 12, 9, plages, q.device) is None


def test_frontiere_insta_dans_une_image_est_abandonnee(monkeypatch):
    """La frontière d'instantané (multiple du pas) qui tombe dans une image
    n'est pas déplacée mais abandonnée : `_eviter_coupe_image` la change,
    donc `_step` la met à None (lu dans le code : coupe = None)."""
    seq = Sequence(list(range(600)), SamplingParams())
    seq.images = [ImageRequete(200, 300, None, "x")]
    assert Engine._eviter_coupe_image(seq, 256) != 256


# --------------------------------------------------------------------------
# (3) clé du cache de préfixe
# --------------------------------------------------------------------------
def test_cle_de_prefixe_porte_le_sha256_des_images():
    ids = list(range(3 * BLOCK_SIZE))
    plage = (BLOCK_SIZE + 2, BLOCK_SIZE + 6)            # dans le bloc 1 seulement
    texte = BlockAllocator.block_hashes(ids, BLOCK_SIZE)
    a = BlockAllocator.block_hashes(ids, BLOCK_SIZE, images=[(*plage, "sha-A")])
    b = BlockAllocator.block_hashes(ids, BLOCK_SIZE, images=[(*plage, "sha-B")])
    a2 = BlockAllocator.block_hashes(ids, BLOCK_SIZE, images=[ImageRequete(*plage, None, "sha-A")])
    assert texte == BlockAllocator.block_hashes(ids, BLOCK_SIZE, images=[])   # texte seul : clé d'avant
    assert a == a2                                                            # même image → même clé
    assert a[0] == texte[0]                                                   # bloc avant l'image : inchangé
    assert a[1] != b[1] and a[1] != texte[1]                                  # deux images ≠ → deux clés
    assert a[2] != b[2]                                                       # et le chaînage propage
    # l'image à cheval sur deux blocs entre dans les deux
    c = BlockAllocator.block_hashes(ids, BLOCK_SIZE, images=[(BLOCK_SIZE - 2, BLOCK_SIZE + 2, "s")])
    assert c[0] != texte[0] and c[1] != texte[1]
    assert BlockAllocator.hash_bloc(0, (1, 2)) == hash((0, (1, 2)))


def test_cle_de_prefixe_du_runner_suit_la_meme_regle():
    """`_register_complete_blocks` recalcule les hachages bloc à bloc : ils
    doivent être ceux de `block_hashes` avec les mêmes images."""
    n = 2 * BLOCK_SIZE
    seq = Sequence(list(range(n)), SamplingParams())
    seq.images = [ImageRequete(3, 5, None, "sha-Z")]
    seq.blocks = [7, 8]
    seq.prefill_len = n
    enregistres = []

    class _Alloc:
        def register(self, blk, h): enregistres.append((blk, h))

    class _Moteur:
        allocator = _Alloc()
    Engine._register_complete_blocks(_Moteur(), seq)
    attendu = BlockAllocator.block_hashes(seq.prompt_ids, BLOCK_SIZE, images=seq.images)
    assert [h for _, h in enregistres] == attendu
    assert attendu != BlockAllocator.block_hashes(seq.prompt_ids, BLOCK_SIZE)


# --------------------------------------------------------------------------
# (4) refus nommé, frontière (b), tour factice, régime
# --------------------------------------------------------------------------
def test_image_sur_modele_sans_tour_refus_nomme(converted):
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    assert loaded.manifest.get("vision") in (None, "non", False)
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256)
    assert engine.vision is None
    with pytest.raises(SansTourVision, match="sans tour de vision"):
        engine.add_request([1, 2, 3, 4], SamplingParams(max_tokens=1),
                           images=[(1, 3, torch.zeros(3, 4, 4), "sha")])
    assert not engine.waiting                     # rien n'est entré en file


class _Frag:
    def __init__(self, debut, fin, pixel_values, sha256):
        self.debut, self.fin, self.pixel_values, self.sha256 = debut, fin, pixel_values, sha256


def test_frontiere_tolerante_et_plages_verifiees():
    im = ImageRequete.depuis(_Frag(2, 5, "pv", "abc"))
    assert (im.debut, im.fin, im.sha256) == (2, 5, "abc")
    assert ImageRequete.depuis((2, 5, "pv", "abc")) == im
    with pytest.raises(ValueError, match="attributs"):
        ImageRequete.depuis(object())
    with pytest.raises(ValueError, match="vide"):
        ImageRequete.depuis((5, 5, None, "s"))


def test_tour_factice_dans_le_runner_end_to_end(converted):
    """Une mini-tour (callable) posée sur le moteur : les traits arrivent dans
    le lot de prefill, aux bonnes lignes, et la clé de préfixe diffère entre
    deux images sur le même texte."""
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256, enable_cuda_graphs=False)
    h = loaded.spec.hidden_size
    appels = []

    def calcul(pv):
        appels.append(pv)
        return torch.full((1, int(pv), h), float(pv))    # [1, n, h] comme HF

    engine.vision = TourVision(calcul, torch.device("cpu"), nom="factice")
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Gemma4ForConditionalGeneration"]}   # famille du masque (20/09 19:01) : le jouet Llama n'en a pas
    assert regime_texte() == "vision=bf16(eager,factice)"
    vus = []
    orig = engine._build_batch

    def espion(seqs, prefill, limite=None):
        b = orig(seqs, prefill, limite)
        vus.append(b)
        return b
    engine._build_batch = espion
    prompt = list(range(1, 9))
    for _ in engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=2),
                             images=[(2, 5, torch.tensor(3.0), "sha-1")]):
        pass
    assert len(appels) == 1                                # une passe par image
    pre = [b for b in vus if b.is_prefill]
    assert pre and pre[0].images is not None
    d, f, e = pre[0].images[0][0]
    assert (d, f) == (2, 5) and e.dtype == torch.bfloat16 and e.shape == (3, h)
    assert all(b.images is None for b in vus if not b.is_prefill)   # jamais au décodage
    # tour en échec → refus nommé, pas de silence
    engine.vision = TourVision(lambda pv: torch.zeros(1, 99, h), torch.device("cpu"))
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Gemma4ForConditionalGeneration"]}   # famille du masque (20/09 19:01) : le jouet Llama n'en a pas
    sorties = list(engine.generate(prompt, SamplingParams(max_tokens=2),
                                   images=[(2, 5, torch.tensor(3.0), "sha-2")]))
    assert sorties and sorties[-1].finish_reason == "refus"


@pytest.fixture(autouse=True)
def _tour_oubliee():
    """`vision._CHARGEE` est un état de processus (la tour nommée sur la ligne de régime) : remis à
    zéro avant et après chaque test, sinon la tour factice d'un test nomme la ligne du suivant."""
    from acvram.engine import vision
    vision._CHARGEE = None
    yield
    vision._CHARGEE = None


def test_sans_transformers_un_alias_texte_se_charge_et_la_tour_le_nomme(converted, monkeypatch):
    """Sage 14 h 10 : transformers est une dépendance ÉPINGLÉE du moteur (5.17.0), importée
    paresseusement — un alias texte n'en importe rien au chargement ; une tour demandée sans
    transformers rend un refus nommé (pas un ImportError anonyme) ; la version relevée porte la ligne."""
    import sys
    from acvram.engine import vision
    monkeypatch.setitem(sys.modules, "transformers", None)          # import transformers → ImportError
    src = open(vision.__file__, encoding="utf-8").read().split("\n")
    tete = [l for l in src if l.startswith(("import ", "from ")) and "transformers" in l]
    assert not tete, f"import de transformers en tête de vision.py : {tete}"
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")    # alias texte : charge
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=64, enable_cuda_graphs=False)
    assert engine.vision is None and vision.regime_texte() == ""
    with pytest.raises(RuntimeError, match="transformers absent"):
        vision.TourVision.depuis_dossier(converted, {"vision": "oui"}, torch.device("cpu"))
    # la version relevée sur la ligne : le nom d'une tour réelle est « transformers <version> »
    vision.TourVision(lambda pv: pv, torch.device("cpu"), nom="transformers 5.17.0")
    assert vision.regime_texte() == "vision=bf16(eager,transformers=5.17.0)"


def test_les_annexes_du_processeur_suivent_l_image_jusqu_a_la_tour():
    """Manon 14 h 40 sur gemma-4-31B-it-nvfp4-vision : « 'bool' object has no attribute 'all' » —
    Gemma4Model.get_image_features(pixel_values, image_position_ids=…) exige les positions que le
    processeur rend à côté de pixel_values ; la tour ne les recevait pas. Ici une tour factice qui
    EXIGE image_position_ids : (1) une ImageRequete avec supplement les lui porte (kwargs, aucun nom
    Gemma en dur dans le moteur) ; (2) sans supplement elle rend une TypeError nommée (jamais un
    silence) ; (3) un tuple (frontière ancienne) porte un supplement vide ; (4) un objet à attributs
    porte le sien tel quel."""
    from acvram.engine.vision import ImageRequete, TourVision
    recus = []

    def calcul(pv, *, image_position_ids):
        recus.append(image_position_ids)
        return torch.zeros(1, int(image_position_ids.shape[1]), 8)
    tour = TourVision(calcul, torch.device("cpu"), nom="factice-gemma")
    pos = torch.tensor([[[0, 0], [0, 1], [0, 2]]])
    im = ImageRequete.depuis(type("Frag", (), {"debut": 2, "fin": 5, "pixel_values": torch.zeros(1, 4),
                                                "sha256": "s", "supplement": {"image_position_ids": pos}})())
    assert im.supplement.keys() == {"image_position_ids"}
    out = tour.traits(im.pixel_values, im.fin - im.debut, supplement=im.supplement)
    assert out.shape == (3, 8) and torch.equal(recus[0], pos)
    with pytest.raises(TypeError):
        tour.traits(im.pixel_values, 3)                                       # positions manquantes : nommé
    assert ImageRequete.depuis((2, 5, torch.zeros(1, 4), "s")).supplement == {}
    tour2 = TourVision(lambda pv: torch.zeros(1, 3, 8), torch.device("cpu"), nom="factice-sans")
    with pytest.raises(TypeError, match="refuse les annexes"):
        tour2.traits(im.pixel_values, 3, supplement={"image_position_ids": pos})   # tour sans kwargs : nommé


@pytest.mark.parametrize("forme,attendue", [((1, 2520, 768), (1, 2520, 768)),    # patches lotis Gemma 4 : inchangé
                                            ((3, 896, 896), (1, 3, 896, 896)),    # image CHW nue : dimension image
                                            ((1, 3, 896, 896), (1, 3, 896, 896)),  # déjà lotie : inchangé
                                            ((2, 2520, 768), (2, 2520, 768))])    # deux images de patches : inchangé
def test_la_forme_du_fragment_va_a_la_tour_telle_quelle(forme, attendue):
    """Manon 14 h 55 (3e essai de (c) sur gemma-4-31B) : `if pv.ndim == 3: unsqueeze(0)` faisait de
    [1, 2520, 768] (patches lotis) une image 4-D fausse. Le fragment garde la dimension image du
    processeur ; seule une image CHW nue [3, H, W] en reçoit une."""
    from acvram.engine.vision import forme_pour_la_tour
    out = forme_pour_la_tour(torch.zeros(*forme, dtype=torch.bfloat16))
    assert tuple(out.shape) == attendue


def test_ce_qui_entre_dans_le_lm_est_la_sortie_projetee_de_la_tour():
    """Gemma 4 (transformers 5.17) : get_image_features rend un ModelOutput — pooler_output = tuple par
    image des traits PROJETÉS (embed_vision, [n, hidden]), last_hidden_state = la tour brute [1, patches,
    1152] (références de Manon). Le moteur prend pooler_output ; un tenseur nu passe ; une tour qui rend
    la dimension brute au lieu de celle du LM est REFUSÉE nommément (jamais dispersée en silence)."""
    from acvram.engine.vision import TourVision, traits_projetes

    class Sortie:                                   # BaseModelOutputWithPooling minimal
        def __init__(self, lhs, po): self.last_hidden_state, self.pooler_output = lhs, po
    lhs = torch.zeros(1, 16, 1152); po = (torch.ones(4, 8), torch.full((2, 8), 2.0))
    out = traits_projetes(Sortie(lhs, po))
    assert out.shape == (6, 8) and out[0, 0] == 1 and out[5, 0] == 2
    assert traits_projetes(torch.zeros(3, 8)).shape == (3, 8)
    assert traits_projetes((torch.zeros(3, 8), None)).shape == (3, 8)
    tour = TourVision(lambda pv: Sortie(lhs, (torch.ones(4, 8),)), torch.device("cpu"), nom="factice", hidden=8)
    assert tour.traits(torch.zeros(1, 16, 3), 4).shape == (4, 8)
    brute = TourVision(lambda pv: Sortie(lhs, None), torch.device("cpu"), nom="factice-brute", hidden=8)
    with pytest.raises(ValueError, match="non projetée"):
        brute.traits(torch.zeros(1, 16, 3), 16)


def test_les_tampons_non_persistants_de_la_tour_sont_reconstruits():
    """20/09 14 h 48, tour réelle Gemma 4 : construite sur meta puis to_empty, `encoder.rotary_emb.inv_freq`
    (tampon NON persistant, absent du state_dict) restait en mémoire non initialisée — traits faux en
    silence (cos 0,13-0,53 contre la référence de Manon). Le sous-module qui porte de tels tampons est
    reconstruit par son constructeur (config, device=) ; un module sans constructeur connu est refusé."""
    import torch.nn as nn
    from acvram.engine.vision import rematerialiser_tampons

    class Cfg:
        base = 3.0

    class Rot(nn.Module):
        def __init__(self, config, device=None):
            super().__init__(); self.config = config
            self.inv_freq = nn.Buffer(torch.full((4,), config.base, device=device), persistent=False)

    class Tour(nn.Module):
        def __init__(self):
            super().__init__(); self.lin = nn.Linear(4, 4); self.enc = nn.Module(); self.enc.rot = Rot(Cfg())
    with torch.device("meta"):
        t = Tour()
    t.to_empty(device="cpu")
    t.enc.rot.inv_freq.fill_(-1.0)                          # la mémoire « non initialisée », rendue visible
    refaits = rematerialiser_tampons(t, torch.device("cpu"))
    assert refaits == ["enc.rot"] and torch.equal(t.enc.rot.inv_freq, torch.full((4,), 3.0))
    assert t.lin.weight.shape == (4, 4)                     # les paramètres (chargés ensuite) ne sont pas touchés

    class Sans(nn.Module):                                  # tampon non persistant, constructeur inconnu → refus
        def __init__(self, k):
            super().__init__(); self.b = nn.Buffer(torch.zeros(k), persistent=False)
    m = nn.Module(); m.s = Sans(2)
    with pytest.raises(RuntimeError, match="constructeur inconnu"):
        rematerialiser_tampons(m, torch.device("cpu"))


def test_la_ligne_du_moteur_porte_vision_et_sans_tour_une_image_est_refusee(converted, capsys):
    """Manon 15 h 00 sur le 31B : la ligne de régime de l'Engine ne portait pas « vision= » et le journal
    n'avait aucune ligne « tour ». (1) Engine.regime_ligne() dit vision=off sans tour et
    vision=bf16(eager,…) avec ; (2) add_request(images=) sur un Engine SANS tour → SansTourVision, jamais
    une réponse texte ; (3) avec tour, l'admission écrit « [engine] tour : … sha=… Σ=… » au journal."""
    from acvram.engine.loader import load_model
    from acvram.engine.vision import SansTourVision, TourVision
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=64, enable_cuda_graphs=False)
    assert engine.vision is None and " vision=off" in engine.regime_ligne()
    with pytest.raises(SansTourVision):
        engine.add_request([1, 2, 3, 4, 5, 6], SamplingParams(temperature=0.0, max_tokens=1),
                           images=[(1, 4, torch.tensor(3.0), "sha-x")])
    h = loaded.spec.hidden_size
    engine.vision = TourVision(lambda pv: torch.full((1, 3, h), 0.5), torch.device("cpu"), nom="factice")
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Gemma4ForConditionalGeneration"]}   # famille du masque (20/09 19:01)
    assert " vision=bf16(eager,factice)" in engine.regime_ligne()
    engine.add_request([1, 2, 3, 4, 5, 6], SamplingParams(temperature=0.0, max_tokens=1),
                       images=[(1, 4, torch.tensor(3.0), "sha-abcdef12")])
    engine.step()
    out = capsys.readouterr().out
    assert "[engine] tour : [1,4) (3, " in out and "sha=sha-abcd" in out


def test_les_prefixes_de_la_tour_couvrent_ceux_de_la_conversion():
    """20/09 17:48 (Qwen3-VL-2B servi dans un trou) : la conversion gardait model.visual.* (VISION_PREFIXES) mais le
    moteur (PREFIXES_TOUR) ne le connaissait pas → « tour déclarée mais aucun tenseur … dans le manifeste ». Tout
    préfixe que la conversion garde, le moteur sait le charger — les deux listes ne divergent plus en silence."""
    from acvram.engine.vision import PREFIXES_TOUR
    from acvram.quant.convert import VISION_PREFIXES
    manquent = sorted(set(VISION_PREFIXES) - set(PREFIXES_TOUR))
    assert not manquent, f"préfixes gardés par la conversion, inconnus du moteur : {manquent}"


def test_la_ligne_du_moteur_porte_mrope_et_deepstack_lus_sur_le_modele(converted, monkeypatch):
    """20/09 17:50 (Qwen3-VL-2B servi) : la ligne de l'Engine disait vision= et kv= mais ni mrope= ni deepstack=
    (seule la ligne de regime.py, depuis le manifeste, les avait). Ils viennent du MODÈLE chargé : mrope de
    spec.mrope_section / rope_scaling, deepstack de la tour (config de la tour, niveaux_deepstack) ; absents sinon."""
    from acvram.engine.loader import load_model
    from acvram.engine.vision import TourVision
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=64, enable_cuda_graphs=False)
    ligne = engine.regime_ligne()
    assert " mrope=" not in ligne and " deepstack=" not in ligne, ligne          # modèle texte 1-D, sans tour
    monkeypatch.setattr(type(loaded.spec), "mrope_section", property(lambda self: [24, 20, 20]), raising=False)
    monkeypatch.setattr(loaded.spec, "rope_scaling", {"mrope_section": [24, 20, 20], "mrope_interleaved": True}, raising=False)
    tour = TourVision(lambda pv: torch.zeros(1, 4, loaded.spec.hidden_size), torch.device("cpu"), nom="factice")
    tour.niveaux_deepstack = 3
    engine.vision = tour
    ligne = engine.regime_ligne()
    assert " mrope=[24,20,20](interleaved)" in ligne and " deepstack=3" in ligne, ligne


# ---- porte de famille du masque de la plage image (Sage, sage-p3-1-scelle-temoin-20-09 ; 20/09 18:57) ----

def _spec_archi(archs):
    from acvram.engine.config import ModelSpec
    s = ModelSpec(name="t", architecture="llama", hidden_size=H, intermediate_size=H, num_layers=1,
                  num_attention_heads=2, num_key_value_heads=2, vocab_size=16, max_position_embeddings=64,
                  rms_norm_eps=1e-6, rope_theta=1e4, head_dim=H // 2)
    s.raw = {"architectures": archs}
    return s


def test_masque_images_famille_gemma_bidir_qwen_causal_inconnu_refuse():
    from acvram.engine.vision import MasqueImageInconnu, masque_images_famille
    assert masque_images_famille(_spec_archi(["Gemma4ForConditionalGeneration"])) == "bidir"
    assert masque_images_famille(_spec_archi(["Gemma4UnifiedForConditionalGeneration"])) == "bidir"
    assert masque_images_famille(_spec_archi(["Gemma3ForConditionalGeneration"])) == "bidir"
    assert masque_images_famille(_spec_archi(["Qwen3VLForConditionalGeneration"])) == "causal"
    assert masque_images_famille(_spec_archi(["Qwen3VLMoeForConditionalGeneration"])) == "causal"
    with pytest.raises(MasqueImageInconnu, match="architectures="):
        masque_images_famille(_spec_archi(["LlavaForConditionalGeneration"]))    # famille absente : refus nommé
    with pytest.raises(MasqueImageInconnu):
        masque_images_famille(_spec_archi([]))


def _logits_prefill(converted, archs, plages_ignorees=False):
    """Logits du prefill d'une requête à image (tour factice, embeds constants) sur le converti jouet, la famille
    du config.json remplacée par ``archs`` ; ``plages_ignorees`` : masque causal pur (référence)."""
    from acvram.engine.loader import load_model
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.vision import TourVision, masque_images_famille
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    loaded.spec.raw = {**(loaded.spec.raw or {}), "architectures": archs}
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=64, enable_cuda_graphs=False)
    h = loaded.spec.hidden_size
    engine.vision = TourVision(lambda pv: torch.full((1, 3, h), 0.25), torch.device("cpu"), nom="factice")
    engine.masque_images = masque_images_famille(loaded.spec)
    if plages_ignorees:
        for l in loaded.model.layers:
            l.self_attn._masque_images = "causal"
    vus = {}

    def force(lg, seqs):
        vus["logits"] = lg[0].float().clone()
        return torch.tensor([1], device=lg.device, dtype=torch.long), torch.zeros(1, device=lg.device)
    engine._sample_only = force
    engine.add_request([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(temperature=0.0, max_tokens=1),
                       images=[(2, 5, torch.tensor(1.0), "sha-1")])
    engine.step()
    return vus["logits"], engine


def test_qwen_plage_image_aucune_cellule_ouverte_gemma_ouvertes(converted):
    """Sage : « Qwen3-VL + plage → aucune cellule ouverte » (= masque causal pur, au bit) ; « Gemma + plage →
    ouvertes » (le bloc bidirectionnel change la sortie). La porte agit au site model.py où les plages sont lues."""
    qwen, eng_q = _logits_prefill(converted, ["Qwen3VLForConditionalGeneration"])
    causal, _ = _logits_prefill(converted, ["Qwen3VLForConditionalGeneration"], plages_ignorees=True)
    assert torch.equal(qwen, causal), "Qwen3-VL : la plage image a ouvert des cellules (masque non causal)"
    gemma, eng_g = _logits_prefill(converted, ["Gemma4ForConditionalGeneration"])
    assert not torch.equal(gemma, causal), "Gemma : le bloc bidirectionnel n'a rien ouvert"
    assert " masque_images=causal" in eng_q.regime_ligne() and " masque_images=bidir" in eng_g.regime_ligne()


def test_alias_a_images_de_famille_inconnue_refuse_au_chargement(converted, monkeypatch):
    """Une tour servie (manifeste vision: oui) sur une famille sans masque connu : MasqueImageInconnu à la
    construction de l'Engine — jamais un bloc bidirectionnel appliqué par défaut."""
    import json, os, shutil
    from acvram.engine.loader import load_model
    from acvram.engine.vision import MasqueImageInconnu, TourVision
    d = os.path.join(os.path.dirname(converted), "inconnu_vision"); shutil.copytree(converted, d, dirs_exist_ok=True)
    man = json.load(open(os.path.join(d, "acvram_manifest.json"))); man["vision"] = "oui"
    json.dump(man, open(os.path.join(d, "acvram_manifest.json"), "w"))
    loaded = load_model(d, dtype=torch.float32, device_override="cpu")
    loaded.spec.raw = {**(loaded.spec.raw or {}), "architectures": ["LlavaForConditionalGeneration"]}
    h = loaded.spec.hidden_size
    monkeypatch.setattr(TourVision, "depuis_dossier", classmethod(lambda cls, p, m, dev: cls(lambda pv: torch.zeros(1, 3, h), dev, nom="factice")))
    with pytest.raises(MasqueImageInconnu, match="Llava"):
        Engine(loaded, None, max_batch_size=1, max_model_len=64, enable_cuda_graphs=False)
