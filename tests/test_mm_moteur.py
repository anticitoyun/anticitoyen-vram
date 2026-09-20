"""Multimodal P1, pièce (c) moteur (poste7-go-multimodal-organisation-20-09 § 2),
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
    assert regime_texte() == "vision=bf16(eager)"
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
    sorties = list(engine.generate(prompt, SamplingParams(max_tokens=2),
                                   images=[(2, 5, torch.tensor(3.0), "sha-2")]))
    assert sorties and sorties[-1].finish_reason == "refus"
