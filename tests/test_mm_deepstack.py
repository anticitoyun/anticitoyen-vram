"""Qwen3-VL deepstack, pièce (c) (poste7-go-qwen3vl-parallele-20-09 § 2), à sec,
formes réduites.

Référence : la BOUCLE de ``Qwen3VLTextModel.forward`` de transformers
(``_deepstack_process`` après les couches 0..k−1), servie par ``from_config``
d'un mini Qwen3-VL dont les couches sont remplacées par les MÊMES couches jouets
que le moteur (modules partagés) : ce qui est comparé, c'est l'ajout des niveaux
— où, sur quoi, dans quel ordre d'arrondi bf16 — pas l'attention (M-RoPE : autre
pièce).

(1) état après la couche 2 = transformers, au bit, sur les lignes image, texte
    inchangé — chemin ordinaire ET résidu différé (C15-prefill) ;
(2) sans image : forward au bit avec le témoin (boucle nue), aucun tenseur touché ;
(3) un morceau de prefill qui coupe la plage image reçoit la bonne tranche ;
(4) le contrôle sait dire FAUX : niveaux permutés 0↔1 détectés ;
(5) ``regime_ligne`` : ``deepstack=3`` déclaré seulement ;
(6) frontière : ``niveaux_deepstack`` / ``traits_niveaux`` / runner → ``ForwardBatch.deepstack``.
"""

import pytest
import torch
import torch.nn as nn

from acvram import regime
from acvram.engine.config import ModelSpec
from acvram.engine.layers import RMSNorm
from acvram.engine.model import (ACVRamModel, DecoderLayer, ForwardBatch,
                                 ajouter_deepstack, niveaux_deepstack)
from acvram.engine.vision import TourVision
from acvram.engine.vision import niveaux_deepstack as niveaux_de_la_tour
from acvram import kernels

H, V, L = 8, 32, 4          # 4 couches : les niveaux vont sur 0, 1, 2 et la couche 3 reste nue
K = 3


# --------------------------------------------------------------------------
# couches jouets (partagées entre le moteur et la référence)
# --------------------------------------------------------------------------
class _Lin(nn.Module):
    """Attention ou MLP jouet : une application linéaire bf16 de h, sans
    dépendance aux positions (un morceau vaut le tout, ligne à ligne)."""
    def __init__(self, g):
        super().__init__()
        self.w = nn.Parameter(torch.randn(H, H, generator=g).to(torch.bfloat16) * 0.5,
                              requires_grad=False)

    def forward(self, h, *args, **kwargs):
        return h @ self.w


def _couches(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(L):
        n1 = RMSNorm((1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16))
        n2 = RMSNorm((1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16))
        out.append(DecoderLayer(i, _Lin(g), _Lin(g), n1, n2, torch.device("cpu")))
    return out


class _Identite(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x):
        return x


def _modele(couches):
    spec = ModelSpec(name="t", architecture="qwen3", hidden_size=H, intermediate_size=H,
                     num_layers=L, num_attention_heads=1, num_key_value_heads=1,
                     vocab_size=V, max_position_embeddings=64)
    embed = torch.randn(V, H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    return ACVRamModel(spec, embed, couches, _Identite(), None, {}, dtype=torch.bfloat16)


def _lot(jetons, images=None, deepstack=None, offset=0):
    n = len(jetons)
    return ForwardBatch(tokens=torch.tensor(jetons), positions=torch.arange(offset, offset + n),
                        seq_lens=[offset + n], query_lens=[n],
                        block_tables=[torch.zeros(4, dtype=torch.long)],
                        slot_mapping=torch.arange(n), is_prefill=True,
                        images=images, deepstack=deepstack)


def _image(d, f, seed=2):
    g = torch.Generator().manual_seed(seed)
    traits = torch.randn(f - d, H, generator=g).to(torch.bfloat16)
    niveaux = torch.randn(K, f - d, H, generator=g).to(torch.bfloat16)
    return traits, niveaux


# --------------------------------------------------------------------------
# référence transformers : la boucle de Qwen3VLTextModel.forward
# --------------------------------------------------------------------------
class _Enveloppe(nn.Module):
    """Une couche jouet du moteur vue par transformers : [B, L, H] → [B, L, H]."""
    def __init__(self, couche):
        super().__init__()
        self.couche = couche

    def forward(self, hidden_states, *args, **kwargs):
        return torch.stack([self.couche(h, None, None) for h in hidden_states], 0)


def _reference_transformers(couches, x, masque, niveaux):
    """État caché après la dernière couche selon transformers (norme finale
    remplacée par l'identité), niveaux ajoutés par ``_deepstack_process``."""
    pytest.importorskip("transformers")  # pièce 267 : extra optionnel (vision/gdn), absent en CI de base
    from transformers import Qwen3VLConfig, Qwen3VLTextModel
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=H, intermediate_size=2 * H, num_hidden_layers=L,
                         num_attention_heads=2, num_key_value_heads=1, vocab_size=V, head_dim=4),
        vision_config=dict(hidden_size=H, out_hidden_size=H, depth=2, num_heads=2,
                           intermediate_size=2 * H, deepstack_visual_indexes=[0, 1, 2]))
    tm = Qwen3VLTextModel(cfg.text_config).to(torch.bfloat16)
    tm.layers = nn.ModuleList([_Enveloppe(c) for c in couches])
    tm.norm = nn.Identity()
    with torch.no_grad():
        out = tm(inputs_embeds=x[None], visual_pos_masks=masque[None],
                 deepstack_visual_embeds=[niveaux[k] for k in range(niveaux.shape[0])],
                 use_cache=False)
    return out.last_hidden_state[0]


def _embeddings(modele, jetons, d, f, traits):
    x = torch.nn.functional.embedding(torch.tensor(jetons), modele.embed_tokens).to(torch.bfloat16)
    x[d:f] = traits
    return x


@pytest.fixture(params=["ordinaire", "residu_differe"])
def chemin(request, monkeypatch):
    """Les deux chemins de la boucle des couches : `layer(x)` et `forward_res` (x, delta)."""
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1 if request.param == "residu_differe" else 0)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "")
    assert kernels.prefill_compact("residu") == (request.param == "residu_differe")
    return request.param


# --------------------------------------------------------------------------
# (1) au bit avec transformers, (4) le contrôle dit FAUX
# --------------------------------------------------------------------------
def test_etat_apres_la_couche_2_egale_transformers(chemin):
    couches = _couches()
    modele = _modele(couches)
    jetons = list(range(1, 11))
    d, f = 3, 7
    traits, niveaux = _image(d, f)
    lot = _lot(jetons, images=[[(d, f, traits)]], deepstack=[[(d, f, niveaux)]])
    with torch.no_grad():
        obtenu = modele(lot, return_hidden=True)

    x0 = _embeddings(modele, jetons, d, f, traits)
    masque = torch.zeros(len(jetons), dtype=torch.bool)
    masque[d:f] = True
    ref = _reference_transformers(couches, x0, masque, niveaux)
    assert obtenu.dtype == torch.bfloat16 and obtenu.shape == ref.shape
    assert torch.equal(obtenu, ref), (obtenu - ref).abs().max()

    # texte : identique à un passage SANS niveaux (les lignes texte ne voient
    # rien du deepstack) — et les lignes image, elles, ont bougé
    with torch.no_grad():
        sans = modele(_lot(jetons, images=[[(d, f, traits)]]), return_hidden=True)
    assert torch.equal(obtenu[~masque], sans[~masque])
    assert not torch.equal(obtenu[masque], sans[masque])

    # (4) faute construite : niveaux 0 et 1 permutés — vue
    permute = niveaux[[1, 0, 2]]
    with torch.no_grad():
        faux = modele(_lot(jetons, images=[[(d, f, traits)]], deepstack=[[(d, f, permute)]]),
                      return_hidden=True)
    assert not torch.equal(faux, ref)
    assert torch.equal(faux[~masque], ref[~masque])


def test_residu_differe_ajout_sur_la_somme_pas_sur_x():
    """(x, delta) : l'ajout va sur x + delta ; sur x seul il tomberait à côté
    (bf16 non associatif) — le test le montre sur des valeurs choisies."""
    # x = 1, delta = 3·2⁻⁹, v = 2⁻⁸ : (1 + delta) + v arrondit à 1 + 2⁻⁶,
    # (1 + v) + delta à 1 + 2⁻⁷ (deux égalités parfaites tranchées au pair)
    x = torch.ones(6, H, dtype=torch.bfloat16)
    delta = torch.full((6, H), 3 * 2.0 ** -9, dtype=torch.bfloat16)
    v = torch.full((1, 2, H), 2.0 ** -8, dtype=torch.bfloat16)
    lot = _lot(list(range(6)), deepstack=[[(2, 4, v)]])
    x2, d2 = ajouter_deepstack(x.clone(), delta.clone(), lot, 0)
    attendu = (x + delta).clone()
    attendu[2:4] = attendu[2:4] + v[0]
    assert torch.equal(x2 + d2, attendu)                       # l'état vrai est juste
    assert torch.equal(d2[2:4], torch.zeros(2, H, dtype=torch.bfloat16))
    assert torch.equal(d2[:2], delta[:2]) and torch.equal(d2[4:], delta[4:])
    assert torch.equal(x2[:2], x[:2]) and torch.equal(x2[4:], x[4:])
    sur_x = x.clone()
    sur_x[2:4] = sur_x[2:4] + v[0]
    assert not torch.equal(sur_x + delta, attendu)             # « sur x » : faux, d'où la règle


# --------------------------------------------------------------------------
# (2) sans image : au bit avec le témoin, aucun tenseur touché
# --------------------------------------------------------------------------
def test_sans_image_au_bit_avec_le_temoin(chemin, monkeypatch):
    couches = _couches()
    modele = _modele(couches)
    jetons = list(range(1, 11))
    lot = _lot(jetons)
    assert lot.images is None and lot.deepstack is None

    import acvram.engine.model as M
    monkeypatch.setattr(M, "ajouter_deepstack",
                        lambda *a, **k: pytest.fail("ajouter_deepstack appelé sans image"))
    monkeypatch.setattr(M, "niveaux_deepstack",
                        lambda *a, **k: pytest.fail("niveaux_deepstack appelé sans image"))
    with torch.no_grad():
        obtenu = modele(lot, return_hidden=True)
        # témoin : la boucle nue, couche après couche (le chemin d'avant)
        x = torch.nn.functional.embedding(torch.tensor(jetons), modele.embed_tokens).to(torch.bfloat16)
        for c in couches:
            x = c(x, lot, None)
    assert torch.equal(obtenu, x)


# --------------------------------------------------------------------------
# (3) un morceau qui coupe la plage image
# --------------------------------------------------------------------------
def test_morceau_qui_coupe_la_plage_image(chemin):
    couches = _couches()
    modele = _modele(couches)
    jetons = list(range(1, 11))
    d, f = 3, 8
    traits, niveaux = _image(d, f)
    entier = _lot(jetons, images=[[(d, f, traits)]], deepstack=[[(d, f, niveaux)]])
    with torch.no_grad():
        ref = modele(entier, return_hidden=True)
        # deux morceaux : [0, 5) puis [5, 10) — la coupe tombe dans l'image
        a = modele(_lot(jetons[:5], images=[[(d, f, traits)]], deepstack=[[(d, f, niveaux)]]),
                   return_hidden=True)
        b = modele(_lot(jetons[5:], images=[[(d, f, traits)]], deepstack=[[(d, f, niveaux)]],
                        offset=5), return_hidden=True)
    assert torch.equal(torch.cat([a, b], 0), ref)
    # la mauvaise tranche (niveaux décalés d'une ligne) est vue
    decale = torch.roll(niveaux, 1, dims=1)
    with torch.no_grad():
        b2 = modele(_lot(jetons[5:], images=[[(d, f, traits)]], deepstack=[[(d, f, decale)]],
                         offset=5), return_hidden=True)
    assert not torch.equal(b2, ref[5:])
    # formes vérifiées : un niveau de mauvaise longueur est refusé, nommément
    with pytest.raises(ValueError, match="deepstack"):
        with torch.no_grad():
            modele(_lot(jetons, images=[[(d, f, traits)]], deepstack=[[(d, f, niveaux[:, :2])]]),
                   return_hidden=True)
    # deux séquences du lot avec des nombres de niveaux différents : refus
    lot2 = ForwardBatch(tokens=torch.tensor(jetons + jetons), positions=torch.cat([torch.arange(10)] * 2),
                        seq_lens=[10, 10], query_lens=[10, 10],
                        block_tables=[torch.zeros(4, dtype=torch.long)] * 2,
                        slot_mapping=torch.arange(20), is_prefill=True,
                        images=[[(d, f, traits)], [(d, f, traits)]],
                        deepstack=[[(d, f, niveaux)], [(d, f, niveaux[:2])]])
    with pytest.raises(ValueError, match="niveaux"):
        niveaux_deepstack(lot2)


def test_deux_sequences_dans_le_lot_chacune_ses_niveaux():
    couches = _couches()
    modele = _modele(couches)
    j1, j2 = list(range(1, 9)), list(range(9, 15))
    t1, n1 = _image(2, 5, seed=3)
    t2, n2 = _image(1, 4, seed=4)
    lot = ForwardBatch(tokens=torch.tensor(j1 + j2),
                       positions=torch.cat([torch.arange(8), torch.arange(6)]),
                       seq_lens=[8, 6], query_lens=[8, 6],
                       block_tables=[torch.zeros(4, dtype=torch.long)] * 2,
                       slot_mapping=torch.arange(14), is_prefill=True,
                       images=[[(2, 5, t1)], [(1, 4, t2)]],
                       deepstack=[[(2, 5, n1)], [(1, 4, n2)]])
    with torch.no_grad():
        obtenu = modele(lot, return_hidden=True)
        s1 = modele(_lot(j1, images=[[(2, 5, t1)]], deepstack=[[(2, 5, n1)]]), return_hidden=True)
        s2 = modele(_lot(j2, images=[[(1, 4, t2)]], deepstack=[[(1, 4, n2)]]), return_hidden=True)
    assert torch.equal(obtenu, torch.cat([s1, s2], 0))


# --------------------------------------------------------------------------
# (6) frontière tour → niveaux → runner → ForwardBatch.deepstack
# --------------------------------------------------------------------------
class _SortieQwen:
    """La sortie de ``get_image_features`` de Qwen3-VL (transformers 5) :
    ``pooler_output`` (tuple par image) + ``deepstack_features`` (liste de k)."""
    def __init__(self, po, ds):
        self.last_hidden_state = torch.zeros(1, 99, 3)
        self.pooler_output = po
        self.deepstack_features = ds


class _SortieGemma:
    def __init__(self, po):
        self.last_hidden_state = torch.zeros(1, 99, 3)
        self.pooler_output = po
        self.deepstack_features = None


def test_frontiere_niveaux_de_la_tour():
    po = (torch.ones(4, H),)
    ds = [torch.full((4, H), float(k)) for k in range(K)]
    n = niveaux_de_la_tour(_SortieQwen(po, ds))
    assert n.shape == (K, 4, H) and all(float(n[k, 0, 0]) == k for k in range(K))
    assert niveaux_de_la_tour(_SortieGemma(po)) is None                     # Gemma 4 : rien
    assert niveaux_de_la_tour(_SortieQwen(po, [])) is None                  # liste vide : rien
    assert niveaux_de_la_tour((torch.ones(4, H), ds)).shape == (K, 4, H)   # forme ancienne (embeds, deepstack)
    assert niveaux_de_la_tour((torch.ones(4, H), None)) is None
    assert niveaux_de_la_tour(torch.ones(4, H)) is None

    tour = TourVision(lambda pv: _SortieQwen(po, ds), torch.device("cpu"), nom="factice", hidden=H)
    traits, niveaux = tour.traits_niveaux(torch.zeros(1), 4)
    assert traits.shape == (4, H) and traits.dtype == torch.bfloat16
    assert niveaux.shape == (K, 4, H) and niveaux.dtype == torch.bfloat16
    assert tour.traits(torch.zeros(1), 4).shape == (4, H)                   # `traits` : inchangé, un tenseur
    gemma = TourVision(lambda pv: _SortieGemma(po), torch.device("cpu"), nom="factice", hidden=H)
    assert gemma.traits_niveaux(torch.zeros(1), 4)[1] is None
    with pytest.raises(ValueError, match="niveaux deepstack"):               # niveaux qui ne collent pas aux traits
        TourVision(lambda pv: _SortieQwen(po, [torch.zeros(3, H)]), torch.device("cpu"),
                   nom="factice", hidden=H).traits_niveaux(torch.zeros(1), 4)


def test_runner_porte_les_niveaux_au_prefill_seulement(converted):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256, enable_cuda_graphs=False)
    h = loaded.spec.hidden_size

    def calcul(pv):
        n = int(pv)
        return _SortieQwen((torch.full((n, h), 1.0),), [torch.full((n, h), float(k + 2)) for k in range(K)])

    engine.vision = TourVision(calcul, torch.device("cpu"), nom="factice")
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Gemma4ForConditionalGeneration"]}   # famille du masque (20/09 19:01) : le jouet Llama n'en a pas
    vus = []
    orig = engine._build_batch

    def espion(seqs, prefill, limite=None):
        b = orig(seqs, prefill, limite)
        vus.append(b)
        return b
    engine._build_batch = espion
    for _ in engine.generate(list(range(1, 9)), SamplingParams(temperature=0.0, max_tokens=2),
                             images=[(2, 5, torch.tensor(3.0), "sha-ds")]):
        pass
    pre = [b for b in vus if b.is_prefill]
    assert pre and pre[0].deepstack is not None and pre[0].images is not None
    d, f, n = pre[0].deepstack[0][0]
    assert (d, f) == (2, 5) and n.dtype == torch.bfloat16 and n.shape == (K, 3, h)
    assert float(n[1, 0, 0]) == 3.0
    assert all(b.deepstack is None for b in vus if not b.is_prefill)      # jamais au décodage
    # tour sans niveaux (Gemma 4) : deepstack absent du lot, images présentes
    vus.clear()
    engine.vision = TourVision(lambda pv: _SortieGemma((torch.full((int(pv), h), 1.0),)),
                               torch.device("cpu"), nom="factice")
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Gemma4ForConditionalGeneration"]}   # famille du masque (20/09 19:01) : le jouet Llama n'en a pas
    for _ in engine.generate(list(range(1, 9)), SamplingParams(temperature=0.0, max_tokens=1),
                             images=[(2, 5, torch.tensor(3.0), "sha-g")]):
        pass
    pre = [b for b in vus if b.is_prefill]
    assert pre and pre[0].images is not None and pre[0].deepstack is None


# --------------------------------------------------------------------------
# (5) la ligne de régime
# --------------------------------------------------------------------------
def test_regime_ligne_deepstack_seulement_declare():
    try:
        regime.declarer_modele_charge(None)
        assert "deepstack=" not in regime.regime_ligne()
        regime.declarer_modele_charge({"vision": "oui"})
        assert "deepstack=" not in regime.regime_ligne()                      # Gemma 4 : pas de mot
        regime.declarer_modele_charge({"vision": "oui", "deepstack": "oui"})
        assert " deepstack=3 " in regime.regime_ligne() + " "
        regime.declarer_modele_charge({"vision": "oui", "deepstack": "oui", "deepstack_niveaux": 2})
        assert " deepstack=2 " in regime.regime_ligne() + " "
        regime.declarer_modele_charge({"vision": "oui", "deepstack": [8, 16, 24]})
        assert " deepstack=3 " in regime.regime_ligne() + " "
        regime.declarer_modele_charge({"vision": "oui", "deepstack": "non"})
        assert "deepstack=" not in regime.regime_ligne()
        regime.declarer_deepstack(4)                                          # la config de la tour prime
        assert " deepstack=4 " in regime.regime_ligne() + " "
    finally:
        regime.declarer_modele_charge(None)
    assert "deepstack=" not in regime.regime_ligne()


# --------------------------------------------------------------------------
# (7) un vrai Qwen3-VL-2B bf16 sur le CPU : la lecture de la tour au bit
# --------------------------------------------------------------------------
def _dossier_qwen3vl():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outils"))
    from racine_modeles import racine_modeles
    return os.path.join(racine_modeles(), "Qwen3-VL-2B-Instruct-bf16")


@pytest.mark.a_sec
def test_tour_reelle_qwen3vl_2b_traits_et_niveaux_au_bit():
    """Qwen3-VL-2B-Instruct bf16 chargé sur le CPU (jamais la carte) : une image
    224×224 de bruit déterministe passe par le processeur du dossier puis par
    ``get_image_features`` ; ``traits_niveaux`` rend (image_embeds concaténés,
    deepstack_image_embeds empilés) au bit, k = len(deepstack_visual_indexes),
    n = grille / merge², h = hidden du LM. Un chargement > 60 s est signalé."""
    import os
    import time
    dossier = _dossier_qwen3vl()
    if not os.path.isdir(dossier):
        pytest.skip(f"Qwen3-VL-2B-Instruct-bf16 absent sous racine_modeles() : {dossier}")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.monotonic()
    modele = Qwen3VLForConditionalGeneration.from_pretrained(dossier, dtype=torch.bfloat16).eval()
    duree = time.monotonic() - t0
    if duree > 60:
        print(f"[test] chargement CPU de Qwen3-VL-2B : {duree:.0f} s (> 60 s)", flush=True)
    proc = AutoProcessor.from_pretrained(dossier)
    g = torch.Generator().manual_seed(7)
    image = (torch.rand(224, 224, 3, generator=g) * 255).to(torch.uint8).numpy()
    entrees = proc.image_processor(images=[image], return_tensors="pt")
    pv, grille = entrees["pixel_values"], entrees["image_grid_thw"]
    vcfg = modele.config.vision_config
    k = len(vcfg.deepstack_visual_indexes)
    n = int(grille.prod(-1).sum()) // vcfg.spatial_merge_size ** 2
    h = modele.config.text_config.hidden_size

    with torch.no_grad():
        ref = modele.model.get_image_features(pixel_values=pv, image_grid_thw=grille)
        image_embeds = torch.cat(list(ref.pooler_output), 0)
        deepstack_image_embeds = ref.deepstack_features
    assert len(deepstack_image_embeds) == k and image_embeds.shape == (n, h)

    tour = TourVision(lambda p, **s: modele.model.get_image_features(pixel_values=p, **s),
                      torch.device("cpu"), hidden=h)
    with torch.no_grad():
        traits, niveaux = tour.traits_niveaux(pv, n, supplement={"image_grid_thw": grille})
    assert traits.shape == (n, h) and niveaux.shape == (k, n, h)
    assert traits.dtype == torch.bfloat16 and niveaux.dtype == torch.bfloat16
    assert torch.equal(traits, image_embeds.to(torch.bfloat16))
    for i in range(k):
        assert torch.equal(niveaux[i], deepstack_image_embeds[i].to(torch.bfloat16))
    assert not torch.equal(niveaux[0], niveaux[1])                     # deux niveaux distincts : un contrôle qui peut dire faux
    print(f"[test] Qwen3-VL-2B CPU : chargement {duree:.1f} s, grille {grille.tolist()}, n={n}, k={k}, h={h}",
          flush=True)
