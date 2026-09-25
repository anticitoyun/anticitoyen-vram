"""M-RoPE Qwen3-VL (pièce (b), contrat revue/poste7-go-qwen3vl-parallele-20-09 § 2), à sec.

Référence : transformers ``modeling_qwen3_vl`` (``Qwen3VLModel.get_rope_index``,
``Qwen3VLTextRotaryEmbedding`` + ``recomposition_frequencies``), reproduite AU BIT
sur formes réduites :
(1) positions 3-D et delta = ``get_rope_index`` / ``rope_deltas`` sur 5 grilles ;
(2) cos/sin de ``RotaryEmbedding`` en mode mrope = transformers au bit ;
(3) décodage après image : position 1-D + delta = les positions « max + 1 » de
    transformers → même rope au bit ;
(4) texte seul : cos/sin ET sortie du modèle au bit avec l'ancien chemin (témoin) ;
(5) le test sait dire FAUX : section permutée, grille fausse → différent / refusé ;
puis le runner (delta porté par la séquence, lu au décodage ; positions_3d au
prefill), la config, la ligne de régime, et le vrai Qwen3-VL-2B sur le CPU.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outils"))
from racine_modeles import alias, alias_absent  # noqa: E402

from acvram.engine.layers import RotaryEmbedding  # noqa: E402
from acvram.engine.model import ForwardBatch  # noqa: E402
from acvram.engine.mrope import axes_interleaved, positions_mrope, section_depuis  # noqa: E402

pytestmark = pytest.mark.a_sec

SECTION = [24, 20, 20]          # Qwen3-VL, head_dim 128
THETA = 5_000_000.0
SCALING = {"mrope_interleaved": True, "mrope_section": SECTION, "rope_type": "default"}
MERGE = 2
CPU = torch.device("cpu")

# (nom, T, images (debut, fin, grid_thw)) — fin − debut = t · (h/2) · (w/2)
GRILLES = [
    ("texte seul", 9, []),
    ("1 image", 14, [(3, 7, (1, 4, 4))]),
    ("2 images", 24, [(2, 6, (1, 4, 4)), (10, 16, (1, 4, 6))]),
    ("image en tete", 11, [(0, 6, (1, 6, 4))]),
    ("image apres du texte", 13, [(8, 12, (1, 4, 4))]),
]


def _modele_ref():
    """Un porteur minimal de ``get_rope_index`` (transformers, sans poids) : la
    config Qwen3-VL par défaut (spatial_merge_size 2, head_dim 128) avec le
    rope_scaling publié du 2B (mrope_interleaved, [24, 20, 20], theta 5e6)."""
    tr = pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig()
    cfg.text_config.rope_parameters = {**SCALING, "rope_theta": THETA}
    assert cfg.vision_config.spatial_merge_size == MERGE and cfg.text_config.head_dim == 128

    class _Porteur:
        config = cfg
        get_vision_position_ids = Qwen3VLModel.get_vision_position_ids

    return tr, cfg, _Porteur()


def _ref_positions(porteur, T, images):
    ids = torch.zeros(1, T, dtype=torch.long)
    types = torch.zeros(1, T, dtype=torch.int)
    for d, f, _ in images:
        types[0, d:f] = 1
    thw = torch.tensor([g for _, _, g in images], dtype=torch.long) if images else None
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    pos, deltas = Qwen3VLModel.get_rope_index(porteur, ids, types, image_grid_thw=thw)
    return pos[:, 0], int(deltas.reshape(-1)[0])


def _ref_cos_sin(cfg, positions3):
    """cos/sin [t, 128] bf16 de transformers pour des positions [3, t]."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    rot = Qwen3VLTextRotaryEmbedding(cfg.text_config)
    c, s = rot(torch.zeros(1, dtype=torch.bfloat16), positions3.unsqueeze(1))
    return c[0], s[0]


def _rope(section=SECTION):
    return RotaryEmbedding(128, 4096, THETA, {**SCALING, "mrope_section": section})


# --------------------------------------------------------------------------
# (1) positions 3-D et delta
# --------------------------------------------------------------------------
@pytest.mark.parametrize("nom,T,images", GRILLES, ids=[g[0] for g in GRILLES])
def test_positions_3d_et_delta_egalent_get_rope_index(nom, T, images):
    _, _, porteur = _modele_ref()
    pos, delta = positions_mrope(T, images, MERGE)
    ref, delta_ref = _ref_positions(porteur, T, images)
    assert pos.shape == (3, T) and pos.dtype == torch.long
    assert torch.equal(pos, ref), f"{nom} : positions ≠ get_rope_index"
    assert delta == delta_ref, f"{nom} : delta {delta} ≠ rope_deltas {delta_ref}"
    if not images:
        assert delta == 0 and torch.equal(pos, torch.arange(T).view(1, -1).expand(3, -1))
    else:
        assert delta < 0                         # une image resserre les positions


def test_grille_fausse_ou_plage_hors_invite_refusee():
    with pytest.raises(ValueError, match="jetons"):
        positions_mrope(14, [(3, 8, (1, 4, 4))], MERGE)       # 5 jetons pour 4 cases
    with pytest.raises(ValueError, match="hors de l'invite"):
        positions_mrope(6, [(3, 7, (1, 4, 4))], MERGE)
    with pytest.raises(ValueError, match="chevauchante"):
        positions_mrope(20, [(3, 7, (1, 4, 4)), (5, 9, (1, 4, 4))], MERGE)
    with pytest.raises(ValueError, match="image_grid_thw"):
        positions_mrope(14, [(3, 7, (4, 4))], MERGE)


# --------------------------------------------------------------------------
# (2) cos/sin entrelacés au bit — (5) section permutée : FAUX
# --------------------------------------------------------------------------
def test_axes_entrelaces_suivent_recomposition_frequencies():
    axes = axes_interleaved(SECTION, 64).tolist()
    assert axes[:60] == [0, 1, 2] * 20 and axes[60:] == [0, 0, 0, 0]
    assert axes.count(0) == 24 and axes.count(1) == 20 and axes.count(2) == 20
    with pytest.raises(ValueError, match="somme"):
        axes_interleaved([24, 20, 21], 64)
    assert section_depuis(None) is None and section_depuis({"rope_type": "yarn"}) is None
    with pytest.raises(ValueError, match="entrelac"):
        section_depuis({"mrope_section": SECTION})       # Qwen2-VL (blocs) : refus nommé


@pytest.mark.parametrize("nom,T,images", GRILLES[1:], ids=[g[0] for g in GRILLES[1:]])
def test_cos_sin_mrope_au_bit_avec_transformers(nom, T, images):
    _, cfg, _ = _modele_ref()
    pos, _ = positions_mrope(T, images, MERGE)
    c_ref, s_ref = _ref_cos_sin(cfg, pos)
    c, s = _rope()(pos, CPU, torch.bfloat16, max_pos=T)
    assert c.shape == (T, 128) and c.dtype == torch.bfloat16
    assert torch.equal(c, c_ref) and torch.equal(s, s_ref), f"{nom} : cos/sin ≠ transformers"
    # (5) faute construite : sections permutées (h, w, t) — le test doit dire FAUX
    c_perm, s_perm = _rope([20, 20, 24])(pos, CPU, torch.bfloat16, max_pos=T)
    assert not (torch.equal(c_perm, c_ref) and torch.equal(s_perm, s_ref)), \
        f"{nom} : une section permutée passe — le test ne sait pas dire FAUX"
    # et les axes h/w comptent : les positions 1-D ×3 ne sont PAS le M-RoPE de l'image
    c_1d, _ = _rope()(pos[0].view(1, -1).expand(3, -1), CPU, torch.bfloat16, max_pos=T)
    assert not torch.equal(c_1d, c_ref)


# --------------------------------------------------------------------------
# (3) décodage après image : 1-D + delta
# --------------------------------------------------------------------------
@pytest.mark.parametrize("nom,T,images", GRILLES[1:], ids=[g[0] for g in GRILLES[1:]])
def test_decodage_apres_image_1d_plus_delta_au_bit(nom, T, images):
    _, cfg, porteur = _modele_ref()
    _, delta = positions_mrope(T, images, MERGE)
    _, delta_ref = _ref_positions(porteur, T, images)
    assert delta == delta_ref
    # transformers en génération : text_positions + rope_deltas sur les trois axes
    pas = torch.arange(T, T + 5)                                 # 5 pas de décodage, positions 1-D
    ref3 = (pas + delta_ref).view(1, -1).expand(3, -1)
    c_ref, s_ref = _ref_cos_sin(cfg, ref3)
    rope = _rope()
    c, s = rope(pas + delta, CPU, torch.bfloat16, max_pos=T + 5)   # chemin 1-D du décodage, inchangé
    assert torch.equal(c, c_ref) and torch.equal(s, s_ref)
    # sans le delta, faux (le test sait le dire)
    c_sans, _ = rope(pas, CPU, torch.bfloat16, max_pos=T + 5)
    assert not torch.equal(c_sans, c_ref)


# --------------------------------------------------------------------------
# (4) texte seul : témoin au bit avec l'ancien chemin
# --------------------------------------------------------------------------
def test_texte_seul_cos_sin_au_bit_avec_le_chemin_1d():
    T = 37
    rope = _rope()
    p1 = torch.arange(T)
    c1, s1 = rope(p1, CPU, torch.bfloat16, max_pos=T)
    c3, s3 = rope(p1.view(1, -1).expand(3, -1), CPU, torch.bfloat16, max_pos=T)
    assert torch.equal(c1, c3) and torch.equal(s1, s3)
    # et un RoPE SANS mrope_section ignore tout : mêmes tables qu'avant
    ancien = RotaryEmbedding(128, 4096, THETA, {"rope_type": "default"})
    assert ancien.mrope_section is None
    c0, s0 = ancien(p1, CPU, torch.bfloat16, max_pos=T)
    assert torch.equal(c0, c1) and torch.equal(s0, s1)
    with pytest.raises(ValueError, match="sans mrope_section"):
        ancien.forward_mrope(p1.view(1, -1).expand(3, -1), CPU, torch.bfloat16, T)


def _mrope_sur_le_jouet(model, section=(6, 5, 5)):
    """Le jouet (head_dim 32, 16 fréquences) reçoit un RoPE à sections [6, 5, 5]
    (rope_type default : mêmes inv_freq, mêmes tables que l'ancien)."""
    from acvram.engine.model import Attention
    for m in model.modules():
        if isinstance(m, Attention) and m.rope is not None:
            r = m.rope
            m.rope = RotaryEmbedding(r.head_dim, r.max_position, r.base,
                                     {**SCALING, "mrope_section": list(section)})


def _lot(prompt, positions_3d=None):
    from acvram.memory.kvcache import BLOCK_SIZE
    n = len(prompt)
    blocks = list(range(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE + 2))
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)])
    return ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n],
                        [torch.tensor(blocks)], slots, True, positions_3d=positions_3d)


def test_texte_seul_sortie_du_modele_au_bit_et_image_change_la_sortie(converted):
    """Le jouet converti : logits du texte seul IDENTIQUES avec l'ancien RoPE,
    avec le RoPE mrope sans positions_3d, et avec positions_3d = 1-D ×3 ; des
    positions 3-D d'une image (grille 4×4 sur [3, 7)) changent les logits."""
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    prompt = list(range(5, 19))
    with torch.no_grad():
        ancien = model(_lot(prompt)).clone()
        _mrope_sur_le_jouet(model)
        sans_3d = model(_lot(prompt)).clone()
        T = len(prompt)
        pos_texte = torch.arange(T).view(1, -1).expand(3, -1).contiguous()
        avec_3d = model(_lot(prompt, pos_texte)).clone()
        pos_im, delta = positions_mrope(T, [(3, 7, (1, 4, 4))], MERGE)
        image = model(_lot(prompt, pos_im)).clone()
    assert torch.equal(ancien, sans_3d) and torch.equal(ancien, avec_3d)
    assert delta == -2 and not torch.equal(ancien, image)


# --------------------------------------------------------------------------
# runner : delta porté par la séquence, positions_3d au prefill, décodage 1-D + delta
# --------------------------------------------------------------------------
class _FragQwen:
    def __init__(self, debut, fin, thw, sha):
        self.debut, self.fin, self.sha256 = debut, fin, sha
        self.pixel_values = torch.tensor(float(fin - debut))
        self.supplement = {"image_grid_thw": torch.tensor([list(thw)])}


def test_runner_delta_par_sequence_et_positions_3d_au_prefill(converted):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.vision import TourVision
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    _mrope_sur_le_jouet(loaded.model)
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256, enable_cuda_graphs=False)
    assert engine._mrope_section is None                       # jouet llama : rien ne change
    engine._mrope_section, engine._mrope_merge = [6, 5, 5], MERGE
    h = loaded.spec.hidden_size
    engine.vision = TourVision(lambda pv, **_: torch.full((1, int(pv), h), 0.5), CPU, nom="factice")
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Qwen3VLForConditionalGeneration"]}   # famille du masque (20/09 19:01)
    vus = []
    orig = engine._build_batch

    def espion(seqs, prefill, limite=None):
        b = orig(seqs, prefill, limite)
        vus.append((b, [(s.rope_delta, s.length) for s in seqs]))
        return b
    engine._build_batch = espion
    prompt = list(range(1, 15))                                # 14 jetons, image [3, 7) grille 1×4×4
    for _ in engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=3),
                             images=[_FragQwen(3, 7, (1, 4, 4), "sha-q")]):
        pass
    attendu, delta = positions_mrope(14, [(3, 7, (1, 4, 4))], MERGE)
    assert delta == -2
    pre = [b for b, _ in vus if b.is_prefill]
    assert len(pre) == 1 and torch.equal(pre[0].positions_3d, attendu) and pre[0].rope_delta == [-2]
    assert torch.equal(pre[0].positions, torch.arange(14))    # 1-D absolu (cache, slots) inchangé
    dec = [(b, d) for b, d in vus if not b.is_prefill]
    assert len(dec) >= 2
    for b, ((rd, longueur),) in dec:
        assert rd == -2 and b.positions_3d is None
        assert b.positions.tolist() == [longueur - 1 + rd]    # 1-D + delta, lu par positions_on
        assert b.positions_on(CPU) is b.positions_on(CPU) and b.positions_rope_on(CPU) is b.positions_on(CPU)
    # texte seul sur le même moteur : ni positions_3d, ni delta — le lot d'avant
    vus.clear()
    for _ in engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=2)):
        pass
    assert all(b.positions_3d is None and b.rope_delta is None for b, _ in vus)
    assert all(rd == 0 for _, d in vus for rd, _ in d)
    dec = [b for b, _ in vus if not b.is_prefill]
    assert dec and dec[0].positions.tolist() == [14]


def test_runner_grille_absente_refus_nomme(converted, capsys):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.vision import TourVision
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256, enable_cuda_graphs=False)
    engine._mrope_section, engine._mrope_merge = [6, 5, 5], MERGE
    h = loaded.spec.hidden_size
    engine.vision = TourVision(lambda pv, **_: torch.full((1, int(pv), h), 0.5), CPU, nom="factice")
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Qwen3VLForConditionalGeneration"]}   # famille du masque (20/09 19:01)
    sorties = list(engine.generate(list(range(1, 15)), SamplingParams(max_tokens=2),
                                   images=[(3, 7, torch.tensor(4.0), "sha-sans-grille")]))
    assert sorties and sorties[-1].finish_reason == "refus"
    assert "image_grid_thw" in capsys.readouterr().out


def test_lot_mixte_texte_et_image_positions_3d_par_sequence():
    from acvram.engine.runner import Engine, Sequence
    from acvram.engine.sampler import SamplingParams
    a = Sequence(list(range(14)), SamplingParams())
    a.mrope_positions, a.rope_delta = positions_mrope(14, [(3, 7, (1, 4, 4))], MERGE)
    b = Sequence(list(range(9)), SamplingParams())
    champs = Engine._mrope_du_lot([a, b], True, [10, 9], [14, 9])   # a reprise à 4 (cache de préfixe)
    assert champs["rope_delta"] == [-2, 0]
    p3 = champs["positions_3d"]
    assert torch.equal(p3[:, :10], a.mrope_positions[:, 4:14])
    assert torch.equal(p3[:, 10:], torch.arange(9).view(1, -1).expand(3, -1))
    assert Engine._mrope_du_lot([b], True, [9], [9]) == {}
    assert Engine._mrope_du_lot([a], False, [1], [15]) == {}     # décodage : rien, le delta est dans positions


# --------------------------------------------------------------------------
# config et régime
# --------------------------------------------------------------------------
def test_config_lit_mrope_section_et_spatial_merge(tmp_path):
    import json
    from acvram.engine.config import load_model_spec
    cfg = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl",
           "text_config": {"model_type": "qwen3_vl_text", "hidden_size": 64, "intermediate_size": 128,
                           "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                           "head_dim": 128, "vocab_size": 100, "max_position_embeddings": 4096,
                           "rope_theta": THETA, "rope_scaling": SCALING},
           "vision_config": {"spatial_merge_size": 2}, "image_token_id": 7}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    spec = load_model_spec(str(tmp_path), "qvl")
    assert spec.mrope_section == SECTION and spec.spatial_merge_size == 2
    assert spec.rope_theta == THETA
    # transformers ≥ 5 : rope_parameters à la place de rope_scaling
    cfg["text_config"].pop("rope_scaling")
    cfg["text_config"]["rope_parameters"] = {**SCALING, "rope_theta": THETA}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    spec2 = load_model_spec(str(tmp_path), "qvl")
    assert spec2.mrope_section == SECTION and spec2.rope_theta == THETA
    # le spec survit au manifeste (ModelSpec(**to_dict()))
    from acvram.engine.config import ModelSpec
    d = spec.to_dict()
    spec3 = ModelSpec(**{k: v for k, v in d.items() if k in ModelSpec.__dataclass_fields__})
    assert spec3.mrope_section == SECTION and spec3.spatial_merge_size == 2


def test_regime_mot_mrope_section_entrelacee():
    """`mrope=[24,20,20](interleaved)` : lu du manifeste écrit par la conversion
    (pièce (a) : `mrope_section` / `mrope_interleaved`), sinon du rope_scaling du
    spec ; rien sur un modèle texte, rien sans modèle (test_defaut_servi)."""
    from acvram import regime
    MOT = "mrope=[24,20,20](interleaved)"
    try:
        regime.declarer_modele_charge({"vision": "oui", "mrope_section": SECTION, "mrope_interleaved": True,
                                       "model": {"rope_scaling": {}}})
        assert regime.mrope_texte() == MOT and MOT in regime.regime_ligne().split()
        regime.declarer_modele_charge({"vision": "oui", "model": {"rope_scaling": SCALING}})
        assert regime.mrope_texte() == MOT
        regime.declarer_modele_charge({"vision": "oui", "mrope_section": [16, 24, 24], "mrope_interleaved": False})
        assert regime.mrope_texte() == "mrope=[16,24,24](blocs)"
        regime.declarer_modele_charge({"vision": "non", "model": {"rope_scaling": {"rope_type": "yarn"}}})
        assert regime.mrope_texte() is None and "mrope=" not in regime.regime_ligne()
    finally:
        regime.declarer_modele_charge(None)
    assert regime.mrope_texte() is None and "mrope=" not in regime.regime_ligne()


@pytest.mark.skipif(bool(alias_absent(_QVL_NOM := "Qwen3-VL-2B-Instruct-bf16")), reason=alias_absent("Qwen3-VL-2B-Instruct-bf16") or "présent")
def test_dossier_qwen3_vl_2b_config_lue_et_ligne_de_regime():
    """Le VRAI dossier, sans manifeste : la lecture que le moteur fait (config.json,
    text_config dépliée) rend mrope_section [24, 20, 20] ET mrope_interleaved,
    spatial_merge_size 2, theta 5e6 — LUS, pas déclarés — et la ligne de régime
    les porte une fois le modèle déclaré (manifeste comme la conversion l'écrit)."""
    from acvram import regime
    from acvram.engine.config import load_model_spec
    chemin = alias(_QVL_NOM)
    assert not os.path.exists(os.path.join(chemin, "acvram_manifest.json"))
    spec = load_model_spec(chemin, _QVL_NOM)
    assert spec.architecture in ("llama",) and spec.head_dim == 128
    assert spec.mrope_section == [24, 20, 20]
    assert spec.rope_scaling["mrope_interleaved"] is True and spec.rope_scaling["rope_type"] == "default"
    assert spec.spatial_merge_size == 2 and spec.rope_theta == 5_000_000
    rope = RotaryEmbedding(spec.head_dim, spec.max_position_embeddings, spec.rope_theta, spec.rope_scaling)
    assert rope.mrope_section == [24, 20, 20]
    manifeste = {"vision": "oui", "model": spec.to_dict(),
                 "mrope_section": [int(x) for x in spec.rope_scaling["mrope_section"]],
                 "mrope_interleaved": bool(spec.rope_scaling["mrope_interleaved"])}
    try:
        regime.declarer_modele_charge(manifeste)
        assert "mrope=[24,20,20](interleaved)" in regime.regime_ligne().split()
    finally:
        regime.declarer_modele_charge(None)


# --------------------------------------------------------------------------
# le vrai Qwen3-VL-2B (bf16, CPU, jamais la carte) : image 224×224 fabriquée
# --------------------------------------------------------------------------
_QVL = "Qwen3-VL-2B-Instruct-bf16"


@pytest.mark.skipif(bool(alias_absent(_QVL)), reason=alias_absent(_QVL) or "présent")
def test_qwen3_vl_2b_cpu_positions_3d_et_delta_contre_get_rope_index():
    """Le modèle réel, chargé par transformers sur le CPU : une image 224×224 de
    bruit déterministe passe par SON processeur (image_grid_thw réel), ses
    positions 3-D + delta sont celles de ``get_rope_index`` ; la frontière lue
    est celle du moteur (debut, fin, supplement["image_grid_thw"], merge)."""
    import time
    pytest.importorskip("transformers")
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from acvram.engine.config import load_model_spec
    from acvram.engine.mrope import grille_de
    from acvram.server.chat import _par_image
    if not torch.cuda.is_available():
        pytest.skip("CI sans carte (CUDA_VISIBLE_DEVICES vide) : rien a attendre, pas de queue")
    verrou = "/tmp/acvram-carte-0.lock.qui"  # lecture seule du vrai verrou : on attend, on n'y écrit jamais
    t0 = time.time()
    # Sous une prise de NOTRE chaîne (ACVRAM_CARTE_TENUE : CI publique lancée sous carte.sh par le chef),
    # attendre la carte serait attendre son propre ancêtre : le test tourne sur le CPU sans attendre.
    while (not os.environ.get("ACVRAM_CARTE_TENUE") and os.path.exists(verrou)
           and os.path.getsize(verrou) > 0 and time.time() - t0 < 1200):
        time.sleep(30)                                       # une prise sur la carte : on attend
    os.nice(19)
    chemin = alias(_QVL)
    spec = load_model_spec(chemin, _QVL)
    assert spec.mrope_section == SECTION and spec.spatial_merge_size == MERGE
    proc = AutoProcessor.from_pretrained(chemin)
    g = torch.Generator().manual_seed(20260920)
    pixels = torch.randint(0, 256, (224, 224, 3), generator=g, dtype=torch.uint8).numpy()
    image = Image.fromarray(pixels)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Décris."}]}]
    texte = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    entrees = proc(text=[texte], images=[image], return_tensors="pt")
    ids = entrees["input_ids"][0]
    thw = entrees["image_grid_thw"]
    assert thw.shape == (1, 3) and int(thw[0, 0]) == 1
    t0 = time.time()
    modele = AutoModelForImageTextToText.from_pretrained(chemin, dtype=torch.bfloat16)
    duree = time.time() - t0
    print(f"[qvl-2b] chargement CPU : {duree:.1f} s, grille {thw.tolist()}")
    types = entrees["mm_token_type_ids"]
    ref, deltas = modele.model.get_rope_index(entrees["input_ids"], types, image_grid_thw=thw)
    # la frontière du moteur : plages depuis les jetons image, grille par fragment
    ou = (ids == spec.raw["image_token_id"]).nonzero().flatten().tolist()
    debut, fin = ou[0], ou[-1] + 1
    assert fin - debut == len(ou) == int(thw.prod()) // MERGE ** 2
    frags = _par_image({"pixel_values": entrees["pixel_values"], "image_grid_thw": thw}, 1)
    assert len(frags) == 1 and torch.equal(frags[0][1]["image_grid_thw"], thw)
    frag = type("F", (), {"supplement": frags[0][1]})()
    pos, delta = positions_mrope(len(ids), [(debut, fin, grille_de(frag))], spec.spatial_merge_size)
    assert torch.equal(pos, ref[:, 0]) and delta == int(deltas.reshape(-1)[0]) and delta < 0
    # cos/sin du texte : le rotary du modèle réel sur ses positions, au bit
    c_ref, s_ref = modele.model.language_model.rotary_emb(torch.zeros(1, dtype=torch.bfloat16), ref)
    rope = RotaryEmbedding(spec.head_dim, spec.max_position_embeddings, spec.rope_theta, spec.rope_scaling)
    c, s = rope(pos, CPU, torch.bfloat16, max_pos=len(ids))
    assert torch.equal(c, c_ref[0]) and torch.equal(s, s_ref[0])
    if duree > 60:                                           # signalé, pas rouge (consigne 16 h 10)
        print(f"[qvl-2b] ATTENTION : chargement CPU du 2B en {duree:.0f} s (> 60 s)")
