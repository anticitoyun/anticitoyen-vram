"""Hooks de fake-quant sur les 7 projections (bead anticitoyen-vram-brd,
étape 1) : couverture et effet, vérifiés sur le modèle jouet, sur
processeur — avant de les lancer sur Llama-2-7B, sur carte."""
import torch

from acvram.engine.loader import load_model
from acvram.engine.model import ForwardBatch
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
from acvram.quant.calibrate import ActStats
from outils.hooks_activations_a4 import (GENRES_ATTN, GENRES_MLP,
                                         installer_hooks,
                                         installer_hooks_genres,
                                         installer_hooks_smoothquant,
                                         modules_a_hooker, retirer_hooks)


def _prefill(model, prompt):
    n = len(prompt)
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
                          for i in range(n)])
    batch = ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n],
                         [torch.tensor(blocks)], slots, True)
    return model(batch)


def test_couvre_les_sept_genres_sur_toutes_les_couches(converted):
    """Le modele jouet est un Llama standard (attention + MLP classiques),
    donc les 7 genres doivent etre trouves sur CHAQUE couche — sinon la
    fonction de decouverte ne verrait pas ce qu'elle devrait."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    trouves = modules_a_hooker(model)
    attendus = len(model.layers) * (len(GENRES_ATTN) + len(GENRES_MLP))
    assert len(trouves) == attendus, (
        f"{len(trouves)} modules trouves pour {attendus} attendus sur "
        f"{len(model.layers)} couches : {sorted(trouves)}")
    for i in range(len(model.layers)):
        for genre in GENRES_ATTN:
            assert f"layers.{i}.self_attn.{genre}" in trouves
        for genre in GENRES_MLP:
            assert f"layers.{i}.mlp.{genre}" in trouves


def test_a16_ne_change_rien_a16_est_le_temoin(converted):
    """Le regime `a16` (aucun hook installe) doit rendre EXACTEMENT les
    memes logits qu'un forward sans hooks du tout — sinon le temoin ne
    temoigne de rien."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    sans_hooks = _prefill(model, [5, 42, 7, 99, 13])

    handles, trouves = installer_hooks(model, "a16")
    assert handles == [], "le regime a16 ne doit installer aucun hook"
    avec_a16 = _prefill(model, [5, 42, 7, 99, 13])
    retirer_hooks(handles)
    assert torch.equal(sans_hooks, avec_a16)


def test_a4_et_a8_changent_reellement_la_sortie(converted):
    """Si les hooks A4/A8 ne modifiaient pas les logits, ils ne seraient
    pas branches — verification directe qu'ils le sont bel et bien, sur un
    modele executable de bout en bout."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    reference = _prefill(model, [5, 42, 7, 99, 13])

    for regime in ("a4", "a8"):
        loaded2 = load_model(converted, dtype=torch.float32, device_override="cpu")
        model2 = loaded2.model
        handles, trouves = installer_hooks(model2, regime)
        assert len(handles) == len(trouves) > 0
        sortie = _prefill(model2, [5, 42, 7, 99, 13])
        retirer_hooks(handles)
        assert torch.isfinite(sortie).all(), regime
        assert not torch.equal(sortie, reference), (
            f"regime {regime} ne change pas les logits : hooks inertes ?")


def test_a4_degrade_plus_que_a8_sur_le_modele_jouet(converted):
    """Coherence minimale avant la vraie campagne : le format a 4 bits doit
    s'ecarter DAVANTAGE de la reference bf16 que le format a 8 bits — sinon
    l'ordre de resolution E2M1 < E4M3 ne se retrouve meme pas sur un
    modele jouet, et la campagne sur Llama-2-7B partirait sur une
    hypothese deja fausse."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    reference = _prefill(loaded.model, [5, 42, 7, 99, 13])

    ecarts = {}
    for regime in ("a4", "a8"):
        l = load_model(converted, dtype=torch.float32, device_override="cpu")
        handles, _ = installer_hooks(l.model, regime)
        sortie = _prefill(l.model, [5, 42, 7, 99, 13])
        retirer_hooks(handles)
        ecarts[regime] = (sortie - reference).norm().item()

    assert ecarts["a4"] > ecarts["a8"], ecarts


def test_regime_inconnu_leve():
    class _Faux:
        layers = []
    import pytest
    with pytest.raises(ValueError, match="inconnu"):
        installer_hooks(_Faux(), "a2")


def test_isolation_down_proj_seul_ne_touche_pas_les_autres_genres(converted):
    """`installer_hooks_genres(model, {'down_proj'}, 'a4')` doit hooker
    EXACTEMENT autant de modules que de couches — pas 7x plus — et changer
    les logits (sinon le hook est inerte)."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    reference = _prefill(model, [5, 42, 7, 99, 13])

    handles, trouves = installer_hooks_genres(model, {"down_proj"}, "a4")
    assert len(trouves) == len(model.layers)
    assert all(chemin.endswith("down_proj") for chemin in trouves)
    sortie = _prefill(model, [5, 42, 7, 99, 13])
    retirer_hooks(handles)
    assert not torch.equal(sortie, reference), "hook down_proj seul inerte"


def test_isolation_gate_up_down_ne_touche_pas_attention(converted):
    """`installer_hooks_genres(model, {'gate_proj','up_proj','down_proj'},
    'a4')` (regime 'moe-proj-seul' du bead brd, recadrage de Jerome 13/09 :
    le noyau MMA ne sert que le MLP/experts) doit hooker exactement 3
    modules par couche, aucun genre d'attention."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    reference = _prefill(model, [5, 42, 7, 99, 13])

    handles, trouves = installer_hooks_genres(
        model, {"gate_proj", "up_proj", "down_proj"}, "a4")
    assert len(trouves) == 3 * len(model.layers)
    assert all(chemin.rsplit(".", 1)[-1] in
              {"gate_proj", "up_proj", "down_proj"} for chemin in trouves)
    assert not any("self_attn" in chemin for chemin in trouves)
    sortie = _prefill(model, [5, 42, 7, 99, 13])
    retirer_hooks(handles)
    assert not torch.equal(sortie, reference), "hook gate/up/down inerte"


def test_smoothquant_replie_le_poids_et_hooke_l_activation(converted):
    """Repliement SmoothQuant + hook sur les 7 genres, avec des
    statistiques d'activation synthetiques (pas de vraie calibration ici —
    l'objet est de verifier le CABLAGE, la vraie calibration vient de
    collect_activation_stats sur le point de controle HF, teste par
    ailleurs dans acvram.quant.collect)."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    reference = _prefill(model, [5, 42, 7, 99, 13])

    trouves_avant = modules_a_hooker(model)
    torch.manual_seed(0)
    act_stats = {
        f"model.{chemin}.weight": ActStats(
            torch.rand(mod.in_features) + 0.1,
            torch.rand(mod.in_features) + 0.1, 64)
        for chemin, mod in trouves_avant.items()
    }

    handles, trouves, manques = installer_hooks_smoothquant(model, act_stats, alpha=0.65)
    assert not manques, manques
    assert len(handles) == len(trouves) == len(trouves_avant)
    sortie = _prefill(model, [5, 42, 7, 99, 13])
    retirer_hooks(handles)
    assert torch.isfinite(sortie).all()
    assert not torch.equal(sortie, reference), "smoothquant+hook inerte"


def test_smoothquant_signale_les_genres_sans_statistiques(converted):
    """Un genre absent des statistiques d'activation est COMPTE, pas
    remplace par une echelle inventee — sinon un trou de calibration
    passerait pour une couverture complete."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    handles, trouves, manques = installer_hooks_smoothquant(model, {}, alpha=0.5)
    assert handles == []
    assert set(manques) == set(trouves)


def test_smoothquant_alpha_differents_rendent_des_sorties_differentes(converted):
    """Le balayage d'alpha doit produire des resultats DISTINCTS — sinon
    balayer alpha ne teste rien de plus qu'un seul point."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    trouves_avant = modules_a_hooker(loaded.model)
    torch.manual_seed(1)
    act_stats = {
        f"model.{chemin}.weight": ActStats(
            torch.rand(mod.in_features) + 0.1,
            torch.rand(mod.in_features) + 0.1, 64)
        for chemin, mod in trouves_avant.items()
    }

    sorties = {}
    for alpha in (0.5, 0.65, 0.8):
        l = load_model(converted, dtype=torch.float32, device_override="cpu")
        handles, _, manques = installer_hooks_smoothquant(l.model, act_stats, alpha)
        assert not manques
        sorties[alpha] = _prefill(l.model, [5, 42, 7, 99, 13])
        retirer_hooks(handles)

    assert not torch.equal(sorties[0.5], sorties[0.65])
    assert not torch.equal(sorties[0.65], sorties[0.8])


def test_apres_chargement_de_perplexity_installe_les_hooks(converted,
                                                            tiny_checkpoint):
    """Point d'extension ajoute a `acvram.evaluate.perplexity` : les hooks
    A4 doivent changer la PPL par rapport au temoin A16, sur le meme
    modele jouet et le meme corpus — la verification de bout en bout,
    AVANT de la refaire sur Llama-2-7B avec le vrai corpus, sur carte."""
    import math
    import os
    import shutil

    from acvram.evaluate import perplexity

    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tiny_checkpoint, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(converted, fn))
    if not os.path.isfile(os.path.join(converted, "tokenizer.json")):
        import pytest
        pytest.skip("no tokenizer available")

    handles_captes = []

    def _brancher_a4(model):
        h, trouves = installer_hooks(model, "a4")
        assert trouves, "aucune projection trouvee : rien n'a ete fake-quantifie"
        handles_captes.extend(h)

    r16 = perplexity(converted, window=64, stride=32, max_tokens=256,
                     device="cpu", dtype=torch.float32)
    r4 = perplexity(converted, window=64, stride=32, max_tokens=256,
                    device="cpu", dtype=torch.float32,
                    apres_chargement=_brancher_a4)
    assert handles_captes, "apres_chargement n'a jamais ete appele"
    assert math.isfinite(r4.perplexity)
    assert r4.perplexity != r16.perplexity, (
        "les hooks A4 n'ont change ni la sortie ni la PPL")
