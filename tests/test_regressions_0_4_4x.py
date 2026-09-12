"""Garde-fous des bogues silencieux corrigés les 3 et 4 septembre 2026.

Chacun de ces trois défauts était invisible : le code s'exécutait, produisait
des sorties correctes, et coûtait seulement un ordre de grandeur de
performance. Ce sont donc des tests de *comportement du chemin pris*, pas de
justesse numérique — celle-ci est couverte ailleurs.
"""

import os

import pytest
import torch

from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator


# --------------------------------------------------------------------------
# v0.4.43 : les blocs étaient rendus avant d'être publiés
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exige un GPU")
def test_une_sequence_qui_finit_publie_son_prefixe(converted):
    """Une requête qui s'arrête au premier jeton doit tout de même laisser son
    invite dans le cache de préfixe.

    ``_finish`` vidait ``seq.blocks`` avant que ``_register_complete_blocks``
    ne s'exécute : le cache ne recevait jamais rien, et les huit gigaoctets de
    cache KV hôte restaient inutilisés.
    """
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    charge = load_model(str(converted), dtype=torch.float32)
    moteur = Engine(charge, None, max_batch_size=1, max_model_len=256,
                    enable_cuda_graphs=False)
    invite = list(range(2, 2 + 4 * BLOCK_SIZE))
    for _ in moteur.generate(invite, SamplingParams(temperature=0.0, max_tokens=1)):
        pass
    assert moteur.allocator._by_hash, \
        "aucun bloc publié : le cache de préfixe ne se remplira jamais"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exige un GPU")
def test_le_prefixe_publie_est_ensuite_retrouve(converted):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    charge = load_model(str(converted), dtype=torch.float32)
    moteur = Engine(charge, None, max_batch_size=1, max_model_len=256,
                    enable_cuda_graphs=False)
    if moteur.est_hybride:
        pytest.skip("les hybrides passent par les instantanés d'état récurrent")
    invite = list(range(2, 2 + 4 * BLOCK_SIZE))
    for _ in range(2):
        for _ in moteur.generate(invite, SamplingParams(temperature=0.0, max_tokens=1)):
            pass
    assert moteur.stats.cached_prompt_tokens > 0, \
        "la seconde requête n'a rien repris du cache"


# --------------------------------------------------------------------------
# v0.4.39 : le seuil de bascule INT8 était resté à sa valeur de mise au point
# --------------------------------------------------------------------------


def test_le_seuil_int8_reste_au_croisement_mesure():
    """Le GEMV multi-N relit les poids une fois par tranche de huit ; la
    déquantification les lit, les réécrit en 16 bits et les relit — un coût
    fixe. Le croisement mesuré est à 88 jetons ; un seuil de 8, valeur d'origine,
    faisait payer la déquantification complète à tout prefill interactif.
    """
    from acvram.kernels import _INT8_GEMV_MAX

    assert 32 <= _INT8_GEMV_MAX <= 88, (
        f"seuil INT8 à {_INT8_GEMV_MAX} : hors de la plage utile, "
        "le croisement est à 88 jetons")


def test_le_seuil_nvfp4_vaut_la_valeur_mesuree():
    """Le seuil NVFP4 vaut 32 depuis `824d2fd`, et sa raison a change de signe.

    Ce test defendait `<= 16` en affirmant que « monter au-dela de 8 degrade le
    temps jusqu'au premier jeton ». La mesure du 10/09 sous verrou de carte
    (douze bras, ABBA, une valeur par processus, chaque bras deux fois) dit
    l'inverse :

        Qwen3-4B    12 seq, debit   18,93 p/s -> 53,66   x2,84
        Qwen3-4B    12 seq, TTFT    1017 ms   ->  925    -9,4 %
        Qwen3-4B     1 seq, TTFT     289 ms   ->  228    -21,3 %
        AWAXIS-31B   1 seq, TTFT     464 ms   ->  434     -6,5 %
        les deux     1 seq, decode  neutre (temoin : n=1 ne franchit
                                     le seuil d'aucun cote)

    Dispersion intra-bras au plus 8 ms : chaque ecart vaut 4 a 200 fois cette
    dispersion. L'ancienne mesure qui posait le 8 etait a UNE sequence, regime
    ou le decodage ne franchit jamais le seuil — elle ne disait rien du seul cas
    ou la constante decide, et elle est infirmee jusque dans son propre regime.

    Ce que ce test defend donc maintenant : la valeur mesuree, et le fait qu'on
    ne remonte pas au-dela SANS MESURE. La borne haute est 32 parce que 64 n'a
    jamais ete mesure, pas parce que 64 serait mauvais.
    """
    from acvram.kernels import _NVFP4_GEMV_MAX

    assert _NVFP4_GEMV_MAX == 32, (
        f"seuil NVFP4 a {_NVFP4_GEMV_MAX} : la seule valeur mesuree est 32 "
        "(824d2fd, x2,84 en debit et -21,3 % de TTFT contre 8). Redescendre "
        "vers 8 annule un gain mesure ; monter vers 64 n'a jamais ete mesure. "
        "Changer cette constante demande une manche, pas une intuition.")


# --------------------------------------------------------------------------
# v0.4.42 : le décodage e2m1 / e4m3 hors Blackwell
# --------------------------------------------------------------------------


def test_les_magnitudes_e2m1_tiennent_dans_la_constante():
    """La table en registres du noyau doit rendre exactement les huit
    magnitudes du format, sans quoi tout poids NVFP4 serait faux sur les
    cartes antérieures à Blackwell."""
    attendu = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    table = 0xC8643210
    obtenu = [0.5 * ((table >> (4 * c)) & 0xF) for c in range(8)]
    assert obtenu == attendu


@pytest.mark.debit_requis
@pytest.mark.skipif(not torch.cuda.is_available(), reason="exige un GPU")
def test_le_gemv_nvfp4_ne_part_pas_en_emulation():
    """Sur toute carte, le noyau NVFP4 doit rester du même ordre que le noyau
    INT8 : c'est le signe que la conversion des quartets se fait en registres
    et non par l'émulation logicielle du toolkit, qui coûtait 5,4x.
    """
    import time

    from acvram.kernels import get_extension

    ext = get_extension()
    if ext is None:
        pytest.skip("noyaux CUDA indisponibles")
    M, K = 4096, 2048
    qw = (torch.arange(M * K // 2) % 256).to(torch.uint8).reshape(M, K // 2).cuda()
    bs = (torch.arange(M * K // 16) % 60 + 50).to(torch.uint8).reshape(M, K // 16).cuda()
    x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")

    def debit():
        for _ in range(3):
            ext.nvfp4_gemv(qw, bs, 1.0, x, K)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            ext.nvfp4_gemv(qw, bs, 1.0, x, K)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 30
        return (qw.numel() + bs.numel()) / dt / 1e9

    go_s = debit()
    assert go_s > 300, (
        f"nvfp4_gemv à {go_s:.0f} Go/s : le décodage des quartets est reparti "
        "en émulation logicielle")
