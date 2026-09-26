"""La chauffe du contexte sollicite les experts comme une vraie requête (poste7 poste7-s2-k48-feu-vert-21-09 § 2 (d)).
À sec, sur le jouet MoE de test_moe_grouped (fixture importée) ; ce module n'est pas gpu_requis."""
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from test_moe_grouped import tiny_moe  # noqa: F401  (fixture)


def test_une_chauffe_aleatoire_sollicite_plus_d_experts_que_des_uns(tiny_moe):
    """poste7 poste7-s2-k48-feu-vert-21-09 § 2 (d) : ``[1]×N`` route tous les jetons vers les mêmes top_k experts par
    couche (2 × 2 couches = 4) ; la séquence pseudo-aléatoire de la chauffe en sollicite plus (jusqu'à 4 × 2 = 8) —
    c'est la grandeur qui a fait mentir la chauffe homogène (GLM k48 : TENUE 32768 puis 500 OOM réel à ctx − 64).
    Si les deux comptes sont égaux, le test est faux, pas la thèse."""
    from acvram.engine.sampler import SamplingParams
    loaded = load_model(tiny_moe, dtype=torch.float32, device_override="cpu", max_model_len=128, max_concurrent_seqs=2)
    eng = Engine(loaded, None, max_batch_size=1, max_model_len=128, enable_cuda_graphs=False)
    p = SamplingParams(max_tokens=1, temperature=0.0)
    uns = eng._experts_sollicites([1] * 126, p); eng._oublier_la_chauffe()
    alea = eng._experts_sollicites(eng.sequence_de_chauffe(126), p); eng._oublier_la_chauffe()
    assert 0 < uns <= 2 * 2, uns
    assert alea > uns, (alea, uns)
