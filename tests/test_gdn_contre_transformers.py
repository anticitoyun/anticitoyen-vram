"""La couche GatedDeltaNet reproduit-elle celle de transformers ?

Juge EXTERIEUR. La coherence interne — bloc contre pas a pas — prouve que les
deux chemins font la meme chose, pas que c'est la bonne. Ce fichier compare la
couche entiere a `Qwen3NextGatedDeltaNet` de transformers, memes poids, sur un
balayage en t.

Ecrit le 8/09/2026 pendant la recherche d'un ecart de facteur neuf sur un
modele hybride. Verdict : la couche est exacte, l'ecart est ailleurs.

Deux pieges de montage ont failli faire annoncer un defaut inexistant, et ils
sont figes ici parce qu'ils se reproduiront :

1. `in_proj_qkvz` est ENTRELACE PAR TETE K — pour chaque tete on trouve
   [q (dk), k (dk), v (r*dv), z (r*dv)] avec r = nv/nk. Un decoupage contigu
   global donne 100 % d'ecart. Idem pour `in_proj_ba`, [b (r), a (r)] par tete.
2. La convolution, elle, s'applique sur `mixed_qkv` DEJA desentrelace : ses
   poids sont dans l'ordre contigu. Lui appliquer le desentrelacement des
   projections donne encore 100 % d'ecart.

Le premier montage a ete eprouve avant d'accuser — q, k, v, z, b, a compares a
`fix_query_key_value_ordering`, zero partout — ce qui a fait porter le soupcon
sur le maillon qui ne l'avait pas ete, la convolution. C'est ce controle-la qui
a evite d'annoncer un faux defaut.
"""
import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from acvram.engine.gdn import GatedDeltaNet

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")


def _paire(dev):
    torch.manual_seed(7)
    cfg = Qwen3NextConfig(
        hidden_size=512, num_attention_heads=8, num_key_value_heads=2,
        linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=64, linear_value_head_dim=64,
        linear_conv_kernel_dim=4, num_hidden_layers=2, intermediate_size=1024)
    ref = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).to(dev).to(torch.float32).eval()
    if ref.conv1d.bias is not None:          # acvram n'a pas de biais de conv
        with torch.no_grad():
            ref.conv1d.bias.zero_()

    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    r, hid = nv // nk, cfg.hidden_size

    def depuis(w):
        m = nn.Linear(w.shape[1], w.shape[0], bias=False, device=dev,
                      dtype=torch.float32)
        with torch.no_grad():
            m.weight.copy_(w)
        return m

    with torch.no_grad():
        W = ref.in_proj_qkvz.weight.view(nk, 2 * dk + 2 * r * dv, hid)
        qkv = depuis(torch.cat([W[:, :dk].reshape(-1, hid),
                                W[:, dk:2 * dk].reshape(-1, hid),
                                W[:, 2 * dk:2 * dk + r * dv].reshape(-1, hid)],
                               0).contiguous())
        gate = depuis(W[:, 2 * dk + r * dv:].reshape(-1, hid).contiguous())
        Wba = ref.in_proj_ba.weight.view(nk, 2 * r, hid)
        notre = GatedDeltaNet(
            qkv, gate,
            depuis(Wba[:, r:].reshape(-1, hid).contiguous()),   # alpha = a
            depuis(Wba[:, :r].reshape(-1, hid).contiguous()),   # beta  = b
            depuis(ref.out_proj.weight),
            ref.conv1d.weight.squeeze(1).clone(),               # deja contigu
            ref.dt_bias.clone(), ref.A_log.clone(),
            ref.norm.weight.clone(),
            nk, nv, dk, dv, eps=ref.norm.variance_epsilon).to(dev)
    return cfg, ref, notre


@CUDA
@pytest.mark.parametrize("t", [1, 2, 8, 32, 64, 128, 513, 1024, 2048])
def test_la_couche_reproduit_transformers(t):
    """Meme entree, memes poids : les deux sorties doivent coincider.

    Jusqu'a 32 jetons c'est exact au bit pres. Au-dela, la regle par blocs de
    transformers (chunk_size 64) reordonne ses sommes et l'ecart passe a
    quelques 1e-5 — l'arrondi de son propre decoupage, pas une divergence.

    Le balayage va jusqu'a 2048 et non 512 : la perplexite du modele hybride
    REMONTE au-dela de ~512 positions de contexte, et huit blocs de 64 font
    justement 512. Mesure le 8/09/2026 — il ne se passe rien a cette frontiere.
    Profil par tranche de 128 positions sur t=2048 : mediane 3,5e-7 de bout en
    bout, sans tendance, sans saut. La couche ne derive pas avec la longueur,
    et la cassure de la perplexite vient d'ailleurs.
    """
    dev = torch.device("cuda:0")
    cfg, ref, notre = _paire(dev)
    x = torch.randn(t, cfg.hidden_size, device=dev, dtype=torch.float32) * 0.5

    with torch.no_grad():
        y_nous, _ = notre(x, None)
        y_ref = ref(x.unsqueeze(0))
        if isinstance(y_ref, tuple):
            y_ref = y_ref[0]
    ecart = float((y_nous.float() - y_ref.reshape(t, -1).float()).norm()
                  / y_ref.float().norm().clamp(min=1e-12))
    assert ecart < 1e-4, f"t={t} : {ecart:.3e}"


@CUDA
def test_le_desentrelacement_est_celui_de_transformers():
    """Le montage lui-meme, avant toute comparaison de couche.

    Un garde-fou qu'on n'eprouve pas laisse accuser le mauvais coupable : c'est
    ce controle qui a montre que l'ecart de 100 % venait du montage et non de
    la couche.
    """
    dev = torch.device("cuda:0")
    cfg, ref, _ = _paire(dev)
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    r, hid = nv // nk, cfg.hidden_size

    x = torch.randn(1, 5, hid, device=dev, dtype=torch.float32)
    with torch.no_grad():
        q_r, k_r, v_r, z_r, b_r, a_r = ref.fix_query_key_value_ordering(
            ref.in_proj_qkvz(x), ref.in_proj_ba(x))
        W = ref.in_proj_qkvz.weight.view(nk, 2 * dk + 2 * r * dv, hid)
        Wba = ref.in_proj_ba.weight.view(nk, 2 * r, hid)
        obtenus = {
            "q": ((x @ W[:, :dk].reshape(-1, hid).t()), q_r),
            "k": ((x @ W[:, dk:2 * dk].reshape(-1, hid).t()), k_r),
            "v": ((x @ W[:, 2 * dk:2 * dk + r * dv].reshape(-1, hid).t()), v_r),
            "z": ((x @ W[:, 2 * dk + r * dv:].reshape(-1, hid).t()), z_r),
            "b": ((x @ Wba[:, :r].reshape(-1, hid).t()), b_r),
            "a": ((x @ Wba[:, r:].reshape(-1, hid).t()), a_r),
        }
    for nom, (mien, sien) in obtenus.items():
        assert torch.equal(mien, sien.reshape(mien.shape)), nom
