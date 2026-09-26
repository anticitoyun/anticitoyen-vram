"""La RMSNorm de l'attention latente passe par le noyau (v0.4.64).

Sept noyaux élémentaires — conversion, carré, moyenne, racine inverse, deux
multiplications, reconversion — pour quelques milliers d'éléments, deux fois
par couche et 47 couches : plus de la moitié des noyaux élémentaires d'un
jeton sur GLM-4.7. Le noyau existant fait le même calcul en un lancement, et
il doit rendre exactement les mêmes bits.
"""
import pytest
import torch


def _reference(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
            ).to(x.dtype) * w


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
@pytest.mark.parametrize("largeur", [512, 576, 768, 2048])
def test_ecart_du_noyau_borne(largeur):
    """Le noyau n'est pas identique au bit près, et il ne peut pas l'être.

    Il reproduit fidèlement les arrondis intermédiaires de la formulation
    PyTorch, mais somme les carrés dans un autre ordre : sa réduction par
    échange de warp donne parfois un dernier bit différent en float32, ce qui
    bascule l'arrondi bf16 d'un ULP. Mesuré sur 31 millions de valeurs :
    0,0006 % d'écarts, tous d'un seul cran de la grille bf16.

    Aucune des deux sommes n'est « la vraie » ; c'est le même noyau qui sert
    déjà toutes les autres normalisations du modèle. Ce test fixe donc ce qui
    est vraiment garanti — l'écart borné — plutôt qu'une égalité que quatre
    lignes de tirage avaient fait croire.
    """
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "rmsnorm_bf16"):
        pytest.skip("extension sans rmsnorm_bf16")
    g = torch.Generator(device="cuda").manual_seed(largeur)
    x = (torch.randn(512, largeur, device="cuda", generator=g) * 2).to(torch.bfloat16)
    w = (torch.randn(largeur, device="cuda", generator=g) * 0.3 + 1).to(torch.bfloat16)
    obtenu, attendu = ext.rmsnorm_bf16(x, w, 1e-6)[0], _reference(x, w, 1e-6)
    differents = (obtenu != attendu).sum().item()
    assert differents / obtenu.numel() < 1e-4          # moins d'un pour dix mille
    if differents:
        # un ULP bf16 vaut 2^-8 en relatif : rien ne doit dépasser ce cran
        ecart = (obtenu.float() - attendu.float()).abs()
        assert (ecart <= attendu.float().abs() * 2 ** -7 + 1e-30).all()


def test_repli_sans_gpu():
    """Sur processeur, ``_norme`` retombe sur la formulation PyTorch."""
    from acvram.engine.mla import MLAttention
    faux = MLAttention.__new__(MLAttention)
    faux.eps = 1e-6
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    w = torch.ones(64, dtype=torch.bfloat16)
    assert torch.equal(faux._norme(x, w), _reference(x, w, 1e-6))


def test_rotation_groupee_identique():
    """q et k tournent d'un seul bloc : mêmes bits, moitié moins de lancements."""
    from acvram.engine.mla import MLAttention

    class Tables:
        def __call__(self, pos, dev, dt, max_pos=None):
            ang = (torch.arange(8, dtype=torch.float32).unsqueeze(0)
                   * pos.unsqueeze(1).float() * 0.01)
            return ang.cos().to(dt), ang.sin().to(dt)

    m = MLAttention.__new__(MLAttention)
    m.rope_emb = Tables()
    q = torch.randn(3, 5, 8, dtype=torch.bfloat16)
    k = torch.randn(3, 8, dtype=torch.bfloat16)
    qa, ka = m._rope(q.clone(), k.clone(), torch.arange(3), 16)

    cos, sin = Tables()(torch.arange(3), None, torch.bfloat16)
    half = cos.shape[-1] // 2
    c, s = cos[..., :half].unsqueeze(1), sin[..., :half].unsqueeze(1)

    def tourner(x):
        x2 = x.reshape(*x.shape[:-1], half, 2)
        x0, x1 = x2[..., 0], x2[..., 1]
        return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).reshape(x.shape)

    assert torch.equal(qa, tourner(q))
    assert torch.equal(ka, tourner(k.unsqueeze(1)).squeeze(1))
