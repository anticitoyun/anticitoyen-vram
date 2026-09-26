"""Un emplacement négatif ne doit RIEN écrire — et le repli ne le savait pas.

Le rembourrage par godets (à venir) donnera au lot une taille prise dans une
petite liste de valeurs, pour que la clé de graphe CUDA cesse d'exploser. Les
lignes ajoutées sont calculées puis jetées ; elles portent un emplacement
`-1` que le noyau d'écriture doit ignorer.

Le noyau CUDA fusionné le fait déjà (`acvram_kernels.cu:1831`). **Le chemin
PyTorch de repli ne le faisait pas**, et l'indexation négative de PyTorch
reboucle : `-1` écrit dans le DERNIER bloc, décalage `bs - 1`. Le repli sert
les caches non-int8, le processeur et l'absence d'extension — donc la
configuration de ces essais mêmes.
"""
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache


def _cache(dtype="bf16"):
    # `quantized` est DERIVE de `dtype` (int8 | fp8_e4m3), il ne se pose pas.
    cfg = KVCacheConfig(num_layers=1, num_blocks=4, block_size=16,
                        num_kv_heads=2, head_dim=8, dtype=dtype, device="cpu")
    return PagedKVCache(cfg)


def _kv(n, valeur):
    k = torch.full((n, 2, 8), float(valeur), dtype=torch.bfloat16)
    return k, k.clone() * 2


def test_une_ligne_de_rembourrage_n_ecrit_nulle_part():
    """Le fait qui compte : le dernier bloc doit rester intact.

    Sans la garde, `-1` y écrit au décalage 15 — dans un bloc qui appartient
    à une autre séquence, et que le cache de préfixe republiera.
    """
    c = _cache()
    temoin = c.k[-1, 15].clone()
    k, v = _kv(2, 3.0)
    c.write(torch.tensor([0, -1], dtype=torch.long), k, v)
    assert torch.equal(c.k[-1, 15], temoin), \
        "la ligne de rembourrage a ecrit dans le dernier bloc"
    assert torch.equal(c.v[-1, 15], torch.zeros_like(c.v[-1, 15]))


def test_les_lignes_reelles_sont_ecrites_normalement():
    """Contrôle : la garde ne doit pas manger les lignes valides — sinon elle
    remplacerait une corruption par une perte, ce qui n'est pas mieux."""
    c = _cache()
    k, v = _kv(3, 5.0)
    c.write(torch.tensor([0, -1, 17], dtype=torch.long), k, v)
    assert c.k[0, 0].max().item() == 5.0, "la ligne 0 n'a pas ete ecrite"
    assert c.k[1, 1].max().item() == 5.0, "l'emplacement 17 n'a pas ete ecrit"


def test_un_lot_entierement_de_rembourrage_ne_fait_rien():
    """Cas limite : toutes les lignes factices. Filtrer laisserait un tenseur
    vide, et une dispersion sur un tenseur vide ne doit pas lever."""
    c = _cache()
    avant_k, avant_v = c.k.clone(), c.v.clone()
    k, v = _kv(2, 7.0)
    c.write(torch.tensor([-1, -1], dtype=torch.long), k, v)
    assert torch.equal(c.k, avant_k) and torch.equal(c.v, avant_v)


def test_la_garde_vaut_aussi_pour_le_cache_quantifie():
    """Le repli quantifié écrit DEUX tenseurs de plus — les échelles. Une
    garde posée sur les seules clés laisserait l'échelle se corrompre, et le
    défaut ne se verrait qu'à la relecture."""
    c = _cache(dtype="int8")
    temoin = c.k_scale[-1, 15].clone()
    k, v = _kv(2, 3.0)
    c.write(torch.tensor([0, -1], dtype=torch.long), k, v)
    assert torch.equal(c.k_scale[-1, 15], temoin), \
        "l'echelle du dernier bloc a ete ecrasee"


def test_sans_sentinelle_le_piege_mordrait():
    """La preuve que ces épreuves gardent quelque chose : on refait à la main
    ce que faisait le code d'avant, et on vérifie que le dernier bloc est bien
    touché. Un essai qui ne casse pas quand on réintroduit la faute ne garde
    rien."""
    c = _cache()
    sm = torch.tensor([0, -1], dtype=torch.long)
    k, _ = _kv(2, 9.0)
    blk = torch.div(sm, 16, rounding_mode="floor")
    off = sm % 16
    assert blk.tolist() == [0, -1] and off.tolist() == [0, 15], \
        "l'indexation negative de PyTorch ne reboucle plus : revoir la garde"
    c.k[blk, off] = k                      # l'ancien code, tel quel
    assert c.k[-1, 15].max().item() == 9.0, \
        "le piege ne mord pas : cet essai ne prouverait rien"
