"""`MoEBlock._tuiles` à grille fixe (bead runner, 14/09 soir — prérequis (ii)
du levier MoE MMA décodage, poste7). L'ancien `int(ntiles.sum())` synchronisait
l'hôte, incapturable dans un graphe CUDA. Fonctionne sur CPU (tensor ops
génériques) : aucun de ces tests n'a besoin de carte.
"""
import math

import torch

from acvram.engine.model import MoEBlock


def _tuiles_ancien(cnt: torch.Tensor, bt: int = 16):
    """Reimplementation FIDELE de l'ancienne fonction (avant le 14/09 soir),
    gardee ici seulement pour verifier l'equivalence -- ne pas reutiliser
    ailleurs, elle synchronise l'hote."""
    dev = cnt.device
    starts = torch.cumsum(cnt, 0) - cnt
    ntiles = (cnt + bt - 1) // bt
    tot = int(ntiles.sum())
    if tot == 0:
        vide = torch.zeros(0, dtype=torch.int32, device=dev)
        return vide, vide, vide
    tile_e = torch.repeat_interleave(torch.arange(cnt.numel(), device=dev), ntiles)
    base = torch.cumsum(ntiles, 0) - ntiles
    idx = torch.arange(tot, device=dev) - torch.repeat_interleave(base, ntiles)
    t0 = starts[tile_e] + idx * bt
    n = torch.clamp(cnt[tile_e] - idx * bt, max=bt)
    return (tile_e.to(torch.int32), t0.to(torch.int32), n.to(torch.int32))


CAS = [
    ("uniforme", [4, 4, 4, 4], 16, 2),
    ("un_seul_expert", [37, 0, 0, 0], 16, 2),
    ("disperse_un_jeton", [1, 1, 1, 1, 1, 1, 1, 1], 16, 4),
    ("tout_vide", [0, 0, 0, 0], 16, 2),
    ("pas_multiple_de_bt", [17, 3, 0, 40], 16, 3),
    ("un_expert_beaucoup", [1, 1, 1, 200], 8, 1),
    ("bt_egal_1", [3, 0, 2], 1, 1),
]


def test_equivalence_portion_valide():
    """Le prefixe [0:tot] de la nouvelle grille (fixe) est BIT-IDENTIQUE à
    l'ancienne sortie (taille variable) -- même ordre, mêmes valeurs. Un
    changement qui casse : inverser l'ordre des experts dans `base`, ou
    utiliser `left` au lieu de `right` dans `searchsorted`, romprait ce
    test (décalage d'un cran aux frontières inter-experts)."""
    for nom, cnt_list, bt, marge in CAS:
        cnt = torch.tensor(cnt_list, dtype=torch.int64)
        e_a, t0_a, n_a = _tuiles_ancien(cnt, bt)
        tot = e_a.numel()
        t_max = tot + marge
        e_n, t0_n, n_n = MoEBlock._tuiles(cnt, bt, t_max)
        assert e_n.numel() == t_max, nom
        assert torch.equal(e_n[:tot], e_a), f"{nom} : tile_e diverge"
        assert torch.equal(t0_n[:tot], t0_a), f"{nom} : tile_t0 diverge"
        assert torch.equal(n_n[:tot], n_a), f"{nom} : tile_n diverge"


def test_remplissage_neutralise_et_valide():
    """Les tuiles au-delà du compte réel : n=0 (aucun jeton, aucune écriture
    dans le noyau -- garde `j < nt` déjà présente pour la dernière tuile
    partielle d'un expert, acvram_kernels.cu:~1899/1940) et `e` reste un
    indice VALIDE (les noyaux groupés déréférencent gscales[e]/table_qw[e]
    AVANT de lire n : acvram_kernels.cu:1879-1882, 2067-2072 -- pas de garde
    e<0 dans CES noyaux, contrairement au routage `bucket_batch`)."""
    for nom, cnt_list, bt, marge in CAS:
        cnt = torch.tensor(cnt_list, dtype=torch.int64)
        E = cnt.numel()
        e_a, _, _ = _tuiles_ancien(cnt, bt)
        tot = e_a.numel()
        t_max = tot + marge
        e_n, _, n_n = MoEBlock._tuiles(cnt, bt, t_max)
        if marge:
            assert torch.all(n_n[tot:] == 0), f"{nom} : remplissage avec n != 0"
        assert torch.all(e_n >= 0) and torch.all(e_n < max(E, 1)), \
            f"{nom} : indice d'expert hors bornes dans le remplissage"


def test_bt_variable_reste_coherent():
    """bt different de 16 (chemin MMA, _MOE_MMA_BT dans {16,32,64,128})."""
    for bt in (16, 32, 64, 128):
        cnt = torch.tensor([9, 0, 200, 5, 130], dtype=torch.int64)
        e_a, t0_a, n_a = _tuiles_ancien(cnt, bt)
        tot = e_a.numel()
        e_n, t0_n, n_n = MoEBlock._tuiles(cnt, bt, tot + 5)
        assert torch.equal(e_n[:tot], e_a)
        assert torch.equal(t0_n[:tot], t0_a)
        assert torch.equal(n_n[:tot], n_a)


def test_borne_ceil_t_sur_bt_plus_e_suffit_toujours():
    """Propriete : pour toute repartition de T jetons (T=t*top_k) sur E
    experts, le nombre reel de tuiles ne depasse JAMAIS ceil(T/bt) + E --
    la borne que les deux sites d'appel (model.py, _forward_prefill_grouped)
    calculent pour construire t_max. Balayage exhaustif de repartitions
    adverses (concentrees, dispersees, mixtes) sur plusieurs (T, E, bt) --
    un changement qui doit casser : reduire la borne a ceil(T/bt) seul (sans
    +E) romprait ce test sur le cas disperse."""
    torch.manual_seed(0)
    for T, E, bt in [(48, 128, 16), (96, 32, 16), (2048 * 8, 128, 16),
                     (1, 4, 16), (0, 8, 16), (17, 3, 4)]:
        bornes = int(math.ceil(T / bt)) + E if bt else E

        # pire cas construit a la main : un jeton par expert tant qu'il en
        # reste, le reste empile sur le dernier expert non vide.
        if E > 0:
            base_par_expert = min(T, E)
            cnt = [1] * base_par_expert + [0] * (E - base_par_expert)
            reste = T - base_par_expert
            if reste > 0 and base_par_expert > 0:
                cnt[base_par_expert - 1] += reste
            cnt_t = torch.tensor(cnt, dtype=torch.int64)
            ntiles = (cnt_t + bt - 1) // bt if bt else cnt_t.clamp(max=1)
            reel = int(ntiles.sum())
            assert reel <= bornes, f"T={T} E={E} bt={bt} : {reel} tuiles > borne {bornes}"

        # quelques repartitions aleatoires en plus, meme (T,E,bt)
        if E > 0 and T > 0:
            for _ in range(20):
                coupures = torch.sort(torch.randint(0, T + 1, (max(E - 1, 0),))).values
                bornes_coupures = torch.cat([torch.tensor([0]), coupures, torch.tensor([T])])
                cnt_t = (bornes_coupures[1:] - bornes_coupures[:-1]).to(torch.int64)
                ntiles = (cnt_t + bt - 1) // bt
                reel = int(ntiles.sum())
                assert reel <= bornes, f"T={T} E={E} bt={bt} (aleatoire) : {reel} > {bornes}"
