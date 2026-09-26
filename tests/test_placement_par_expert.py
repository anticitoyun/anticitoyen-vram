"""Placement par expert (bead anticitoyen-vram-pds, point 4 — poste7 §4) :
sortie bit-identique entre une couche 100 % résidente et la MÊME couche à
moitié exilée PAR EXPERT (pas par couche entière). « Placement = vitesse,
jamais sémantique » — REGLES.md : ce test EST cette règle, appliquée au
grain le plus fin que ce chantier introduit.

Style repris de test_moe_chemins_decodage.py (couche réduite mais réelle,
poids NVFP4 identiques entre les deux configurations, comparaison au chemin
`t == 1` du décodage) ; NVFP4 ici plutôt que q3n, parce que c'est le format
du contrat de table (`memory/table_adresses.py`) et du noyau de poste4.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu_requis

CACHE, INTER, N_EXPERTS, TOP_K = 256, 128, 8, 4


def _lineaire(sortie, entree, graine, dev, pool):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4

    g = torch.Generator().manual_seed(graine)
    w = torch.randn(sortie, entree, generator=g) * 0.02
    t = quantize_nvfp4(w)
    lin = QuantLinear(t, out_features=sortie, in_features=entree)
    return lin.to_device(dev, streamed=pool is not None, pool=pool)


def _couche(dev, residents: "set[int] | None"):
    """`residents=None` : tout résident (comportement d'aujourd'hui).
    Sinon : les experts HORS `residents` sont streamés via un `ExpertPool`,
    exactement le chemin qu'un `mlp_storage` de couche entière emprunte déjà
    — appliqué ici par expert."""
    from acvram.engine.layers import ExpertPool
    from acvram.engine.model import MLP, MoEBlock

    pool = ExpertPool(dev, 2 * TOP_K + 2) if residents is not None else None
    experts = []
    for e in range(N_EXPERTS):
        froid = residents is not None and e not in residents
        p = pool if froid else None
        gate = _lineaire(INTER, CACHE, 10 * e + 1, dev, p)
        up = _lineaire(INTER, CACHE, 10 * e + 2, dev, p)
        down = _lineaire(CACHE, INTER, 10 * e + 3, dev, p)
        experts.append(MLP(gate, up, down))

    g = torch.Generator().manual_seed(999)
    routeur_w = torch.randn(N_EXPERTS, CACHE, generator=g) * 0.02
    routeur = _lineaire2(routeur_w, dev)
    bloc = MoEBlock(routeur, experts, TOP_K)
    bloc._stack_state = "non"          # isole le placement du choix pile/boucle
    return bloc


def _lineaire2(w, dev):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4
    t = quantize_nvfp4(w)
    return QuantLinear(t, out_features=w.shape[0],
                       in_features=w.shape[1]).to_device(dev)


@pytest.mark.parametrize("residents", [
    {0, 2, 4, 6},               # les moitiés paire/impaire, arbitraire
    {1, 2, 3},                  # 5/8 exilés
])
def test_moitie_exilee_par_expert_est_bit_identique(residents):
    """`t == 1` (décodage) : le chemin qui appelle `self.experts[e](x)`
    directement, celui que `MoEBlock.forward` emprunte pour un seul jeton."""
    dev = torch.device("cuda")
    tout_resident = _couche(dev, None)
    moitie = _couche(dev, residents)

    torch.manual_seed(42)
    x = torch.randn(1, CACHE, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        y1 = tout_resident(x.clone())
        y2 = moitie(x.clone())

    assert torch.equal(y1, y2), (
        f"placement par expert a changé la sortie (écart max "
        f"{(y1 - y2).abs().max().item()}) — violation de « placement = "
        f"vitesse, jamais sémantique »")


def test_repin_reel_echange_sans_changer_la_sortie():
    """Bead pds point 2 : un échange RÉEL (`_repin_echanger_reel`) — copie
    hôte↔VRAM + mise à jour des trois tables — laisse la sortie inchangée
    pour un jeton qui route l'un OU l'autre des deux experts échangés.
    « Placement = vitesse, jamais sémantique », version dynamique."""
    from acvram.engine.runner import _repin_echanger_reel
    from acvram.memory.table_adresses import construire_table, verifier_table

    dev = torch.device("cuda")
    residents = {0, 1, 2, 3}
    bloc = _couche(dev, residents)
    bloc._table_qw = {}
    bloc._table_bscale = {}
    for nom in ("gate_proj", "up_proj", "down_proj"):
        tq, tb = construire_table(bloc.experts, nom, device=dev)
        bloc._table_qw[nom] = tq
        bloc._table_bscale[nom] = tb

    torch.manual_seed(11)
    x = torch.randn(1, CACHE, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        y_avant = bloc(x.clone())

    # Échange 2 (résident) <-> 5 (froid) : les tables doivent rester valides
    # (verifier_table le lèverait sinon) et le calcul, identique.
    _repin_echanger_reel(bloc, sortant=2, entrant=5)
    for nom in ("gate_proj", "up_proj", "down_proj"):
        verifier_table(bloc._table_qw[nom])
        verifier_table(bloc._table_bscale[nom])
    # L'expert 2 est maintenant froid : le noyau existant (qui ignore encore
    # la table) doit continuer à le lire correctement via `.streamed`.
    assert bloc.experts[2].gate_proj.streamed is not None
    assert bloc.experts[5].gate_proj.streamed is None

    with torch.no_grad():
        y_apres = bloc(x.clone())

    assert torch.equal(y_avant, y_apres), (
        "un échange REPIN réel a changé la sortie — violation de "
        "« placement = vitesse, jamais sémantique »")


def test_chemin_groupe_table_bit_identique_au_chemin_groupe_pile():
    """Suite du point 4 (chef, 13/09) : `_try_build_stacks` ne doit plus
    refuser une couche hétérogène — le chemin groupé TABLE (décodage,
    `nvfp4_gemv_grouped_gateup_table` / `nvfp4_gemv_grouped_table`) doit
    rester bit-identique au chemin groupé PILE existant (même noyau
    `nvfp4_row_dot_warp`, seule l'adresse change).

    Comparé au chemin groupé, PAS à la boucle par expert : les deux
    empruntent des noyaux différents (accumulation d'ordre différent), un
    écart de l'ordre du dernier bit y est déjà connu et accepté (mémoire
    « une divergence n'est pas une preuve », NVFP4 resserre le SNR par
    construction) — ce test isole la seule question qui compte ici :
    l'adressage par table change-t-il le résultat du MÊME noyau ?"""
    from acvram.memory.table_adresses import construire_table

    dev = torch.device("cuda")
    pile = _couche(dev, None)                    # tout résident : chemin groupé pile
    pile._stack_state = "?"

    table = _couche(dev, {0, 2, 4, 6})             # hétérogène : chemin groupé table
    table._table_qw = {}
    table._table_bscale = {}
    for nom in ("gate_proj", "up_proj", "down_proj"):
        tq, tb = construire_table(table.experts, nom, device=dev)
        table._table_qw[nom] = tq
        table._table_bscale[nom] = tb
    table._stack_state = "?"

    torch.manual_seed(123)
    x = torch.randn(1, CACHE, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        y_pile = pile(x.clone())
        y_table = table(x.clone())

    assert pile._stacks["gate_proj"][0] == "nvfp4", "témoin : pile attendue"
    assert table._stack_state == "oui", "le chemin groupé aurait dû s'activer"
    assert table._stacks["gate_proj"][0] == "nvfp4_table", (
        "pendant table non construit -- retombé sur autre chose sans le dire")
    assert torch.equal(y_pile, y_table), (
        f"adressage par table != pile contiguë, même noyau (écart max "
        f"{(y_pile - y_table).abs().max().item()}) — violation de "
        f"« placement = vitesse, jamais sémantique »")


def test_tous_exiles_est_aussi_bit_identique():
    """Cas limite : `residents` vide (aucun résident). Doit rester correct —
    équivalent à l'exil de couche entière d'aujourd'hui, à ce détail près
    que le placement est décrit par expert plutôt que par `mlp_storage`."""
    dev = torch.device("cuda")
    tout_resident = _couche(dev, None)
    tout_exile = _couche(dev, set())

    torch.manual_seed(7)
    x = torch.randn(1, CACHE, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        y1 = tout_resident(x.clone())
        y2 = tout_exile(x.clone())
    assert torch.equal(y1, y2)


def test_repin_sous_graphe_cuda():
    """Suite du point 4 (chef, 14/09) : lever la garde `graphs.py:224`
    pour le chemin table exige de prouver trois conditions avant, pas après.

    (1) Visibilité rejeu : un REPIN réel entre deux `replay()` (même flux,
        même ordre d'émission — aucune synchro ajoutée) doit se refléter au
        rejeu suivant, identique à un bloc frais construit avec le nouveau
        placement d'emblée.
    (2) Formes stables : les tenseurs que le graphe capture (les tables)
        gardent la MÊME adresse après l'échange — seul leur contenu change.
    (3) Un changement qui doit casser (REGLES.md règle 5) : un rejeu SANS
        échange redonne le même résultat (stable au repos) ; une table
        CORROMPUE à la main (une adresse changée à tort) doit, elle, se voir
        au rejeu — pas « le REPIN change la sortie », qu'il ne doit JAMAIS
        faire (placement = vitesse, jamais sémantique) et que (1) vérifie
        déjà en creux, mais « le noyau relit vraiment la table à chaque
        lancement », sans quoi (1) pourrait passer par coïncidence.
    """
    from acvram.engine.runner import _repin_echanger_reel
    from acvram.memory.table_adresses import construire_table

    dev = torch.device("cuda")
    residents = {0, 2, 4, 6}
    bloc = _couche(dev, residents)
    bloc._table_qw = {}
    bloc._table_bscale = {}
    for nom in ("gate_proj", "up_proj", "down_proj"):
        tq, tb = construire_table(bloc.experts, nom, device=dev)
        bloc._table_qw[nom] = tq
        bloc._table_bscale[nom] = tb
    bloc._stack_state = "?"        # laisse _try_build_stacks choisir le groupé

    # Le jeton doit router l'expert échangé (2) : TOP_K=4 sur N_EXPERTS=8 ne
    # le garantit pas pour une graine quelconque -- sans lui l'échange serait
    # invisible en sortie, et (3b) ne prouverait rien qu'un hasard de graine.
    for graine in range(1000):
        torch.manual_seed(graine)
        x = torch.randn(1, CACHE, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            _, topi = bloc._route(x)
        if 2 in topi.reshape(-1).tolist():
            break
    else:
        raise AssertionError("aucune graine testée ne route l'expert échangé")

    with torch.no_grad():
        bloc(x.clone())             # échauffement hors capture : piles, tampons
    assert bloc._stack_state == "oui"
    assert bloc._stacks["gate_proj"][0] == "nvfp4_table", (
        "pendant table non construit -- ce test ne porterait pas sur ce "
        "qu'il pretend tester")

    adr_avant = {nom: (bloc._table_qw[nom].data_ptr(),
                       bloc._table_bscale[nom].data_ptr())
                for nom in bloc._table_qw}

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            bloc(x.clone())         # échauffement sur le flux annexe (contrat capture)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_out = bloc(x.clone())

    # (3a) témoin : sans échange, un second rejeu redonne le même résultat.
    g.replay()
    torch.cuda.synchronize()
    y_avant = static_out.clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, y_avant), (
        "deux rejeux sans échange diffèrent -- le graphe n'est même pas "
        "stable au repos")

    # L'échange REPIN réel, EXACTEMENT comme en service (_repin_pass, fin de
    # step()) : sur le flux courant, sans synchronisation ajoutée -- c'est
    # l'ordre d'émission par flux qui doit suffire (condition 1).
    _repin_echanger_reel(bloc, sortant=2, entrant=5)

    # (2) formes/adresses stables : le graphe a capturé CES tenseurs.
    for nom in bloc._table_qw:
        assert bloc._table_qw[nom].data_ptr() == adr_avant[nom][0]
        assert bloc._table_bscale[nom].data_ptr() == adr_avant[nom][1]

    g.replay()
    torch.cuda.synchronize()
    y_apres = static_out.clone()

    # Référence indépendante : un bloc FRAIS, MÊME chemin groupé table (pas
    # la boucle par expert -- noyau différent, écart au dernier bit déjà
    # connu et accepté, cf. test_chemin_groupe_table_bit_identique_au_
    # chemin_groupe_pile), placement échangé dès le départ.
    reference = _couche(dev, {0, 4, 5, 6})       # 2 dehors, 5 dedans
    reference._table_qw = {}
    reference._table_bscale = {}
    for nom in ("gate_proj", "up_proj", "down_proj"):
        tq, tb = construire_table(reference.experts, nom, device=dev)
        reference._table_qw[nom] = tq
        reference._table_bscale[nom] = tb
    reference._stack_state = "?"
    with torch.no_grad():
        y_reference = reference(x.clone())
    assert reference._stacks["gate_proj"][0] == "nvfp4_table"

    # (1) le rejeu APRÈS échange doit refléter le NOUVEAU placement.
    assert torch.equal(y_apres, y_reference), (
        f"le rejeu après REPIN ne reflète pas le nouveau placement (écart "
        f"max {(y_apres - y_reference).abs().max().item()}) -- condition 1 "
        f"violée : le graphe garde un contenu périmé")

    # (3b) un changement qui DOIT casser : une table CORROMPUE à la main
    # (l'adresse de l'expert 2 pointée, à tort, sur celle de l'expert 0) doit
    # se voir au rejeu suivant. Pas « le REPIN change la sortie » -- il ne le
    # doit JAMAIS (placement = vitesse, jamais sémantique, `y_avant ==
    # y_apres` est attendu et déjà vérifié en creux par (1)) -- mais « le
    # noyau relit vraiment la table, pas une valeur mise en cache ailleurs ».
    # Sans cette preuve, (1) ci-dessus pourrait passer par coïncidence.
    table_gate = bloc._table_qw["gate_proj"]
    adr_correcte = table_gate[2].item()
    table_gate[2] = table_gate[0].item()          # adresse de l'expert 0, a tort
    g.replay()
    torch.cuda.synchronize()
    y_corrompu = static_out.clone()
    table_gate[2] = adr_correcte                  # remis en etat avant de conclure

    assert not torch.equal(y_corrompu, y_apres), (
        "une table délibérément corrompue n'a AUCUN effet au rejeu -- le "
        "noyau ne relit pas la table à chaque lancement, ce que (1) "
        "supposait sans le prouver")


def test_creneaux_fantomes_masques_bit_identique_et_compte_exclu():
    """Bead pds, 14/09 (masquage du remplissage godet, chef) : la sortie
    des jetons RÉELS d'un lot rembourré (b_reel=3 -> godet 4, `bucket_batch`)
    doit être bit-identique au même lot non rembourré, et le remplissage ne
    doit RIEN ajouter à `_usage_routage` -- l'inverse de la cause mesurée le
    14/09 (les 48 couches de Coder-30B routaient x=0 vers 1 à 7 experts
    froids, sans masque, une lecture PCIe inutile par couche à chaque pas).

    « Un changement qui doit casser » (REGLES.md règle 5) : SANS masque, le
    même lot rembourré ajoute bien une sélection au compte -- sinon ce test
    ne prouverait rien, qu'il passe ou non.
    """
    from acvram.memory.table_adresses import construire_table

    dev = torch.device("cuda")
    residents = {0, 2, 4, 6}

    def _table_couche():
        b = _couche(dev, residents)
        b._table_qw = {}
        b._table_bscale = {}
        for nom in ("gate_proj", "up_proj", "down_proj"):
            tq, tb = construire_table(b.experts, nom, device=dev)
            b._table_qw[nom] = tq
            b._table_bscale[nom] = tb
        b._stack_state = "?"
        return b

    torch.manual_seed(202)
    x_reel = torch.randn(3, CACHE, device=dev, dtype=torch.bfloat16)
    x_pad = torch.zeros(4, CACHE, device=dev, dtype=torch.bfloat16)
    x_pad[:3] = x_reel
    valid = torch.tensor([True, True, True, False], device=dev)

    bloc_reel = _table_couche()
    with torch.no_grad():
        y_reel = bloc_reel(x_reel.clone())
    assert bloc_reel._stacks["gate_proj"][0] == "nvfp4_table"
    compte_reel = bloc_reel._usage_routage.clone()

    bloc_masque = _table_couche()
    with torch.no_grad():
        y_masque = bloc_masque(x_pad.clone(), valid=valid)
    compte_masque = bloc_masque._usage_routage.clone()

    assert torch.equal(y_reel, y_masque[:3]), (
        f"le remplissage change la sortie des jetons réels (écart max "
        f"{(y_reel - y_masque[:3]).abs().max().item()})")
    assert torch.equal(compte_reel, compte_masque), (
        "le créneau fantôme masqué a quand même été compté")

    bloc_nu = _table_couche()
    with torch.no_grad():
        bloc_nu(x_pad.clone())            # valid=None : PAS de masquage
    compte_nu = bloc_nu._usage_routage.clone()

    assert not torch.equal(compte_nu, compte_reel), (
        "sans masque, le compte aurait dû différer (le fantôme ajoute une "
        "sélection) -- sinon ce test ne prouve rien")
