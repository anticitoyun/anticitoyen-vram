"""Godets sur `b` — la preuve à sec du chantier C4 (revue/chantier-c4-19-09),
celle qui manquait le 10/09 (MECANISMES « prérequis bloquant »).

Sur une couche GDN RÉELLE (`GatedDeltaNet`, projections, convolution à état,
`decode_static` qui mute les tampons en place — gdn.py:221-225), liée par le
`_bind_hybrid` RÉEL de `GraphRunner` et le `static_bind` RÉEL de
`DecoderLayerGDN`, un pas de décodage à godet > lot réel doit rendre :

1. le créneau de rembourrage lié à une sentinelle, à l'état ZÉRO — à chaque
   cycle, pas seulement au premier ;
2. l'état récurrent et la sortie des séquences vivantes IDENTIQUES AU BIT à
   ceux du même pas au lot exact (godet = lot) ;
3. un magasin d'états qui ne porte jamais une clé absente du lot (ni la
   sentinelle, ni un ancien propriétaire réécrit).

Témoins cassants : l'ancienne liaison sur `sids` (enumerate) fait échouer
(2) — le créneau de rembourrage garde le propriétaire évincé, `_la_decode`
avance son état avec la ligne factice, il repart corrompu ; l'ancien
`static_bind` (avant le 19/09, copié ici) fait échouer (1) et (3) — la
sentinelle était exportée sous `store[-1]` puis rechargée.

Sans `transformers` dans le venv (c'est le cas du venv de mesure), la règle
récurrente de référence (`torch_recurrent_gated_delta_rule`) est remplacée par
sa transcription torch locale, pas à pas : les invariants éprouvés ici portent
sur la LIAISON et l'EXPORT des états, pas sur la mathématique de la règle —
n'importe quelle récurrence déterministe qui mute l'état en place les exerce.
Quand `transformers` est là, la référence réelle sert.
"""
import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import acvram.engine.gdn as gdn_module
from acvram import regime
from acvram.engine import graphs
from acvram.engine.gdn import GatedDeltaNet, gdn_available
from acvram.engine.graphs import GraphRunner, bucket_batch, godet_hybride, godet_lot
from acvram.engine.layers import RMSNorm
from acvram.engine.model import DecoderLayerGDN, _STATIC, _sid_fantome

torch.set_num_threads(min(8, torch.get_num_threads()))   # une mesure d'un pair tourne peut-être

H, NK, NV, DK, DV, KER = 32, 2, 2, 8, 8, 4
MAX_SLOTS = 4


# -- règle récurrente de secours (transformers absent) -------------------------

def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _regle_recurrente_locale(query, key, value, g=None, beta=None, initial_state=None,
                             output_final_state=True, use_qk_l2norm_in_kernel=True):
    """Transcription de `torch_recurrent_gated_delta_rule` (transformers,
    modeling_qwen3_next) pour t jetons : q, k [b, t, h, dk], v [b, t, h, dv],
    g, beta [b, t, h] ; rend (sortie [b, t, h, dv], état [b, h, dk, dv])."""
    dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                                  for x in (query, key, value, beta, g)]
    b, h, t, dk = key.shape
    dv = value.shape[-1]
    query = query * dk ** -0.5
    S = (torch.zeros(b, h, dk, dv, dtype=torch.float32) if initial_state is None
         else initial_state.to(torch.float32))
    out = torch.zeros(b, h, t, dv, dtype=torch.float32)
    for i in range(t):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        S = S * g[:, :, i].exp()[..., None, None]
        kv = (S * k_t[..., None]).sum(dim=-2)
        delta = (v_t - kv) * beta[:, :, i][..., None]
        S = S + k_t[..., None] * delta[..., None, :]
        out[:, :, i] = (S * q_t[..., None]).sum(dim=-2)
    return out.transpose(1, 2).contiguous().to(dtype), (S if output_final_state else None)


@pytest.fixture(autouse=True)
def _regle_de_reference(monkeypatch):
    if not gdn_available():
        monkeypatch.setattr(gdn_module, "_refs",
                            lambda: (_regle_recurrente_locale, _regle_recurrente_locale))
    monkeypatch.setattr(gdn_module, "_GDN_VOIE", "torch")   # jamais fla ici : CPU, pas à pas


# -- montage : une couche GDN réelle, un runner réduit à _bind_hybrid ----------

def _couche_gdn(seed: int = 20260919) -> DecoderLayerGDN:
    torch.manual_seed(seed)

    def lin(o, i):
        m = nn.Linear(i, o, bias=False)
        m.weight.data.uniform_(-0.4, 0.4)
        return m

    key_dim, value_dim = NK * DK, NV * DV
    conv_dim = 2 * key_dim + value_dim
    gdn = GatedDeltaNet(qkv=lin(conv_dim, H), gate=lin(value_dim, H), alpha=lin(NV, H),
                        beta=lin(NV, H), out=lin(H, value_dim),
                        conv_weight=torch.empty(conv_dim, KER).uniform_(-0.4, 0.4),
                        dt_bias=torch.empty(NV).uniform_(-0.5, 0.5),
                        a_log=torch.empty(NV).uniform_(-2.0, 1.0),
                        norm_weight=torch.ones(DV),
                        num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV)
    norm = RMSNorm(torch.ones(H), eps=1e-6)
    return DecoderLayerGDN(index=0, gdn=gdn, mlp=None, input_norm=norm, post_norm=norm,
                           device=torch.device("cpu"))


class _Modele:
    dtype = torch.float32


class _Runner(GraphRunner):
    """`_bind_hybrid` réel, sans carte ni modèle : la surface qu'il touche."""

    def __init__(self, couche, ancien: bool = False):
        self.hybrid_layers = [couche]
        self.model = _Modele()
        self.max_model_len = 128
        self.max_slots = MAX_SLOTS
        self.ancien = ancien

    def _bind_hybrid_ancien(self, batch, lb):
        """graphs.py avant 1213554 (11/09) : liaison du LOT RÉEL, pas du godet."""
        sids = batch.seq_ids or list(range(batch.batch_size))
        for layer in self.hybrid_layers:
            store = batch.gdn_store.setdefault(layer.index, {})
            for slot, sid in enumerate(sids):
                layer.static_bind(slot, sid, store, 256, self.model.dtype)

    def lier(self, batch, godet):
        if self.ancien:
            self._bind_hybrid_ancien(batch, 0)
        else:
            self._bind_hybrid(batch, lb=0, b_godet=godet)


class _Lot:
    def __init__(self, seq_ids, store):
        self.seq_ids = seq_ids
        self.batch_size = len(seq_ids)
        self.gdn_store = {0: store}


def _entrees(sids: list, pas: int) -> torch.Tensor:
    """Une ligne déterministe par séquence et par pas (fonction du sid seul :
    les deux bras lisent les mêmes lignes quel que soit le godet)."""
    g = torch.Generator().manual_seed(1000 * pas + 7)
    lignes = {}
    for sid in sorted(set(sids)):
        lignes[sid] = torch.randn(H, generator=g)
    return torch.stack([lignes[s] for s in sids])


def _pas(couche, runner, store, sids, godet, pas) -> torch.Tensor:
    """Un pas de décodage à formes fixes : liaison puis `decode_fixed` réel
    sur `godet` lignes — les lignes de rembourrage à zéro, comme `_fill`."""
    runner.lier(_Lot(sids, store), godet)
    x = torch.zeros(godet, H)
    x[:len(sids)] = _entrees(sids, pas)
    positions = torch.zeros(godet, dtype=torch.long)
    y = couche.decode_fixed(x, positions, positions, positions, positions, 1, None, q_len=1)
    return y[:len(sids)].clone()


def _etats(couche, store, sids) -> dict:
    """L'état de chaque sid, où qu'il vive (créneau ou magasin) : (conv, S)."""
    out = {}
    for sid in sids:
        e = store.get(sid)
        if e is _STATIC:
            e = couche.linear_attn.static_export(couche.statics[couche.static_owners.index(sid)])
        out[sid] = tuple(t.clone() for t in e)
    return out


def _scenario(ancien: bool, godet_fn):
    """[10,20,30,40] → [10,20,30] (un créneau de rembourrage) → [10,20,30,40] :
    le scénario du 10/09. Rend (sorties par pas, états finaux, magasin)."""
    couche = _couche_gdn()
    runner = _Runner(couche, ancien=ancien)
    store: dict = {}
    lots = [[10, 20, 30, 40], [10, 20, 30], [10, 20, 30, 40]]
    sorties = [_pas(couche, runner, store, sids, godet_fn(len(sids)), i)
               for i, sids in enumerate(lots)]
    return sorties, _etats(couche, store, [10, 20, 30, 40]), store, couche


def _godet(b):
    g = godet_hybride(b, MAX_SLOTS)
    assert g is not None
    return g


def _exact(b):
    return b


# -- 1 + 2 + 3 : le pas à godet > lot réel ------------------------------------

def test_pas_a_godet_superieur_au_lot_reel():
    """Godet 4 pour un lot de 3 : sentinelle à zéro, séquences vivantes au
    bit près comme au lot exact, magasin sans clé étrangère au lot."""
    couche_g, couche_e = _couche_gdn(), _couche_gdn()
    rg, re_ = _Runner(couche_g), _Runner(couche_e)
    sg, se = {}, {}
    sids = [10, 20, 30]
    assert _godet(3) == 4 > len(sids)

    rg.lier(_Lot(sids, sg), 4)
    assert couche_g.static_owners == [10, 20, 30, -1]
    assert _sid_fantome(couche_g.static_owners[3])
    for k in ("conv", "S"):
        assert couche_g.statics[3][k].abs().max().item() == 0.0, "créneau de rembourrage non neutre"

    yg = _pas(couche_g, rg, sg, sids, 4, 0)
    ye = _pas(couche_e, re_, se, sids, 3, 0)
    assert torch.equal(yg, ye), "sortie des séquences vivantes ≠ lot exact"
    eg, ee = _etats(couche_g, sg, sids), _etats(couche_e, se, sids)
    for sid in sids:
        for a, b in zip(eg[sid], ee[sid]):
            assert torch.equal(a, b), f"état de {sid} ≠ lot exact"
    assert set(sg) == set(sids), f"clés étrangères au lot dans le magasin : {sorted(sg)}"


def test_scenario_du_10_09_eviction_retour_identique_au_lot_exact():
    """40 quitte le lot un pas (son ancien créneau devient rembourrage), puis
    revient : sorties et états identiques au bit à trois pas au lot exact."""
    sg, eg, store_g, _ = _scenario(False, _godet)
    se, ee, store_e, _ = _scenario(False, _exact)
    for i, (a, b) in enumerate(zip(sg, se)):
        assert torch.equal(a, b), f"pas {i} : sortie ≠ lot exact"
    for sid in (10, 20, 30, 40):
        for a, b in zip(eg[sid], ee[sid]):
            assert torch.equal(a, b), f"état final de {sid} ≠ lot exact"
    assert set(store_g) == {10, 20, 30, 40}, sorted(store_g)
    assert not any(_sid_fantome(k) for k in store_g)


def test_temoin_ancienne_liaison_sur_sids_corrompt_40():
    """Témoin cassant de (2) : avec `enumerate(sids)`, le créneau 3 garde 40
    pendant le pas à trois, `_la_decode` l'avance avec la ligne factice, 40
    repart corrompu — le test précédent DOIT échouer sur ce code."""
    sa, ea, _, couche = _scenario(True, _godet)
    se, ee, _, _ = _scenario(False, _exact)
    assert torch.equal(sa[0], se[0]) and torch.equal(sa[1], se[1]), \
        "les pas sans rembourrage lu doivent rester égaux — sinon le témoin ne prouve rien"
    assert not torch.equal(sa[2], se[2]), (
        "l'ancienne liaison n'a pas corrompu 40 : le scénario ne discrimine plus")
    assert not torch.equal(ea[40][1], ee[40][1])
    assert all(torch.equal(a, b) for a, b in zip(ea[10], ee[10])), "10 ne doit pas bouger"


# -- 1 + 3 sur plusieurs cycles : la sentinelle ne rentre jamais dans le magasin
#
# Sur un GDN, une ligne factice à zéro laisse l'état du créneau de rembourrage
# à zéro (projections et convolution sans biais : q = k = v = 0, S ← S·e^g = 0) :
# l'ancien export sous `store[-1]` y était invisible en valeur. Il ne l'est pas
# sur Mamba2 (mamba2.py:68 `conv_b`, :160 `dt_bias` : l'état bouge sous une
# entrée nulle) — d'où le second montage, un état qui bouge à chaque pas.

class _EtatMobile(nn.Module):
    """Récurrence factice dont l'état avance sous toute entrée (S += 1)."""

    def forward(self, x, etat):
        return x, etat

    def new_static(self, device):
        return {"conv": torch.zeros(4, KER - 1), "S": torch.zeros(1, 2, 2)}

    @staticmethod
    def static_load(st, etat):
        if etat is None:
            st["conv"].zero_(); st["S"].zero_()
            return
        st["conv"].copy_(etat[0]); st["S"].copy_(etat[1])

    @staticmethod
    def static_export(st):
        return (st["conv"].clone(), st["S"].clone())

    def decode_static(self, x, st):
        st["S"].add_(1.0); st["conv"].add_(0.5)
        return x


def _couche_mobile() -> DecoderLayerGDN:
    norm = RMSNorm(torch.ones(H), eps=1e-6)
    return DecoderLayerGDN(index=0, gdn=_EtatMobile(), mlp=None, input_norm=norm,
                           post_norm=norm, device=torch.device("cpu"))


def _cycles(couche):
    """Cinq lots alternés à godet 4, `_bind_hybrid` réel ; relevés du créneau
    3 (propriétaire, |S| max, |conv| max) à chaque pas où il est du rembourrage."""
    runner = _Runner(couche)
    store: dict = {}
    releves = []
    for i, sids in enumerate([[10, 20, 30, 40], [10, 20, 30], [10, 20, 30, 40],
                              [10, 20, 30], [10, 20, 30, 50]]):
        runner.lier(_Lot(sids, store), 4)
        if len(sids) < 4:
            releves.append((couche.static_owners[3],
                            couche.statics[3]["S"].abs().max().item(),
                            couche.statics[3]["conv"].abs().max().item()))
        x = torch.zeros(4, H)
        x[:len(sids)] = _entrees(sids, i)
        p = torch.zeros(4, dtype=torch.long)
        couche.decode_fixed(x, p, p, p, p, 1, None, q_len=1)
    return releves, store


@pytest.mark.parametrize("montage", [_couche_gdn, _couche_mobile], ids=["gdn", "etat-mobile"])
def test_sentinelle_neutre_a_chaque_cycle_et_absente_du_magasin(montage):
    releves, store = _cycles(montage())
    assert [r[0] for r in releves] == [-1, -1]
    for _owner, s_max, conv_max in releves:
        assert s_max == 0.0 and conv_max == 0.0, (
            f"créneau de rembourrage rechargé avec un état non neutre : S {s_max}, conv {conv_max}")
    assert set(store) == {10, 20, 30, 40, 50}, sorted(store)
    assert store[40] is not _STATIC and store[40][1].abs().max().item() > 0.0, \
        "40 évincé par 50 doit être exporté, intact, sous sa propre clé"


def _static_bind_ancien(self, slot, sid, store, max_len, dtype):
    """model.py `static_bind` tel qu'avant le 19/09 (47812d9:2341-2366) :
    la sentinelle suit le chemin ordinaire — exportée sous sa clé à
    l'éviction, rechargée au cycle suivant."""
    la = self.linear_attn
    while len(self.statics) <= slot:
        self.statics.append(self._nouveau_static(max_len, dtype))
        self.static_owners.append(None)
    if self.static_owners[slot] == sid and store.get(sid) is _STATIC:
        return
    prev = self.static_owners[slot]
    if prev is not None and prev != sid and store.get(prev) is _STATIC:
        store[prev] = la.static_export(self.statics[slot])
    self.static_owners[slot] = None
    etat = store.get(sid)
    if etat is _STATIC:
        etat = self._reprendre(sid)
    la.static_load(self.statics[slot], etat)
    store[sid] = _STATIC
    self.static_owners[slot] = sid


def test_temoin_ancien_static_bind_exportait_la_sentinelle(monkeypatch):
    """Témoin cassant de (1) et (3) : l'ancien `static_bind` écrit `store[-1]`
    (les deux montages) et, quand l'état bouge sous une entrée nulle, le
    recharge au second cycle — le test précédent DOIT échouer sur ce code."""
    monkeypatch.setattr(DecoderLayerGDN, "static_bind", _static_bind_ancien)
    releves_gdn, store_gdn = _cycles(_couche_gdn())
    assert -1 in store_gdn, "l'ancien code n'écrit plus sous la sentinelle : témoin caduc"
    releves, store = _cycles(_couche_mobile())
    assert -1 in store
    assert releves[0][1] == 0.0, "au premier cycle l'ancien code partait bien de zéro"
    assert releves[1][1] > 0.0 and releves[1][2] > 0.0, (
        "au second cycle l'ancien code aurait dû recharger un état non neutre")


# -- ACVRAM_GODETS_B : régime ---------------------------------------------------

def test_godets_b_defaut_arrondit_comme_avant():
    """Défaut (variable absente) : comportement en place depuis le 11/09."""
    assert graphs._GODETS_B is True
    for b in range(1, 13):
        assert godet_lot(b) == bucket_batch(b)
        assert godet_hybride(b, 12) == min(bucket_batch(b), 12)
    assert godet_hybride(13, 12) is None


def test_godets_b_zero_lot_exact(monkeypatch):
    monkeypatch.setattr(graphs, "_GODETS_B", False)
    for b in range(1, 13):
        assert godet_lot(b) == b
        assert godet_hybride(b, 12) == b
    assert godet_hybride(13, 12) is None, "au-delà des créneaux : refus, comme au défaut"
    src = inspect.getsource(GraphRunner.preparer)
    assert "godet_lot(b_reel)" in src and "godet_hybride(b_reel, self.max_slots)" in src, \
        "preparer ne passe plus par les fonctions gouvernées : la variable serait inerte"


def test_godets_b_declaree_et_masquable(monkeypatch):
    var = {v.nom: v for v in regime.VARIABLES}["GODETS_B"]
    assert var.defaut == "1" and var.torch == "0"
    assert var.lu_a == ("acvram.engine.graphs", "_GODETS_B")
    monkeypatch.delenv("ACVRAM_GODETS_B", raising=False)
    monkeypatch.setattr(graphs, "_GODETS_B", True)
    fait = regime.masquer(["GODETS_B"])
    assert fait["GODETS_B"] == "ACVRAM_GODETS_B=0"
    assert graphs._GODETS_B is False and godet_lot(3) == 3
    monkeypatch.delenv("ACVRAM_GODETS_B", raising=False)
