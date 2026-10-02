"""g6r (poste6, 02/10, scellé revue/poste6-g6r-scelle-02-10.md) : cible KV du planificateur sous l'anneau — poids résidents
d'abord. G1 (01/10) : gemma-4-31B à 65 536 servait 39/60 MLP en RAM hôte pour loger 9,44 Gio de cache là où une séquence en
anneau en demande 5,45 ; `kv_min` était calculé au KV plein, avant la pose de l'anneau. À sec, CUDA simulé (comme
test_fenetre_qui_tient_kv31b), sur une réplique de gemma-4-31B aux octets du vrai manifeste (attention 4,44 Gio, MLP 10,90,
embed 2,62, tête liée 1,34, annexes 1,07), carte de 31,84 Gio dont 30,7 libres.
(1) 65 536 : le plan passe sous l'anneau, budget KV = plancher d'UNE séquence, ≤ 16 MLP exilés (ancien code : 39), une séquence
entière loge dans les blocs. (2) 27 648 : anneau, 0 exilé (ancien : KV plein, 49-60). (3) 4 096, témoin : KV plein, plan identique
à `ACVRAM_KV_ANNEAU=0`. (4) anneau interdit : jamais posé. (5) la réserve du plan reste celle des morceaux (la régression de
2808a8190 la portait à 11,7 Gio à 65 536 → refus) et la chauffe compare un seul tenant à la formule du seul tenant.
(6) planificateur : sous l'anneau `kv_max_tokens ≥ max_model_len` ⟺ une séquence loge ; sans anneau, rien ne change."""
import pytest
import torch

from acvram.engine import attention as A
from acvram.engine import loader as LD
from acvram.engine import runner as R
from acvram.engine.config import ModelSpec
from acvram.hardware import detect as DT
from acvram.memory.tiering import PlannerOptions, plan_placement

G, M = 2 ** 30, 2 ** 20
N = 60
TOTAL, LIBRE, LIBRE_PLAN = 32607 * M, int(30.7 * G), int(31.19 * G)


def _spec():
    return ModelSpec(name="gemma31b-replique-g6r", architecture="llama", hidden_size=5376, intermediate_size=21504,
                     num_layers=N, num_attention_heads=32, num_key_value_heads=16, vocab_size=262144,
                     max_position_embeddings=262144, head_dim=256, sliding_window=1024, tie_word_embeddings=True,
                     layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10)


def _manifest():
    t = {}
    for i in range(N):
        t[f"model.layers.{i}.self_attn.q_proj.weight"] = {"format": "nvfp4", "bpw": 4.5, "shape": [8192, 17236]}   # 75,8 Mio
        t[f"model.layers.{i}.mlp.gate_proj.weight"] = {"format": "nvfp4", "bpw": 4.5, "shape": [21504, 16128]}     # 186 Mio
    t["model.embed_tokens.weight"] = {"format": "bf16", "shape": [262144, 5376]}                                   # 2,62 Gio
    t["annexe.poids"] = {"format": "bf16", "shape": [262144, 2192]}                                                # 1,07 Gio
    return {"vision": "non", "tensors": t, "plan": {"tiers": [{"name": "cuda:0", "kind": "gpu"}]},
            "model": {"name": "gemma31b-replique-g6r", "tie_word_embeddings": True, "vocab_size": 262144, "hidden_size": 5376}}


@pytest.fixture
def carte(monkeypatch):
    """Une 5090 simulée, et les seuils globaux de préfill rendus à la sortie (`_plan_from_manifest` les pose)."""
    for v in ("ACVRAM_MTP", "ACVRAM_PLAN_FIGE", "ACVRAM_SANS_REPLAN", "ACVRAM_EXIL_COUCHES"):
        monkeypatch.delenv(v, raising=False)
    gpu = DT.Gpu(index=0, name="RTX 5090 (simulée)", total_mem=TOTAL, used_mem=TOTAL - LIBRE_PLAN, sm=120)
    rig = DT.Rig(gpus=[gpu], host=DT.HostMemory(total=128 * G, available=100 * G), cpu=DT.Cpu(), source="simulée")
    monkeypatch.setattr(DT, "detect_rig", lambda profile=None: rig)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (LIBRE, TOTAL))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(LD, "_KV_ANNEAU_MODE", "auto")
    monkeypatch.setattr(A, "_MLP_MORCEAU", 4096)
    monkeypatch.setattr(R, "_MORCEAU_AU_DELA", 4096)
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    yield rig
    A.definir_seuil(None); R.definir_seuil_morceaux(None)


def _charger(ctx):
    """La chaîne de `load_model` jusqu'aux blocs : plan, pose de l'anneau, borne et exil, découpage en blocs."""
    spec, man = _spec(), _manifest()
    plan = LD._plan_from_manifest(man, spec, max_model_len=ctx)
    dev = lambda nom: nom                                                                   # noqa: E731
    LD._poser_anneau(spec, plan, man, dev, ctx, False)
    LD._borner_kv_avec_exil(plan, man, dev, spec, ctx, reserve=LD._reserve_prefill(spec, ctx, man, plan))
    return spec, plan, LD._kv_blocks_per_device(plan, spec, ctx)


def test_65536_poids_residents_d_abord(carte, capsys):
    spec, plan, blocs = _charger(65536)
    exiles = LD._mlp_exiles(plan)
    plancher = spec.kv_bytes_pour_sequence(65536, anneau=67)
    assert spec.kv_anneau == 67 and plan.kv_anneau == 67
    assert plan.kv_budget["cuda:0"] == LD._kv_plancher(plan, spec, 65536, "cuda:0") == plancher, "le KV cède jusqu'au plancher"
    assert exiles <= 16, f"{exiles} MLP exilés : le KV n'a pas cédé avant les poids (G1 : 39)"
    assert blocs["cuda:0"] * 16 >= 65536 and LD.ANNEAU_SEQS == {"cuda:0": 1}, (blocs, LD.ANNEAU_SEQS)
    assert "ne tient pas ; sous l'anneau" in capsys.readouterr().out
    print(f"g6r 65 536 : {exiles} MLP exilés, budget KV {plancher / G:.2f} Gio, {blocs['cuda:0']} blocs")


def test_27648_anneau_sans_exil(carte):
    spec, plan, blocs = _charger(27648)
    assert spec.kv_anneau == 67 and LD._mlp_exiles(plan) == 0
    assert blocs["cuda:0"] * 16 >= 27648


def test_temoin_4096_plan_identique_sans_l_anneau(carte, monkeypatch):
    spec, plan, blocs = _charger(4096)
    monkeypatch.setattr(LD, "_KV_ANNEAU_MODE", "0")
    spec0, plan0, blocs0 = _charger(4096)
    assert spec.kv_anneau == 0 and plan.kv_anneau == 0 and LD._mlp_exiles(plan) == 0
    assert (plan.kv_budget, blocs, spec.mlp_prefill_plafond) == (plan0.kv_budget, blocs0, spec0.mlp_prefill_plafond)
    assert [l.mlp_storage for l in plan.layers] == [l.mlp_storage for l in plan0.layers]


def test_anneau_interdit_jamais_pose(carte, monkeypatch):
    monkeypatch.setattr(LD, "_KV_ANNEAU_MODE", "0")
    spec, man = _spec(), _manifest()
    plan = LD._plan_from_manifest(man, spec, max_model_len=65536)
    assert plan.kv_anneau == 0 and LD._poser_anneau(spec, plan, man, lambda n: n, 65536, False) == 0
    assert LD._kv_plancher_prevu(plan, spec, 65536, "cuda:0") == spec.kv_bytes_pour_sequence(65536, anneau=0)


def test_creneaux_jamais_pris_sur_le_plancher(carte):
    spec, plan, _ = _charger(65536)
    plancher = LD._kv_plancher(plan, spec, 65536, "cuda:0")
    par_seq = plan.kv_bytes_per_token * 50 * 67 * 16 // 60                    # les 50 anneaux d'une séquence
    plan.kv_budget["cuda:0"] = plancher + 3 * par_seq + par_seq // 2          # trois anneaux de plus, pas quatre
    blocs = LD._kv_blocks_per_device(plan, spec, 65536)
    assert LD.ANNEAU_SEQS["cuda:0"] == 4 and blocs["cuda:0"] * 16 >= 65536, (LD.ANNEAU_SEQS, blocs)


def test_reserve_du_plan_par_morceaux_et_chauffe_au_seul_tenant(monkeypatch):
    s = _spec()
    s.mlp_prefill_plafond = s.prefill_morceau_plafond = 4096
    plan65 = s.activations_prefill_bytes(65536)
    assert 1.93 * G <= plan65 <= 3.5 * G, f"{plan65 / G:.2f} Gio : G1 a mesuré 1,93 par morceaux (régression : 11,7)"
    seul = s.activations_prefill_bytes(16384, seul_tenant=True)
    assert seul >= 3.91 * G > s.activations_prefill_bytes(16384), "seul tenant à 16 384 : pic mesuré 3,91 Gio"
    dense = ModelSpec(name="dense", architecture="llama", hidden_size=5120, intermediate_size=32768, num_layers=40,
                      num_attention_heads=32, num_key_value_heads=8, vocab_size=131072, max_position_embeddings=131072, head_dim=128)
    dense.mlp_prefill_plafond = dense.prefill_morceau_plafond = 4096
    assert dense.activations_prefill_bytes(65536, seul_tenant=True) == dense.activations_prefill_bytes(65536), "sans fenêtre : inchangé"

    # la chauffe : tenu atteint d'un seul tenant → formule du seul tenant ; atteint par morceaux → réserve du plan
    from acvram.engine.contexte import ChauffeContexte
    vus = []
    monkeypatch.setattr(LD, "enregistrer_chauffe", lambda nom, jetons, pic, formule, **k: vus.append(formule) or {"exces_par_jeton": 0})

    class Moteur(ChauffeContexte):
        max_model_len = 16384
    m = Moteur()
    m.model = type("M", (), {"spec": s})()
    m.pic_chauffe, m.morceaux_seuil = (16384, int(3.91 * G)), None
    m._enregistrer_pic_de_chauffe(16384)
    m.morceaux_seuil = 8192
    m._enregistrer_pic_de_chauffe(16384)
    assert vus == [seul, s.activations_prefill_bytes(16384)], [v / G for v in vus]


def test_planificateur_cible_par_sequence(carte):
    s = _spec()
    plein = plan_placement(s, carte, PlannerOptions(max_model_len=65536, max_concurrent_seqs=8, kv_vram_fraction=0.22))
    sous = plan_placement(s, carte, PlannerOptions(max_model_len=65536, max_concurrent_seqs=8, kv_vram_fraction=0.22, kv_anneau=67))
    par_seq = s.kv_bytes_pour_sequence(65536, anneau=67)
    assert plein.kv_anneau == 0 and plein.kv_max_tokens == sum(plein.kv_budget.values()) // plein.kv_bytes_per_token < 65536
    assert sous.kv_anneau == 67 and sous.kv_budget == plein.kv_budget, "même fraction, même budget : seule la lecture change"
    assert sum(sous.kv_budget.values()) >= par_seq and sous.kv_max_tokens >= 65536
    court = plan_placement(s, carte, PlannerOptions(max_model_len=65536, max_concurrent_seqs=8, kv_vram_fraction=0.10, kv_anneau=67))
    assert sum(court.kv_budget.values()) < par_seq and court.kv_max_tokens < 65536
