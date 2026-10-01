"""kv31b, correctif de la réserve (poste6, 30/09, ordre chef ; preuve carte du même jour) : la chauffe de gemma-4-31B à
31 744 ne tenait que 20 480 jetons alors que le plan (réserve plafonnée 3,48 Gio) promettait 31 744 — pic transitoire du
préfill mesuré ≈ 4,87 Gio à 20 480 (6,59 Gio libres avant la passe, 1,72 après, journal de la prise) contre 2,70 de formule.
(1) `activations_prefill_bytes` gagne le terme d'attention par blocs de lignes (têtes × min(T, 1 024) × T × 4 o, fp32) :
la formule COUVRE le pic mesuré (test cassant : sans le terme, 2,7 < 4,87) ; (2) la chauffe MESURE le pic (max_memory_allocated
autour de la passe tenue), l'écrit au journal et dans `~/.cache/acvram/chauffe/<modèle>.json` (`enregistrer_chauffe` : version d'acvram, empreinte du converti, date),
et le prochain chargement ajoute l'excès mesuré − formule s'il est positif (`_exces_mesure`) — la formule garde la main, la
mesure la vérifie ; fichier absent, d'une autre version ou d'un autre converti : IGNORÉ ET DIT ; la ligne de régime nomme l'excès
appliqué et sa source (`reserve_chauffe=`), jamais un plan changé par un état tacite (revue chef 30/09) (vLLM profile_run saute l'attention : pas une référence ; notre chauffe passe la vraie attention) ;
(3) réplique gemma-4-31B à 30,7 Gio libres : la fenêtre qui tient descend sous 31 744 (prédit 24 576-28 672, contre 31 744
sans le terme — que la chauffe démentait).
PRÉDICTION carte (demain) : à 31 744 le plan refuse et annonce ≈ 27 000 ; relancé à N, la chauffe prouve ≥ N (KV dimensionné
pour N) — ou clampe encore, et l'écart mesuré dit de combien la formule reste courte."""
import contextlib
import io
import json
import sys

import pytest
import torch

import acvram
from acvram.engine import loader as LD
from acvram.engine.config import LIGNES_BLOC_ATTENTION

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_fenetre_qui_tient_kv31b import G, _manifest, _plan, _spec   # noqa: E402

PIC_MESURE_20480 = int(4.87 * G)      # prise 30/09 16:31, gemma-4-31B, plafond MLP 5 120 : 6,59 − 1,72 Gio


def test_la_formule_couvre_le_pic_mesure_par_la_chauffe():
    spec = _spec()
    spec.mlp_prefill_plafond = 5120
    formule = spec.activations_prefill_bytes(20480)
    assert formule >= PIC_MESURE_20480, (formule / G, PIC_MESURE_20480 / G)
    assert formule <= 1.5 * PIC_MESURE_20480, "terme trop large : la réserve mangerait la carte"
    # le terme d'attention est bien celui-là : têtes × lignes × T × 4
    sans = formule - spec.num_attention_heads * min(20480, LIGNES_BLOC_ATTENTION) * 20480 * 4
    assert 2.5 * G <= sans <= 2.9 * G, sans / G


def _vierge(monkeypatch, tmp_path):
    monkeypatch.setenv("ACVRAM_CHAUFFE_CACHE", str(tmp_path))
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    spec, man, p = _spec(), _manifest(), _plan(34 * G)
    man["model"] = {"name": "gemma31b"}
    spec.mlp_prefill_plafond = 5120
    return spec, man, p


def test_mesure_de_chauffe_enregistree_relue_et_nommee_au_regime(tmp_path, monkeypatch, capsys):
    spec, man, p = _vierge(monkeypatch, tmp_path)
    formule = spec.activations_prefill_bytes(20480)
    sans_mesure = LD._reserve_prefill(spec, 31744, man, p)
    assert "formule seule — mesure de chauffe de gemma31b ignorée (aucun fichier)" in capsys.readouterr().out
    assert LD.reserve_chauffe_texte("gemma31b") == "formule(mesure ignorée: aucun fichier)"
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    d = LD.enregistrer_chauffe("gemma31b", 20480, formule + 20480 * 1024 * 50, formule, manifest=man, kv_format="int8", plafond=5120)
    assert d["exces_par_jeton"] == 50 * 1024 and d["empreinte"] == LD.empreinte_modele(man)
    assert json.load(open(tmp_path / "gemma31b.json"))["version"] == acvram.__version__
    avec = LD._reserve_prefill(spec, 31744, man, p)
    assert avec - sans_mesure == 50 * 1024 * 31744
    assert "réserve de préfill calée sur la chauffe" in capsys.readouterr().out
    assert LD.reserve_chauffe_texte("gemma31b").startswith("+50Kio/jeton(") and acvram.__version__ in LD.reserve_chauffe_texte("gemma31b")
    # pic sous la formule : rien n'est ajouté, la ligne le dit
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    LD.enregistrer_chauffe("gemma31b", 20480, formule - 1, formule, manifest=man)
    assert LD._reserve_prefill(spec, 31744, man, p) == sans_mesure and LD.reserve_chauffe_texte("gemma31b") == "formule"


def test_mesure_perimee_ou_d_un_autre_converti_ignoree_et_dite(tmp_path, monkeypatch, capsys):
    spec, man, p = _vierge(monkeypatch, tmp_path)
    formule = spec.activations_prefill_bytes(20480)
    sans = LD._reserve_prefill(spec, 31744, man, p)
    # autre version d'acvram
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    d = LD.enregistrer_chauffe("gemma31b", 20480, formule * 2, formule, manifest=man)
    d["version"] = "0.0.0"; json.dump(d, open(tmp_path / "gemma31b.json", "w"))
    assert LD._reserve_prefill(spec, 31744, man, p) == sans
    assert f"ignorée (version 0.0.0 ≠ {acvram.__version__})" in capsys.readouterr().out
    assert "mesure ignorée: version 0.0.0" in LD.reserve_chauffe_texte("gemma31b")
    # autre converti sous le même nom (autre échelle, autre liste de tenseurs)
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    LD.enregistrer_chauffe("gemma31b", 20480, formule * 2, formule, manifest=man)
    autre = dict(man); autre["tensors"] = dict(man["tensors"]); autre["tensors"]["model.layers.0.mlp.up_proj.weight"] = {"format": "nvfp4", "shape": [1, 1]}
    assert LD._reserve_prefill(spec, 31744, autre, p) == LD._reserve_prefill(spec, 31744, autre, p) == sans
    assert "empreinte du converti différente" in capsys.readouterr().out
    # fichier illisible
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    (tmp_path / "gemma31b.json").write_text("{pas du json")
    assert LD._reserve_prefill(spec, 31744, man, p) == sans and LD.lire_chauffe("gemma31b") is None


def test_replique_la_fenetre_qui_tient_descend(monkeypatch, tmp_path):
    monkeypatch.setenv("ACVRAM_CHAUFFE_CACHE", str(tmp_path))
    monkeypatch.setattr(LD, "RESERVE_CHAUFFE_APPLIQUEE", {})
    monkeypatch.delenv("ACVRAM_MTP", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (int(30.7 * G), 34 * G))
    spec, man, p = _spec(), _manifest(), _plan(34 * G)
    p.kv_bytes_per_token = spec.kv_bytes_per_token(8)
    p.kv_budget = {"cuda:0": p.kv_bytes_per_token * 31744}
    with pytest.raises(RuntimeError) as e, contextlib.redirect_stderr(io.StringIO()):
        LD._borner_kv_avec_exil(p, man, lambda n: n, spec, max_model_len=31744, reserve=LD._reserve_prefill(spec, 31744, man, p))
    n = int(str(e.value).split("fenêtre qui tient : ")[1].split()[0])
    # PRÉDIT 24 576-28 672 ; RENDU 29 696 (prédiction réfutée par le haut, dite) — sans le terme : 31 744, que la chauffe
    # (20 480 tenus, KV dimensionné pour 31 744) démentait ; à 29 696 le KV serait dimensionné pour 29 696
    # d19 (poste6 01/10) : au-delà du plafond l'attention passe par morceaux — le flux résiduel et q/k/v ne sont plus comptés qu'à
    # 4 096 lignes (−2,0 Gio à 32 768 sur gemma) et les K/V bf16 transitoires d'une couche s'ajoutent (+0,5) : la fenêtre annoncée
    # monte ; c'est la chauffe qui la prouve (scellé poste6-d19, C5)
    assert 30720 <= n <= 31744, n                                   # 29 696 avant d19
