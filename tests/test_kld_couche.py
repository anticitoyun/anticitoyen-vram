"""KLD couche-par-couche : option ajoutee au convertisseur, jamais un
remplacement du SNR.

Consigne Jerome 12/09/2026, suite de la campagne duck.ai Manon
(revue/duck-manon-12-09.md) : trois modeles a recherche web concordent, le
SNR-bloc seul ne classe pas toujours les tenseurs comme le ferait le KLD au
logit final (Spearman 0,96-0,97 avec les inversions de jeton, methodologie
Fireworks). Le convertisseur continue de decider par SNR ; le KLD est
mesure et publie a cote, a activer par `--mesurer-kld` ou
`ConversionOptions.mesurer_kld`.
"""
import math

import pytest
import torch

from acvram.quant.calibrate import ActStats, kld_couche_bits, quantize_with_calibration


def _snr_db(y_ref: torch.Tensor, y_q: torch.Tensor) -> float:
    """Meme formule que `out_snr_db` dans calibrate.py, isolee ici pour
    comparer les deux metriques sans dupliquer tout le pipeline."""
    err = (y_q - y_ref).norm() / y_ref.norm().clamp(min=1e-12)
    return 20 * math.log10(1.0 / max(err.item(), 1e-12))


def test_kld_est_nul_quand_les_sorties_coincident():
    """Deux lois identiques : rien ne distingue le format quantifie de la
    reference, le KLD doit etre nul."""
    y = torch.randn(64)
    assert kld_couche_bits(y, y) == pytest.approx(0.0, abs=1e-9)


def test_kld_croit_avec_la_perturbation():
    """Sur un bruit croissant ajoute a la MEME sortie de reference, le KLD
    doit croitre : sans cette monotonie, la metrique ne distinguerait pas
    « un peu casse » de « tres casse » et ne vaudrait rien pour classer des
    tenseurs entre eux."""
    torch.manual_seed(0)
    y_ref = torch.randn(128)
    valeurs = []
    for bruit in (0.0, 0.1, 0.5, 1.0, 2.0):
        y_q = y_ref + bruit * torch.randn(128)
        valeurs.append(kld_couche_bits(y_ref, y_q))
    assert valeurs == sorted(valeurs), valeurs


def test_kld_n_est_pas_symetrique():
    """KL(p||q) != KL(q||p) en general. Si les deux sens coincidaient
    toujours, le choix documente dans calibrate.py (KL(p_fp16||p_quant), le
    sens utilise par la litterature de calibration) serait arbitraire."""
    y1 = torch.tensor([3.0, 0.0, 0.0])
    y2 = torch.tensor([0.0, 3.0, 1.0])
    a = kld_couche_bits(y1, y2)
    b = kld_couche_bits(y2, y1)
    assert abs(a - b) > 1e-6, (a, b)


def test_mesurer_kld_est_une_option_qui_ne_change_rien_par_defaut():
    """Sans le drapeau, aucun champ ni aucun cout supplementaire ; avec, le
    reste du resultat (quantification, echelle, tous les autres champs de
    metrics) est identique au bit pres. Sinon ce ne serait pas une option
    mais un chemin de decision different."""
    torch.manual_seed(1)
    w = torch.randn(16, 32)
    st = ActStats(torch.rand(32).clamp(min=0.1), None, 64)

    torch.manual_seed(2)
    _, _, sans = quantize_with_calibration(w, "int4_awq", st, group_size=8)
    torch.manual_seed(2)
    _, _, avec = quantize_with_calibration(w, "int4_awq", st, group_size=8,
                                           mesurer_kld=True)

    assert "out_kld_bits" not in sans
    assert "out_kld_bits" in avec
    assert avec["out_kld_bits"] >= 0.0
    for cle in sans:
        assert sans[cle] == avec[cle], cle


def test_le_snr_et_le_kld_peuvent_classer_deux_candidats_en_sens_oppose():
    """Justifie l'existence des DEUX metriques plutot que de n'en garder
    qu'une : sur un lot de candidats (sortie de reference + bruit), il
    existe au moins une paire ou le classement par SNR et le classement par
    KLD s'inversent. Recherche deterministe (graine fixe) sur des vecteurs
    generes, pas une affirmation a priori sur des exemples choisis a la
    main pour faire mentir le test."""
    torch.manual_seed(3)
    candidats = []
    for _ in range(200):
        n = torch.randint(4, 12, (1,)).item()
        y_ref = torch.randn(n) * (torch.rand(1).item() * 10)
        y_q = y_ref + torch.randn(n) * torch.rand(1).item()
        candidats.append((_snr_db(y_ref, y_q), kld_couche_bits(y_ref, y_q)))

    inversion = any(
        (candidats[i][0] - candidats[j][0]) * (candidats[i][1] - candidats[j][1]) < 0
        for i in range(len(candidats))
        for j in range(i + 1, len(candidats))
    )
    assert inversion, "SNR et KLD ordonnent toujours pareil sur cet echantillon"


def test_ordre_sac_kld_exige_mesurer_kld(tiny_checkpoint, target_rig,
                                        tmp_path_factory, monkeypatch):
    """`ACVRAM_ORDRE_SAC=kld` sans `mesurer_kld` degenererait en silence sur
    une cle jamais calculee — meme garde que les autres modes du sac a dos
    (voir la garde `liste` juste au-dessus dans convert.py)."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("kld_sans_mesure"))
    monkeypatch.setenv("ACVRAM_ORDRE_SAC", "kld")
    try:
        with pytest.raises(ValueError, match="exige.*mesurer_kld"):
            convert_checkpoint(tiny_checkpoint, plan,
                               ConversionOptions(out_dir=out, bits_budget_gib=1.0),
                               spec=spec)
    finally:
        monkeypatch.delenv("ACVRAM_ORDRE_SAC", raising=False)


def test_ordre_sac_kld_publie_son_nom_et_change_les_promotions(
        tiny_checkpoint, target_rig, tmp_path_factory, monkeypatch):
    """Le sac a dos tourne avec `mesurer_kld=True` et `ACVRAM_ORDRE_SAC=kld`,
    le manifeste avoue le mode utilise, et — a budget serre, comme pour
    `genre`/`cout_decroissant` (test_ordre_genre_et_cout) — le jeu de
    promotions differe de celui du SNR par defaut. Si les deux modes
    promouvaient toujours le meme ensemble, le tri par KLD ne servirait a
    rien de plus qu'une reecriture du SNR."""
    import json
    import pathlib

    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))

    def convertir(mode, budget, mesurer_kld):
        out = str(tmp_path_factory.mktemp(f"kld_{mode}"))
        monkeypatch.setenv("ACVRAM_ORDRE_SAC", mode)
        try:
            r = convert_checkpoint(
                tiny_checkpoint, plan,
                ConversionOptions(out_dir=out, bits_budget_gib=budget,
                                  mesurer_kld=mesurer_kld),
                spec=spec)
        finally:
            monkeypatch.delenv("ACVRAM_ORDRE_SAC", raising=False)
        m = json.loads((pathlib.Path(out) / "acvram_manifest.json").read_text())
        return m, sorted(p["name"] for p in r.promotions)

    # meme fenetre de budget que test_ordre_genre_et_cout : assez serree pour
    # qu'un ordre different change reellement l'ensemble des promus.
    m_ref, _ = convertir("snr", 1.0, mesurer_kld=True)
    plancher = m_ref["budget"]["plancher_gib"]
    plafond = m_ref["budget"]["plafond_gib"]
    budget_serre = plancher + (plafond - plancher) * 0.4

    m_snr, promus_snr = convertir("snr", budget_serre, mesurer_kld=True)
    m_kld, promus_kld = convertir("kld", budget_serre, mesurer_kld=True)

    assert m_kld["budget"]["ordre_glouton"] == "kld_couche_par_octet_decroissant"
    assert m_snr["budget"]["ordre_glouton"] == "snr_par_octet_decroissant"
    assert promus_kld, "le mode kld ne promeut rien"
    assert promus_snr != promus_kld, (
        "SNR et KLD promeuvent exactement le meme ensemble a budget serre : "
        f"{promus_snr}")
