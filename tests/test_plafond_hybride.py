"""Plafond des créneaux hybrides d'un graphe (GDN/KDA/Mamba2/MLA) : suit le
--max-batch du moteur quand ACVRAM_HYBRID_SLOTS n'est pas posé (G1 19/09 :
GLM tombait en eager dès b=5 sous `acvram serve` avec le défaut fixe 4 —
155 t/s au lieu de 568 —, `certifie` posant 12 avant l'import ne le voyait
pas) ; la variable posée garde la main ; le régime du moteur porte le plafond."""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from acvram.engine.graphs import GraphRunner                                    # noqa: E402


def test_plafond_suit_le_lot_servi_sauf_variable_posee():
    assert GraphRunner.plafond_hybride(12, env="") == 12
    assert GraphRunner.plafond_hybride(1, env="") == 4                  # jamais sous 4 (l'ancien défaut)
    assert GraphRunner.plafond_hybride(None, env="") == 4
    assert GraphRunner.plafond_hybride(12, env="4") == 4                # la variable posée garde la main
    assert GraphRunner.plafond_hybride(2, env="16") == 16


def test_le_regime_porte_le_plafond():
    src = (pathlib.Path(__file__).resolve().parent.parent / "acvram" / "engine" / "runner.py").read_text()
    assert "max_batch_size=self.max_batch_size" in src                  # le GraphRunner reçoit le lot servi
    assert "hybrides≤" in src                                            # visible dans regime_ligne()


def test_couverture_experts_par_couche():
    """`experts_layout` du régime dit la couverture Marlin PAR COUCHE (GLM :
    33 Marlin / 13 refusées) — « marlin » seul quand toutes le sont, « naturel »
    quand aucune, « marlin(N/M) » sinon."""
    import torch
    from acvram.engine.runner import _couverture_experts
    from acvram.engine.model import MoEBlock

    class Faux(torch.nn.Module):
        def __init__(self, dispositions):
            super().__init__()
            self.blocs = torch.nn.ModuleList()
            for d in dispositions:
                b = MoEBlock.__new__(MoEBlock)
                torch.nn.Module.__init__(b)
                if d is not None:
                    b.__dict__["experts_layout"] = d
                self.blocs.append(b)
    assert _couverture_experts(Faux(["marlin", "marlin"])) == "marlin"
    assert _couverture_experts(Faux([None, None])) == "naturel"
    assert _couverture_experts(Faux(["marlin"] * 33 + [None] * 13)) == "marlin(33/46)"
    assert _couverture_experts(Faux([])) == "aucun"


def test_tout_repli_eager_est_compte():
    """Sage 19/09 : trois replis eager silencieux en une soirée — le GraphRunner
    compte CHAQUE repli (`replis_eager`), pas seulement la première raison ;
    `regime()` porte `repli_eager` et les raisons ; `regime_ligne()` l'imprime."""
    gr = GraphRunner.__new__(GraphRunner)
    gr._raisons_eager_vues = set()
    gr.replis_eager = 0
    gr._eager("lot hybride 12 au-dela du plafond")
    gr._eager("lot hybride 12 au-dela du plafond")
    gr._eager("forme non rejouable")
    assert gr.replis_eager == 3 and len(gr._raisons_eager_vues) == 2
    src = (pathlib.Path(__file__).resolve().parent.parent / "acvram" / "engine" / "runner.py").read_text()
    assert '"repli_eager"' in src and "repli_eager=" in src
    app = (pathlib.Path(__file__).resolve().parent.parent / "acvram" / "server" / "app.py").read_text()
    assert '"repli_eager"' in app
