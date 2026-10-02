"""Comparateur de la chaîne S1 (`outils/gpu/mesure/s1-morceaux-comparer.py`) : l'écart se lit sur le logprob du MÊME candidat.

Faute trouvée le 02/10 (revue/poste6-bf16-reduction-verdict-carte-02-10.md, ordre chef) : quand le premier jeton
basculait entre deux sorties, le script soustrayait les logprobs des deux jetons CHOISIS — deux jetons différents. Sur
l'invite dense de S1, deux candidats X et Y se disputent la tête à 0,01-0,08 de logprob ; le témoin reprise affichait
0,0299 (réel 0,041) sous la réduction exacte et 0,00403 (réel 0,014) sous la réduite, et le seuil « 2 × témoin reprise »
de REGLES § 4 reposait sur ces chiffres. Les valeurs ci-dessous sont celles de la prise (logprobs seulement, jetons
remplacés par X, Y, Z). Cassant : chaque test rougit si l'on remet la soustraction des jetons choisis à la position 0."""
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "outils" / "gpu" / "mesure" / "s1-morceaux-comparer.py"
_spec = importlib.util.spec_from_file_location("s1_morceaux_comparer", _SCRIPT)
CMP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CMP)

SUITE = {"Z": -0.5, "X": -4.0, "Y": -4.2}                       # positions suivantes : le choix n'y fait pas débat


def _ecrire(base: Path, bras: str, positions: list[dict], fichier: str = "completion.json", morceaux: int = 0) -> None:
    """Un bras de prise : à chaque position un top {jeton: logprob}, le jeton choisi est le plus probable."""
    d = base / f"poste6-s1-{bras}"
    d.mkdir(parents=True, exist_ok=True)
    choisis = [max(p, key=p.get) for p in positions]
    (d / fichier).write_text(json.dumps({
        "choices": [{"finish_reason": "length", "logprobs": {
            "tokens": choisis, "token_logprobs": [p[c] for p, c in zip(positions, choisis)], "top_logprobs": positions}}],
        "usage": {"prompt_tokens": 7865}}), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps({"engine": {"prefill_morceaux": morceaux}}), encoding="utf-8")
    (d / "serveur.log").write_text("[acvram] régime NOMINAL — couches_exilées=0/40"
                                   + (" prefill=bf16(morceaux@4096)" if morceaux else "") + "\n", encoding="utf-8")


def _lu(tmp_path: Path, positions: list[dict], nom: str) -> dict:
    _ecrire(tmp_path, nom, positions)
    return CMP.lire(tmp_path, nom)


def test_premier_jeton_bascule_meme_candidat_pas_deux_jetons(tmp_path):
    """Reprise sous réduction exacte : X passe de −3,1604 à −3,2010 (0,0406), Y prend la tête à −3,1903."""
    ref = _lu(tmp_path, [{"X": -3.1604, "Y": -3.1981, "Z": -3.4492}, SUITE], "A1")
    rep = _lu(tmp_path, [{"X": -3.2010, "Y": -3.1903, "Z": -3.4565}, {"Z": -9.0, "X": -0.1, "Y": -5.0}], "A2")
    d, n, pos = CMP.ecart_candidats(ref, rep)
    assert pos == 0 and n == 3                                  # la position 1 n'a plus le même contexte : non comparée
    assert d == pytest.approx(0.0406, abs=1e-9)
    assert abs(d - 0.0299) > 0.005, "0,0299 = |X choisi par l'un − Y choisi par l'autre| : deux jetons différents"


def test_un_ecart_affiche_petit_cachait_un_ecart_trois_fois_plus_grand(tmp_path):
    """Reprise sous réduction réduite : l'ancien calcul rendait 0,0041, le même candidat (Y) s'écarte de 0,0141."""
    ref = _lu(tmp_path, [{"X": -3.1997, "Y": -3.1904}, SUITE], "A1")
    rep = _lu(tmp_path, [{"X": -3.1945, "Y": -3.2045}, SUITE], "A2")
    d, n, pos = CMP.ecart_candidats(ref, rep)
    assert pos == 0 and n == 2 and d == pytest.approx(0.0141, abs=1e-9)


def test_sans_divergence_toutes_les_positions_comptent(tmp_path):
    ref = _lu(tmp_path, [{"X": -3.16, "Y": -3.20}, SUITE], "A1")
    aut = _lu(tmp_path, [{"X": -3.17, "Y": -3.20}, {"Z": -0.5, "X": -4.0, "Y": -4.25}], "A2")
    d, n, pos = CMP.ecart_candidats(ref, aut)
    assert pos is None and n == 5 and d == pytest.approx(0.05, abs=1e-9)


def test_le_candidat_de_la_reference_absent_ne_se_remplace_pas(tmp_path):
    ref = _lu(tmp_path, [{"X": -3.16, "Y": -3.20}, SUITE], "A1")
    aut = _lu(tmp_path, [{"Y": -3.19, "Z": -3.40}, SUITE], "A2")       # X hors du top de l'autre : rien de comparable
    d, _, pos = CMP.ecart_candidats(ref, aut)
    assert d == float("inf") and pos == 0


def _prise(base: Path, reprise: list[dict], b: list[dict]) -> None:
    a = [{"X": -3.1604, "Y": -3.1981, "Z": -3.4492}, SUITE]
    _ecrire(base, "A1", a)
    _ecrire(base, "A1", reprise, fichier="completion-2.json")
    _ecrire(base, "A2", a)
    _ecrire(base, "B", b, morceaux=4)


def test_p3_juge_sur_le_meme_candidat(tmp_path, capsys):
    """B bascule le premier jeton : son Y choisi (−3,1650) est à 0,0046 du X choisi par A1 — l'ancien calcul disait « tenu »
    (seuil 0,0598) ; le même candidat X s'écarte de 0,1396 pour un seuil de 2 × 0,0406 : FAUX."""
    _prise(tmp_path, reprise=[{"X": -3.2010, "Y": -3.1903, "Z": -3.4565}, SUITE],
           b=[{"X": -3.3000, "Y": -3.1650, "Z": -3.4500}, SUITE])
    assert CMP.main([str(tmp_path)]) == 1
    sortie = capsys.readouterr().out
    assert "Δ 0.0406 sur 3 valeurs, jetons choisis divergents à la position 0" in sortie      # témoin reprise
    assert "P3 B/A1 : Δ 0.14 sur 3 valeurs" in sortie and "= 0.0812 : FAUX" in sortie


def test_p3_tenu_quand_b_reste_dans_deux_fois_le_temoin(tmp_path):
    _prise(tmp_path, reprise=[{"X": -3.2010, "Y": -3.1903, "Z": -3.4565}, SUITE],
           b=[{"X": -3.2100, "Y": -3.1650, "Z": -3.4500}, SUITE])        # bascule aussi, X à 0,0496 ≤ 0,0812
    assert CMP.main([str(tmp_path)]) == 0


def test_un_temoin_incomparable_ne_rend_pas_p3_tenu(tmp_path, capsys):
    _prise(tmp_path, reprise=[{"Y": -3.19, "Z": -3.40}, SUITE], b=[{"X": -3.1604, "Y": -3.1981, "Z": -3.4492}, SUITE])
    assert CMP.main([str(tmp_path)]) == 1
    assert "= inf : FAUX" in capsys.readouterr().out
