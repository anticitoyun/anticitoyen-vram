"""anticitoyen-vram-6it (ordre chef, 24/09) : `flock` seul ne garantit pas l'ordre d'arrivée
entre plusieurs attendeurs de `outils/carte.sh` (fd 9). Bead réel : poste4 (L, la plus ancienne
attente) bloquée > 540 s pendant qu'poste6 et poste5 enchaînaient des prises b1/b8 courtes,
chacune se remettant aussitôt dans la file après sa propre libération — plusieurs attendeurs
bloqués EN MÊME TEMPS, le gagnant à chaque libération tiré au hasard parmi eux plutôt que le
plus ancien. Correctif : ticket + un seul prétendant en vol à la fois (`outils/carte-ticket.sh`),
voir `revue/poste3-piece-6it-fifo-carte-24-09.md`.

Scénario (`scenario2.sh`) : L + 3 « chaîneurs » se bloquent TOUS EN MÊME TEMPS derrière un
occupant ; chaque chaîneur se remet dans la file dès sa propre libération (« rendue »), jusqu'à
20 passages au total. Métrique : le RANG de passage de L (1 = premier).

Reproduction du défaut ORIGINAL (flock nu) tentée sur deux formes (arrivées fraîches décalées,
puis blocage simultané + rechaînage immédiat) : **0 inversion sur 15 essais** au total — le
défaut de production n'a pas pu être reproduit en isolé (charge système calme, peu
d'attendeurs), contrairement au terrain (charge réelle, davantage de postes, accumulation sur
des minutes). Décision chef (24/09, borne 30 min épuisée sans inversion) : fusionner avec
cette note, le test « 5/5 » (L toujours rang 1 avec le correctif) reste le garde de non-
régression — le correctif est sûr PAR CONSTRUCTION (ticket = un seul prétendant en vol),
indépendamment du mécanisme exact de la famine réelle.

À sec (CUDA_VISIBLE_DEVICES="", `ACVRAM_VERROU` isolé dans `tmp_path` — aucune interférence avec
la vraie carte partagée)."""
from __future__ import annotations

import os
import pathlib
import subprocess

RACINE = pathlib.Path(__file__).resolve().parent.parent
SCENARIO = RACINE / "tests" / "aux" / "carte_ticket_scenario2.sh"


def _rang_de_l(tmp_path, essai, ticket_desactive):
    verrou = tmp_path / f"verrou{essai}.lock"
    # Sans la marque de chaîne tenante : lancé sous une prise carte.sh (tests du chef), le
    # scénario refuserait sinon « attendre son propre ancêtre » (carte.sh:203).
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_CARTE_TENUE"}
    r = subprocess.run(["bash", str(SCENARIO), str(RACINE), str(verrou),
                         "1" if ticket_desactive else "0", "20"],
                        capture_output=True, text=True, timeout=40, env=env)
    assert r.returncode == 0, r.stderr[-500:]
    lignes = [l for l in r.stdout.splitlines() if l]
    assert "L" in lignes, (essai, lignes, r.stderr[-300:])
    return lignes.index("L") + 1, lignes


def test_ticket_l_passe_toujours_au_rang_1(tmp_path):
    """Correctif actif : L (bloqué en premier, avant les 3 chaîneurs) obtient TOUJOURS le
    verrou en premier — déterministe par construction (un seul prétendant en vol vers fd 9 à la
    fois), pas une question de chance. C'est le garde de non-régression de cette pièce."""
    for essai in range(5):
        rang, lignes = _rang_de_l(tmp_path, essai, ticket_desactive=False)
        assert rang == 1, (essai, rang, lignes)


def test_flock_nu_defaut_non_reproduit_en_isole(tmp_path):
    """Documente la tentative de reproduction du défaut ORIGINAL (`ACVRAM_TICKET_DESACTIVE=1`,
    flock nu) — voir la note en tête de fichier. N'ASSERT PAS d'inversion (décision chef,
    24/09) : sur ce banc calme, L gagne aussi sous flock nu (kernel apparemment FIFO pour un
    blocage simultané isolé) — imprime la distribution des rangs pour archive, ne fait pas
    échouer le test si tout est à 1 (le défaut réel demande une charge que ce banc ne produit
    pas). Échoue seulement si le scénario lui-même casse (crash, L absent du journal)."""
    rangs = []
    for essai in range(5):
        rang, _ = _rang_de_l(tmp_path, essai, ticket_desactive=True)
        rangs.append(rang)
    print(f"[6it] rangs de L sous flock nu (5 essais) : {rangs}")
