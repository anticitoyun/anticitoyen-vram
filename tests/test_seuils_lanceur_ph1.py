"""ph1 (poste6, 30/09, ordre chef) : la GUI claude-modeles recopiait `seuil_minimum = 15096` (poste3-gui, 29/09)
alors que claude-modele, depuis poste6-menus (3fb9af100), refuse par mesure de l'invite et précharge acvram à
CTX_CLIENT_MIN = 29 120 : entre 15 096 et 29 120 la GUI lançait, puis le lanceur refusait. Une seule source
désormais : `claude-modele --seuils` imprime ses seuils, la GUI les lit (menu_modeles.config.seuils_lanceur).

Cassures (chacune vérifiée à la main avant d'écrire ce fichier) :
  * remettre `"seuil_minimum": 15096` (ou tout chiffre) dans claude-modeles → test_gui_sans_copie_chiffree ;
  * remettre `CTX_CLIENT_MIN=29120` en dur dans claude-modele → test_prechargement_utilise_fenetre_min ;
  * changer FENETRE_MIN sans changer ce que la GUI lit (impossible par construction : elle lit le lanceur) —
    test_gui_lit_le_lanceur le prouve avec un faux lanceur, pas avec une seconde copie du chiffre."""
import os
import pathlib
import re
import subprocess

RACINE = pathlib.Path(__file__).resolve().parent.parent
LANCEUR = RACINE / "parc" / "bin" / "claude-modele"
GUI = RACINE / "parc" / "bin" / "claude-modeles"
KIMI = RACINE / "parc" / "bin" / "kimi-modele"
KIMI_GUI = RACINE / "parc" / "bin" / "kimi-modeles"
PY = "/usr/bin/python3"


def _seuils_du_lanceur() -> dict:
    r = subprocess.run([str(LANCEUR), "--seuils"], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    return {k: int(v) for k, v in (l.split("=", 1) for l in r.stdout.splitlines())}


def _lire_par_la_gui(tmp_path, nom) -> str:
    """Joue menu_modeles.config.seuils_lanceur dans un processus neuf (config.py charge le parc à l'import),
    sous un parc.toml minimal, et rend le dict imprimé."""
    parc = tmp_path / "parc.toml"
    parc.write_text(f'[chemins]\nkimi_dir="{tmp_path}"\ntsv_dir="{tmp_path}"\n')
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(parc), "PYTHONPATH": str(RACINE / "parc" / "lib"),
           "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([PY, "-c", f"from menu_modeles.config import seuils_lanceur; print(seuils_lanceur({nom!r}))"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip().splitlines()[-1]


def test_lanceur_imprime_ses_seuils_sans_charger_le_parc(tmp_path):
    """`--seuils` sort avant l'eval python du parc : il répond sous un HOME vide, sans parc.toml, en < 10 s."""
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    r = subprocess.run([str(LANCEUR), "--seuils"], capture_output=True, text=True, env=env, timeout=10)
    assert r.returncode == 0 and r.stdout == "seuil_complet=45000\nseuil_minimum=29120\n", r.stdout + r.stderr


def test_seuil_minimum_est_la_somme_que_sortie_pour_exige():
    """29 120 = PROMPT_BASE_ESSENTIEL 11 000 + TAMPON_COMPACTION 13 000 + MARGE_CONVERSATION 4 096 + SORTIE_MIN 1 024
    (menus, 29/09) — lu dans le lanceur lui-même (`bash -c`), pas recopié ici."""
    r = subprocess.run(["bash", "-c", f'set -a; . <(sed -n "/^SEUIL_COMPLET=/,/^FENETRE_MIN=/p" "{LANCEUR}"); '
                        'echo $((PROMPT_BASE_ESSENTIEL + TAMPON_COMPACTION + MARGE_CONVERSATION + SORTIE_MIN)) $FENETRE_MIN'],
                       capture_output=True, text=True, timeout=10)
    somme, fenetre_min = r.stdout.split()
    assert somme == fenetre_min == str(_seuils_du_lanceur()["seuil_minimum"])


def test_gui_lit_le_lanceur(tmp_path):
    """La GUI rend ce que le lanceur imprime : le vrai lanceur → ses seuils ; un faux lanceur → les siens
    (preuve qu'elle lit, et ne recopie pas)."""
    assert _lire_par_la_gui(tmp_path, "claude-modele") == str(_seuils_du_lanceur())
    faux = tmp_path / "faux-lanceur"
    faux.write_text('#!/bin/sh\n[ "$1" = --seuils ] && printf "seuil_complet=7\\nseuil_minimum=5\\n"\n')
    faux.chmod(0o755)
    assert _lire_par_la_gui(tmp_path, str(faux)) == "{'seuil_complet': 7, 'seuil_minimum': 5}"


def test_gui_lanceur_muet_aucun_seuil(tmp_path):
    """Lanceur en erreur : aucun seuil (0 / None), la GUI n'affirme rien — jamais un chiffre de secours codé en dur."""
    muet = tmp_path / "muet"
    muet.write_text("#!/bin/sh\nexit 3\n")
    muet.chmod(0o755)
    assert _lire_par_la_gui(tmp_path, str(muet)) == "{'seuil_complet': 0, 'seuil_minimum': None}"


def test_gui_sans_copie_chiffree():
    """Aucun `"seuil_complet": <chiffre>` ni `"seuil_minimum": <chiffre>` dans claude-modeles : les deux viennent
    de seuils_lanceur("claude-modele")."""
    texte = GUI.read_text(encoding="utf-8")
    assert not re.search(r'"seuil_(complet|minimum)"\s*:\s*\d', texte), "copie chiffrée d'un seuil du lanceur dans la GUI"
    assert 'seuils_lanceur("claude-modele")' in texte
    # 90q : kimi-modeles de même (seuil_complet 65536 recopié de KIMI_MCP_CTX_MIN, poste3-gui 29/09) — et pas de `None` codé non plus
    texte = KIMI_GUI.read_text(encoding="utf-8")
    assert not re.search(r'"seuil_(complet|minimum)"\s*:', texte), "copie d'un seuil du lanceur dans kimi-modeles"
    assert 'seuils_lanceur("kimi-modele")' in texte


def test_kimi_lanceur_imprime_ses_seuils(tmp_path):
    """90q : `kimi-modele --seuils` sous HOME vide → KIMI_MCP_CTX_MIN (65 536) et seuil_minimum VIDE (aucun plancher de refus) ;
    l'env KIMI_MCP_CTX_MIN reste honoré (même valeur que kimi_local() lira), preuve qu'il n'y a qu'une définition."""
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    r = subprocess.run([str(KIMI), "--seuils"], capture_output=True, text=True, env=env, timeout=10)
    assert r.returncode == 0 and r.stdout == "seuil_complet=65536\nseuil_minimum=\n", r.stdout + r.stderr
    r = subprocess.run([str(KIMI), "--seuils"], capture_output=True, text=True, env={**env, "KIMI_MCP_CTX_MIN": "40000"}, timeout=10)
    assert r.stdout == "seuil_complet=40000\nseuil_minimum=\n"
    texte = KIMI.read_text(encoding="utf-8")
    assert len(re.findall(r"65536", texte)) == 1, "KIMI_MCP_CTX_MIN défini à plus d'un endroit"
    assert _lire_par_la_gui(tmp_path, "kimi-modele") == "{'seuil_complet': 65536, 'seuil_minimum': None}"


def test_prechargement_utilise_fenetre_min():
    """acvram-serveur est préchargé à `CTX_CLIENT_MIN=$FENETRE_MIN`, jamais à un chiffre en dur : le seuil affiché par
    la GUI et la fenêtre demandée au serveur ne peuvent pas diverger."""
    texte = LANCEUR.read_text(encoding="utf-8")
    assert re.search(r'^\s*CTX_CLIENT_MIN=\$FENETRE_MIN ', texte, re.M)
    assert not re.search(r'^\s*CTX_CLIENT_MIN=\d', texte, re.M), "CTX_CLIENT_MIN en dur dans claude-modele"
