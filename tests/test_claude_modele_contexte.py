"""D2 puis iqm — ``mode_outils`` de claude-modele : jeu d'outils selon la fenêtre servie, refus nommé si trop petite.

Depuis sf2, acvram rend les outils au gabarit : l'invite de démarrage de claude pèse 24 593-26 076 jetons avec ses
40 outils, 9 764-10 300 avec l'essentiel sans MCP (q40.tsv) (mesures à sec, scratchpad/poste1-iqm-28-09). Complet à partir d'une
fenêtre de 45 000 (chef 28/09), essentiel en dessous. Les tests exécutent la fonction EXTRAITE du script (pas une
copie) : une constante changée dans le script se voit ici.
"""
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "parc" / "bin" / "claude-modele"


def _fonctions() -> str:
    texte = SCRIPT.read_text()
    err = re.search(r"^err\(\) \{.*\}$", texte, re.M).group(0)
    mode = re.search(r"^mode_outils\(\) \{.*?^\}$", texte, re.M | re.S).group(0)
    # ph1 (30/09) : bloc « Seuils » en tête du script (SEUIL_COMPLET … FENETRE_MIN), une seule source lue aussi par la GUI
    constantes = re.search(r"^SEUIL_COMPLET=.*?^FENETRE_MIN=[^\n]*$", texte, re.M | re.S).group(0)
    sortie = re.search(r"^sortie_pour\(\) \{.*?^\}$", texte, re.M | re.S).group(0)
    return f"c_r=; c_0=\n{err}\n{mode}\n{constantes}\n{sortie}\n"


def _mode(ctx: str, *args: str) -> tuple[int, str]:
    prog = _fonctions() + ('mode_outils "$@" || exit 1\n'
                           'printf "%s|%s|%s|%s\\n" "$MODE_OUTILS" "$PROMPT_BASE" "${OUTILS_ARGS[*]}" "$LIGNE_OUTILS"\n')
    r = subprocess.run(["bash", "-c", prog, "--", ctx, "mon-alias", *args],
                       capture_output=True, text=True, timeout=5)
    return r.returncode, r.stdout + r.stderr


def test_fenetre_45k_complet():
    rc, out = _mode("65536")
    assert rc == 0 and out.startswith("complet|27000||complet (fenêtre 65536 ≥ 45000)")


def test_fenetre_32k_essentiel():
    """32 768 : l'invite complète (≥ 24,6 k) plus la sortie ne tiendrait pas — jeu essentiel, dit à l'écran."""
    rc, out = _mode("32768")
    assert rc == 0
    mode, base, args, ligne = out.strip().split("|")
    assert (mode, base, args) == ("essentiel", "11000", "--tools Read,Edit,Bash,Grep --disallowedTools mcp__*")
    assert ligne == "essentiel Read, Edit, Bash, Grep, sans MCP (fenêtre 32768 < 45000)"


def test_seuil_complet_exact():
    assert _mode("45000")[1].startswith("complet|")
    assert _mode("44999")[1].startswith("essentiel|")


def test_fenetre_trop_petite_refusee():
    """Essentiel : 11 000 + 4 096 = 15 096 au moins ; en dessous, refus avec les chiffres."""
    rc, out = _mode("15095")
    assert rc != 0 and "15095" in out and "15096" in out and "essentiel" in out
    assert _mode("15096")[0] == 0


def test_fenetre_inconnue_complet():
    rc, out = _mode("")
    assert rc == 0 and out.startswith("complet|27000||complet (fenêtre inconnue")


def test_tools_utilisateur_prime():
    rc, out = _mode("32768", "-p", "x", "--tools", "Read")
    assert rc == 0 and out.startswith("utilisateur|27000||")


def test_commande_porte_les_outils_du_mode():
    """Le jeu choisi atteint la commande claude (et --afficher la montre)."""
    texte = SCRIPT.read_text()
    assert '"${OUTILS_ARGS[@]}" "$@")' in texte
    assert 'mode_outils "$ctx" "$modele" "$@" || exit 1' in texte
    assert "outils : %s" in texte


def test_fenetre_servie_au_dessus_de_la_colonne_3(tmp_path):
    """edz 28/09 : acvram-serveur relève le contexte à CTX_CLIENT_MIN (29 120 depuis menus, 29 096 alors) pour une colonne 3 à 15 360 ; claude-modele
    ne prenait la fenêtre servie que si elle BAISSAIT le contexte → refus « 15360 insuffisant » d'un serveur qui tient 29 096."""
    import os, stat, subprocess
    racine = Path(__file__).resolve().parents[1]
    def exe(p, corps):
        p.write_text(corps); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"; home.mkdir(); tsv = home / "TSV"; tsv.mkdir(); b = tmp_path / "bin"; b.mkdir()
    (tsv / "acvram-chemins.tsv").write_text(f"acvram-un\t{tmp_path}\t15360\n")
    exe(b / "acvram-serveur", "#!/bin/bash\nexit 0\n")
    exe(b / "curl", "#!/bin/bash\ncase \"$*\" in *models*) printf '%s' "
        "'{\"data\":[{\"id\":\"acvram-un\",\"acvram\":{\"max_model_len\":29120}}]}' ;; esac\n")
    exe(b / "faux-claude", "#!/bin/bash\necho \"CTX=$CLAUDE_CODE_MAX_CONTEXT_TOKENS\"\n")
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\ntsv_dir = "{tsv}"\nbin = "{b}"\nsecrets = "{tmp_path}/absent.env"\n'
                   f'lib = "{racine}/parc/share/kimi-menu.lib.sh"\n[moteurs.acvram]\npresent = true\nport = 8090\n')
    env = {**os.environ, "HOME": str(home), "PATH": f"{b}:/usr/bin:/bin", "ACVRAM_PARC_CONFIG": str(cfg),
           "PARC_EXEC": str(b / "faux-claude")}
    r = subprocess.run(["bash", str(racine / "parc" / "bin" / "claude-modele"), "acvram-un", "-p", "OK"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "CTX=29120" in r.stdout, r.stdout[-400:] + r.stderr[-400:]


def _sortie(ctx: str, invite: str) -> tuple[int, str]:
    prog = _fonctions() + 'MODE_OUTILS=essentiel; sortie_pour "$@"\n'
    r = subprocess.run(["bash", "-c", prog, "--", ctx, invite, "mon-alias"], capture_output=True, text=True, timeout=5)
    return r.returncode, r.stdout + r.stderr


def test_sortie_laisse_la_compaction_possible():
    """menus (poste6 29/09) : claude 2.1.284 compacte à fenêtre − sortie − 13 000 ; la sortie laisse 4 096 jetons de
    conversation avant ce seuil. Invite edz 13 938 sur 32 768 : 32 768 − 13 000 − 13 938 − 4 096 = 1 734."""
    assert _sortie("32768", "13938") == (0, "1734\n")
    assert _sortie("34816", "15128") == (0, "2592\n")


def test_sortie_plafonnee_a_8192():
    assert _sortie("65536", "26000") == (0, "8192\n")


def test_fenetre_trop_petite_pour_l_invite_mesuree():
    """Invite 15 128 (crochets du projet) sur 32 768 : 544 < 1 024 de sortie minimale → refus avec la fenêtre
    nécessaire 15 128 + 13 000 + 4 096 + 1 024 = 33 248 ; à 33 248 exactement, sortie 1 024."""
    rc, out = _sortie("32768", "15128")
    assert rc != 0 and "33248" in out and "15128" in out and "13000" in out and "essentiel" in out
    assert _sortie("33248", "15128") == (0, "1024\n")


def test_lancement_mesure_puis_choisit():
    texte = SCRIPT.read_text()
    assert '{ read -r invite; read -r ORIGINE_INVITE; } < <(mesurer_invite "$url" "$cle" "$modele" "$@")' in texte
    assert 'sortie=$(sortie_pour "$ctx" "$invite" "$modele") || exit 1' in texte
    assert 'export CLAUDE_CODE_MAX_OUTPUT_TOKENS="$sortie"' in texte
