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
    return f"c_r=; c_0=\n{err}\n{mode}\n"


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
