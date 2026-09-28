"""edz 28/09 : le menu masque les alias en PANNE au test réel des menus (outils/test-menus-reels.py) pour CE client ou
au chargement ; config intacte, alias masqué tapé en entier accepté, ACVRAM_MENU_TOUT=1 montre tout. À sec (bash)."""
import os
import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "parc" / "share" / "kimi-menu.lib.sh"
TSV = ("# alias\tdate\tmoteur\tverdict\tetape\tcause\n"
       "acvram-a\t28/09 16:00\tacvram\tPANNE\tclaude\tcontexte\n"         # masqué pour claude seulement
       "acvram-b\t28/09 16:00\tacvram\tPANNE\tpréchargement\tOOM\n"       # masqué pour tous
       "acvram-c\t28/09 16:00\tacvram\tOK\t\t\n"
       "acvram-d\t28/09 16:00\tacvram\tPANNE\tclaude+kimi\tsans Paris\n")  # masqué pour les deux


def _menu(tmp_path, client, entree, **env_sup):
    (tmp_path / "menus-reels.tsv").write_text(TSV)
    script = (f"set -uo pipefail\nerr() {{ echo \"$*\" >&2; }}\nc_t= c_d= c_v= c_r= c_0=\n"
              "lister_alias() { for a in a b c d; do printf 'acvram-%s\\tm\\t32768\\t-\\t-\\t-\\t-\\n' $a; done; }\n"
              f". {LIB}\nchoisir_alias 'Menu' ''\n")
    env = {**os.environ, "PARC_CLIENT": client, "TMR_RESULTATS": str(tmp_path / "menus-reels.tsv"), **env_sup}
    env.pop("ACVRAM_MENU_TOUT", None) if "ACVRAM_MENU_TOUT" not in env_sup else None
    return subprocess.run(["bash", "-c", script], input=entree, capture_output=True, text=True, env=env, timeout=30)


def test_claude_ne_voit_que_c(tmp_path):
    r = _menu(tmp_path, "claude-modele", "\n")
    assert r.returncode == 0 and r.stdout.strip() == "acvram-c", r.stdout + r.stderr[-600:]
    assert "3 alias masqués" in r.stderr and "acvram-a" not in r.stderr


def test_kimi_voit_a_et_c(tmp_path):
    r = _menu(tmp_path, "kimi-modele", "2\n")
    assert r.returncode == 0 and r.stdout.strip() == "acvram-c", r.stdout + r.stderr[-600:]
    assert "2 alias masqués" in r.stderr and "acvram-a" in r.stderr and "acvram-b" not in r.stderr


def test_alias_masque_tape_en_entier_et_tout_voir(tmp_path):
    r = _menu(tmp_path, "claude-modele", "acvram-b\n")
    assert r.returncode == 0 and r.stdout.strip() == "acvram-b", r.stdout + r.stderr[-600:]
    r = _menu(tmp_path, "claude-modele", "\n", ACVRAM_MENU_TOUT="1")
    assert r.returncode == 0 and r.stdout.strip() == "acvram-a" and "masqués" not in r.stderr
