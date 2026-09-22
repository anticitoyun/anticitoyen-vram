"""D2 — lancer_claude refuse les alias dont le contexte est trop petit pour claude CLI.

Mesure à sec du prompt de démarrage : ~12 682 tokens (tiktoken cl100k, session 22/09).
Constantes dans le script : PROMPT_BASE=15000, MIN_REPONSE=4096 → seuil=19096.

Les tests extraient et testent directement la logique de vérification sans
sourcer l'initialisation complète du script (qui requiert acvram_parc.py).
"""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "parc" / "bin" / "claude-modele"


# Logique de vérification extraite du script — doit rester synchronisée.
# Cassure si PROMPT_BASE ou MIN_REPONSE changent sans mettre à jour ce test.
_VERIF_CTX = r"""
err() { printf '\033[0;31m%s\033[0m\n' "$*" >&2; }
PROMPT_BASE=15000; MIN_REPONSE=4096
ctx="$1"; modele="$2"
if [ -n "$ctx" ] && [ "$ctx" -lt $(( PROMPT_BASE + MIN_REPONSE )) ] 2>/dev/null; then
    err "alias $modele : contexte $ctx tokens insuffisant (prompt de démarrage ~${PROMPT_BASE}, réponse mini ${MIN_REPONSE} — il faut ≥ $(( PROMPT_BASE + MIN_REPONSE )) tokens)"; exit 1
fi
echo "ok"
"""


def _verif(ctx: int, alias: str = "mon-alias") -> tuple[int, str]:
    r = subprocess.run(
        ["bash", "-c", _VERIF_CTX, "--", str(ctx), alias],
        capture_output=True, text=True, timeout=5
    )
    return r.returncode, r.stdout + r.stderr


def test_contexte_trop_petit_refuse():
    """ctx=10000 < 19096 → refus avec le chiffre."""
    rc, out = _verif(10000)
    assert rc != 0, f"doit échouer pour ctx=10000"
    assert "10000" in out, f"le chiffre ctx doit apparaître : {out}"
    assert "19096" in out, f"le seuil 19096 doit apparaître : {out}"


def test_contexte_suffisant_passe():
    """ctx=32768 ≥ 19096 → ok."""
    rc, out = _verif(32768)
    assert rc == 0, f"ne doit pas échouer pour ctx=32768 : {out}"
    assert "ok" in out


def test_contexte_limite_inferieur_refuse():
    """ctx=19095 = seuil-1 → refus."""
    rc, out = _verif(19095)
    assert rc != 0
    assert "19095" in out


def test_contexte_limite_exact_passe():
    """ctx=19096 = seuil exact → ok."""
    rc, out = _verif(19096)
    assert rc == 0


def test_contexte_absent_passe():
    """ctx vide (modèle sans limite) → pas de refus."""
    rc, out = _verif.__wrapped__ if hasattr(_verif, '__wrapped__') else (None, None)
    # Appel direct avec ctx=""
    r = subprocess.run(
        ["bash", "-c", _VERIF_CTX, "--", "", "alias-sans-ctx"],
        capture_output=True, text=True, timeout=5
    )
    assert r.returncode == 0, f"ctx vide ne doit pas refuser : {r.stdout + r.stderr}"
