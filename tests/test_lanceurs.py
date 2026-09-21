"""Vérification que les lanceurs (vllm-serveur, etc.) sont dans parc/bin et affichent leurs paramètres."""
import subprocess
import pathlib

PARC_BIN = pathlib.Path(__file__).resolve().parent.parent / "parc" / "bin"


def test_vllm_serveur_affiche_le_backend_choisi():
    """vllm-serveur --afficher <alias> imprime le backend d'attention (ou 'auto' s'il laisse vLLM choisir)."""
    vllm = PARC_BIN / "vllm-serveur"
    if not vllm.exists():
        raise AssertionError(f"vllm-serveur manquant : {vllm}")

    # Modèle standard AWQ pour test
    modele = "/mnt/4TO_SATACMR_2022/Modeles/models_awq/Devstral-Small-2507-AWQ"

    r = subprocess.run(
        ["bash", str(vllm), "--afficher", modele],
        capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, f"vllm-serveur --afficher échoué : {r.stderr}"

    # Doit contenir « attention_backend » dans la sortie
    output = r.stdout + r.stderr
    assert "attention_backend=" in output, \
        f"vllm-serveur --afficher n'affiche pas attention_backend. Output: {output}"


def test_lanceurs_existent_dans_parc_bin():
    """Les lanceurs (vllm-serveur, llamacpp-serveur, llamacpp-appoint) doivent être dans parc/bin."""
    for nom in ("vllm-serveur", "llamacpp-serveur", "llamacpp-appoint"):
        lanceur = PARC_BIN / nom
        assert lanceur.is_file(), f"{nom} manquant dans {PARC_BIN}"
        assert lanceur.stat().st_mode & 0o111, f"{nom} n'est pas exécutable"
