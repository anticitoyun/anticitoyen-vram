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


def test_les_trois_serveurs_prennent_le_verrou_carte():
    """Cassant (trou du 21/09, REGLES § 2) : acvram/llamacpp/vllm-serveur lancent
    leur serveur via carte.sh en mode service (ACVRAM_TYPE=service), jamais par un
    `setsid nohup` nu qui laisserait une mesure croire la carte libre. Le repli
    `setsid nohup` reste permis SEULEMENT dans la branche « carte.sh introuvable »."""
    for nom in ("acvram-serveur", "llamacpp-serveur", "vllm-serveur"):
        src = (PARC_BIN / nom).read_text()
        assert "ACVRAM_TYPE=service" in src and "CARTE_SH" in src, \
            f"{nom} ne prend pas le verrou carte.sh en mode service"
        # tout `setsid nohup` EXECUTE (hors commentaire) doit suivre l'avertissement « SANS verrou »
        lignes = src.splitlines()
        for i, ligne in enumerate(lignes):
            if "setsid nohup" in ligne and not ligne.lstrip().startswith("#"):
                contexte = "\n".join(lignes[max(0, i - 3):i + 1])
                assert "SANS verrou" in contexte, \
                    f"{nom}:{i+1} garde un `setsid nohup` hors de la branche de repli :\n{contexte}"
