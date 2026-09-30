"""Vérification que les lanceurs (vllm-serveur, etc.) sont dans parc/bin et affichent leurs paramètres."""
import subprocess
import pathlib

import pytest

PARC_BIN = pathlib.Path(__file__).resolve().parent.parent / "parc" / "bin"


def test_vllm_serveur_affiche_le_backend_choisi():
    """vllm-serveur --afficher <alias> imprime le backend d'attention (ou 'auto' s'il laisse vLLM choisir)."""
    vllm = PARC_BIN / "vllm-serveur"
    if not vllm.exists():
        raise AssertionError(f"vllm-serveur manquant : {vllm}")

    # Modèle standard AWQ pour test
    modele = "/mnt/4TO_SATACMR_2022/Modeles/models_awq/Devstral-Small-2507-AWQ"
    # Pièce 267c (CI GitHub) : ce test lit le catalogue RÉEL de la machine de dev
    # (`/mnt/4TO_SATACMR_2022`, ~250 Go de modèles), rien de portable à simuler ici —
    # aucun runner CI n'a ce montage. `vllm-serveur` échoue même avant de chercher le
    # modèle (`mkdir /mnt/AI_GENERATOR` refusé, permission denied), preuve que c'est
    # la disposition de stockage de la machine qui est testée, pas la logique du script.
    if not pathlib.Path(modele).exists():
        pytest.skip(f"modèle absent (catalogue réel de la machine cible) : {modele}")

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


@pytest.mark.parametrize("taille,attendu", [(20_000_000_000, "0"), (33_125_992_448, "0,1"), (74_000_000_000, "0,1")])
def test_llamacpp_grand_modele_voit_les_deux_cartes(taille, attendu):
    """poste5-menus 29/09 : au-delà de 30 Go le lanceur répartit sur 5090 + 3080 Ti, mais carte.sh reçoit
    ACVRAM_CARTE=${CUDA_VISIBLE_DEVICES:-0} — sans export dans la branche, la session (CVD vide) donnait « 0 » et
    le chargement mourait en OOM sur la 5090 seule. Joue le bloc de choix des cartes, pas le script (qui tue le 8080)."""
    src = (PARC_BIN / "llamacpp-serveur").read_text()
    debut = src.index('if [ "$taille" -gt 40000000000 ]; then')
    bloc = src[debut:src.index("\nfi\n", debut) + 4]
    r = subprocess.run(["bash", "-c", f'taille={taille}; MODE=""; gpu_args=(); c_d=; c_0=\n{bloc}\n'
                        'echo "carte=${ACVRAM_CARTE:-${CUDA_VISIBLE_DEVICES:-0}}"'],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""})
    assert r.stdout.strip().splitlines()[-1] == f"carte={attendu}", r.stdout + r.stderr


@pytest.mark.parametrize("sonde,attendu", [("q4_0 q4_0", "q4_0 q4_0"), ("q4_0 f16", "f16 f16"), ("f16 q4_0", "f16 f16"),
                                           ("", "q4_0 q4_0")])
def test_llamacpp_cache_kv_jamais_de_types_melanges(sonde, attendu):
    """poste5-menus 29/09 : Kimi-Linear (têtes V de 72) — la sonde rend « q4_0 f16 » et llama.cpp meurt au démarrage
    (« does not support different K and V cache types »). Joue le bloc du lanceur avec la sonde remplacée."""
    src = (PARC_BIN / "llamacpp-serveur").read_text()
    debut = src.index('lu=$("$HOME/.local/bin/gguf-cache-compatible"')
    bloc = src[debut:src.index("\n", src.index('[ "$CV" = f16 ]', debut)) + 1]
    bloc = bloc.replace('"$HOME/.local/bin/gguf-cache-compatible" "$MODELE" 2>/dev/null', f"echo {sonde}")
    r = subprocess.run(["bash", "-c", f"c_d=; c_0=\n{bloc}\necho \"$CK $CV\""], capture_output=True, text=True)
    assert r.stdout.strip().splitlines()[-1] == attendu, r.stdout + r.stderr


@pytest.mark.parametrize("gabarit,rc,attendu", [("illisible", 1, "Gabarit illisible : "), ("lisible", 0, "--chat-template-file"),
                                                ("", 0, "TMPL=0")])
def test_llamacpp_gabarit_illisible_refus_nomme(tmp_path, gabarit, rc, attendu):
    """kimi-linear 30/09 : GABARIT pointant vers le home d'un autre poste, ignoré en silence (`[ -f ] &&`), le modèle
    servait le gabarit embarqué. Joue le bloc du lanceur (le script entier tue le 8080). Cassant sur l'ancien bloc."""
    src = (PARC_BIN / "llamacpp-serveur").read_text()
    debut = src.index("TMPL=()")
    fin = src.index("# Sans gabarit imposé", debut)          # le bloc s'arrête au commentaire suivant
    bloc = src[debut:fin]
    chemin = {"illisible": "/nulle/part/kimi-linear-hermes.jinja", "lisible": str(tmp_path / "g.jinja"), "": ""}[gabarit]
    (tmp_path / "g.jinja").write_text("{{ x }}")
    r = subprocess.run(["bash", "-c", f'err() {{ echo "$*" >&2; }}\nGABARIT="{chemin}"\n{bloc}\necho "TMPL=${{#TMPL[@]}} ${{TMPL[*]:-}}"'],
                       capture_output=True, text=True)
    assert r.returncode == rc and attendu in r.stdout + r.stderr, r.stdout + r.stderr
