"""Pièce 256 (chef, 26/09, sur 237/poste1) : REGLES § 4 interdit le texte répété dans une
cellule MoE, pas même en remplissage de longueur — l'invite répétée déplace le coût du GEMM
experts de ± 0,6-0,9 ms/pas SELON LE SENS (237 : PAR_LIGNE=1 −12,6 % en brut/répété contre
+12,3 % en brutchat/réel, même protocole, seule l'invite change). Cette garde refuse toute
invite dont la part de trigrammes répétés dépasse le seuil mesuré (0,5, entre le 0,0 du texte
réel de la 226 et le 0,89 d'une phrase courte tuilée à 256 jetons — 229/237).

Pièce 256b : `_INVITE_TEXTE_REELLE` est désormais sept questions techniques réelles et
distinctes (277 jetons), assez longue pour que `invite_reelle(gguf, 256)` ne tuile plus jamais
— **la 229 et la 237 (invite tuilée à 256 jetons d'une phrase courte) ne sont donc plus
reproductibles À L'IDENTIQUE avec ce banc, voulu : reproduire une invite répétée n'a plus de
sens une fois la règle qui l'interdit écrite.** Le témoin (tuilage forcé au-delà de 277 jetons)
prouve que le mécanisme de détection marche toujours, pas que l'ancien défaut est encore
atteignable en usage normal."""
import importlib.util
import os
import pathlib

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent
MODELE = "/mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4"


def _charger(chemin, nom, env_sup=None):
    env_sup = env_sup or {}
    ancien = {k: os.environ.get(k) for k in env_sup}
    os.environ.update(env_sup)
    try:
        spec = importlib.util.spec_from_file_location(nom, chemin)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        for k, v in ancien.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def banc_llamacpp():
    return _charger(str(RACINE / "scratchpad/banc-llamacpp-16-09.py"), "banc_llamacpp_256")


@pytest.fixture(scope="module")
def _tokenizer_disponible():
    if not pathlib.Path(MODELE, "tokenizer.json").exists():
        pytest.skip(f"tokenizer absent : {MODELE}")


def test_ratio_ngrammes_mesure_texte_reel_et_ancienne_phrase_courte_tuilee(banc_llamacpp, _tokenizer_disponible):
    """Les deux chiffres qui fixent le seuil (0,0 et 0,89) — pas supposés, mesurés ici même. Le
    témoin tuilé reprend l'ANCIENNE phrase courte (28 jetons, pièce 256 initiale) : la nouvelle
    `_INVITE_TEXTE_REELLE` (277 jetons) ne tuile plus à 256, c'est tout le point de la 256b."""
    from acvram.server.chat import load_tokenizer
    tok = load_tokenizer(MODELE)
    ancienne_phrase_courte = ("Explique en detail le fonctionnement d'un cache a correspondance "
                               "directe, puis compare-le a un cache associatif par ensembles.")
    ids_courte = tok.encode(ancienne_phrase_courte)
    r_reel = banc_llamacpp._ratio_ngrammes_repetes(tok.encode(banc_llamacpp._INVITE_TEXTE_REELLE))
    assert r_reel < 0.1, (f"le texte reel (277 jetons, sept questions distinctes) devrait avoir "
                          f"peu de trigrammes repetes (tournures communes entre questions), ratio={r_reel}")

    tuile = [ids_courte[i % len(ids_courte)] for i in range(256)]
    r_tuile = banc_llamacpp._ratio_ngrammes_repetes(tuile)
    assert r_tuile > 0.8, f"la phrase courte tuilee a 256 jetons (229/237) devrait etre tres repetee, ratio={r_tuile}"
    assert r_reel < banc_llamacpp.SEUIL_REPETITION_NGRAMMES < r_tuile, (
        "le seuil doit separer net le texte reel du tuilage d'une phrase courte")


def test_invite_reelle_256_jetons_est_maintenant_acceptee(banc_llamacpp, _tokenizer_disponible):
    """Pièce 256b : `_INVITE_TEXTE_REELLE` (277 jetons) couvre 256 sans tuiler — accepté."""
    ids = banc_llamacpp.invite_reelle(MODELE, 256)
    assert len(ids) == 256
    assert banc_llamacpp._ratio_ngrammes_repetes(ids) < banc_llamacpp.SEUIL_REPETITION_NGRAMMES


def test_invite_reelle_au_dela_de_sa_longueur_naturelle_tuile_et_est_refusee(banc_llamacpp, _tokenizer_disponible):
    """Témoin : au-delà de sa propre longueur (277), `invite_reelle` tuile encore par construction
    (`ids[i % len(ids)]`) — le mécanisme de garde doit toujours l'attraper, même sur ce texte-ci."""
    with pytest.raises(RuntimeError, match="REFUS.*trigrammes"):
        banc_llamacpp.invite_reelle(MODELE, 2000)


def test_invite_reelle_a_sa_longueur_naturelle_est_acceptee(banc_llamacpp, _tokenizer_disponible):
    """Le texte réel de la 226, non tuilé (n = sa propre longueur) : aucune répétition, accepté."""
    from acvram.server.chat import load_tokenizer
    tok = load_tokenizer(MODELE)
    n = len(tok.encode(banc_llamacpp._INVITE_TEXTE_REELLE))
    ids = banc_llamacpp.invite_reelle(MODELE, n)
    assert len(ids) == n


def test_banc_chat_openai_invite_226_est_acceptee_a_l_import():
    """`INVITE_TEXTE` (banc chat 102, texte réel de la 226) passe la garde au chargement du
    module — un import qui lève voudrait dire que le texte réel lui-même est jugé répété,
    ce qui serait le signe d'un seuil mal choisi."""
    _charger(str(RACINE / "scratchpad/poste2-piece102-etalon-hf-24-09/banc-chat-openai.py"),
             "banc_chat_openai_256", env_sup={"BANC_URL": "http://127.0.0.1:1", "BANC_MOTEUR": "acvram"})


def test_banc_chat_openai_refuse_un_texte_tuile():
    """Témoin : si `INVITE_TEXTE` était tuilé (même défaut que 229/237 côté banc chat), la
    garde doit refuser — sinon elle ne protège que banc-llamacpp-16-09.py."""
    m = _charger(str(RACINE / "scratchpad/poste2-piece102-etalon-hf-24-09/banc-chat-openai.py"),
                 "banc_chat_openai_256_temoin",
                 env_sup={"BANC_URL": "http://127.0.0.1:1", "BANC_MOTEUR": "acvram"})
    mots = m.INVITE_TEXTE.split()
    tuile = " ".join(mots * 10)
    with pytest.raises(RuntimeError, match="REFUS.*trigrammes"):
        m._refuser_si_trop_repete(tuile, "temoin")
