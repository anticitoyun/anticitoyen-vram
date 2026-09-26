"""Pièce 256 (chef, 26/09, sur 237/poste1) : REGLES § 4 interdit le texte répété dans une
cellule MoE, pas même en remplissage de longueur — l'invite répétée déplace le coût du GEMM
experts de ± 0,6-0,9 ms/pas SELON LE SENS (237 : PAR_LIGNE=1 −12,6 % en brut/répété contre
+12,3 % en brutchat/réel, même protocole, seule l'invite change). Cette garde refuse toute
invite dont la part de trigrammes répétés dépasse le seuil mesuré (0,5, entre le 0,0 du texte
réel de la 226 et le 0,89 de la même invite tuilée à 256 jetons — 229/237)."""
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


def test_ratio_ngrammes_mesure_texte_reel_et_invite_tuilee(banc_llamacpp, _tokenizer_disponible):
    """Les deux chiffres qui fixent le seuil (0,0 et 0,89) — pas supposés, mesurés ici même."""
    from acvram.server.chat import load_tokenizer
    tok = load_tokenizer(MODELE)
    texte_reel = banc_llamacpp._INVITE_TEXTE_REELLE
    ids_reels = tok.encode(texte_reel)
    r_reel = banc_llamacpp._ratio_ngrammes_repetes(ids_reels)
    assert r_reel == 0.0, f"le texte reel de la 226 devrait n'avoir aucun trigramme repete, ratio={r_reel}"

    tuile = [ids_reels[i % len(ids_reels)] for i in range(256)]
    r_tuile = banc_llamacpp._ratio_ngrammes_repetes(tuile)
    assert r_tuile > 0.8, f"l'invite tuilee a 256 jetons (229/237) devrait etre tres repetee, ratio={r_tuile}"
    assert r_reel < banc_llamacpp.SEUIL_REPETITION_NGRAMMES < r_tuile, (
        "le seuil doit separer net le texte reel de son propre tuilage a 256 jetons")


def test_invite_reelle_256_jetons_tuilee_de_la_229_est_refusee(banc_llamacpp, _tokenizer_disponible):
    with pytest.raises(RuntimeError, match="REFUS.*trigrammes"):
        banc_llamacpp.invite_reelle(MODELE, 256)


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
