"""L'alias servi du Coder est un contrat par version, comme les défauts (`test_defaut_servi.py`) : quel dossier
sert la cellule b=12 publiée, et ce que son manifeste doit contenir. Un alias qui change sans version, ou une
version sans alias déclaré, casse ici. Dossier : revue/poste6-remplacement-i8c-par-s1b-23-09.md § 4.

La bascule i8c → S1b (experts calibrés AWQ, mêmes formats) n'entre qu'avec la pièce 123 (échelle AWQ portée par
le chemin tensor) : le test qui la constate est `xfail(strict=True)` tant qu'elle n'est pas fusionnée — il passe
au vert (et casse le xfail, donc oblige à l'enlever) le jour où `forme_tensor_refus` accepte les tables AWQ."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

import acvram

RACINE = Path(__file__).resolve().parents[1]
LISTE_ALIAS_SERVIS = RACINE / "outils" / "poste" / "alias-servis-20-09.txt"

# version → (dossier de l'alias servi, sha256 de son acvram_manifest.json, experts sans statistiques AWQ).
# Ajouter une ligne par version qui change l'alias ; ne jamais modifier une ligne existante.
ALIAS_PAR_VERSION = {
    "0.6.38": ("Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c", "efcc12e3417e8d53c9377956e38e65efa2d92344467c4afac499c77276832e15", 18432),
    # 0.7.0 (26/09, pièce 232) : même alias servi qu'en 0.6.38 (la bascule S1b attend la 123) ; sha256 relevé sur le disque le 26/09.
    "0.7.0": ("Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c", "efcc12e3417e8d53c9377956e38e65efa2d92344467c4afac499c77276832e15", 18432),
    # 0.7.1 (26/09, pièce 272) : même alias servi qu'en 0.7.0 (aucune pièce de la 0.7.1 ne change l'alias).
    "0.7.1": ("Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c", "efcc12e3417e8d53c9377956e38e65efa2d92344467c4afac499c77276832e15", 18432),
    # 0.7.2 (26/09, pièces 070 b/273) : doctor et paquets seulement, même alias.
    "0.7.2": ("Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c", "efcc12e3417e8d53c9377956e38e65efa2d92344467c4afac499c77276832e15", 18432),
    # 0.7.3 (26/09, pièce 269 c) : guet d'admission au défaut, même alias.
    "0.7.3": ("Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c", "efcc12e3417e8d53c9377956e38e65efa2d92344467c4afac499c77276832e15", 18432),
}
# Formats par famille, identiques pour i8c et S1b (107 bis § 6) : ce qui distingue les deux est dans les experts
# (calibrés ou non), pas dans les formats.
FORMATS_ATTENDUS = {"proj_int8": 192, "tete": "int8", "experts_nvfp4": 18432, "attn_int8": "canal"}


def _alias_servi_declare() -> str:
    for l in LISTE_ALIAS_SERVIS.read_text(encoding="utf-8").splitlines():
        if l.strip() and not l.startswith("#"):
            return l.strip()
    raise AssertionError(f"{LISTE_ALIAS_SERVIS} : aucun alias")


def _racine_parc() -> Path:
    sys.path.insert(0, str(RACINE / "outils"))
    from racine_modeles import racine_modeles
    return Path(os.environ.get("ACVRAM_PARC") or racine_modeles())


def test_la_version_a_son_alias_servi():
    assert acvram.__version__ in ALIAS_PAR_VERSION, (
        f"version {acvram.__version__} sans alias servi déclaré : ajouter son entrée (dossier, sha256, experts)")


def test_la_liste_des_alias_servis_commence_par_l_alias_de_la_version():
    dossier, _, _ = ALIAS_PAR_VERSION[acvram.__version__]
    assert _alias_servi_declare() == dossier


def test_le_manifeste_de_l_alias_servi_est_celui_qui_a_ete_qualifie():
    dossier, sha, sans_stats = ALIAS_PAR_VERSION[acvram.__version__]
    manifeste = _racine_parc() / dossier / "acvram_manifest.json"
    if not manifeste.is_file():
        pytest.skip(f"parc absent : {manifeste}")
    assert hashlib.sha256(manifeste.read_bytes()).hexdigest() == sha, "manifeste changé : ce n'est plus l'artefact qualifié"
    m = json.loads(manifeste.read_text(encoding="utf-8"))
    t = m["tensors"]
    assert m.get("attn_int8") == FORMATS_ATTENDUS["attn_int8"]
    assert sum(1 for n, x in t.items() if "self_attn" in n and x.get("format") == "int8") == FORMATS_ATTENDUS["proj_int8"]
    assert t["lm_head.weight"]["format"] == FORMATS_ATTENDUS["tete"]
    assert sum(1 for n, x in t.items() if "experts" in n and x.get("format") == "nvfp4") == FORMATS_ATTENDUS["experts_nvfp4"]
    assert m.get("experts_sans_stats") == sans_stats


@pytest.mark.xfail(strict=True, reason="pièce 123 (poste1) non fusionnée : les tables AWQ d'activation par expert "
                                       "renvoient encore le chemin tensor au GEMV (moe.py forme_tensor_refus) ; "
                                       "S1b ne peut pas remplacer i8c avant — retirer ce xfail avec la 123")
def test_les_tables_awq_des_experts_sont_acceptees_par_le_chemin_tensor():
    from acvram.engine.moe import forme_tensor_refus
    assert forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, False) == ""
