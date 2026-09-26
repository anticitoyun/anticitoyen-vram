"""Contrôle : 'avec spéculation n-gram' ne doit apparaître que sur les entrées acvram.cli serve.
Ne jamais annoter vLLM (vllm-direct, NVFP4 majuscules), llama.cpp (GGUF natif), EXL3/TabbyAPI natif.
"""
import re
from pathlib import Path

import pytest

REVUE = Path(__file__).parent.parent / "acvram-memoire" / "revue"
ANNOTATION = "avec spéculation n-gram"

# Exceptions acvram connues dont le nom est trompeur
EXCEPTIONS_ACVRAM = {
    "nemo-12b-thinking-exl3",  # bloc1 = acvram seul, format source EXL3 mais servi acvram
}


def _est_non_acvram(alias: str) -> bool:
    """Retourne True si l'alias identifie un moteur NON acvram.

    Règle de nommage :
    - Se termine par -nvfp4 → acvram (même si srcQ4_K_M, srcexl3 dans le nom)
    - Se termine par -int8  → acvram (Falcon-H1R-7B-int8, Nemotron-Nano-9B-int8)
    - Contient vllm-direct  → vLLM
    - Suffixe NVFP4 ou FP4 en MAJUSCULES (fin de nom) → vLLM
    - Suffixe GGUF natif (Q4_K_M, Q5_K_M, Q6_K, IQ4_XS, etc.) en fin de nom → llama.cpp
    - Contient EXL3-N.NN ou exl3-N.NN (bitrate) ou .exl3 en fin → TabbyAPI
    """
    a = alias.lower()
    if a in {x.lower() for x in EXCEPTIONS_ACVRAM}:
        return False
    if a.endswith('-nvfp4') or a.endswith('_nvfp4'):
        return False
    if a.endswith('-int8') or a.endswith('_int8'):
        return False
    if 'vllm-direct' in a:
        return True
    # vLLM utilise NVFP4 en MAJUSCULES (acvram utilise -nvfp4 minuscules)
    if re.search(r'[-_]NVFP4', alias):   # n'importe où dans le nom, casse intentionnelle
        return True
    if re.search(r'[-_]FP4$', alias):    # fin de nom en majuscules
        return True
    # GGUF natifs : Qx_K_y ou IQx_y en fin de nom
    if re.search(r'[-_](Q[0-9]+_K[^`_-]*)$', alias, re.IGNORECASE):
        return True
    if re.search(r'[-_](IQ[0-9]+_[A-Z]+)$', alias, re.IGNORECASE):
        return True
    # EXL3 avec bitrate explicite (ex: exl3-4.00bpw, EXL3-6.0bpw)
    if re.search(r'(exl3|EXL3)[-_][0-9]', alias):
        return True
    # .exl3 en fin de nom
    if a.endswith('.exl3'):
        return True
    return False


def trouver_violations(chemin: Path) -> list[tuple[int, str]]:
    """Retourne les (numéro de ligne, contenu) qui ont une annotation interdite."""
    if not chemin.exists():
        # instantané public : publier-github.sh garde ces tableaux privés (ils nomment les modèles de sessions)
        pytest.skip(f"{chemin.name} absent de cet arbre")
    violations = []
    for i, ligne in enumerate(chemin.read_text().splitlines(), 1):
        if ANNOTATION not in ligne:
            continue
        m = re.search(r"`([^`]+)`", ligne)
        if m:
            alias = m.group(1)
            if _est_non_acvram(alias):
                violations.append((i, ligne[:120]))
    return violations


def test_claude_modeles_aucune_annotation_non_acvram():
    chemin = REVUE / "claude-modeles.md"
    violations = trouver_violations(chemin)
    assert violations == [], (
        f"{len(violations)} annotation(s) sur entrées non-acvram dans {chemin.name}:\n"
        + "\n".join(f"  L{n}: {l}" for n, l in violations)
    )


def test_kimi_modeles_aucune_annotation_non_acvram():
    chemin = REVUE / "kimi-modeles.md"
    violations = trouver_violations(chemin)
    assert violations == [], (
        f"{len(violations)} annotation(s) sur entrées non-acvram dans {chemin.name}:\n"
        + "\n".join(f"  L{n}: {l}" for n, l in violations)
    )
