"""Les clés T() de packaging/acvram-gui (source française) ont chacune leur traduction dans TOUTES les langues
de packaging/langues/ (31 fichiers + le français = 32 langues), et _cles.json est exactement l'ensemble des clés
du script — test cassant (Sage, sage-menus-test-reel-contexte-20-09 § 4) : une clé nouvelle sans ses traductions
rend rouge ; une traduction orpheline (clé disparue du script) aussi. Les clés sont extraites du SOURCE (regex sur
T(...)), pas déclarées à la main. À sec, sans GTK."""
import ast
import json
import os
import pathlib
import re

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent
GUI = RACINE / "packaging" / "acvram-gui"
LANGUES = RACINE / "packaging" / "langues"


def cles_du_script() -> set[str]:
    """Toutes les chaînes littérales passées en premier argument à T(...) dans le script (AST)."""
    arbre = ast.parse(GUI.read_text(encoding="utf-8"))
    cles = set()
    for n in ast.walk(arbre):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "T" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                cles.add(a.value)
    # les T("...") écrits DANS des f-strings HTML (return f"""…{T("x")}…""") sont vus par ast comme des appels
    # dans la FormattedValue : ast.walk les parcourt aussi. Contrôle : le regex textuel en trouve au moins autant.
    texte = GUI.read_text(encoding="utf-8")
    par_regex = len(re.findall(r'\bT\(\s*(?:"|\')', texte))
    assert len(cles) >= 20 and par_regex >= len(cles) - 2, (len(cles), par_regex)
    return cles


def fichiers_langues() -> dict[str, dict]:
    out = {}
    for f in sorted(LANGUES.glob("*.json")):
        if f.name.startswith("_"):
            continue
        out[f.stem] = json.load(open(f, encoding="utf-8"))
    return out


def test_le_script_a_ses_32_langues():
    assert len(fichiers_langues()) == 31, sorted(fichiers_langues())   # + le français source = 32


def _presente_dans_le_script(k: str, texte: str) -> bool:
    """La clé est un littéral du script — passée à T() directement, ou par une variable (les infobulles de
    la barre : `for ico, tip, cb in (("view-refresh", "recharger (F5)", …)` puis `T(tip)`)."""
    return json.dumps(k, ensure_ascii=False) in texte or repr(k) in texte or f'"{k}"' in texte or f"'{k}'" in texte


def test_cles_json_couvre_les_appels_T_et_n_a_pas_d_orpheline():
    cles = cles_du_script()
    declarees = set(json.load(open(LANGUES / "_cles.json", encoding="utf-8")))
    texte = GUI.read_text(encoding="utf-8")
    manquent = sorted(cles - declarees)
    orphelines = sorted(k for k in declarees if not _presente_dans_le_script(k, texte))
    assert not manquent and not orphelines, {"manquent dans _cles.json": [k[:50] for k in manquent[:6]],
                                              "orphelines (plus dans le script)": [k[:50] for k in orphelines[:6]]}


def test_toute_cle_T_a_ses_31_traductions():
    cles = cles_du_script() | set(json.load(open(LANGUES / "_cles.json", encoding="utf-8")))
    fautes = {}
    for code, d in fichiers_langues().items():
        manque = sorted(k for k in cles if k not in d or not str(d[k]).strip())
        if manque:
            fautes[code] = manque
    assert not fautes, "clés sans traduction : " + json.dumps({c: [k[:50] for k in m[:4]] + ([f"… (+{len(m) - 4})"] if len(m) > 4 else []) for c, m in fautes.items()}, ensure_ascii=False)


def test_les_traductions_gardent_les_champs_de_format():
    """Une traduction qui perd un {champ} casse T(...).format(...) à l'exécution : jamais en silence."""
    cles = cles_du_script(); fautes = []
    for code, d in fichiers_langues().items():
        for k in cles:
            if k in d:
                if set(re.findall(r"\{(\w+)\}", k)) != set(re.findall(r"\{(\w+)\}", str(d[k]))):
                    fautes.append(f"{code}: {k[:40]}")
    assert not fautes, fautes[:10]


def test_aucun_chemin_de_machine_dans_l_exemple():
    """L'exemple Agent OS va dans le paquet : aucun chemin de la machine de développement."""
    texte = GUI.read_text(encoding="utf-8")
    interdits = [m for m in re.findall(r"/mnt/\S+|/home/\w+\S*", texte) if "127.0.0.1" not in m]
    assert not interdits, interdits[:5]


def test_le_test_sait_dire_faux(tmp_path, monkeypatch):
    """Une clé ajoutée au script sans traduction rend rouge (faute construite sur une copie du script)."""
    import sys
    copie = tmp_path / "acvram-gui"
    copie.write_text(GUI.read_text(encoding="utf-8") + '\nX = T("clé fabriquée le 20/09 sans traduction")\n', encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "GUI", copie)
    with pytest.raises(AssertionError, match="clés sans traduction"):
        test_toute_cle_T_a_ses_31_traductions()
