#!/usr/bin/python3
"""acvram_parc — lecture de ~/.config/acvram-parc/parc.toml, le seul endroit où
vivent les chemins, ports et moteurs d'un poste. Les scripts du paquet
acvram-parc ne contiennent aucun chemin de machine : tout vient d'ici.

    from acvram_parc import charger
    P = charger()                 # défauts + parc.toml (ACVRAM_PARC_CONFIG le remplace)
    P.kimi_dir, P.tsv_dir, P.tsv("gguf"), P.moteurs["llamacpp"]["port"], P.extras.get("comfy_url")

    python3 acvram_parc.py --env  # assignations shell pour les scripts bash (eval)
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

CONFIG_DEFAUT = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "acvram-parc" / "parc.toml"

# Moteurs connus : clé → (nom affiché, port par défaut, route de sondage, variable de clé, clé par défaut)
MOTEURS_CONNUS = {
    "acvram":   ("acvram",          8090, "/v1/models", "CLE_ACVRAM",   "acvram-local"),
    "vllm":     ("vLLM",            8000, "/v1/models", "CLE_VLLM",     "vllm-local"),
    "rapide":   ("appoint (2e GPU)", 8081, "/v1/models", "CLE_RAPIDE",   "rapide-local"),
    "llamacpp": ("llama.cpp",       8080, "/v1/models", "CLE_LLAMACPP", "llamacpp-local"),
    "tabby":    ("TabbyAPI",        5000, "/v1/model",  "CLE_TABBY",    ""),
    "yals":     ("YALS",            5011, "/v1/models", "CLE_YALS",     ""),
    "jan":      ("Jan",             1337, "/v1/models", "CLE_JAN",      ""),
}
ORDRE_MOTEUR = {"acvram": 0, "vllm": 1, "rapide": 2, "llamacpp": 3, "tabby": 4, "yals": 5, "jan": 6}

DEFAUTS = {
    "chemins": {
        "kimi_dir": "~/.kimi-code",
        "tsv_dir": "~/TSV",
        "secrets": "~/.config/ia-secrets.env",
        "bin": "~/.local/bin",
        "icone_claude": "~/.local/share/icons/hicolor/256x256/apps/claude-local.png",
        "mcp_claude": "~/.config/claude-local/mcp.json",
        "dossiers_lancement": ["~"],
        "lib": "/usr/share/acvram-parc/kimi-menu.lib.sh",
    },
    "racines": {"modeles": [], "profondeur": 4},
    "moteurs": {},
    "outils": {"kimi": "", "claude": "", "hf": ""},
    "extras": {},
}


def _p(s: str | None) -> Path | None:
    return Path(os.path.expanduser(s)) if s else None


class Parc:
    def __init__(self, brut: dict, source: Path | None):
        self.source = source
        ch = {**DEFAUTS["chemins"], **brut.get("chemins", {})}
        self.kimi_dir = _p(ch["kimi_dir"])
        self.tsv_dir = _p(ch["tsv_dir"])
        self.secrets = _p(ch["secrets"])
        self.bin = _p(ch["bin"])
        self.icone_claude = _p(ch["icone_claude"])
        self.mcp_claude = _p(ch["mcp_claude"])
        self.lib = _p(ch["lib"])
        if self.lib and not self.lib.exists():
            _cand = Path(__file__).parent.parent / "share" / "kimi-menu.lib.sh"
            if _cand.exists():
                self.lib = _cand
        self.dossiers_lancement = [_p(d) for d in ch["dossiers_lancement"]] or [Path.home()]
        ra = {**DEFAUTS["racines"], **brut.get("racines", {})}
        self.racines = [_p(r) for r in ra["modeles"]]
        self.profondeur = int(ra["profondeur"])
        self.outils = {k: (os.path.expanduser(v) if isinstance(v, str) and v else v) for k, v in {**DEFAUTS["outils"], **brut.get("outils", {})}.items()}
        self.extras = dict(brut.get("extras", {}))
        self.moteurs: dict[str, dict] = {}
        for cle, (nom, port, route, var_cle, cle_defaut) in MOTEURS_CONNUS.items():
            m = dict(brut.get("moteurs", {}).get(cle, {}))
            if not m.get("present", False):
                continue
            m.setdefault("nom", nom); m.setdefault("port", port); m.setdefault("route", route)
            m.setdefault("var_cle", var_cle); m.setdefault("cle_defaut", cle_defaut)
            m.setdefault("journal", ""); m.setdefault("chemin", ""); m.setdefault("tokens", "")
            self.moteurs[cle] = m

    # fichiers TSV : "gguf" | "vllm" | "acvram" | "notes" | "vision"
    def tsv(self, quoi: str) -> Path:
        noms = {"gguf": "gguf-chemins.tsv", "vllm": "vllm-chemins.tsv", "acvram": "acvram-chemins.tsv",
                "notes": "notes-modeles.tsv", "vision": "vision-modeles.tsv"}
        return self.tsv_dir / noms[quoi]

    @property
    def config_kimi(self) -> Path:
        return self.kimi_dir / "config.toml"

    def port(self, moteur: str) -> int | None:
        m = self.moteurs.get(moteur)
        return int(m["port"]) if m else None

    def present(self, moteur: str) -> bool:
        return moteur in self.moteurs

    def env_shell(self) -> str:
        """Assignations pour les scripts bash (eval "$(python3 acvram_parc.py --env)")."""
        def q(v) -> str:
            return "'" + str(v).replace("'", "'\\''") + "'"
        lignes = [
            f"PARC_CONFIG={q(self.config_kimi)}", f"PARC_TSV_DIR={q(self.tsv_dir)}",
            f"PARC_GGUF_TSV={q(self.tsv('gguf'))}", f"PARC_VLLM_TSV={q(self.tsv('vllm'))}",
            f"PARC_ACVRAM_TSV={q(self.tsv('acvram'))}", f"PARC_NOTES_TSV={q(self.tsv('notes'))}",
            f"PARC_MCP_CLAUDE={q(self.mcp_claude)}", f"PARC_SECRETS={q(self.secrets)}",
            f"PARC_LIB={q(self.lib)}", f"PARC_KIMI={q(self.outils.get('kimi') or (self.kimi_dir / 'bin/kimi'))}",
            f"PARC_CLAUDE={q(self.outils.get('claude') or 'claude')}", f"PARC_HF={q(self.outils.get('hf') or 'hf')}",
            f"PARC_BIN={q(self.bin)}", f"PARC_DOSSIER_LANCEMENT={q(self.dossiers_lancement[0])}",
            f"PARC_TOKENS_TABBY={q(self.moteurs.get('tabby', {}).get('tokens', ''))}",
            f"PARC_RACINE_TELECHARGEMENT={q(self.racines[0] if self.racines else Path.home() / 'Modeles')}",
            f"PARC_MOTEURS={q(' '.join(sorted(self.moteurs, key=lambda k: ORDRE_MOTEUR.get(k, 99))))}",
        ]
        for cle, m in self.moteurs.items():
            K = cle.upper()
            lignes += [f"PARC_PORT_{K}={int(m['port'])}", f"PARC_CLE_{K}={q(m['cle_defaut'])}",
                       f"PARC_ROUTE_{K}={q(m['route'])}", f"PARC_JOURNAL_{K}={q(m.get('journal', ''))}"]
        for k, v in self.extras.items():
            if isinstance(v, (str, int, float)):
                lignes.append(f"PARC_EXTRA_{k.upper()}={q(v)}")
        return "\n".join(lignes) + "\n"


def charger(chemin: str | os.PathLike | None = None) -> Parc:
    src = Path(chemin) if chemin else Path(os.environ.get("ACVRAM_PARC_CONFIG", CONFIG_DEFAUT))
    if src.exists():
        with open(src, "rb") as f:
            return Parc(tomllib.load(f), src)
    return Parc({}, None)


if __name__ == "__main__":
    if "--env" in sys.argv:
        sys.stdout.write(charger().env_shell())
    else:
        P = charger()
        print(f"source={P.source or 'défauts (aucun parc.toml)'} moteurs={sorted(P.moteurs)} racines={[str(r) for r in P.racines]}")
