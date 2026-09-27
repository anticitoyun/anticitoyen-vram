# Pièce 273 — extra `vision` : ALERTE du doctor (pas ÉCHEC) et embarqué dans le Flatpak : verdict

poste6, 26/09/2026, décision chef (a + b). Branche poste6-273, code **9e1de7ab9**, fusion **4d7b88e87** (0.7.2, avec
070 b), suite complète 6 082 verts (6423f55f6).

**Quoi.** (a) `release.yml` : `transformers` et `pillow` dans la liste de `sources-pypi.py` (44 roues cp314/abi3/py3
résolues à sec en 18 s, aucune sdist) ; commentaire du manifeste. (b) `acvram/cli.py` : `_doctor_modules()` — requis
(safetensors, fastapi, uvicorn, tokenizers, jinja2) en ECHEC ; `_EXTRAS = {"vision": (transformers, PIL)}` en
`alerte vision indisponible (… absent) : pip install 'acvram[vision]'`, code 0. **Renverse le 21/09** (« trou P3 » :
ECHEC), écrit noir sur blanc dans le code, le CHANGELOG et le commit ; un modèle multimodal sans vision échoue toujours au
chargement, en clair. Avec elle : 0.7.2 (versions, metainfo, tables de version, CHANGELOG 0.7.2 = 070 b + 273).

**Tests.** `tests/test_doctor_modules_273.py` (imports factices : sans vision → True + alerte ; requis absent → False ;
vision présente → ok ; `_EXTRAS` ⊆ optional-dependencies de pyproject) ; `tests/test_flathub_vision_273.py`. 34 verts
ciblés + 138 cli/régime ; témoins 4 + 1 rouges sur l'ancien.

**Résultat sous carte.sh.** verif-072 (`poste6-serie266-flatpak-verdict-26-09.md` § 5) : doctor du bac à sable Flatpak
0.7.2 **code 0**, `ok    vision (transformers, PIL)`, `noyaux CUDA fusionnes compiles pour sm_120`. **Décision** : livrée
en 0.7.2 ; le `.deb` ne déclare pas transformers non plus — même question, non mesurée, à qui reprend le paquet Debian.
