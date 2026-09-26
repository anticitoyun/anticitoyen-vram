"""Instrument multi-tranches `outils/gpu/mesure/ppl-decode-kv.py` (chantier 3.2, 20/09), à sec.

(a) la découpe (préfixe de séquence en tête, compté dans P, tronqué à P+N) est
    celle de `scratchpad/ppl-decode-kv-17-09.py:31-33`, rejouée ici mot pour mot
    sur un tokenizer jouet ; (b) --depuis / --reprendre sautent ce qu'il faut ;
(c) le RESULTAT d'une tranche est écrit avant que la suivante commence ;
(d) témoin : deux tranches différentes rendent deux PPL différentes."""
import importlib.util, io, json, os
import pytest
import torch

_ICI = os.path.dirname(os.path.abspath(__file__))
_OUTIL = os.path.join(_ICI, "..", "outils", "gpu", "mesure", "ppl-decode-kv.py")


@pytest.fixture(scope="module")
def outil():
    spec = importlib.util.spec_from_file_location("ppl_decode_kv", _OUTIL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TokJouet:
    """Un jeton par mot ; add_special_tokens=False n'ajoute rien — comme
    `load_tokenizer(GLM).encode` pour le préfixe (vérifié à sec le 20/09)."""
    VOCAB = 64

    def encode(self, texte, add_special_tokens=True):
        return [hash(m) % self.VOCAB for m in texte.split()]


def _decoupe_ancienne(tokenizer, texte, prefixe_seq, P, N):
    # ppl-decode-kv-17-09.py:31-33, recopié tel quel (os.environ remplacé par l'argument)
    _PFX = tokenizer.encode(prefixe_seq, add_special_tokens=False) if prefixe_seq else []
    ids = (_PFX + tokenizer.encode(texte))[:P + N]
    assert len(ids) == P + N, len(ids)
    return ids, _PFX


@pytest.mark.parametrize("prefixe_seq", ["", "gMASK sop"])
def test_decoupe_identique_a_l_ancien_script(outil, prefixe_seq):
    tok = TokJouet()
    texte = " ".join(f"mot{i}" for i in range(40))
    ids, pfx = outil.decouper_ids(tok, texte, prefixe_seq, 12, 8)
    ids0, pfx0 = _decoupe_ancienne(tok, texte, prefixe_seq, 12, 8)
    assert ids == ids0 and pfx == pfx0 and len(ids) == 20
    if prefixe_seq:
        assert ids[:2] == tok.encode(prefixe_seq, add_special_tokens=False)   # en tête, compté dans P
    with pytest.raises(ValueError):
        outil.decouper_ids(tok, "trois mots seulement", prefixe_seq, 12, 8)


def test_depuis_et_reprendre(outil, tmp_path):
    tranches = ["a.txt", "b.txt", "c.txt", "d.txt"]
    assert outil.tranches_a_faire(tranches, str(tmp_path), "x") == [(1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt")]
    assert outil.tranches_a_faire(tranches, str(tmp_path), "x", depuis=3) == [(3, "c.txt"), (4, "d.txt")]
    (tmp_path / "ppl-x-t2.json").write_text("{}")
    assert outil.tranches_a_faire(tranches, str(tmp_path), "x", reprendre=True) == [(1, "a.txt"), (3, "c.txt"), (4, "d.txt")]
    assert outil.tranches_a_faire(tranches, str(tmp_path), "y", reprendre=True) == [(k, t) for k, t in enumerate(tranches, 1)]
    assert outil.tranches_a_faire(tranches, str(tmp_path), "x", depuis=2, reprendre=True) == [(3, "c.txt"), (4, "d.txt")]


class Seq:
    def __init__(self, ids, n):
        self.prompt_ids, self.restant, self.prefilled = ids, n, False


class MoteurFactice:
    """Un moteur dont les logits dépendent des ids de l'invite (par une graine) :
    deux tranches différentes rendent deux PPL différentes. Il note, à chaque
    `add_request`, les JSON déjà présents sur disque (test c)."""
    VOCAB = 64

    def __init__(self, sortie):
        self.running, self.waiting, self.sortie = [], [], sortie
        self.json_vus_a_l_admission, self.requetes = [], []

    def add_request(self, ids, params, request_id=""):
        self.json_vus_a_l_admission.append(sorted(os.listdir(self.sortie)))
        self.requetes.append(request_id)
        s = Seq(list(ids), params.max_tokens)
        self.waiting.append(s)
        self.gen = torch.Generator().manual_seed(sum(ids) % 100003)
        return s

    def step(self):
        if self.waiting:
            s = self.waiting.pop(); s.prefilled = True; self.running.append(s)
        s = self.running[0]
        logits = torch.randn(1, self.VOCAB, generator=self.gen)
        self._sample_only(logits, [s])
        s.restant -= 1
        if s.restant == 0:
            self.running.remove(s)


class Params:
    def __init__(self, temperature, max_tokens):
        self.max_tokens = max_tokens


def _tranches(tmp_path, n=3):
    fichiers = []
    for k in range(n):
        f = tmp_path / f"tranche-{k}.txt"
        f.write_text(" ".join(f"t{k}m{i}" for i in range(30)))
        fichiers.append(str(f))
    return fichiers


def test_resultat_ecrit_avant_la_tranche_suivante_et_temoin(outil, tmp_path):
    sortie = tmp_path / "sortie"; sortie.mkdir()
    fichiers = _tranches(tmp_path, 3)
    moteur = MoteurFactice(str(sortie))
    flux = io.StringIO()
    res = outil.courir(fichiers, moteur, TokJouet(), lambda p: open(p).read(), torch, Params, bras="essai",
                       sortie=str(sortie), prefixe=10, notes=6, prefixe_seq="", base={"regime_ligne": "sec"},
                       sortie_txt=flux)
    assert len(res) == 3 and moteur.requetes == ["t1", "t2", "t3"]
    # (c) : à l'admission de la tranche k, les JSON 1..k-1 existent déjà
    assert moteur.json_vus_a_l_admission == [[], ["ppl-essai-t1.json"], ["ppl-essai-t1.json", "ppl-essai-t2.json"]]
    resultats = [json.loads(l.split(" ", 1)[1]) for l in flux.getvalue().splitlines() if l.startswith("RESULTAT ")]
    assert [r["k"] for r in resultats] == [1, 2, 3]
    for r in resultats:
        assert set(("ppl", "nll_moy", "n_jetons_notes", "pas", "regime_ligne", "prefixe_sequence", "t_prefill", "t_decode")) <= set(r)
        assert r["n_jetons_notes"] == 6 and r["pas"] == 6
        disque = json.load(open(sortie / f"ppl-essai-t{r['k']}.json"))
        assert set(outil.CLES_SCHEMA) | {"ppl", "nll_moy", "n_jetons_notes", "pas", "nll_par_jeton"} <= set(disque)
        assert disque["ppl"] == r["ppl"] and disque["prefixe_ids"] == [] and disque["capacity_tokens"] is None
        assert disque["regime_ligne"] == "sec" and len(disque["nll_par_jeton"]) == 6
    # (d) témoin : deux tranches différentes, deux PPL différentes ; la même tranche rejouée, la même PPL
    assert resultats[0]["ppl"] != resultats[1]["ppl"]
    rejeu = outil.courir([fichiers[0]], MoteurFactice(str(sortie)), TokJouet(), lambda p: open(p).read(), torch, Params,
                         bras="rejeu", sortie=str(sortie), prefixe=10, notes=6, prefixe_seq="", base={}, sortie_txt=io.StringIO())
    assert rejeu[0]["ppl"] == resultats[0]["ppl"]


def test_reprendre_saute_les_json_presents(outil, tmp_path):
    sortie = tmp_path / "sortie"; sortie.mkdir()
    fichiers = _tranches(tmp_path, 3)
    (sortie / "ppl-essai-t2.json").write_text("{}")
    moteur = MoteurFactice(str(sortie))
    res = outil.courir(fichiers, moteur, TokJouet(), lambda p: open(p).read(), torch, Params, bras="essai",
                       sortie=str(sortie), prefixe=10, notes=6, prefixe_seq="", base={}, reprendre=True,
                       sortie_txt=open(os.devnull, "w"))
    assert [r["k"] for r in res] == [1, 3] and moteur.requetes == ["t1", "t3"]


def test_cache_de_prefixe_desactive_a_la_construction_du_moteur():
    """runner.py:984 publie les blocs d'une requête finie au cache de préfixe :
    avec le cache actif, la tranche suivante sur le même préfixe sauterait le
    prefill et la PPL ne jugerait plus « KV écrit par le chemin de prefill »."""
    import ast
    arbre = ast.parse(open(_OUTIL, encoding="utf-8").read())
    appels = [n for n in ast.walk(arbre) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Engine"]
    assert len(appels) == 1
    kw = {k.arg: k.value for k in appels[0].keywords}
    # défaut = régime servi (cache ON : poste2 07 h 55, −1,2 % cache éteint = _frontiere_insta) ; --sans-cache-prefixe = bras
    src = ast.unparse(kw["enable_prefix_cache"])
    assert src == "not args.sans_cache_prefixe", src


def test_analyser_arguments_et_repli_env(outil, monkeypatch):
    monkeypatch.setenv("PPL_PREFIXE", "gMASK sop")
    monkeypatch.setenv("ACVRAM_MODELE_MESURE", "/nulle/part")
    monkeypatch.setenv("PPL_DECODE_CORPUS", "/corpus/unique.txt")
    a = outil.analyser(["--bras", "fp8", "--depuis", "4", "--reprendre"])
    assert a.prefixe_seq == "gMASK sop" and a.modele == "/nulle/part" and a.prefixe == 8192 and a.notes == 512
    assert a.depuis == 4 and a.reprendre and outil.lister_tranches(a) == ["/corpus/unique.txt"]
    b = outil.analyser(["--tranches", "x", "y", "--prefixe-seq", "", "--prefixe", "1024", "--notes", "64"])
    assert outil.lister_tranches(b) == ["x", "y"] and b.prefixe_seq == "" and b.prefixe == 1024
