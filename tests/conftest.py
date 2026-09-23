import json
import os

# Sans carte visible, les noyaux Triton (kernels/gemm_groupe, gemm_etroit,
# attn_paginee) tournent dans l'interpréteur numpy : la variable doit être
# posée AVANT tout import de triton — la collecte de la suite l'importe (via
# torch) avant que le premier test Triton ne la pose, et un @triton.jit
# décoré sans elle rend « Cannot call @triton.jit'd outside of the scope of
# a kernel » (vu en suite complète seulement, pas en fichier isolé).
# Éco d'horloge (sage-eco-2700-defaut-19-09) : un test qui charge un moteur sur
# carte ne pose pas `-lgc` de lui-même — la suite n'est pas un service ; un
# test d'éco pose ACVRAM_ECO explicitement.
os.environ.setdefault("ACVRAM_ECO", "off")
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    os.environ.setdefault("TRITON_INTERPRET", "1")
    # Les noyaux `fla` (flash-linear-attention) sont autoréglés : sans
    # pilote, le banc de l'autoréglage échoue (« 0 active drivers »). Sous
    # l'interpréteur, la première configuration suffit — les valeurs ne
    # dépendent pas du réglage, seul le temps en dépend.
    try:
        from triton.runtime import autotuner as _at

        _init = _at.Autotuner.__init__

        def _init_une_config(self, *a, **kw):
            _init(self, *a, **kw)
            if len(self.configs) > 1:
                self.configs = self.configs[:1]
        _at.Autotuner.__init__ = _init_une_config
    except Exception:                                    # noqa: BLE001
        pass

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Un point de contrôle de 4 couches, de forme llama, assez petit pour être
    converti en une seconde.

    Avec une graine fixe. Sans elle, toute la suite est non déterministe, et les
    tests qui comparent des formats de quantification sur ce modèle sont assez
    proches du plancher de bruit pour qu'un tirage différent les fasse basculer
    — c'est exactement ainsi qu'une exécution verte dans un répertoire de travail
    est devenue rouge dans une extraction propre.
    """
    from safetensors.torch import save_file

    torch.manual_seed(20260830)
    H, I, L, NH, NKV, V = 256, 688, 4, 8, 2, 1024
    d = tmp_path_factory.mktemp("hf")
    json.dump({
        "architectures": ["LlamaForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
    }, open(d / "config.json", "w"))

    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * 0.02}
    for i in range(L):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(NH * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.k_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.v_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * hd, dtype=torch.bfloat16) * .02
        sd[p + "mlp.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16) * .02
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture(scope="session")
def target_rig():
    from acvram.hardware.profiles import load_profile
    return load_profile("rig-14900k-5090-3080ti")


@pytest.fixture(scope="session")
def converted(tiny_checkpoint, target_rig, tmp_path_factory):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("acvram"))
    convert_checkpoint(tiny_checkpoint, plan,
                       ConversionOptions(out_dir=out), spec=spec)
    return out


@pytest.fixture(scope="session")
def tiny_checkpoint_qknorm(tmp_path_factory):
    """Le même modèle minuscule, mais de forme Qwen3 : une RMSNorm par tête sur
    Q et sur K, appliquée avant la RoPE.

    Les poids de ces normalisations sont volontairement tirés au hasard autour
    de 1 : à un, la normalisation reste visible mais un test qui les ignorerait
    resterait proche, ce qui masquerait l'erreur.
    """
    from safetensors.torch import save_file

    torch.manual_seed(20260831)
    H, I, L, NH, NKV, V = 256, 688, 4, 8, 2, 1024
    hd = H // NH
    d = tmp_path_factory.mktemp("hf_qknorm")
    json.dump({
        "architectures": ["Qwen3ForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 2048, "head_dim": hd,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "torch_dtype": "bfloat16",
        "model_type": "qwen3",
    }, open(d / "config.json", "w"))

    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * 0.02}
    for i in range(L):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(NH * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.k_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.v_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * hd, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.q_norm.weight"] = (1 + torch.randn(hd) * .1).to(torch.bfloat16)
        sd[p + "self_attn.k_norm.weight"] = (1 + torch.randn(hd) * .1).to(torch.bfloat16)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16) * .02
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture(scope="session")
def converted_qknorm(tiny_checkpoint_qknorm, target_rig, tmp_path_factory):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint_qknorm, "tiny-qknorm")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("acvram_qknorm"))
    convert_checkpoint(tiny_checkpoint_qknorm, plan,
                       ConversionOptions(out_dir=out), spec=spec)
    return out


@pytest.fixture(autouse=True)
def _carte_disponible(request):
    """Une carte saturée par une mesure en cours rend « ignoré », pas « échec ».

    Le 8/09, sept tests graphes/MoE ont échoué pendant qu'une perplexité
    occupait la 5090 : graphes non capturés (OOM silencieux), pile d'experts
    refusée faute de place. Trois sessions se partagent la machine ; une carte
    occupée est l'état normal. Un test qui exige de la VRAM la vérifie avant
    de courir — le seuil couvre le modèle-jouet, ses graphes et la marge de
    capture.
    """
    if not any(m.name == "gpu_requis" for m in request.node.iter_markers()):
        return
    if not torch.cuda.is_available():
        pytest.skip("pas de carte CUDA")
    libre, _ = torch.cuda.mem_get_info()
    if libre < 4 << 30:
        pytest.skip(f"carte occupée : {libre / (1 << 30):.1f} Gio libres, "
                    "4 requis — mesure en cours ailleurs ?")


@pytest.fixture(autouse=True)
def _regime_des_marqueurs(request, monkeypatch):
    """T4 20/09 11 h 40 sous enveloppe (carte visible) : 46 rouges de fixture —
    des tests écrits à sec dont le régime tenait à l'ABSENCE de carte. Un
    test porte son régime par son marqueur, pas par le poste qui le joue.
    - a_sec : noyaux Triton interprétés ; TRITON_INTERPRET se pose avant
      l'import de triton (tête de ce fichier), donc carte visible → ignoré,
      pas rouge ; la passe CUDA_VISIBLE_DEVICES= le joue.
    - sans_extension : repli torch (get_extension() → None) sur tenseurs CPU,
      carte ou pas — « d/q/g doit résider sur un périphérique CUDA » ×8.
    - pile_naturelle : `_stacks[nom][1]` lu par les tests de la pile, que la
      disposition Marlin (défaut servi depuis le 18/09) rend (None) — NoneType
      / tenseur de taille 0 ×20."""
    marqueurs = {m.name for m in request.node.iter_markers()}
    if "a_sec" in marqueurs and torch.cuda.is_available():
        pytest.skip("test à sec (Triton interprété) : carte visible — passe CUDA_VISIBLE_DEVICES=")
    if "sans_extension" in marqueurs:
        from acvram import kernels as K
        monkeypatch.setattr(K, "get_extension", lambda: None)
    if "pile_naturelle" in marqueurs:
        from acvram.engine import model as M
        from acvram.engine import moe as MOE
        monkeypatch.setattr(MOE, "_GEMV_LAYOUT", "naturel")
        monkeypatch.setattr(MOE, "_PREFILL_GROUPED", "grouped_mm")
        monkeypatch.setenv("ACVRAM_GEMV_LAYOUT", "naturel")
        monkeypatch.setenv("ACVRAM_PREFILL_GROUPED", "grouped_mm")


# Au-dela de cette occupation, un debit mesure n'est plus celui du code teste.
OCCUPATION_MAX = 50


def occupation_gpu(index: "int | None" = None) -> "int | None":
    """Occupation de la carte courante en pour cent, ou None si indisponible.

    Separee de la fixture pour etre eprouvee : un garde-fou qu'on ne peut pas
    faire mordre a la demande n'est pas un garde-fou.
    """
    import subprocess
    if index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    try:
        lignes = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")
        return int(lignes[index].strip())
    except Exception:
        return None                 # pas de nvidia-smi : on laisse courir


@pytest.fixture(autouse=True)
def _carte_au_repos(request):
    """Un test de DEBIT exige une carte au repos, pas seulement de la place.

    La garde ci-dessus regarde la memoire libre. Un debit ne s'effondre pas
    par manque de place mais par partage des multiprocesseurs : le 8/09,
    `test_le_gemv_nvfp4_ne_part_pas_en_emulation` a rendu 57 Go/s au lieu de
    300 avec huit gigaoctets libres et une conversion voisine a cent pour cent
    d'occupation. Le test annoncait « reparti en emulation logicielle » — un
    diagnostic faux tire d'une mesure vraie, exactement ce que cette journee a
    produit trois fois ailleurs.

    On mesure donc l'instrument avant la mesure : si la carte travaille deja
    pour quelqu'un d'autre, le chiffre ne dira rien du code teste.
    """
    if not any(m.name == "debit_requis" for m in request.node.iter_markers()):
        return
    if not torch.cuda.is_available():
        pytest.skip("pas de carte CUDA")
    occupation = occupation_gpu()
    if occupation is not None and occupation > OCCUPATION_MAX:
        pytest.skip(f"carte a {occupation} % d'occupation — un debit mesure "
                    f"pendant le travail d'autrui ne dit rien du code teste")


# Seuil de KL entre deux jeux de logits : nos propres mesures du 10/09
# donnent 1,9e-4 a 7,0e-3 nats pour un changement d'ordre d'accumulation
# legitime (noyau fusionne vs torch, decodage batche vs par sequence), et
# 0,8 a 5,0 nats pour un vrai defaut de traitement par lot. `torch.equal`
# ne discriminait plus rien depuis que graphe et eager peuvent emprunter
# des chemins numeriquement differents mais tous deux corrects (fusion de
# noyau, batching) : le KL, qui pese chaque composante par sa probabilite
# plutot que de comparer bit a bit, reste discriminant entre les deux
# regimes. Le seuil est pose a 1e-2, un ordre de grandeur au-dessus du
# plus grand ecart legitime mesure et deux ordres en dessous du plus petit
# defaut reel observe.
KL_LOGITS_MAX = 1e-2


def assert_logits_proches(a: torch.Tensor, b: torch.Tensor, msg: str = "",
                          seuil: float = KL_LOGITS_MAX) -> None:
    """Remplace un ``torch.equal`` devenu aveugle : compare deux jeux de
    logits par KL plutot que bit a bit. Voir ``KL_LOGITS_MAX`` pour l'origine
    du seuil."""
    pa = F.log_softmax(a.double().reshape(-1, a.shape[-1]), dim=-1)
    pb = F.softmax(b.double().reshape(-1, b.shape[-1]), dim=-1)
    kl = F.kl_div(pa, pb, reduction="batchmean").item()
    assert kl < seuil, f"{msg} (KL={kl:.4e}, seuil={seuil:.4e})"


# ---------------------------------------------------------------------------
# LE VERROU DE CARTE, PRIS PAR LA SESSION PYTEST ENTIERE
#
# Le 10/09/2026 a 20:09:48, `pytest tests/ -q` a tourne 127,57 s pendant une
# manche de mesure sur la 5090 et a pollue quatre tours sur quinze. Une suite
# qui alloue des gigaoctets et capture des graphes CUDA est un TEST DE CARTE,
# quel que soit son nom.
#
# Trois remedes, et deux sont mauvais :
#
#   sauter les tests GPU quand la carte est prise
#       -> la couverture tombe EN SILENCE, et precisement quand la machine est
#          chargee, c'est-a-dire quand les tests de contention comptent le plus.
#          Un garde qui cesse de pouvoir rendre « faux » au pire moment.
#   un verrou par test
#       -> dix-neuf fichiers attendent chacun leur tour, la suite devient
#          impraticable et personne ne la lance plus.
#   un verrou pour la SESSION pytest
#       -> UNE attente, couverture complete. C'est celui-ci.
#
# Meme fichier de verrou que `outils/carte.sh` (`/tmp/acvram-carte-0.lock`,
# surchargeable par ACVRAM_VERROU), donc la suite et les mesures s'excluent
# reellement au lieu de s'exclure chacune de son cote.
_VERROUS = []


def _verrou_tenu_en_mesure():
    """Une carte est-elle tenue en TYPE=mesure, MAINTENANT, par un vivant ?

    Contrairement à `_gpu_demande` ci-dessous, ne dépend PAS de
    `CUDA_VISIBLE_DEVICES` : le lanceur de session exporte cette variable
    vide par défaut (Sage 14/09), donc une suite lancée normalement se
    croit sans carte et ne prend jamais le flock plus bas — mais elle
    tourne quand même sur le même PROCESSEUR qu'une fenêtre HTTP mesurée
    sous `carte.sh`. Trouvé le 18/09 (load 70,8) : trois `pytest` de pairs
    pendant une fenêtre HTTP, aucun n'a rien vu venir puisque aucun ne
    touchait le GPU. REGLES §2 le disait en consigne ; ceci en fait un
    refus dur — lecture seule, jamais de verrou pris ici."""
    import glob
    import re
    motif = os.environ.get("ACVRAM_VERROU_GLOB", "/tmp/acvram-carte-*.lock")
    for verrou in sorted(glob.glob(motif)):
        info = verrou + ".qui"
        try:
            champs = open(info).read().split(None, 3)
            pid, _pris_a, nom = int(champs[0]), champs[1], champs[2]
            type_ = champs[3].strip() if len(champs) > 3 else "?"
        except (OSError, ValueError, IndexError):
            continue
        if type_ != "mesure":
            continue
        # le detenteur est NOTRE ancetre (pytest lance SOUS carte.sh, qui
        # exporte ACVRAM_CARTE_TENUE) : ce n'est pas la fenetre d'un pair,
        # c'est la notre — la refuser serait refuser sa propre prise (19/09,
        # Oceane : `carte.sh pytest tests/test_gemv_marlin.py` refuse par
        # lui-meme, meme geste que le code 66 plus bas, mais pose ici a moitie)
        if str(pid) == os.environ.get("ACVRAM_CARTE_TENUE", ""):
            continue
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            continue   # detenteur disparu, info perimee (meme logique que carte.sh qui_tient())
        return verrou, pid, nom
    return None


def _suite_ciblee(config) -> bool:
    """Un chemin/nodeid explicite, un `-k` ou un `-m` : la personne a dit ce
    qu'elle veut lancer. Rien de tout ça (pytest nu, `testpaths` de
    `pyproject.toml` par défaut) = suite complète. `file_or_dir` est déjà
    séparé des valeurs de flags par l'analyseur de pytest lui-même — pas de
    re-parsing fragile de la ligne de commande brute."""
    return bool(config.getoption("file_or_dir") or config.getoption("keyword")
               or config.getoption("markexpr"))


def _gpu_demande(config):
    """La suite touche-t-elle la carte ? Oui des qu'un GPU est visible et que
    rien ne l'interdit. On ne cherche PAS a deviner quels tests allouent : le
    seul qui n'alloue pas est celui qu'on n'a pas encore ecrit, et se tromper
    dans ce sens rend le verrou inutile."""
    if os.environ.get("ACVRAM_TESTS_SANS_VERROU"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _chemins_de_verrou():
    """LE VERROU EST PAR CARTE, ET LE DEFAUT DE `carte.sh` NE L'EST PAS.

    `carte.sh` prend `/tmp/acvram-carte-0.lock` sauf si ACVRAM_VERROU est pose
    a la main. Vu le 10/09 a 20h24 : une session mesurant sur la 3080 Ti avait
    pense a poser `ACVRAM_VERROU=/tmp/acvram-carte-1.lock` — la justesse
    dependait de sa memoire. Ma premiere version de ce verrou codait `-0` en
    dur : une suite lancee avec `CUDA_VISIBLE_DEVICES=1` aurait bloque la carte
    qu'elle n'emploie pas ET laisse sans protection celle qu'elle emploie.
    Faux dans les deux sens a la fois.

    On derive donc du parc VISIBLE, un verrou par carte, dans un ordre FIXE :
    deux sessions qui verrouillent deux cartes en ordres opposes se
    bloqueraient mutuellement.
    """
    force = os.environ.get("ACVRAM_VERROU")
    if force:
        return [force]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        # un UUID n'est pas un index : on ne le traduit pas, on le garde tel
        # quel dans le nom, ce qui reste exclusif entre deux sessions qui
        # nomment la carte de la meme facon.
        ids = sorted(x.strip().replace("/", "_") for x in vis.split(",")
                     if x.strip())
    else:
        try:
            import torch
            ids = [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            ids = ["0"]
    return [f"/tmp/acvram-carte-{i}.lock" for i in ids]


def pytest_configure(config):
    """Le verrou de carte, pris par la SESSION pytest entiere.

    Le 10/09/2026 a 20:09:48, la suite a tourne 127,57 s pendant une manche de
    mesure sur la 5090 et a pollue quatre tours sur quinze. Une suite qui
    alloue des gigaoctets et capture des graphes CUDA est un TEST DE CARTE,
    quel que soit son nom.

    Trois remedes, et deux sont mauvais :

      sauter les tests GPU quand la carte est prise
          -> la couverture tombe EN SILENCE, et precisement quand la machine
             est chargee, c'est-a-dire quand les tests de contention comptent
             le plus. Un garde qui cesse de pouvoir rendre « faux » au pire
             moment.
      un verrou par test
          -> dix-neuf fichiers attendent chacun leur tour, la suite devient
             impraticable et personne ne la lance plus.
      un verrou pour la SESSION pytest
          -> UNE attente, couverture complete. C'est celui-ci.
    """
    # 18/09 : load 90/32 cœurs, trois suites COMPLETES de pairs en parallele
    # pendant une fenetre P1 -- le verrou-mesure ci-dessous ne protege que les
    # worktrees qui l'ont deja fusionne (REGLES §1), donc un second filet,
    # INCONDITIONNEL, qui ne depend d'aucune fusion : une suite NON CIBLEE
    # (aucun chemin/nodeid, aucun -k, aucun -m -- pytest nu, testpaths par
    # defaut) est le signal le plus fiable qu'une session a lance « la suite »
    # plutot que ce qu'elle vient de changer. Refuse a load1 > nproc/4 (plus
    # bas que le seuil nproc/2 d'energie.py : ici on protege TOUTE la
    # machine, pas seulement une fenetre HTTP).
    # Ordre voulu : l'override se lit AVANT `_suite_ciblee(config)`, en
    # court-circuit -- un `config` factice (tests, `None`) ne doit pas être
    # sollicité quand l'override suffit à trancher.
    if os.environ.get("ACVRAM_TESTS_SOUS_CHARGE") != "1" and not _suite_ciblee(config):
        load1 = os.getloadavg()[0]
        nproc = os.cpu_count() or 1
        if load1 > nproc / 4:
            raise pytest.UsageError(
                f"suite complète sous charge : cible tes tests ou attends "
                f"(load1={load1:.1f} > nproc/4={nproc / 4:.1f}, {nproc} cœurs). "
                f"ACVRAM_TESTS_SOUS_CHARGE=1 pour la CI finale.")

    # Symétrique de ce qui précède, et INCONDITIONNEL (pas derrière
    # `_gpu_demande`, qui ne voit rien sans CUDA_VISIBLE_DEVICES) : une
    # fenêtre HTTP mesurée sous `carte.sh` (TYPE=mesure) se fait fausser par
    # le PROCESSEUR qu'une suite pytest lui prend, même une suite qui ne
    # touche jamais le GPU (18/09, load 70,8, REGLES §2).
    if os.environ.get("ACVRAM_TESTS_PENDANT_MESURE") != "1":
        tenue = _verrou_tenu_en_mesure()
        if tenue:
            verrou, pid, nom = tenue
            raise pytest.UsageError(
                f"carte tenue en TYPE=mesure par PID {pid} ({nom}, {verrou}) : "
                f"la suite ne demarre pas — une suite pytest, meme sans GPU, "
                f"prend du processeur a une fenetre HTTP mesuree (18/09, load "
                f"70,8). ACVRAM_TESTS_PENDANT_MESURE=1 pour passer outre en "
                f"connaissance de cause.")

    if not _gpu_demande(config):
        return

    # REFUS DE DOUBLE PRISE, repris de carte.sh (code 66) : un descripteur
    # herite ne peut pas etre repris par `flock -n`, donc attendre ici serait
    # attendre son propre ancetre. Un pytest lance SOUS carte.sh ne reprend
    # donc pas le verrou — il l'a deja.
    if os.environ.get("ACVRAM_CARTE_TENUE"):
        print(f"\n[tests] carte deja tenue par le PID "
              f"{os.environ['ACVRAM_CARTE_TENUE']} plus haut dans la meme "
              f"arborescence — la suite ne reprend pas le verrou.", flush=True)
        return

    import fcntl
    import time
    attente = float(os.environ.get("ACVRAM_TESTS_ATTENTE", "1800"))
    chemins = _chemins_de_verrou()
    debut = time.monotonic()
    dernier_dit = -60.0
    for chemin in chemins:
        fd = open(chemin, "a+")
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                ecoule = time.monotonic() - debut
                if ecoule >= attente:
                    fd.close()
                    _relacher()
                    # ABANDON EXPLICITE, jamais un saut silencieux : la suite
                    # n'a pas tourne, et ca doit se voir comme un REFUS.
                    # `UsageError` et non `RuntimeError` : la seconde sort en
                    # INTERNALERROR avec une pile de pluggy, donc se lit comme
                    # un plantage alors que c'est une decision — et un refus
                    # qui ressemble a un bug sera contourne, pas respecte.
                    raise pytest.UsageError(
                        f"carte tenue par un autre depuis {ecoule:.0f} s "
                        f"({chemin}) : la suite ne demarre pas. Attendre, ou "
                        f"poser ACVRAM_TESTS_SANS_VERROU=1 en sachant que la "
                        f"mesure voisine sera polluee.")
                if ecoule - dernier_dit >= 60.0:
                    dernier_dit = ecoule
                    print(f"[tests] {chemin} occupe, attente {ecoule:.0f} s "
                          f"/ {attente:.0f} s", flush=True)
                time.sleep(2.0)
        _VERROUS.append(fd)
    # marqueur d'ancetre, pour qu'un carte.sh lance DEPUIS un test refuse au
    # lieu d'attendre le verrou que cette session tient deja
    os.environ["ACVRAM_CARTE_TENUE"] = str(os.getpid())
    print(f"\n[tests] carte obtenue apres {time.monotonic() - debut:.1f} s "
          f"({', '.join(chemins)})", flush=True)


def _relacher():
    while _VERROUS:
        try:
            _VERROUS.pop().close()   # fermer relache le flock
        except Exception:
            pass
    os.environ.pop("ACVRAM_CARTE_TENUE", None)


def pytest_unconfigure(config):
    if _VERROUS:
        _relacher()
        print("[tests] carte relachee", flush=True)


def attendre_chemin(bloc, nom: str, avant: int = 0) -> int:
    """REGLES § 7, « noyau atteint, pas fonction appelée » (Sage, 18/09, après
    trois tests d'équivalence qui comparaient sans atteindre le chemin) :
    asserte que le préfill du ``bloc`` (MoEBlock) vient de prendre le chemin
    ``nom`` (compteur `_chemin`) et que son compte a AVANCÉ depuis ``avant``.
    Rend le compte courant. À appeler AVANT toute comparaison de sorties."""
    chemins = getattr(bloc, "chemins", {})
    assert getattr(bloc, "dernier_chemin", None) == nom, \
        f"chemin pris : {getattr(bloc, 'dernier_chemin', None)!r}, attendu {nom!r} (compteurs {chemins})"
    assert chemins.get(nom, 0) > avant, f"le chemin {nom!r} n'a pas avancé : {chemins}"
    return chemins[nom]
