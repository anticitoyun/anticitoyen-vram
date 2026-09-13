"""Ligne de commande acvram.

    acvram doctor                     cette machine est-elle prête, et pour quoi
    acvram detect                     quel matériel est présent
    acvram plan MODELE                où irait chaque couche
    acvram convert MODELE -o REP      quantifie, un format par GPU de destination
    acvram serve REP                  serveur compatible avec l'API OpenAI
    acvram eval REP [REP ...]         perplexité, pour classer les formats
    acvram bench REP                  mesure ce que le plan ne faisait qu'estimer
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Optional

# La version a UNE seule source, acvram/__init__.py. Elle etait ici en dur et
# a derive : le paquet installe le 10/09/2026 annoncait 0.5.0 par dpkg, 0.3.0
# par acvram.__version__ et 0.2.0 par `acvram --version` — trois copies, trois
# valeurs, et celle que l utilisateur voit etait la plus ancienne des trois.
from . import __version__            # noqa: E402  (source unique de verite)



# Variables d'environnement que le code lit REELLEMENT. Le 9/09/2026,
# `MAXTOK=65536` a ete pose dans l'environnement d'une mesure de perplexite
# que rien ne lisait : la mesure s'est arretee a 16 fenetres au lieu de 128,
# sans que rien ne le signale. Une variable posee qui ne va nulle part est
# une consigne silencieusement ignoree.
#
# Cette liste se met a jour avec le code ; une epreuve verifie qu'elle ne
# derive pas.
#
# Tu viens d'ecrire `os.environ.get("ACVRAM_...")` ailleurs dans le depot ?
# Ajoute le nom ici avant de committer -- cinq variables l'ont deja oublie
# le 10/09, la garde ne les a signalees qu'apres coup, jamais au moment ou
# elles ont ete ecrites.
VARIABLES_LUES = {
    "ACVRAM_ALLOC_EXTENSIBLE",
    "ACVRAM_BANC_ACCEPTE_REPLAN",
    "ACVRAM_KERNEL_CACHE",
    "ACVRAM_FOND_COOL",
    "ACVRAM_FOND_ZEN",
    "ACVRAM_GALERIE_DIR",
    "ACVRAM_LOGITS_BF16",
    "ACVRAM_MAX_GRAPHS",
    "ACVRAM_PARC",
    "ACVRAM_MODELES",
    "ACVRAM_SEUIL_EXIL",
    "ACVRAM_REPIN",
    "ACVRAM_VERROU",
    "ACVRAM_MLA_BATCH",
    "ACVRAM_NOM",
    "ACVRAM_MLA_DEBUG_ECART",
    "ACVRAM_MLA_EAGER_TORCH",
    # Bras du banc a trois bras de l'attention paginee. Lu par le noyau, donc
    # il DOIT etre declare ici : la garde l'a signale comme inconnu, ce qui est
    # exactement son role — une variable posee qui ne va nulle part est une
    # consigne silencieusement ignoree.
    "ACVRAM_PA_ARM",
    "ACVRAM_PREFILL_BATCH",
    "ACVRAM_BUDGET_JETONS",
    # Echappement de mesure de la fusion, pour les QUATRE empileurs.
    # ACVRAM_SANS_FUSION_BF16 ne coupait que le chemin bf16 : le gain de la
    # fusion n'etait donc mesurable qu'en bf16, et c'est ainsi qu'un +2,60 %
    # mesure la ou 100 % des groupes fusionnent a ete transporte sur un int8
    # ou 7,8 % seulement fusionnent.
    "ACVRAM_SANS_FUSION",
    # Renverse le signe de l'ordre du glouton budgetaire, RIEN D'AUTRE. Sert a
    # eprouver si le critere (gain de SNR par octet) est bien oriente pour un
    # objectif de perplexite : si oui le bras inverse est nettement pire, si non
    # il est meilleur ou equivalent. Instrument de mesure, pas reglage.
    "ACVRAM_ORDRE_SAC_INVERSE",
    # Choisit la cle de tri du glouton budgetaire : `snr` (defaut, gain de
    # decibels par octet) ou `erreur` (erreur de sortie evitee par octet). La
    # seconde n'est PAS une transformation monotone de la premiere : elle
    # applique 10^(-snr/20) aux deux SNR AVANT la soustraction, donc elle
    # privilegie les tenseurs a faible SNR de base — verifie sur nos donnees,
    # correlation -0,66 entre SNR de base et deplacement de rang. Un mode
    # inconnu leve une erreur au lieu de retomber en silence sur le defaut.
    "ACVRAM_ORDRE_SAC",
    "ACVRAM_LISTE_PROMUS",
    "ACVRAM_LISTE_CLE",
    "ACVRAM_MAX_PROMUS",
    "ACVRAM_SCALER_SANS_CACHE",
    # ACVRAM_SRC_HASH n'est PAS une variable d'environnement : c'est une option
    # de compilation (-D) portant le sha du .cu. Elle figure ici parce que
    # l'epreuve la reconnait au motif ACVRAM_[A-Z0-9_]+ dans le source, et
    # qu'une liste incomplete fait echouer la garde — pas parce qu'on la lit.
    "ACVRAM_SRC_HASH",
    "ACVRAM_ARCH_FAMILY",
    "ACVRAM_CUDA_HOME",
    "ACVRAM_DISABLE_CPU_KERNELS",
    "ACVRAM_DISABLE_CUDA_GRAPHS",
    "ACVRAM_DISABLE_FP4_GEMM",
    "ACVRAM_DISABLE_KERNELS",
    "ACVRAM_DISABLE_PAGED_ATTN",
    "ACVRAM_EXIL_COUCHES",
    "ACVRAM_EXIL_EXPERTS_FRACTION",
    "ACVRAM_FUSION_NVFP4",
    "ACVRAM_FUSION_PARTIELLE",
    "ACVRAM_GC_FREEZE",
    "ACVRAM_GDN",
    "ACVRAM_GRAPHES_MUETS",
    "ACVRAM_GRAPHS_EAGER",
    "ACVRAM_GW_WARPS",
    "ACVRAM_HYBRID_KERNELS",
    "ACVRAM_HYBRID_SLOTS",
    "ACVRAM_INSTA_MAX",
    "ACVRAM_INSTA_PAS",
    "ACVRAM_INT8_GEMV_MAX",
    "ACVRAM_KDA_CHUNK",
    "ACVRAM_MAMBA_CHUNK",
    "ACVRAM_MLA_BUCKET",
    "ACVRAM_MLA_NORME_NOYAU",
    "ACVRAM_MLP_HOTE_CPU",
    "ACVRAM_MODELS_DIR",
    "ACVRAM_MOE_DECODE_MASQUES",
    "ACVRAM_MOE_GEMM_MAX",
    "ACVRAM_MOE_GLUE_TORCH",
    "ACVRAM_MOE_GROUPED_MAX",
    "ACVRAM_MOE_MMA",
    "ACVRAM_MOE_MMA_BT",
    "ACVRAM_MTP",
    "ACVRAM_NVFP4_GEMV_MAX",
    "ACVRAM_PLAN_FIGE",
    "ACVRAM_POOL_SYNC",
    "ACVRAM_PREFILL",
    "ACVRAM_PREFILL_DEQUANT",
    "ACVRAM_SANS_FUSION_BF16",
    "ACVRAM_SANS_PRECHARGE",
    "ACVRAM_SANS_REPLAN",
    "ACVRAM_SEUIL_FUSION",
    "ACVRAM_SYNC_COUCHES",
    "ACVRAM_TETE_LIEE",
    "ACVRAM_TRACEBACK",
    "ACVRAM_TRACE_PTRS",
    "ACVRAM_TRACE_ROUTAGE",
    "ACVRAM_CHRONO_SYNC",
    "ACVRAM_TRACE_STEPS",
    "ACVRAM_VERBOSE_BUILD",
    "ACVRAM_WARM_GRAPHS",
    "ACVRAM_WARM_SPEC",
}


def _avertir_variables_inconnues() -> None:
    """Signale toute ACVRAM_* posee que le code ne lit pas."""
    inconnues = sorted(v for v in os.environ
                       if v.startswith("ACVRAM_") and v not in VARIABLES_LUES)
    if inconnues:
        print(red("  variables ignorees (le code ne les lit nulle part) : "
                  + ", ".join(inconnues)), flush=True)

def _tty() -> bool:
    """Une barre de progression a sa place sur un terminal, pas dans un tube ni un journal."""
    return sys.stderr.isatty() and not os.environ.get("NO_COLOR")


def _progress(text: str) -> None:
    """Ligne d'avancement. Sur un terminal elle s'ecrase ; redirigee vers un
    fichier elle s'empile, faute de quoi une conversion lancee en tache de fond
    ne montre plus rien du tout."""
    if _tty():
        sys.stderr.write(f"\r{text[:100]:<100}")
    else:
        sys.stderr.write(text.strip() + "\n")
    sys.stderr.flush()


def _progress_done() -> None:
    if _tty():
        sys.stderr.write("\r" + " " * 100 + "\r")
        sys.stderr.flush()


def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


def _h(n: float) -> str:
    for unite in ("o", "Kio", "Mio", "Gio", "Tio"):
        if abs(n) < 1024 or unite == "Tio":
            return f"{int(n)} o" if unite == "o" else f"{n:.1f} {unite}"
        n /= 1024
    return f"{n:.1f} Tio"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_detect(args: argparse.Namespace) -> int:
    from .hardware.detect import detect_rig
    rig = detect_rig(args.profile)
    if args.json:
        print(rig.to_json())
        return 0

    print(bold("materiel"))
    print(f"  source        {rig.source}")
    print(f"  distribution  {rig.distro}")
    print(f"  noyau         {rig.kernel}")
    print(f"  processeur    {rig.cpu.model}")
    if rig.cpu.efficiency_cores:
        print(f"                {rig.cpu.performance_cores} coeurs P + "
              f"{rig.cpu.efficiency_cores} coeurs E, "
              f"epinglez les fils sur le cpuset {rig.cpu.p_core_cpuset}")
    print(f"  memoire vive  {_h(rig.host.total)} au total, "
          f"{_h(rig.host.available)} disponibles")
    print(f"  pilote / cuda {rig.driver_version or '?'} / {rig.cuda_version or '?'}")
    print()
    if not rig.gpus:
        print(yellow("  aucun GPU NVIDIA detecte"))
        return 0
    print(bold("cartes graphiques"))
    for g in rig.gpus:
        caps = g.caps
        print(f"  [{g.index}] {g.name}")
        print(f"       {_h(g.total_mem)} de VRAM, ~{g.vram_bandwidth_gbps:.0f} Go/s")
        print(f"       sm_{caps.sm} ({caps.arch_name})   "
              f"fp4={_yn(caps.fp4_tensor_core)} fp8={_yn(caps.fp8_tensor_core)} "
              f"int8={_yn(caps.int8_tensor_core)} bf16={_yn(caps.bf16)}")
        print(f"       PCIe gen{g.pcie_gen_cur or g.pcie_gen_max} "
              f"x{g.pcie_width_cur or g.pcie_width_max} "
              f"-> {g.host_link_gbps:.1f} Go/s vers l'hote")
        print(f"       {green('poids ' + caps.weight_format)}, cache KV {caps.kv_format}")
    if len(rig.gpus) > 1 and not any(rig.p2p_matrix[0][1:]):
        print()
        print(dim("  pas de pair-a-pair entre GPU (attendu sur GeForce) : les "
                  "tenseurs transitent par la memoire hote epinglee"))
    return 0


def _yn(b: bool) -> str:
    return green("oui") if b else dim("non")


def cmd_doctor(args: argparse.Namespace) -> int:
    from .hardware.detect import detect_rig
    ok = True
    print(bold("acvram doctor"))

    try:
        import torch
        print(f"  {green('ok')}    torch {torch.__version__}, "
              f"cuda {torch.version.cuda or 'cpu-only'}")
    except ImportError:
        print(f"  {red('ECHEC')} torch n'est pas installe")
        return 1

    rig = detect_rig()
    if not rig.gpus:
        print(f"  {yellow('alerte')} aucun peripherique CUDA ; acvram tournera sur processeur seul")
    for g in rig.gpus:
        caps = g.caps
        line = f"  {green('ok')}    [{g.index}] {g.name} sm_{caps.sm} -> {caps.weight_format}"
        print(line)
        if caps.sm >= 120:
            cuda = torch.version.cuda or "0.0"
            major, _, minor = cuda.partition(".")
            if (int(major or 0), int(minor or 0)) < (12, 8):
                ok = False
                print(f"  {red('ECHEC')} {g.name} est Blackwell (sm_{caps.sm}) mais "
                      f"torch est compile pour CUDA {cuda}. sm_120 exige 12.8+. "
                      f"Installez : pip install torch --index-url "
                      f"https://download.pytorch.org/whl/cu128")
        if g.pcie_width_cur and g.pcie_width_cur < g.pcie_width_max:
            print(f"  {yellow('alerte')} [{g.index}] le lien tourne en x"
                  f"{g.pcie_width_cur} au lieu de x{g.pcie_width_max} ; les couches "
                  f"transferees seront {g.pcie_width_max / g.pcie_width_cur:.0f}x plus lentes")

    from . import kernels
    info = kernels.build_info()
    if info["available"]:
        print(f"  {green('ok')}    noyaux CUDA fusionnes compiles pour "
              f"{', '.join(info['device_caps']) or 'n/a'}")
    else:
        print(f"  {yellow('alerte')} noyaux CUDA fusionnes indisponibles "
              f"({info['error']}) ; chemin de reference utilise")

    cpu = info["cpu"]
    if cpu["available"]:
        simd = "AVX2+FMA" if cpu["avx2"] else "scalaire"
        note = "" if cpu["avx2"] else "  (pas d'AVX2 sur ce processeur -- bien plus lent)"
        print(f"  {green('ok')}    noyaux processeur compiles, chemin {simd}{note}")
    else:
        print(f"  {yellow('alerte')} noyaux processeur indisponibles ({cpu['error']}) ; "
              f"l'etage hote sera lent")

    fp4 = info["fp4_tensorcore"]
    if fp4["available"]:
        print(f"  {green('ok')}    produit FP4 sur tensor cores via {fp4['impl']}")
    else:
        print(f"  {yellow('alerte')} pas de produit FP4 sur tensor cores : {fp4['reason']}")
        print(f"        le prefill retombe sur dequantification + cuBLAS ; le "
              f"decodage n'est pas affecte")

    from .kernels import backends as bk
    print(f"  {green('ok')}    backends retenus par (format, peripherique) :")
    for row in bk.table():
        print(f"           {row['device']:<8} {row['format']:<9} -> "
              f"{' > '.join(row['backends'])}")

    for mod in ("safetensors", "fastapi", "uvicorn", "tokenizers", "jinja2"):
        try:
            __import__(mod)
            print(f"  {green('ok')}    {mod}")
        except ImportError:
            ok = False
            print(f"  {red('ECHEC')} {mod} est absent (pip install {mod})")

    if rig.host.total and rig.host.total < 32 * 1024 ** 3:
        print(f"  {yellow('alerte')} {_h(rig.host.total)} de memoire vive limitent "
              f"l'etage hote")
    return 0 if ok else 1


def cmd_plan(args: argparse.Namespace) -> int:
    from .engine.config import load_model_spec
    from .hardware.detect import detect_rig
    from .memory.tiering import PlannerOptions, auto_plan
    rig = detect_rig(args.profile)
    spec = load_model_spec(args.model, args.name)
    opts = PlannerOptions(
        max_model_len=args.max_model_len,
        max_concurrent_seqs=args.max_seqs,
        kv_bits=args.kv_bits,
        host_fraction=args.host_fraction,
        group_size=args.group_size,
        allow_host_tier=not args.no_host,
        force_format=args.format,
        gpus=args.gpus,
        host_exec=args.host_exec,
        host_compute_gb_s=args.host_gb_s,
    )
    plan, trials = auto_plan(spec, rig, opts)
    if args.json:
        print(json.dumps({"spec": spec.to_dict(), "plan": plan.to_dict(),
                          "trials": trials}, indent=2))
        return 0
    print(bold(spec.summary()))
    print()
    print(plan.render())
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from .engine.config import load_model_spec
    from .hardware.detect import detect_rig
    from .memory.tiering import PlannerOptions, auto_plan
    from .quant.convert import ConversionOptions, convert_checkpoint

    rig = detect_rig(args.profile)
    spec = load_model_spec(args.model, args.name)
    if not args.out:
        # Les chemins de la machine de developpement etaient codes ici en
        # dur, avec le nom d'utilisateur dedans : ils partaient tels quels
        # dans le paquet .deb, chez quiconque l'installe. Le repli se lit
        # desormais dans un fichier de configuration, absent par defaut.
        # ACVRAM_MODELS_DIR d'abord (heritage), sinon ACVRAM_MODELES partage
        # avec outils/racine_modeles.py et acvram/server/app.py.
        base = os.environ.get("ACVRAM_MODELS_DIR") or os.environ.get("ACVRAM_MODELES")
        if not base:
            conf = os.path.join(
                os.environ.get("XDG_CONFIG_HOME",
                               os.path.expanduser("~/.config")),
                "acvram", "modeles")
            try:
                with open(conf) as f:
                    for ligne in f:
                        ligne = ligne.strip()
                        if ligne and not ligne.startswith("#") and os.path.isdir(ligne):
                            base = ligne
                            break
            except OSError:
                pass
        if not base:
            print(red("aucun repertoire de sortie : passez -o, posez "
                      "ACVRAM_MODELS_DIR, ou ecrivez un chemin par ligne "
                      f"dans {conf}"))
            return 2
        args.out = os.path.join(base, spec.name.replace("/", "--"))
        print(f"  sortie : {args.out}")
    plan, _ = auto_plan(spec, rig, PlannerOptions(
        max_model_len=args.max_model_len, max_concurrent_seqs=args.max_seqs,
        group_size=args.group_size, force_format=args.format, gpus=args.gpus,
        host_exec=args.host_exec, host_compute_gb_s=args.host_gb_s))
    print(bold(spec.summary()))
    print()
    print(plan.render())
    print()
    # Drapeau structuré : chercher une sous-chaîne dans un message destiné à
    # l'utilisateur casse dès que ce message change de langue.
    if plan.overflowed and not args.force:
        print(red("conversion refusee : le modele ne tient pas. "
                  "Relancez avec --force pour ecrire les fragments malgre tout."))
        return 2

    # AWQ sans statistiques d'activation est inopérant : la recherche sur
    # grille n'a rien pour pondérer les canaux et se fixe sur une échelle plate.
    # Les statistiques sont donc relevées d'abord et, si c'est impossible, on le
    # dit et l'on retombe sur l'arrondi au plus proche, plutôt que d'annoncer une
    # mise à l'échelle qui n'a jamais eu lieu.
    stats = None
    use_awq = not args.no_awq
    from .quant.exl3 import is_exl3
    from .quant.gguf import is_gguf
    if use_awq and is_exl3(args.model):
        print(yellow("  source EXL3 : arrondi au plus proche, sans AWQ"))
        use_awq = False
    if use_awq and is_gguf(args.model):
        # Le collecteur de statistiques lit des safetensors ; et re-calibrer des
        # poids deja quantifies par llama.cpp apporterait peu de toute facon.
        print(yellow("  source GGUF : arrondi au plus proche, sans AWQ"))
        use_awq = False
    if use_awq:
        from .quant.collect import collect_activation_stats, load_calib_ids
        from .server.chat import load_tokenizer
        try:
            tokenizer = load_tokenizer(args.model)
            calib = load_calib_ids(tokenizer, args.calib_file, args.calib_seqs,
                                   args.calib_len, spec.vocab_size)
            print(f"  calibration sur {len(calib)} sequences "
                  f"({sum(len(c) for c in calib)} jetons) ...")

            def cprog(done: int, total: int) -> None:
                _progress(f"  calibration couche {done}/{total}")

            stats = collect_activation_stats(
                args.model, spec, calib, device=args.calib_device,
                progress=cprog)
            _progress_done()
            print(f"  statistiques relevees pour {len(stats)} tenseurs")
        except Exception as exc:                      # noqa: BLE001
            print(yellow(f"  calibration indisponible ({exc}) ; "
                         f"repli sur l'arrondi au plus proche"))
            use_awq = False
            stats = None

    opts = ConversionOptions(
        out_dir=args.out, awq=use_awq, use_hadamard=args.hadamard,
        group_size=args.group_size, n_grid=args.grid,
        lm_head_format=args.lm_head_format, dry_run=args.dry_run,
        q3n_table=(tuple(float(v) for v in args.q3n_table.split(","))
                   if args.q3n_table else None),
        mixed_precision=args.mixed_precision, snr_floor=args.snr_floor,
        max_promotions=args.max_promotions,
        autoriser_grossissement=args.autoriser_grossissement,
        quant_device=args.quant_device, bits_budget_gib=args.bits_budget,
        garder_grille=args.grille_erreurs,
        promotion_cout_max_mib=args.promotion_cout_max,
        format_impose=args.format, mesurer_kld=args.mesurer_kld)

    last = [0.0]

    def progress(name: str, n: int, _: int) -> None:
        now = time.time()
        if now - last[0] < 0.5:
            return
        last[0] = now
        _progress(f"  {n} tenseurs quantifies  {name}")

    report = convert_checkpoint(args.model, plan, opts, spec=spec, stats=stats,
                                progress=progress)
    _progress_done()
    print(report.render())
    if not args.dry_run:
        print()
        print(f"  servez-le avec : {bold(f'acvram serve {args.out}')}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import torch
    import uvicorn

    from .engine.loader import load_model
    from .engine.runner import Engine
    from .server.app import create_app
    from .server.chat import load_tokenizer

    print(f"chargement de {args.model} ...")
    t0 = time.time()
    loaded = load_model(args.model, dtype=torch.bfloat16 if not args.fp16
                        else torch.float16,
                        max_model_len=args.max_model_len,
                        device_override=args.device)
    tokenizer = load_tokenizer(args.model)
    speculator = None
    if args.speculative == "auto":
        # La tete du modele si elle existe, le n-gramme sinon.
        args.speculative = "mtp" if getattr(loaded.model, "mtp", None) is not None \
            else "ngram"
    if args.speculative == "ngram":
        from .engine.speculative import NGramProposer
        speculator = NGramProposer()
    elif args.speculative == "draft":
        if not args.draft_model:
            print(red("--speculative draft exige --draft-model"))
            return 2
        from .engine.speculative import DraftModelProposer
        draft = load_model(args.draft_model, dtype=torch.bfloat16,
                           device_override=args.draft_device)
        speculator = DraftModelProposer(draft, max_model_len=args.max_model_len)
        print(f"  modele brouillon : {args.draft_model} "
              f"({_h(draft.model.nbytes)})")
    elif args.speculative == "mtp":
        if getattr(loaded.model, "mtp", None) is None:
            print(red("--speculative mtp : ce modele n'a pas de tete nextn"))
            return 2
        from .engine.speculative import MTPProposer
        speculator = MTPProposer(loaded.model, max_model_len=args.max_model_len)
        print("  brouillon : tete de prediction multi-jetons du modele")

    engine = Engine(loaded, tokenizer, max_batch_size=args.max_batch,
                    max_model_len=args.max_model_len,
                    enable_prefix_cache=not args.no_prefix_cache,
                    speculator=speculator, spec_k=args.spec_k,
                    enable_cuda_graphs=not args.no_cuda_graphs,
                    host_kv_gib=args.host_kv_gib)
    if engine.graphs is not None:
        n = engine.warm_graphs(int(os.environ.get("ACVRAM_WARM_GRAPHS", "2048")))
        print(f"  graphes CUDA : actifs (decodage), {n} godets capturés d'avance")
    # Le tas est énorme après le chargement (manifeste, tokenizer, modules) :
    # une collecte de génération 2 le parcourt entier — plus de 100 ms toutes
    # les quelques dizaines de pas. Geler ces objets les sort du parcours.
    if os.environ.get("ACVRAM_GC_FREEZE", "1") != "0":
        import gc
        gc.collect()
        gc.freeze()
        # et l'on espace les collectes : le pas de décodage crée peu d'objets
        # cycliques, inutile de balayer toutes les 700 allocations
        gc.set_threshold(50000, 20, 20)
    print(f"  charge en {time.time() - t0:.1f} s, "
          f"{_h(loaded.model.nbytes)} de poids")
    print(f"  blocs KV : {engine.allocator.num_blocks} "
          f"({engine.allocator.num_blocks * 16} jetons par couche)")
    print(f"  cache prefixe : {'desactive' if args.no_prefix_cache else 'actif'}")
    print(f"  speculation   : {args.speculative}"
          f"{'' if args.speculative == 'none' else f', k={args.spec_k}'}")
    if tokenizer:
        print(f"  gabarit chat  : {tokenizer.template_source}")
    else:
        print(yellow("  aucun tokenizer.json ; les points d'entree /v1 qui "
                     "prennent du texte echoueront"))

    name = args.served_name or os.path.basename(os.path.abspath(args.model))
    app = create_app(engine, tokenizer, name,
                     {"model_path": args.model, "version": __version__})
    print()
    print(f"  {bold('OpenAI API')}  http://{args.host}:{args.port}/v1")
    print(f"  {dim('models')}      curl http://{args.host}:{args.port}/v1/models")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch

    from .evaluate import perplexity, render

    # Le cadrage se declare TOUJOURS, avec le corpus et la fenetre : une
    # perplexite ne se compare qu'a une autre prise au meme cadrage, et rien
    # dans le nombre publie ne dit lequel a servi. Imprime avant la mesure,
    # il part dans le journal meme si la sortie est redirigee.
    # Un chiffre qui peut sortir SEUL sera compare a tort. Deux defauts du
    # 9/09 en sont la preuve : min_context a 0 sans avertissement, et un plan
    # degrade rendant un nombre d'allure normale. Dans les deux cas le chiffre
    # voyageait sans ses conditions. La parade generique n'est pas de garder
    # chaque cas, c'est que la configuration EFFECTIVE sorte a cote du
    # resultat — cadrage, corpus et son sha, VRAM libre au chargement.
    _avertir_variables_inconnues()
    sha = "?"
    try:
        import hashlib
        h = hashlib.sha256()
        with open(args.corpus, "rb") as fh:
            for bloc in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloc)
        sha = h.hexdigest()[:12]
    except OSError:
        pass
    print(f"  cadrage : min_context={args.min_context} window={args.window} "
          f"stride={args.stride} max_tokens={args.max_tokens}", flush=True)
    print(f"  corpus  : {os.path.basename(args.corpus)} sha256:{sha}", flush=True)
    if torch.cuda.is_available():
        libre, total = torch.cuda.mem_get_info()
        print(f"  carte   : {libre / 2**30:.2f} Gio libres sur "
              f"{total / 2**30:.2f}", flush=True)
    if not args.min_context:
        # La garde de `evaluate` n'avertit que si le corpus est plus court que
        # la fenetre — jamais sur wiki.test.raw (1,29 Mo). Sans ce message,
        # oublier le cadrage donne un chiffre faux d'un facteur proche de 2
        # (9,525 contre 7,233 au protocole) sans le moindre signe.
        # Precision du 10/09 : il y a DEUX protocoles, et ce message n'en
        # nommait qu'un. Dire « le protocole » envoie chercher un defaut la ou
        # il n'y en a pas — meme famille que la divergence corpus reperee le
        # meme jour entre docs/BARRIERE-QUALITE-PROTOCOLE.md et la chaine de
        # l'etalon. Un avertissement doit nommer CE QU'IL INVALIDE.
        print(red("  min_context=0 : les premieres positions sont notees avec "
                  "un contexte quasi vide."), flush=True)
        print(red("    non comparable a la barriere de qualite de noyau, qui "
                  "impose --min-context 256 --window 512 --stride 512 sur "
                  "wiki.test.raw ;"), flush=True)
        print(red("    COMPARABLE en revanche a l'etalon exterieur "
                  "transformers/GPTQ, qui note des segments disjoints de 2048 "
                  "sans contexte reporte — c'est meme le seul cadrage qui lui "
                  "corresponde (--window 2048 --stride 2048 --min-context 0 "
                  "sur wiki-gptq.txt, reference 5,4141)."), flush=True)
    # Les modeles se chargeaient l'un apres l'autre dans le MEME processus sans
    # que le precedent soit libere. Le 9/09/2026, une barriere de qualite a
    # mesure un nvfp4 puis charge un bf16 par-dessus :
    #   plan reajuste : 32 MLP de plus en RAM hote (15,9 Gio pour 18,1 libres)
    #   OutOfMemoryError : 111,88 MiB libres sur 31,36 Gio
    # Le second modele est donc mesure EN REGIME DEGRADE, ou pas du tout — et
    # une perplexite prise sur un plan degrade n'est comparable a rien.
    #
    # Liberer entre deux ne suffit pas a garantir un plan identique : le
    # cache de l'allocateur et la fragmentation survivent. Un modele par
    # PROCESSUS reste la seule mesure propre, et c'est ce que dit
    # l'avertissement.
    if len(args.models) > 1:
        print(red(f"  {len(args.models)} modeles dans un seul processus : le "
                  f"plan du second depend de ce que le premier a laisse. "
                  f"Pour une mesure comparable, un appel par modele."),
              flush=True)
    results = []
    for i, path in enumerate(args.models):
        def prog(done: int, total: int, _p: str = path) -> None:
            _progress(f"  {os.path.basename(_p)}: window {done}/{total}")
        results.append(perplexity(
            path, args.corpus, window=args.window, stride=args.stride,
            max_tokens=args.max_tokens, device=args.device, progress=prog,
            min_context=args.min_context))
        _progress_done()
        if i + 1 < len(args.models):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    results.sort(key=lambda r: r.perplexity)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    print(render(results))
    if len(results) > 1:
        print()
        print(f"  meilleur : {bold(results[0].model)} a {results[0].perplexity:.3f}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmarks
    return run_benchmarks(args)


def cmd_profiles(args: argparse.Namespace) -> int:
    from .hardware.profiles import list_profiles
    for name, desc in list_profiles().items():
        print(f"  {bold(name)}\n      {desc}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acvram",
        description="anticitoyen VRAM/RAM — inference etagee, quantifiee par GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--version", action="version", version=f"acvram {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="rapporte le materiel local")
    d.add_argument("--profile", help="utilise un profil declare au lieu de sonder")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_detect)

    doc = sub.add_parser("doctor", help="verifie que cette machine peut faire tourner acvram")
    doc.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("profiles", help="liste les profils materiels declares")
    pr.set_defaults(func=cmd_profiles)

    def add_plan_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("model", help="repertoire de modele HF (avec config.json)")
        sp.add_argument("--name", help="remplace le nom du modele")
        sp.add_argument("--profile", help="planifie pour un profil declare")
        sp.add_argument("--max-model-len", type=int, default=8192)
        sp.add_argument("--max-seqs", type=int, default=8,
                        help="sequences simultanees que le cache KV doit tenir")
        sp.add_argument("--group-size", type=int, default=128)
        sp.add_argument("--format", help="impose un seul format de poids partout")
        sp.add_argument("--gpus", default="auto",
                        help="auto (le debit tranche), all, ou des indices "
                             "comme 0,1")
        sp.add_argument("--host-exec", choices=["auto", "stream", "cpu"],
                        default="auto",
                        help="comment sont calcules les poids en RAM : copies "
                             "vers le GPU, ou sur place par le processeur")
        sp.add_argument("--host-gb-s", type=float, default=70.0,
                        help="bande passante DDR mesuree ; acvram bench la rapporte")

    pl = sub.add_parser("plan", help="montre ou serait placee chaque couche")
    add_plan_args(pl)
    pl.add_argument("--kv-bits", type=int, default=8)
    pl.add_argument("--host-fraction", type=float, default=0.85)
    pl.add_argument("--no-host", action="store_true",
                    help="refuse d'utiliser la memoire vive comme etage")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_plan)

    cv = sub.add_parser("convert", help="quantifie un point de controle en fragments acvram")
    add_plan_args(cv)
    cv.add_argument("-o", "--out",
                    help="repertoire de sortie (defaut : "
                         "$ACVRAM_MODELS_DIR/<nom>, sinon models_acvram/<nom> "
                         "sur le SSD 2TO_2023_980PRO1, sinon sur le HDD "
                         "4TO_SATACMR_2022)")
    cv.add_argument("--no-awq", action="store_true",
                    help="simple arrondi au plus proche, sans mise a l'echelle AWQ")
    cv.add_argument("--hadamard", choices=["auto", "always", "never"],
                    default="auto")
    cv.add_argument("--grid", type=int, default=20,
                    help="finesse de la grille de recherche AWQ")
    cv.add_argument("--lm-head-format", help="format de la projection de sortie")
    cv.add_argument("--q3n-table", help=("niveaux q3n de ce modèle, huit "
                    "flottants séparés par des virgules (symétriques, bornes "
                    "±1) ; défaut : table de la spécification"))
    cv.add_argument("--dry-run", action="store_true",
                    help="rapporte tailles et erreurs sans ecrire de fragments")
    cv.add_argument("--force", action="store_true",
                    help="convertit meme si le modele ne tient pas")
    cv.add_argument("--calib-file", help="fichier texte de calibration "
                                         "(par defaut : un petit corpus integre)")
    cv.add_argument("--calib-seqs", type=int, default=16)
    cv.add_argument("--calib-len", type=int, default=512)
    cv.add_argument("--calib-device", default="cuda:0",
                    help="appareil sur lequel executer les passes de calibration")
    cv.add_argument("--promotion-cout-max", type=float, default=0.0, metavar="MIO",
                    help="prix plafond d'une promotion, en Mio ajoutes "
                         "(0 = aucun) : ecarte les gros tenseurs, dont la "
                         "promotion coute des octets relus a chaque jeton")
    cv.add_argument("--grille-erreurs", action="store_true",
                    help="conserve l'erreur des 21 valeurs de la grille AWQ "
                         "par tenseur, au manifeste : sert a calculer le prix "
                         "d'un exposant commun a un groupe empilable, sans "
                         "reconvertir")
    cv.add_argument("--bits-budget", type=float, default=0.0,
                    help="budget total de poids en Gio : les promotions sont "
                         "choisies par gain de SNR par octet (sac a dos), au "
                         "lieu du plancher SNR fixe")
    cv.add_argument("--quant-device", default="auto",
                    help="appareil de la recherche AWQ et de la quantification "
                         "(auto, cpu, cuda:0 ...)")
    cv.add_argument("--mixed-precision", choices=["auto", "off"], default="auto",
                    help="promeut vers un format plus large les tenseurs mal quantifies")
    cv.add_argument("--autoriser-grossissement", action="store_true",
                    help="autorise une conversion plus grosse que sa source "
                         "(refusée par défaut depuis le 8/09/2026)")
    cv.add_argument("--max-promotions", type=float, default=0.15,
                    metavar="PART",
                    help="part maximale de tenseurs promus (0,15 par defaut). "
                         "Le message de saturation conseillait de relever "
                         "cette option, qui n'existait pas : le plafond etait "
                         "atteint sur 27 modeles du parc sur 110, et au-dela "
                         "c'est l'ordre de parcours qui decide a la place du "
                         "SNR (5333 inversions mesurees sur Agents-A1-4B). "
                         "1.0 ne borne plus rien")
    cv.add_argument("--snr-floor", type=float, default=0.0,
                    help="SNR en sortie de couche (dB) sous lequel un tenseur est "
                         "promu ; 0 (defaut) ne promeut rien. Mesure sur un 27B : "
                         "25 dB coute 13,4 %% de memoire et 10,6 %% de debit pour "
                         "2,0 %% de perplexite")
    cv.add_argument("--mesurer-kld", action="store_true",
                    help="publie le KLD couche-par-couche (proxy softmax, "
                         "duck.ai 12/09) au manifeste, a cote du SNR. "
                         "N'AFFECTE AUCUNE DECISION : le convertisseur promeut "
                         "toujours sur le SNR. Sert au protocole A/B")
    cv.set_defaults(func=cmd_convert)

    sv = sub.add_parser("serve", help="lance le serveur compatible OpenAI")
    sv.add_argument("model", help="repertoire de modele converti")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--max-model-len", type=int, default=8192)
    sv.add_argument("--max-batch", type=int, default=16)
    sv.add_argument("--served-name", help="nom annonce par /v1/models")
    sv.add_argument("--device", help="force toutes les couches sur un seul appareil")
    sv.add_argument("--fp16", action="store_true",
                    help="calcule en float16 au lieu de bfloat16")
    sv.add_argument("--log-level", default="info")
    sv.add_argument("--speculative", choices=["none", "ngram", "draft", "mtp", "auto"],
                    default="ngram",
                    help="ngram ne coute rien et paie quand la sortie recopie "
                         "l'entree ; draft exige --draft-model")
    sv.add_argument("--draft-model", help="repertoire converti d'un petit modele "
                                          "charge de proposer des jetons")
    sv.add_argument("--draft-device", help="appareil du modele brouillon "
                                           "(par defaut : le GPU le plus oisif)")
    sv.add_argument("--spec-k", type=int, default=4,
                    help="jetons proposes par etape")
    sv.add_argument("--host-kv-gib", type=float, default=8.0,
                    help="etage hote du cache KV en Gio (0 = desactive) : les "
                         "prefixes evinces de la VRAM descendent en RAM et "
                         "remontent au reemploi au lieu d'etre recalcules")
    sv.add_argument("--no-cuda-graphs", action="store_true",
                    help="rejoue chaque pas de decodage en eager plutot qu'en "
                         "graphe CUDA capture")
    sv.add_argument("--no-prefix-cache", action="store_true",
                    help="desactive la reutilisation du KV entre requetes")
    sv.set_defaults(func=cmd_serve)

    ev = sub.add_parser("eval", help="perplexite d'un ou plusieurs modeles convertis")
    ev.add_argument("models", nargs="+", help="repertoires de modeles convertis")
    ev.add_argument("--corpus", help="fichier texte servant a l'evaluation")
    ev.add_argument("--window", type=int, default=512)
    ev.add_argument("--stride", type=int, default=256)
    ev.add_argument("--max-tokens", type=int, default=8192)
    # DEFAUT 0 CONSERVE, mais il ne passe plus en silence. Une perplexite a
    # min_context 0 n'est PAS comparable a une perplexite cadree : sur
    # wiki.test.raw la table du protocole donne 9,525 contre 7,233, un facteur
    # proche de 2. Le seul garde-fou existant n'avertit que si le corpus est
    # plus court que la fenetre — ce qui n'arrive jamais sur ce corpus de
    # 1,29 Mo. La barriere reposait donc sur la memoire de l'operateur.
    ev.add_argument("--min-context", type=int, default=0,
                    help="n'note que les positions ayant au moins tant de "
                         "jetons de contexte (0 = tout, NON COMPARABLE a une "
                         "mesure cadree ; le protocole impose 256)")
    ev.add_argument("--device", help="impose un appareil")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval)

    bn = sub.add_parser("bench", help="mesure noyaux, bande passante et debit")
    bn.add_argument("model", nargs="?", help="repertoire de modele converti")
    bn.add_argument("--topology-out",
                    help="ou ecrire la topologie mesuree "
                         "(defaut ~/.config/acvram/acvram-topology.json)")
    bn.add_argument("--what", default="all",
                    choices=["all", "kernels", "bandwidth", "topology", "decode"])
    bn.add_argument("--json", action="store_true")
    bn.set_defaults(func=cmd_bench)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrompu", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(red(f"introuvable : {exc}"), file=sys.stderr)
        return 2
    except Exception as exc:                          # noqa: BLE001
        print(red(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        if os.environ.get("ACVRAM_TRACEBACK"):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
