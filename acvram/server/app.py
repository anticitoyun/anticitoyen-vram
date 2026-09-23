"""Le serveur HTTP compatible avec l'API OpenAI.

Le moteur est monothread et synchrone ; l'API est asynchrone et concurrente. La
jonction entre les deux est un unique fil d'arrière-plan qui fait tourner la
boucle d'étapes du moteur et pousse les sorties dans des files asyncio propres à
chaque requête, via ``loop.call_soon_threadsafe``. Les requêtes ne touchent
jamais le modèle directement, et c'est ce qui permet à une douzaine de clients
en flux de partager un seul pipeline réparti sur deux GPU et la mémoire vive.
"""

from __future__ import annotations

import logging
import asyncio
import glob
import os
import sys
import json
import re
import signal
import subprocess
import threading
from pathlib import Path
import time
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from .. import __version__
from .. import regime_ligne as _regime_ligne
from . import capteurs as _capteurs
from .console import GALERIE, PAGE
from ..engine.runner import Engine, GenerationOutput
from ..engine.sampler import SamplingParams
from .chat import Tokenizer, render_chat
from .chat import extraire_appels, messages_pour_gabarit
from .chat import ProcesseurVision, charger_processeur_vision, preparer_images
from .protocol import (ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
                       ChatCompletionResponse, ChoiceMessage, ChunkChoice,
                       CompletionChoice, CompletionRequest, CompletionResponse,
                       DeltaMessage, EmbeddingData, EmbeddingRequest,
                       EmbeddingResponse, ErrorResponse, ModelCard, ModelList,
                       Usage, charger_image, new_id)

__all__ = ["create_app", "EngineService"]


class EngineService:
    """Anime le moteur depuis un fil d'arrière-plan et redistribue les résultats."""

    def __init__(self, engine: Engine, tokenizer: Optional[Tokenizer],
                 model_name: str) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._queues: dict[str, asyncio.Queue] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._loop = loop
            self._thread = threading.Thread(target=self._run,
                                            name="acvram-engine", daemon=True)
            self._thread.start()

    def ensure_started(self) -> None:
        """Démarre à la première requête si le crochet de cycle de vie n'a jamais
        été déclenché.

        Certains hôtes ASGI — et tout harnais de test qui instancie
        l'application sans entrer dans son cycle de vie — sautent les événements
        de démarrage. Un serveur dont le fil moteur ne démarre jamais accepte les
        requêtes puis reste bloqué indéfiniment, ce qui est une panne bien pire
        que de démarrer paresseusement un fil.
        """
        if self._thread is None:
            self.start(asyncio.get_running_loop())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        trace = bool(os.environ.get("ACVRAM_TRACE_STEPS"))
        durees: list[float] = []
        if trace:
            import gc
            debut = {}

            def _gc_cb(phase: str, info: dict) -> None:
                if phase == "start":
                    debut["t"] = time.perf_counter()
                else:
                    d = (time.perf_counter() - debut.get("t", time.perf_counter())) * 1000
                    if d > 5:
                        print(f"[gc] gen{info.get('generation')} {d:.1f} ms "
                              f"({info.get('collected')} objets)", flush=True)
            gc.callbacks.append(_gc_cb)
        while not self._stop.is_set():
            if self.engine.idle:
                time.sleep(0.002)
                continue
            try:
                t0 = time.perf_counter()
                outputs = self.engine.step()
                if trace:
                    durees.append((time.perf_counter() - t0) * 1000)
                    if len(durees) >= 100:
                        durees.sort()
                        print(f"[pas] med {durees[50]:.1f} p90 {durees[90]:.1f} "
                              f"max {durees[-1]:.1f} ms", flush=True)
                        durees.clear()
            except Exception as exc:                 # noqa: BLE001
                self._broadcast_error(exc)
                continue
            t1 = time.perf_counter()
            for out in outputs:
                self._deliver(out)
            if trace and (time.perf_counter() - t1) * 1000 > 2:
                print(f"[livraison] {(time.perf_counter() - t1)*1000:.1f} ms", flush=True)

    def _deliver(self, out: GenerationOutput) -> None:
        with self._lock:
            q = self._queues.get(out.request_id)
        if q is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(q.put_nowait, out)

    def _broadcast_error(self, exc: Exception) -> None:
        with self._lock:
            queues = list(self._queues.values())
        if self._loop is None:
            return
        for q in queues:
            self._loop.call_soon_threadsafe(q.put_nowait, exc)

    # -- request lifecycle -------------------------------------------------
    async def submit(self, prompt_ids: list[int], params: SamplingParams,
                     images: Optional[list] = None) -> tuple[str, asyncio.Queue]:
        self.ensure_started()
        request_id = new_id("req")
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._queues[request_id] = q
        if images:
            self.engine.add_request(prompt_ids, params, request_id, images=images)
        else:
            self.engine.add_request(prompt_ids, params, request_id)
        return request_id, q

    # -- images ------------------------------------------------------------
    def vision_servie(self) -> bool:
        """Le manifeste de l'alias servi porte ``vision: oui`` (pièce (a))."""
        man = getattr(getattr(self.engine, "loaded", None), "manifest", None) or {}
        v = man.get("vision")
        return str(v).strip().lower() in ("oui", "true", "1")

    def processeur_vision(self) -> ProcesseurVision:
        """Chargé au premier ``image_url``, jamais sur le chemin texte."""
        pv = getattr(self, "_processeur_vision", None)
        if pv is None:
            chemin = getattr(getattr(self.engine, "loaded", None), "path", "") or ""
            pv = charger_processeur_vision(chemin)
            self._processeur_vision = pv
        return pv

    def release(self, request_id: str) -> None:
        with self._lock:
            self._queues.pop(request_id, None)

    async def collect(self, request_id: str, q: asyncio.Queue
                      ) -> AsyncIterator[GenerationOutput]:
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            self.release(request_id)


_CHAMPS_SIGNALES: set[str] = set()


def signaler_champs_inconnus(req: Any, route: str) -> list[str]:
    """Journalise les champs de la requête que le serveur ne lit pas.
    WARNING à la première occurrence de chaque nom (une fois par session) ;
    DEBUG à chaque requête qui en contient — aucun champ n'est avalé sans trace.
    Un client qui envoie « temprature » ou « reasoning_effort » doit pouvoir
    le voir dans le journal du service, sinon il croit régler ce que le moteur
    sert au défaut."""
    inconnus = req.champs_inconnus()
    if not inconnus:
        return inconnus
    log = logging.getLogger("acvram.server")
    nouveaux = [c for c in inconnus if c not in _CHAMPS_SIGNALES]
    if nouveaux:
        _CHAMPS_SIGNALES.update(nouveaux)
        log.warning("%s : champs ignorés par ce serveur (sans effet sur la réponse) : %s",
                    route, ", ".join(nouveaux))
    else:
        log.debug("%s : champs ignorés (déjà signalés) : %s", route, ", ".join(inconnus))
    return inconnus


def _garde_contexte(engine, prompt_ids: list[int], detail: str = "", *,
                    params: Optional[Any] = None) -> None:
    """400 NOMMÉ pour une invite au-delà de ``max_model_len`` (jamais un 500 CUDA OOM) ; quand la chauffe a clampé
    le contexte, le message porte le demandé et le tenu (Sage sage-s2-k48-feu-vert-21-09 § 2 (c)).
    Quand ``params`` est fourni, borne ``params.max_tokens`` à ``max_model_len − len(prompt_ids)``
    plutôt que de refuser — comme llama.cpp."""
    if len(prompt_ids) >= engine.max_model_len:
        demande = getattr(engine, "ctx_demande", None)
        clamp = f" (demandé {demande}, tenu par la chauffe)" if demande not in (None, engine.max_model_len) else ""
        raise HTTPException(400, f"invite de {len(prompt_ids)} jetons{detail} au-delà de max_model_len "
                                 f"{engine.max_model_len}{clamp}")
    if params is not None:
        max_restant = engine.max_model_len - len(prompt_ids)
        if params.max_tokens > max_restant:
            params.max_tokens = max_restant


def _params_from(req: Any, default_max: int) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=req.min_p,
        repetition_penalty=req.repetition_penalty,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        max_tokens=req.token_budget(default_max),
        stop=req.stop_list(),
        seed=req.seed,
        n=req.n,
        ignore_eos=req.ignore_eos,
        logprobs=_n_logprobs(getattr(req, "logprobs", None)),
    )


def _n_logprobs(v: Any) -> Optional[int]:
    """Nombre de top-logprobs demandé (pièce 36). /v1/completions : entier ;
    /v1/chat : booléen (+ top_logprobs). None si non demandé. True → 0 (le
    logprob du jeton choisi, sans top-K)."""
    if v is None or v is False:
        return None
    if v is True:
        return 0
    return int(v)


_LOGPROBS_GRAPHE_AVERTI = False


def _avertir_logprobs_graphe() -> None:
    """Sous le défaut sampler=graphe, les top-K logprobs ne sont pas rapatriés
    (None). Le logprob du jeton choisi reste servi ; pour les top-K, relancer le
    serveur avec ACVRAM_SAMPLER_LENT=1. Averti une fois (ligne de régime)."""
    global _LOGPROBS_GRAPHE_AVERTI
    if not _LOGPROBS_GRAPHE_AVERTI:
        _LOGPROBS_GRAPHE_AVERTI = True
        print("acvram: top-K logprobs demandés sous sampler=graphe → indisponibles "
              "(top_logprobs=null) ; relancer avec ACVRAM_SAMPLER_LENT=1 pour les servir "
              "(logprobs=graphe-sans-topk)", file=sys.stderr)


def create_app(engine: Engine, tokenizer: Optional[Tokenizer],
               model_name: str, served_paths: Optional[dict] = None) -> FastAPI:
    service = EngineService(engine, tokenizer, model_name)
    app = FastAPI(title="acvram", version="0.1.0",
                  description="anticitoyen VRAM/RAM — inférence compatible OpenAI")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    app.state.service = service
    app.state.info = served_paths or {}
    app.state.anneau_energie = _capteurs.AnneauEnergie()   # ajout n°3, corrigé le 18/09

    @app.on_event("startup")
    async def _startup() -> None:
        service.start(asyncio.get_running_loop())
        app.state.anneau_energie.start(lambda: engine.stats.decode_tokens)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        service.stop()
        app.state.anneau_energie.stop()
        if hasattr(engine, "fermer"):
            engine.fermer()                               # rend l'horloge éco (-rgc)

    # -- console ----------------------------------------------------------
    # Le paquet s'installait sans rien de visible : ni entree de menu, ni
    # icone, ni page. Un serveur qui ne repond qu'a des clients OpenAI est
    # muet pour qui vient de l'installer et veut verifier qu'il tourne.
    # La console ne recalcule rien : elle affiche ce que /metrics rend.
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def console() -> str:
        return PAGE

    # -- introspection ----------------------------------------------------
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": model_name}

    # -- fonds d'ambiance --------------------------------------------------
    # Deux images d'agrement, servies par le serveur plutot que codees dans la
    # page. TROIS RAISONS, et la premiere est un defaut qu'on vient de corriger
    # ailleurs ce soir :
    #
    # 1. un chemin absolu dans un fichier suivi porte un nom d'utilisateur, et
    #    le crochet de construction du .deb refuse ce motif — le paquet ne se
    #    construirait plus ;
    # 2. une image en base64 dans la page ferait grossir une chaine que le
    #    module garde en memoire pour la vie du serveur ;
    # 3. la page doit rester utilisable quand aucune image n'est configuree :
    #    l'absence rend 404, le bouton se desactive, rien ne casse.
    # Les chemins viennent de l'environnement, puis d'un fichier de reglages
    # que la console peut ecrire. L'ordre compte : ce que l'utilisateur a
    # regle DANS l'interface doit survivre au redemarrage, sinon le reglage
    # n'en est pas un.
    _REGLAGES = Path.home() / ".config" / "acvram" / "console.json"

    def _charger_fonds() -> dict:
        d = {"zen": os.environ.get("ACVRAM_FOND_ZEN", ""),
             "cool": os.environ.get("ACVRAM_FOND_COOL", "")}
        try:
            enr = json.loads(_REGLAGES.read_text(encoding="utf-8"))
            for c in ("zen", "cool"):
                if enr.get(c):
                    d[c] = enr[c]
        except Exception:
            pass                      # absent ou illisible : les defauts suffisent
        return d

    _FONDS = _charger_fonds()

    # Extensions servables. LISTE BLANCHE et non liste noire : la console
    # ecrit un chemin choisi par qui la regarde, et une liste noire oublie
    # toujours le format ajoute apres elle. Refuser un .jpeg legitime coute
    # moins qu'ouvrir /etc/shadow parce qu'il n'etait pas dans la liste.
    _EXT_IMAGE = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".png": "image/png", ".webp": "image/webp",
                  ".gif": "image/gif", ".avif": "image/avif"}

    @app.get("/fond/{cle}", include_in_schema=False)
    async def fond(cle: str):
        chemin = _FONDS.get(cle, "")
        if not chemin or not os.path.isfile(chemin):
            raise HTTPException(status_code=404, detail="fond non configure")
        # Type devine par l'extension : servir une image en octet-stream la
        # ferait telecharger au lieu de l'afficher.
        mime = _EXT_IMAGE.get(os.path.splitext(chemin)[1].lower())
        if mime is None:
            raise HTTPException(status_code=415, detail="format non servi")
        return FileResponse(chemin, media_type=mime)

    @app.get("/fonds", include_in_schema=False)
    async def fonds() -> dict:
        """Quels fonds sont REELLEMENT servables — pas ceux qui sont declares.

        Une variable posee vers un fichier absent rendrait un bouton actif qui
        ne montre rien : la page saurait qu'il est configure sans savoir qu'il
        est introuvable.
        """
        return {c: bool(v and os.path.isfile(v)) for c, v in _FONDS.items()}

    # -- materiel ----------------------------------------------------------
    # DEUX SOURCES, ET ELLES NE VOIENT PAS LA MEME CHOSE.
    #
    #   nvidia-smi  ne connait QUE les cartes NVIDIA, mais donne leur etat :
    #               memoire, occupation, temperature, puissance, lien PCIe.
    #   lspci       voit TOUTES les cartes, y compris l'iGPU Intel, mais ne
    #               dit rien de leur charge.
    #
    # Une console qui n'interrogerait que la premiere laisserait croire que la
    # machine n'a que des cartes NVIDIA — l'iGPU disparaitrait de l'inventaire
    # alors qu'il porte l'affichage. On croise donc les deux, et on dit pour
    # chaque carte D'OU vient l'information.
    _CACHE_MAT: dict = {"t": 0.0, "v": None}

    def _lspci_gpu() -> list:
        try:
            sortie = subprocess.run(["lspci", "-mm"], capture_output=True,
                                    text=True, timeout=3).stdout
        except Exception:                                   # noqa: BLE001
            return []
        out = []
        for ligne in sortie.splitlines():
            bas = ligne.lower()
            if not any(k in bas for k in ('"vga compatible controller"',
                                          '"3d controller"', '"display controller"')):
                continue
            # `lspci -mm` rend : ADRESSE "classe" "fabricant" "modele" -rXX ...
            # Un split sur '" "' laisse la queue collee au modele (les
            # revisions et le sous-systeme). On coupe au premier guillemet
            # restant plutot que de decouper plus finement : le reste ne nous
            # sert pas, et un parseur ambitieux casserait sur la premiere
            # carte au nom inhabituel.
            champs = [c.strip('"') for c in ligne.split('" "')]
            if len(champs) >= 3:
                adr = champs[0].split()[0]
                modele = champs[3].split('"')[0].strip() if len(champs) > 3 else ""
                out.append({"adresse": adr, "fabricant": champs[2],
                            "modele": modele})
        return out

    def _nvidia_smi() -> list:
        champs = ("index,name,memory.total,memory.used,utilization.gpu,"
                  "temperature.gpu,power.draw,power.limit,"
                  "pcie.link.width.current,pci.bus_id")
        try:
            sortie = subprocess.run(
                ["nvidia-smi", f"--query-gpu={champs}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4).stdout
        except Exception:                                   # noqa: BLE001
            return []
        out = []
        for ligne in sortie.splitlines():
            c = [x.strip() for x in ligne.split(",")]
            if len(c) < 10:
                continue
            def f(v, d=0.0):
                try:
                    return float(v)
                except ValueError:
                    return d
            out.append({
                "index": int(f(c[0])), "nom": c[1],
                "mio_total": f(c[2]), "mio_pris": f(c[3]),
                "occupation": f(c[4]), "temperature": f(c[5]),
                "watts": f(c[6]), "watts_max": f(c[7]),
                "pcie_largeur": c[8],
                # Le bus_id de nvidia-smi est 00000000:01:00.0, lspci rend
                # 01:00.0 : la meme carte, un domaine PCI en plus.
                #
                # Premiere version : split(":",1)[-1].lstrip("0") — elle
                # rendait « 1:00.0 » parce que le lstrip mangeait AUSSI le
                # zero de « 01 ». Le croisement ne trouvait donc jamais, et
                # chaque carte NVIDIA apparaissait DEUX FOIS : une fois « non
                # pilotee » via lspci, une fois pilotee via nvidia-smi. Cinq
                # cartes annoncees pour trois reelles — un inventaire faux qui
                # avait l air complet.
                #
                # On prend les DEUX DERNIERS segments, qui sont bus:slot.fonc.
                "adresse": ":".join(c[9].split(":")[-2:]).lower(),
            })
        return out

    # -- capteurs -----------------------------------------------------------
    # Toutes les puces hwmon, les cartes NVIDIA avec leurs raisons de bridage,
    # et le systeme (psutil). Cache d'une seconde : hwmon est instantane mais
    # nvidia-smi coute ~30 ms, et l'interface rafraichit toutes les deux
    # secondes depuis plusieurs fenetres.
    _CACHE_CAPT: dict = {"t": 0.0, "v": None}

    @app.get("/capteurs", include_in_schema=False)
    async def capteurs() -> dict:
        from . import capteurs as _c
        maintenant = time.time()
        if _CACHE_CAPT["v"] is not None and maintenant - _CACHE_CAPT["t"] < 1.0:
            return _CACHE_CAPT["v"]
        rep = _c.relever()
        _CACHE_CAPT.update(t=maintenant, v=rep)
        return rep

    @app.get("/materiel", include_in_schema=False)
    async def materiel() -> dict:
        """Toutes les cartes de la machine, pas seulement celles qu'acvram sert.

        Rafraichi au plus une fois par seconde : `nvidia-smi` coute une
        trentaine de millisecondes, et la console rafraichit plus souvent que
        cela.
        """
        maintenant = time.time()
        if _CACHE_MAT["v"] is not None and maintenant - _CACHE_MAT["t"] < 1.0:
            return _CACHE_MAT["v"]
        nv = _nvidia_smi()
        pci = _lspci_gpu()
        par_adresse = {c["adresse"]: c for c in nv}
        cartes = []
        for p in pci:
            a = p["adresse"].lower()
            n = par_adresse.pop(a, None)
            nom = (n["nom"] if n else
                   (p["modele"] or p["fabricant"]).replace("Corporation", "").strip())
            cartes.append({
                "adresse": p["adresse"], "nom": nom,
                "fabricant": p["fabricant"],
                "pilotee": n is not None,
                # « pilotee » veut dire : nvidia-smi la voit, donc acvram peut
                # s'en servir. Une carte presente mais non pilotee n'est pas
                # une panne — l'iGPU porte l'affichage et c'est tout.
                **({k: n[k] for k in ("index", "mio_total", "mio_pris",
                                      "occupation", "temperature", "watts",
                                      "watts_max", "pcie_largeur")} if n else {}),
            })
        # Une carte vue par nvidia-smi et absente de lspci existe quand meme :
        # ne pas la perdre parce que le croisement a rate.
        for reste in par_adresse.values():
            cartes.append({"adresse": reste["adresse"], "nom": reste["nom"],
                           "fabricant": "NVIDIA", "pilotee": True, **reste})
        rep = {"cartes": cartes, "servie": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
        _CACHE_MAT.update(t=maintenant, v=rep)
        return rep

    # -- moteurs voisins ---------------------------------------------------
    # QUI D'AUTRE OCCUPE LA CARTE. Une machine de mesure porte souvent trois
    # ou quatre serveurs d'inference en meme temps, et un voisin qui calcule
    # pendant qu'on mesure fabrique le resultat. La console doit donc les
    # montrer — et permettre de les arreter, ce qui est une action
    # DESTRUCTRICE et se traite comme telle.
    #
    # TROIS GARDES, et la deuxieme est la seule qui compte vraiment :
    #   1. on n'arrete qu'un processus VU SUR LA CARTE. Un PID quelconque
    #      envoye a cette route ne sera pas tue.
    #   2. les services PERMANENTS sont proteges par defaut. 8081 (llama-server),
    #      8082 (embeddings), 8083 (reranker) tournent depuis des semaines et
    #      les tuer est une panne, pas un nettoyage.
    #   3. SIGTERM d'abord, jamais SIGKILL : un moteur qui meurt brutalement
    #      laisse de la VRAM et des fichiers de verrou derriere lui.
    _PORTS_PERMANENTS = {8081, 8082, 8083}

    _SIGNATURES = [
        ("llama.cpp", ("llama-server", "llama-cli", "llama-bench", "server.cpp")),
        ("tabbyAPI", ("tabbyapi", "tabby_api", "tabbyAPI")),
        ("vLLM", ("vllm.entrypoints", "vllm serve", "-m vllm")),
        ("SGLang", ("sglang.launch_server", "sglang")),
        ("Ollama", ("ollama",)),
        ("TGI", ("text-generation-server", "text_generation_server")),
        ("ExLlamaV2", ("exllamav2", "exllama")),
        ("acvram", ("acvram",)),
        ("PyTorch", ("torchrun", "-m torch")),
    ]

    def _ports_par_pid() -> dict:
        """Quels ports TCP ecoute chaque PID — sans quoi on ne peut pas savoir
        qu'un processus EST un des services permanents."""
        par_pid: dict[int, set] = {}
        try:
            sortie = subprocess.run(["ss", "-tlnp"], capture_output=True,
                                    text=True, timeout=3).stdout
        except Exception:                                   # noqa: BLE001
            return par_pid
        for ligne in sortie.splitlines()[1:]:
            m_port = re.search(r":(\d+)\s", ligne)
            for m_pid in re.finditer(r"pid=(\d+)", ligne):
                if m_port:
                    par_pid.setdefault(int(m_pid.group(1)), set()).add(int(m_port.group(1)))
        return par_pid

    # SAGE, 18/09 (sage-gui-ajouts-18-09 § 2, ajout n°1) : le verrou
    # outils/carte.sh est la seule verite sur l'etat du GPU (REGLES § 2),
    # mais rien avant cette route ne le rendait visible dans la console —
    # « une annonce a une minute de retard, une fenetre GUI non ». Six
    # manches perdues le 10/09, deux incidents rejoues le 14 et le 15 sur ce
    # meme defaut d'observabilite.
    def _verrous() -> list:
        """Un verrou par fichier `/tmp/acvram-carte-*.lock` trouve — le nom
        de fichier PORTE l'index de carte (carte.sh:41), donc pas besoin de
        connaitre les cartes a l'avance pour lister les verrous poses.
        Motif substituable (ACVRAM_VERROU_GLOB) : les tests ne doivent pas
        lire /tmp/acvram-carte-*.lock, partage par tout le circuit en
        service."""
        motif = os.environ.get("ACVRAM_VERROU_GLOB", "/tmp/acvram-carte-*.lock")
        out = []
        for verrou in sorted(glob.glob(motif)):
            m = re.search(r"acvram-carte-(\d+)\.lock$", verrou)
            carte = int(m.group(1)) if m else None
            info = verrou + ".qui"
            pid = nom = type_ = None
            depuis = None
            vivant = False
            try:
                champs = Path(info).read_text().split(None, 3)
                pid = int(champs[0])
                pris_a = int(champs[1])
                nom = champs[2] if len(champs) > 2 else "?"
                type_ = champs[3].strip() if len(champs) > 3 else "?"
                depuis = max(0, int(time.time() - pris_a))
                vivant = _pid_vivant(pid)
            except Exception:                                # noqa: BLE001
                pass                                          # .qui absent/perime : verrou sans detenteur lisible
            out.append({
                "carte": carte, "pid": pid if vivant else None,
                "nom": nom if vivant else None, "type": type_ if vivant else None,
                "depuis_secondes": depuis if vivant else None,
                "tenu": vivant,
            })
        return out

    def _pid_vivant(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # le PID existe mais appartient a un autre utilisateur — vivant
            # quand meme, kill(0) ne l'a pas dit "mort".
            return True
        except Exception:                                    # noqa: BLE001
            return False

    def _moteurs_gpu() -> list:
        try:
            sortie = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4).stdout
        except Exception:                                   # noqa: BLE001
            return []
        index_par_uuid = {}
        try:
            for l in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                                     "--format=csv,noheader"],
                                    capture_output=True, text=True,
                                    timeout=4).stdout.splitlines():
                i, u = [x.strip() for x in l.split(",", 1)]
                index_par_uuid[u] = int(i)
        except Exception:                                   # noqa: BLE001
            pass
        ports = _ports_par_pid()
        moi = os.getpid()
        pids_verrous = {v["pid"] for v in _verrous() if v["tenu"]}
        out = []
        for ligne in sortie.splitlines():
            c = [x.strip() for x in ligne.split(",")]
            if len(c) < 2:
                continue
            pid = int(c[0])
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", "replace").strip()
                debut = Path(f"/proc/{pid}").stat().st_ctime
            except Exception:                               # noqa: BLE001
                cmd, debut = "", 0.0
            bas = cmd.lower()
            nom = next((n for n, motifs in _SIGNATURES
                        if any(m.lower() in bas for m in motifs)), "inconnu")
            p = sorted(ports.get(pid, ()))
            permanent = any(x in _PORTS_PERMANENTS for x in p)
            out.append({
                "pid": pid, "mio": float(c[1]),
                "carte": index_par_uuid.get(c[2] if len(c) > 2 else "", None),
                "moteur": nom, "commande": cmd[:220], "ports": p,
                "permanent": permanent,
                # LEGITIME = service permanent OU detenteur du verrou de la
                # carte ou'il tourne (sage-gui-ajouts-18-09 § 2, n°1) — calcule
                # ici, pas dans la console (REGLES : la console montre ce que
                # le serveur rend, jamais un tri/calcul en JS).
                "legitime": permanent or pid in pids_verrous,
                "moi": pid == moi,
                "secondes": max(0, int(time.time() - debut)) if debut else None,
            })
        return out

    @app.get("/verrou", include_in_schema=False)
    async def verrou() -> dict:
        return {"verrous": _verrous()}

    @app.get("/moteurs", include_in_schema=False)
    async def moteurs() -> dict:
        return {"moteurs": _moteurs_gpu(),
                "ports_permanents": sorted(_PORTS_PERMANENTS)}

    @app.post("/moteurs/arreter", include_in_schema=False)
    async def arreter(req: Request) -> dict:
        corps = await req.json()
        pid = int(corps.get("pid", 0))
        force = bool(corps.get("force"))
        vus = {m["pid"]: m for m in _moteurs_gpu()}
        cible = vus.get(pid)
        if cible is None:
            # Ne jamais tuer un PID que la carte ne montre pas : la route
            # deviendrait un tueur de processus arbitraire.
            raise HTTPException(status_code=404,
                                detail="ce PID n'occupe aucune carte")
        if cible["moi"]:
            raise HTTPException(status_code=409,
                                detail="c'est ce serveur — arretez-le par son terminal")
        if cible["permanent"] and not force:
            raise HTTPException(
                status_code=409,
                detail=(f"service permanent (ports {cible['ports']}) : "
                        "confirmez explicitement pour l'arreter"))
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return {"pid": pid, "etat": "deja parti"}
        except PermissionError:
            raise HTTPException(status_code=403,
                                detail="pas le droit d'arreter ce processus")
        # On ne CONSTATE pas la mort ici : un moteur met plusieurs secondes a
        # liberer sa VRAM, et repondre « arrete » tout de suite serait une
        # affirmation non verifiee. La console relit /moteurs.
        return {"pid": pid, "etat": "SIGTERM envoye",
                "moteur": cible["moteur"], "mio": cible["mio"]}

    # -- galerie de photos -------------------------------------------------
    # Un DOSSIER entier, pas deux fichiers. 633 images ici, 585 Mio : la page
    # ne peut donc pas les porter, elle demande une liste et le serveur sert
    # chaque fichier a la demande.
    #
    # DEUX GARDES, et la premiere est la seule qui protege vraiment :
    #   1. on ne sert QUE des fichiers dont le chemin reel est SOUS le dossier
    #      declare. Sans ce test, « ../../etc/passwd » sortirait du dossier —
    #      et une console locale reste un serveur HTTP.
    #   2. on ne sert que les extensions de la liste blanche, comme les fonds.
    _GALERIE_DIR = os.environ.get("ACVRAM_GALERIE_DIR", "")

    def _photos() -> list:
        d = _GALERIE_DIR
        if not d or not os.path.isdir(d):
            return []
        racine = os.path.realpath(d)
        out = []
        try:
            for nom in sorted(os.listdir(racine)):
                if os.path.splitext(nom)[1].lower() not in _EXT_IMAGE:
                    continue
                chemin = os.path.realpath(os.path.join(racine, nom))
                # Le realpath resout les liens : un lien vers /etc sortirait
                # du dossier sans que le nom ne le montre.
                if not chemin.startswith(racine + os.sep):
                    continue
                if os.path.isfile(chemin):
                    out.append((nom, chemin, os.path.getsize(chemin)))
        except Exception:                                   # noqa: BLE001
            return []
        return out

    # -- parc de modeles ---------------------------------------------------
    # Quels modeles converti tiennent sur quelle carte. Lu dans les MANIFESTES,
    # jamais devine depuis les noms : un nom de dossier ne dit ni
    # l'architecture, ni les octets, ni les parametres actifs — et une
    # classification par nom nous a deja rendu un modele dense pour un MoE.
    # ACVRAM_PARC gagne, sinon ACVRAM_MODELES partage avec outils/racine_modeles.py,
    # sinon le defaut historique ~/Modeles/models_acvram.
    _PARC_DIR = os.environ.get("ACVRAM_PARC") or os.environ.get(
        "ACVRAM_MODELES", str(Path.home() / "Modeles" / "models_acvram"))
    _CACHE_PARC: dict = {"t": 0.0, "v": None}

    # Marge au-dessus des poids : contexte KV, activations, arene de
    # l'allocateur. Sans elle, un modele « qui tient a 11,9 sur 12 Gio »
    # tombe en OOM au premier lot — ce qui est arrive ce soir sur les
    # creneaux MLA.
    _MARGE = 1.25

    def _parc() -> list:
        d = _PARC_DIR
        if not os.path.isdir(d):
            return []
        out = []
        for nom in sorted(os.listdir(d)):
            man = os.path.join(d, nom, "acvram_manifest.json")
            if not os.path.isfile(man):
                continue
            try:
                j = json.loads(Path(man).read_text(encoding="utf-8"))
            except Exception:                               # noqa: BLE001
                continue
            m, plan = j.get("model", {}), j.get("plan", {})
            octets = plan.get("total_weight_bytes", 0) or 0
            moe = (m.get("num_experts", 0) or 0) > 0
            mla = (m.get("kv_lora_rank", 0) or 0) > 0
            gdn = (m.get("linear_num_value_heads", 0) or 0) > 0
            genres = [g for g, b in (("MoE", moe), ("MLA", mla), ("GDN", gdn)) if b]
            fmts = sorted({lp.get("fmt", "?") for lp in plan.get("layers", [])})
            out.append({
                "nom": nom, "chemin": os.path.join(d, nom),
                "gio": round(octets / 1024 ** 3, 2),
                "gio_requis": round(octets * _MARGE / 1024 ** 3, 2),
                "genres": genres or ["dense"],
                "formats": fmts,
                "params_total": round((m.get("total_params", 0) or 0) / 1e9, 2),
                "params_actifs": round((m.get("active_params", 0) or 0) / 1e9, 2),
                "couches": m.get("num_layers", 0),
                "contexte_max": m.get("max_position_embeddings", 0),
            })
        return out

    @app.get("/parc", include_in_schema=False)
    async def parc() -> dict:
        maintenant = time.time()
        if _CACHE_PARC["v"] is not None and maintenant - _CACHE_PARC["t"] < 30:
            return _CACHE_PARC["v"]
        modeles = _parc()
        cartes = [c for c in (await materiel())["cartes"] if c.get("pilotee")]
        for c in cartes:
            libre_gio = (c["mio_total"] - c["mio_pris"]) / 1024
            total_gio = c["mio_total"] / 1024
            c["compatibles"] = [
                m["nom"] for m in modeles if m["gio_requis"] <= total_gio]
            c["compatibles_maintenant"] = [
                m["nom"] for m in modeles if m["gio_requis"] <= libre_gio]
        rep = {"dossier": _PARC_DIR, "marge": _MARGE,
               "modeles": modeles, "cartes": cartes,
               "servi": {"nom": model_name,
                         "chemin": getattr(engine.loaded, "path", "")}}
        _CACHE_PARC.update(t=maintenant, v=rep)
        return rep

    @app.get("/photos", include_in_schema=False)
    async def photos() -> dict:
        p = _photos()
        return {"dossier": _GALERIE_DIR, "n": len(p),
                "photos": [{"nom": n, "octets": o} for n, _, o in p]}

    @app.get("/photo/{index}", include_in_schema=False)
    async def photo(index: int):
        p = _photos()
        if not (0 <= index < len(p)):
            raise HTTPException(status_code=404, detail="hors de la liste")
        nom, chemin, _ = p[index]
        # On sert par INDEX et non par nom : aucun nom de fichier fourni par
        # le client n'atteint le systeme de fichiers, donc aucune traversee
        # possible, meme si la garde ci-dessus etait un jour affaiblie.
        return FileResponse(chemin,
                            media_type=_EXT_IMAGE[os.path.splitext(nom)[1].lower()])

    @app.get("/galerie", response_class=HTMLResponse, include_in_schema=False)
    async def galerie() -> str:
        return GALERIE

    @app.get("/reglages", include_in_schema=False)
    async def lire_reglages() -> dict:
        return {"fonds": {c: {"chemin": v, "servable": bool(v and os.path.isfile(v))}
                          for c, v in _FONDS.items()},
                "fichier": str(_REGLAGES),
                "extensions": sorted(_EXT_IMAGE)}

    @app.post("/reglages", include_in_schema=False)
    async def ecrire_reglages(req: Request) -> dict:
        """Change les chemins d'ambiance depuis la console, et les garde.

        CE QUI EST REFUSE, ET POURQUOI CHAQUE REFUS EXISTE : un chemin qui
        n'est pas un fichier REGULIER (un tube nomme ou /dev/zero se lirait
        indefiniment), une extension hors liste blanche (le serveur ne doit
        pas devenir un lecteur de fichiers arbitraires), et un chemin
        introuvable (accepter en silence poserait un bouton actif qui ne
        montre rien — le defaut qu'on a corrige dans la page).
        """
        corps = await req.json()
        nouveaux, refus = {}, {}
        for cle in ("zen", "cool"):
            if cle not in corps:
                continue
            brut = str(corps[cle] or "").strip()
            if not brut:                       # vider est un reglage legitime
                nouveaux[cle] = ""
                continue
            chemin = os.path.realpath(os.path.expanduser(brut))
            if not os.path.isfile(chemin):
                refus[cle] = "fichier introuvable"
            elif os.path.splitext(chemin)[1].lower() not in _EXT_IMAGE:
                refus[cle] = ("extension non servie ; attendu "
                              + ", ".join(sorted(_EXT_IMAGE)))
            else:
                nouveaux[cle] = chemin
        _FONDS.update(nouveaux)
        try:
            _REGLAGES.parent.mkdir(parents=True, exist_ok=True)
            _REGLAGES.write_text(json.dumps(_FONDS, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            garde = True
        except Exception as e:                              # noqa: BLE001
            # Un reglage applique mais non garde doit se DIRE : sinon il
            # disparait au redemarrage et personne ne sait pourquoi.
            garde = False
            refus["_fichier"] = f"non ecrit : {e}"
        return {"fonds": {c: {"chemin": v, "servable": bool(v and os.path.isfile(v))}
                          for c, v in _FONDS.items()},
                "garde": garde, "refus": refus}

    @app.get("/repartition", include_in_schema=False)
    async def repartition() -> dict:
        """Le plan de placement, agrege par appareil.

        La console montre OU le travail se fait : quelles couches sur quel
        etage, dans quel format, et combien d octets y vivent. C est la
        question que le projet existe pour resoudre — donner a chaque GPU le
        format que son silicium sait lire — et elle n etait visible nulle part.
        """
        plan = engine.loaded.plan
        par_dev: dict[str, dict] = {}
        for lp in plan.layers:
            d = par_dev.setdefault(lp.exec_device, {
                "couches": 0, "formats": {}, "attn_octets": 0,
                "mlp_octets": 0, "mlp_hote": 0, "moe": 0})
            d["couches"] += 1
            d["formats"][lp.fmt] = d["formats"].get(lp.fmt, 0) + 1
            d["attn_octets"] += lp.attn_bytes if lp.attn_storage != "cpu" else 0
            d["mlp_octets"] += lp.mlp_bytes if lp.mlp_storage != "cpu" else 0
            d["mlp_hote"] += lp.mlp_bytes if lp.mlp_storage == "cpu" else 0
            d["moe"] += 1 if lp.is_moe else 0
        return {
            "modele": plan.model,
            "appareils": par_dev,
            "embed": plan.embed_device,
            "lm_head": plan.lm_head_device,
            "kv_octets_par_jeton": plan.kv_bytes_per_token,
            "kv_jetons_max": plan.kv_max_tokens,
            "etages": [{"nom": t.name, "octets": plan.kv_budget.get(t.name, 0)}
                       for t in plan.tiers],
            "estimation_decode_j_s": round(plan.est_decode_tok_s, 1),
            "estimation_prefill_j_s": round(plan.est_prefill_tok_s, 1),
            "avertissements": list(plan.warnings),
        }

    def _energie_par_jeton() -> dict:
        """J/jeton en direct (ajout GUI n°3, sage-gui-ajouts-18-09 § 3),
        corrigé le 18/09 (sage-metrics-energie-fenetre-18-09) : lu depuis
        `app.state.anneau_energie`, un anneau d'échantillons alimenté par
        un fil de fond — jamais consommé ici, pour que deux lecteurs
        concurrents de `/metrics` rendent la MÊME valeur sur la même
        fenêtre (l'ancienne version, qui consommait son propre échantillon
        précédent à chaque appel, les faisait diverger)."""
        cartes = _capteurs.nvidia()
        anneau = app.state.anneau_energie
        return {
            "j_par_jeton_10s": anneau.j_par_jeton(),
            "fenetre_s": anneau.fenetre_s,
            "jetons_fenetre": anneau.jetons_fenetre(),
            "cartes": [{"index": c["index"], "horloge_sm": c["horloge_sm"],
                        "watts_plafond": c["watts_max"]} for c in cartes],
        }

    @app.get("/metrics")
    async def metrics() -> dict:
        # `version` sert a la console, qui l'affiche en tete : sans elle on ne
        # sait pas quelle version repond, et deux versions ont deja coexiste
        # sur cette machine (paquet 0.5.0, venv 0.2.0).
        plan = engine.loaded.plan
        # Ajout n°2 (sage-gui-ajouts-18-09) : `sequences_tronquees_budget`
        # (EngineStats) est deja compte, jamais affiche — « toutes les
        # cellules b > 8 faussees jusqu'au 17/09 sans une ligne d'erreur ».
        # `kv_max_tokens / kv_planned_seqs` donne le budget REEL par sequence
        # planifiee (slots x contexte) : sans lui, un compteur > 0 ne dit pas
        # si le budget est structurellement sous-dimensionne ou accidentel.
        kv_seqs = getattr(plan, "kv_planned_seqs", 0) or 0
        return {"engine": engine.stats.to_dict(),
                "kv_max_tokens": plan.kv_max_tokens,
                "kv_planned_seqs": kv_seqs,
                "kv_tokens_par_sequence_planifiee": (
                    round(plan.kv_max_tokens / kv_seqs, 1) if kv_seqs else None),
                # sage-profil-verdict-18-09 §4 : sans elle, un client HTTP de
                # mesure (hors carte.sh) ne peut pas savoir quelle carte est
                # RÉELLEMENT servie et somme celles qu'il voit lui-même —
                # 18/09, acvram [0,1] contre llama.cpp [0], énergie faussée
                # par le repos de la carte inutilisée.
                "cartes": engine.regime()["cartes"],
                "repli_eager": engine.regime().get("repli_eager", 0),
                "replis_eager_raisons": engine.regime().get("replis_eager_raisons", []),
                # pièce 90 : le nombre de clés de graphe refusées et leur raison
                # principale, structurés — pas seulement noyés dans regime_ligne
                "graphes": engine.regime().get("graphes"),
                "graphes_refus_n": engine.regime().get("graphes_refus_n", 0),
                "graphes_refus_principale": engine.regime().get("graphes_refus_principale"),
                "energie": _energie_par_jeton(),
                # Ajout n°4 (sage-gui-ajouts-18-09 § 4) : la meme ligne,
                # octet pour octet, que le "[regime]" ecrit dans le JSON
                # d'une mesure (`acvram.regime_ligne`) — variables ACVRAM_*
                # hors defaut + versions torch/triton/fla.
                "regime_ligne": _regime_ligne(),
                # pièce 49 : régime spéculatif visible dans /metrics (même source que
                # regime_ligne — mode + état garde + gain moyen glissant)
                "speculation": engine.regime().get("speculation"),
                "version": __version__, **app.state.info}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        plan = engine.loaded.plan
        return ModelList(data=[ModelCard(
            id=model_name,
            acvram={
                "formats": sorted({lp.fmt for lp in plan.layers}),
                "devices": sorted({lp.exec_device for lp in plan.layers}),
                "max_model_len": engine.max_model_len,
                "kv_tokens": plan.kv_max_tokens,
                "chat_template": tokenizer.template_source if tokenizer else None,
                "regime": engine.regime(),
            })])

    # -- chat -------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw: Request):
        signaler_champs_inconnus(req, "/v1/chat/completions")
        # Trois états pour une image, jamais un silence : acceptée (alias
        # vision), refus nommé (source, dépassement), alias sans vision.
        urls = [u for m in req.messages for u in m.images()]
        if urls and not service.vision_servie():
            raise HTTPException(400, f"modèle sans tour de vision : l'alias « {model_name} » "
                                     "ne porte pas vision: oui dans son manifeste, "
                                     f"{len(urls)} image_url refusée(s)")
        messages = messages_pour_gabarit(req.messages, avec_images=bool(urls))
        extra = dict(req.chat_template_kwargs or {})
        if req.tools:
            extra["tools"] = req.tools          # les gabarits HF les rendent eux-mêmes
        prompt = render_chat(tokenizer, messages, req.add_generation_prompt, extra)
        params = _params_from(req, 512)
        images = None
        if urls:
            octets = [charger_image(u) for u in urls]           # ImageRefusee -> 400
            prompt_ids, images = preparer_images(service.processeur_vision(), prompt, octets)
            n_img = sum(f.n_jetons for f in images)
            _garde_contexte(engine, prompt_ids, f" dont {n_img} jetons image ({len(images)} image(s))", params=params)
        else:
            prompt_ids = _encode(tokenizer, prompt)
            _garde_contexte(engine, prompt_ids, params=params)
        request_id, q = await service.submit(prompt_ids, params, images)

        if req.stream:
            return StreamingResponse(
                _stream_chat(service, request_id, q, model_name, len(prompt_ids),
                             bool((req.stream_options or {}).get("include_usage")),
                             bool(req.tools)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        text, reason, n_out = "", "stop", 0
        async for out in service.collect(request_id, q):
            text += out.text_delta
            n_out = out.completion_tokens
            if out.finished:
                reason = out.finish_reason or "stop"
        return ChatCompletionResponse(
            model=model_name,
            choices=[ChatChoice(message=_message_finale(text, bool(req.tools)),
                                finish_reason=_raison_finale(text, reason, bool(req.tools)))],
            usage=Usage(prompt_tokens=len(prompt_ids), completion_tokens=n_out,
                        total_tokens=len(prompt_ids) + n_out))


    # -- API Anthropic (/v1/messages) --------------------------------------
    # Claude Code parle cette API-là, pas celle d'OpenAI : l'exposer permet
    # aux menus locaux de brancher Claude directement sur ce serveur. Champs
    # couverts : system, messages (texte ou blocs), stop_sequences,
    # temperature/top_p/top_k, stream (evenements message_start,
    # content_block_delta, message_delta, message_stop).
    @app.post("/v1/messages")
    async def anthropic_messages(raw: Request):
        req = await raw.json()
        messages = []
        sys_prompt = req.get("system")
        if sys_prompt:
            if isinstance(sys_prompt, list):
                sys_prompt = "".join(b.get("text", "") for b in sys_prompt)
            messages.append({"role": "system", "content": sys_prompt})
        for m in req.get("messages", []):
            contenu = m.get("content", "")
            if isinstance(contenu, list):
                contenu = "".join(b.get("text", "") for b in contenu
                                  if isinstance(b, dict)
                                  and b.get("type") == "text")
            messages.append({"role": m.get("role", "user"),
                             "content": contenu})
        prompt = render_chat(tokenizer, messages, True)
        prompt_ids = _encode(tokenizer, prompt)
        params = SamplingParams(
            temperature=float(req.get("temperature", 1.0)),
            top_p=float(req.get("top_p", 1.0)),
            top_k=int(req.get("top_k", 0) or 0),
            max_tokens=int(req.get("max_tokens", 512)),
            stop=list(req.get("stop_sequences") or []),
        )
        _garde_contexte(engine, prompt_ids, params=params)
        request_id, q = await service.submit(prompt_ids, params)
        mid = new_id("msg")

        def stop_reason(r: str) -> str:
            return {"stop": "end_turn", "length": "max_tokens"}.get(r, "end_turn")

        if req.get("stream"):
            async def flux():
                def ev(nom: str, data: dict) -> str:
                    return (f"event: {nom}\n"
                            f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
                yield ev("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant",
                    "content": [], "model": model_name, "stop_reason": None,
                    "usage": {"input_tokens": len(prompt_ids),
                              "output_tokens": 0}}})
                yield ev("content_block_start",
                         {"type": "content_block_start", "index": 0,
                          "content_block": {"type": "text", "text": ""}})
                n_out, raison = 0, "end_turn"
                async for out in service.collect(request_id, q):
                    n_out = out.completion_tokens
                    if out.text_delta:
                        yield ev("content_block_delta",
                                 {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta",
                                            "text": out.text_delta}})
                    if out.finished:
                        raison = stop_reason(out.finish_reason or "stop")
                yield ev("content_block_stop",
                         {"type": "content_block_stop", "index": 0})
                yield ev("message_delta", {"type": "message_delta",
                         "delta": {"stop_reason": raison,
                                   "stop_sequence": None},
                         "usage": {"output_tokens": n_out}})
                yield ev("message_stop", {"type": "message_stop"})
            return StreamingResponse(
                flux(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache",
                         "X-Accel-Buffering": "no"})

        text, raison, n_out = "", "end_turn", 0
        async for out in service.collect(request_id, q):
            text += out.text_delta
            n_out = out.completion_tokens
            if out.finished:
                raison = stop_reason(out.finish_reason or "stop")
        return {"id": mid, "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": model_name, "stop_reason": raison,
                "stop_sequence": None,
                "usage": {"input_tokens": len(prompt_ids),
                          "output_tokens": n_out}}

    # -- legacy completions ------------------------------------------------
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        signaler_champs_inconnus(req, "/v1/completions")
        prompt = req.prompt
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt_ids = list(prompt)
            prompt_text = ""
        else:
            prompt_text = prompt[0] if isinstance(prompt, list) else prompt
            prompt_ids = _encode(tokenizer, str(prompt_text), brut=True)
        params = _params_from(req, 256)
        _garde_contexte(engine, prompt_ids, params=params)
        request_id, q = await service.submit(prompt_ids, params)

        if req.stream:
            return StreamingResponse(
                _stream_completion(service, request_id, q, model_name,
                                   len(prompt_ids),
                                   bool((req.stream_options or {}).get("include_usage"))),
                media_type="text/event-stream")

        nlp = _n_logprobs(getattr(req, "logprobs", None))
        text, reason, n_out = "", "stop", 0
        # Pièce 36 : logprobs par pas de décodage, remplis SEULEMENT si demandés
        # (sinon la sortie est identique au bit — cf. sampler `veut_logprobs`).
        lp_tok: list[str] = []; lp_val: list = []; lp_top: list = []; lp_off: list = []
        async for out in service.collect(request_id, q):
            if nlp is not None and out.text_delta:
                lp_off.append(len(text)); lp_tok.append(out.text_delta)
                lp_val.append(getattr(out, "logprob", None))
                top = getattr(out, "top_logprobs", None)
                lp_top.append({tokenizer.decode([i]): float(v) for i, v in top} if top else None)
            text += out.text_delta
            n_out = out.completion_tokens
            if out.finished:
                reason = out.finish_reason or "stop"

        logprobs_champ = None
        if nlp is not None:
            # decode → offsets et jetons de génération sont posés au fil de la boucle,
            # sur le texte de génération. echo : préfixer par les logprobs de l'invite.
            toks, vals, tops, offs = lp_tok, lp_val, lp_top, lp_off
            if req.echo and prompt_ids:
                inv = engine.logprobs_invite(prompt_ids, top_k=nlp)
                dt = [tokenizer.decode([i]) for i in inv["ids"]]
                off_inv, cur = [], 0
                for t in dt:
                    off_inv.append(cur); cur += len(t)
                toks = dt + [t for t in lp_tok]
                vals = list(inv["logprobs"]) + lp_val
                itops = ([({tokenizer.decode([i]): float(v) for i, v in pos} if pos else None)
                          for pos in inv["top"]]
                         if inv.get("top") else [None] * len(dt))
                tops = itops + lp_top
                base = len(str(prompt_text)) if prompt_text else cur
                offs = off_inv + [base + o for o in lp_off]
            logprobs_champ = {"tokens": toks, "token_logprobs": vals,
                              "top_logprobs": tops, "text_offset": offs}
            if nlp > 0 and any(t is None for t in lp_top):
                _avertir_logprobs_graphe()

        if req.echo:
            text = str(prompt_text) + text
        return CompletionResponse(
            model=model_name,
            choices=[CompletionChoice(text=text, finish_reason=reason, logprobs=logprobs_champ)],
            usage=Usage(prompt_tokens=len(prompt_ids), completion_tokens=n_out,
                        total_tokens=len(prompt_ids) + n_out))

    # -- embeddings --------------------------------------------------------
    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest):
        import torch
        from ..engine.model import ForwardBatch
        from ..memory.kvcache import BLOCK_SIZE

        inputs = req.input if isinstance(req.input, list) else [req.input]
        if inputs and isinstance(inputs[0], int):
            inputs = [inputs]

        data, total = [], 0
        for i, item in enumerate(inputs):
            ids = item if isinstance(item, list) else _encode(tokenizer, str(item), brut=True)
            ids = ids[: engine.max_model_len]
            if not ids:
                raise HTTPException(400, "entrée vide")
            total += len(ids)
            vec = await asyncio.to_thread(_embed_once, engine, ids)
            if req.dimensions:
                vec = vec[: req.dimensions]
            data.append(EmbeddingData(index=i, embedding=vec))
        return EmbeddingResponse(data=data, model=model_name,
                                 usage=Usage(prompt_tokens=total,
                                             total_tokens=total))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        return JSONResponse(status_code=400,
                            content=ErrorResponse.make(f"requête invalide — {detail}").model_dump())

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400,
                            content=ErrorResponse.make(str(exc)).model_dump())

    @app.exception_handler(HTTPException)
    async def _http_error(req: Request, exc: HTTPException) -> JSONResponse:
        # /v1/messages suit le format d'erreur Anthropic, les autres routes OpenAI
        if req.url.path.startswith("/v1/messages"):
            body = {"type": "error",
                    "error": {"type": "invalid_request_error",
                               "message": str(exc.detail)}}
        else:
            body = ErrorResponse.make(str(exc.detail)).model_dump()
        return JSONResponse(status_code=exc.status_code, content=body)

    return app


def _encode(tokenizer: Optional[Tokenizer], text: str, brut: bool = False) -> list[int]:
    """``brut`` : invite de complétion brute (pas de gabarit rendu) — le
    tokeniseur pose alors ses jetons spéciaux et le préfixe du gabarit
    (``[gMASK]<sop>`` sur GLM) ; le chemin conversation les a déjà."""
    if tokenizer is None:
        raise HTTPException(500, "aucun tokeniseur trouvé dans le répertoire du modèle")
    ids = tokenizer.encode_brut(text) if brut else tokenizer.encode(text)
    if not ids:
        raise HTTPException(400, "l'invite s'est encodée en zéro jeton")
    return ids


def _embed_once(engine: Engine, ids: list[int]) -> list[float]:
    """Moyenne les états cachés finaux, puis normalise en L2.

    S'exécute hors de la boucle de lot : une requête de plongement est un simple
    prefill sans KV à conserver, elle n'a donc pas besoin d'un emplacement de
    cache et ne doit pas en retirer un à une génération qui, elle, en a besoin.
    """
    import torch
    from ..engine.model import ForwardBatch
    from ..memory.kvcache import BLOCK_SIZE

    n = len(ids)
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks = engine.allocator.allocate(n_blocks)
    try:
        slots = torch.tensor(
            [blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
             for i in range(n)], dtype=torch.long)
        batch = ForwardBatch(
            tokens=torch.tensor(ids, dtype=torch.long),
            positions=torch.arange(n, dtype=torch.long),
            seq_lens=[n], query_lens=[n],
            block_tables=[torch.tensor(blocks, dtype=torch.long)],
            slot_mapping=slots, is_prefill=True)
        hidden = engine.model(batch, return_hidden=True)
        pooled = hidden.to(torch.float32).mean(dim=0)
        pooled = pooled / pooled.norm().clamp(min=1e-12)
        return pooled.tolist()
    finally:
        engine.allocator.free(blocks)


def _message_finale(text: str, outils: bool) -> ChoiceMessage:
    if outils:
        reste, appels = extraire_appels(text)
        if appels:
            return ChoiceMessage(content=reste or None, tool_calls=appels)
    return ChoiceMessage(content=text)


def _raison_finale(text: str, reason: str, outils: bool) -> str:
    return "tool_calls" if outils and extraire_appels(text)[1] else reason


async def _stream_chat(service: EngineService, request_id: str,
                       q: asyncio.Queue, model: str, prompt_tokens: int,
                       include_usage: bool, outils: bool = False) -> AsyncIterator[str]:
    cid = new_id("chatcmpl")
    first = ChatCompletionChunk(
        id=cid, model=model,
        choices=[ChunkChoice(delta=DeltaMessage(role="assistant", content=""))])
    yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"

    n_out = 0
    # Avec des outils, tout ce qui suit un « <tool_call> » est retenu et rendu
    # à la fin en appels structurés ; un « < » isolé attend un peu, le temps
    # de savoir s'il ouvre la balise.
    total, pend, retenu = "", "", False
    premier_jeton = None
    try:
        async for out in service.collect(request_id, q):
            n_out = out.completion_tokens
            if out.text_delta and premier_jeton is None:
                # Journalisé pour les mesures d'énergie et de bus : la fenêtre
                # d'un instrument extérieur se borne sur ces deux instants,
                # sans messagerie ni horloge partagée (protocole du 8/09).
                import datetime as _dt
                premier_jeton = _dt.datetime.now()
                print(f"[mesure] {cid} premier jeton "
                      f"{premier_jeton.strftime('%H:%M:%S.%f')[:-3]}", flush=True)
            if out.text_delta:
                total += out.text_delta
                if not outils:
                    delta = out.text_delta
                elif retenu:
                    delta = ""
                else:
                    pend += out.text_delta
                    if "<tool_call>" in pend:
                        retenu = True
                        delta, pend = pend[:pend.index("<tool_call>")], ""
                    elif "<" in pend and len(pend) < 64:
                        delta = ""
                    else:
                        delta, pend = pend, ""
                if delta:
                    chunk = ChatCompletionChunk(
                        id=cid, model=model,
                        choices=[ChunkChoice(delta=DeltaMessage(content=delta))])
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            if out.finished:
                raison = out.finish_reason or "stop"
                if outils:
                    reste, appels = extraire_appels(total)
                    if appels:
                        chunk = ChatCompletionChunk(
                            id=cid, model=model,
                            choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[
                                {"index": i, **a} for i, a in enumerate(appels)]))])
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        raison = "tool_calls"
                    elif pend:
                        chunk = ChatCompletionChunk(
                            id=cid, model=model,
                            choices=[ChunkChoice(delta=DeltaMessage(content=pend))])
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                done = ChatCompletionChunk(
                    id=cid, model=model,
                    choices=[ChunkChoice(delta=DeltaMessage(), finish_reason=raison)])
                if include_usage:
                    done.usage = Usage(prompt_tokens=prompt_tokens,
                                       completion_tokens=n_out,
                                       total_tokens=prompt_tokens + n_out)
                yield f"data: {done.model_dump_json(exclude_none=True)}\n\n"
    except Exception as exc:                          # noqa: BLE001
        # Le message seul ne dit pas d'où vient une erreur CUDA : sans la
        # pile, quatre séries de banc ont été perdues le 7/09 à deviner.
        import traceback
        traceback.print_exc()
        err = ErrorResponse.make(str(exc), "server_error")
        yield f"data: {json.dumps(err.model_dump())}\n\n"
    if premier_jeton is not None:
        import datetime as _dt
        print(f"[mesure] {cid} dernier jeton "
              f"{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
              f"({n_out} jetons)", flush=True)
    yield "data: [DONE]\n\n"


async def _stream_completion(service: EngineService, request_id: str,
                             q: asyncio.Queue, model: str, prompt_tokens: int = 0,
                             include_usage: bool = False) -> AsyncIterator[str]:
    cid = new_id("cmpl")
    try:
        async for out in service.collect(request_id, q):
            resp = CompletionResponse(
                id=cid, model=model,
                choices=[CompletionChoice(
                    text=out.text_delta,
                    finish_reason=out.finish_reason if out.finished else None)])
            if out.finished and include_usage:
                # même contrat que le flux de chat : l'usage sur le dernier morceau
                resp.usage = Usage(prompt_tokens=prompt_tokens,
                                   completion_tokens=out.completion_tokens,
                                   total_tokens=prompt_tokens + out.completion_tokens)
            yield f"data: {resp.model_dump_json(exclude_none=True)}\n\n"
    except Exception as exc:                          # noqa: BLE001
        # Le message seul ne dit pas d'où vient une erreur CUDA : sans la
        # pile, quatre séries de banc ont été perdues le 7/09 à deviner.
        import traceback
        traceback.print_exc()
        err = ErrorResponse.make(str(exc), "server_error")
        yield f"data: {json.dumps(err.model_dump())}\n\n"
    yield "data: [DONE]\n\n"
