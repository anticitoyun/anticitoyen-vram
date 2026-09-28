"""
title: ComfyUI médias
description: Images (Flux.2 Klein 9B, DreamShaper 8, SD 1.5) et vidéo (Wan 2.2 TI2V-5B, Wan 2.2 14B I2V/T2V, Wan VACE, LTX-2.3) par ComfyUI, comme modèles du sélecteur.
version: 0.2.0
licence: MIT
"""
# Pipe « manifold » d'Open WebUI (pièce 53j-b, 27/09) : le réglage Images natif ne sert qu'UN workflow, et Klein
# (modèle de diffusion + encodeur Qwen3) ne se charge pas comme un checkpoint SD 1.5 — d'où un modèle du
# sélecteur par graphe. Le message de l'utilisateur est l'invite ; le fichier rendu par ComfyUI est rangé dans le
# stockage d'OWUI (lien /api/v1/files/<id>/content, survit à l'arrêt de ComfyUI) et affiché dans la réponse.
# Installé par parc/bin/openwebui-medias ; graphes testés à sec par tests/test_openwebui_medias.py.
import asyncio
import base64
import copy
import json
import random
import re
import subprocess
import time
import urllib.parse
import urllib.request
import uuid

from pydantic import BaseModel, Field

NEGATIF = "blurry, deformed, low quality, watermark, text"

MODELES = {
    "klein9b": {"nom": "Image · Flux.2 Klein 9B", "type": "klein",
                "fichier": "Flux.2 Klein 9B/flux-2-klein-9b-fp8.safetensors", "taille": (1024, 1024), "pas": 4},
    "dreamshaper8": {"nom": "Image · DreamShaper 8 (SD 1.5)", "type": "sd15",
                     "fichier": "SD 1.5/dreamshaper_8.safetensors", "taille": (512, 512), "pas": 25},
    "sd15": {"nom": "Image · Stable Diffusion 1.5", "type": "sd15",
             "fichier": "SD 1.5/v1-5-pruned-emaonly.ckpt", "taille": (512, 512), "pas": 25},
    "wan22": {"nom": "Vidéo · Wan 2.2 TI2V-5B", "type": "video",
              "fichier": "Wan Video 2.2 TI2V-5B/wanVideo22_ti2v5BFp16.safetensors", "taille": (832, 480), "pas": 20},
    # Pièce 296 : workflows testés fournis par poste6 (parc/share/openwebui/videos/<gabarit>.{api,noeuds}.json),
    # inlinés dans GABARITS par openwebui-medias. `entree` : pièces jointes exigées (image ; video+image : vidéo de
    # contrôle ET image du sujet, les deux obligatoires dans le graphe VACE) ; `multiple` : longueur = multiple·k + 1.
    # Pas ni négatif : ceux du workflow testé (LTX distillé : 8 sigmas fixes, sans pas).
    "wan22_i2v": {"nom": "Vidéo · Wan 2.2 14B image → vidéo", "type": "gabarit", "gabarit": "wan22-14b-i2v",
                  "usage": "Joignez une image et décrivez le mouvement : l'image ouvre la vidéo. Meilleure qualité, lent.",
                  "entree": "image", "taille": (832, 480), "fps": 16, "multiple": 4, "duree": 5.0},
    "wan22_t2v": {"nom": "Vidéo · Wan 2.2 14B texte → vidéo", "type": "gabarit", "gabarit": "wan22-14b-t2v",
                  "usage": "Décrivez la scène ; « 1280x720 » ou « 3 s » dans le message règlent taille et durée.",
                  "entree": None, "taille": (832, 480), "fps": 16, "multiple": 4, "duree": 5.0},
    "wan_vace": {"nom": "Vidéo · Wan VACE (contrôle par référence)", "type": "gabarit", "gabarit": "wan-vace",
                 "usage": "Joignez une vidéo de référence (mouvement, pose) ET une image du sujet : le sujet reprend le mouvement.",
                 "entree": "video+image", "taille": (832, 480), "fps": 16, "multiple": 4, "duree": 5.0},
    "ltx23": {"nom": "Vidéo · LTX-2.3 image → vidéo (rapide)", "type": "gabarit", "gabarit": "ltx23",
              "usage": "Joignez une image et décrivez la scène : rapide (secondes à chaud), avec son ; qualité sous Wan 14B.",
              "entree": "image", "taille": (1024, 576), "fps": 24, "multiple": 8, "duree": 5.0},
    "ltx23_t2v": {"nom": "Vidéo · LTX-2.3 texte → vidéo (rapide)", "type": "gabarit", "gabarit": "ltx23-t2v",
                  "usage": "Décrivez la scène : rapide (secondes à chaud), avec son ; qualité sous Wan 14B.",
                  "entree": None, "taille": (1024, 576), "fps": 24, "multiple": 8, "duree": 5.0},
}

# {gabarit: {"graphe": workflow API ComfyUI, "noeuds": [{"type", "node_ids", "key"}]}} — REMPLI À L'INSTALLATION par
# parc/bin/openwebui-medias (la Pipe vit dans la base d'OWUI, pas à côté des fichiers du parc).
GABARITS = {}

# « 1024x768 », « 3 s » ou « 3 secondes » dans l'invite règlent taille et durée ; le reste est l'invite.
_TAILLE = re.compile(r"\b(\d{3,4})\s*[x×]\s*(\d{3,4})\b")
_DUREE = re.compile(r"\b(\d{1,2}(?:[.,]\d)?)\s*(?:s|sec|secondes?)\b", re.I)


def lire_options(texte, defaut_taille, defaut_duree=2.0):
    """(invite, largeur, hauteur, durée) ; dimensions ramenées au multiple de 16 dans [256, 2048], durée [1, 10] s."""
    l, h = defaut_taille
    m = _TAILLE.search(texte)
    if m:
        l, h = int(m.group(1)), int(m.group(2))
        texte = texte[:m.start()] + texte[m.end():]
    duree = defaut_duree
    m = _DUREE.search(texte)
    if m:
        duree = float(m.group(1).replace(",", "."))
        texte = texte[:m.start()] + texte[m.end():]
    borne = lambda v: max(256, min(2048, v // 16 * 16))
    return " ".join(texte.split()), borne(l), borne(h), max(1.0, min(10.0, duree))


def graphe_klein(m, invite, l, h, graine):
    # Gabarit officiel « Flux.2 Klein distilled » (comfyui_workflow_templates, image_flux2_klein_text_to_image) en 9B :
    # 4 pas, CFG 1, négatif remis à zéro — c'est aussi parc/share/openwebui/flux2-klein-9b.api.json (réglage natif).
    return {
        "70": {"class_type": "UNETLoader", "inputs": {"unet_name": m["fichier"], "weight_dtype": "default"}},
        "71": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b.safetensors", "type": "flux2", "device": "default"}},
        "72": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "74": {"class_type": "CLIPTextEncode", "inputs": {"text": invite, "clip": ["71", 0]}},
        "76": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["74", 0]}},
        "63": {"class_type": "CFGGuider", "inputs": {"cfg": 1.0, "model": ["70", 0], "positive": ["74", 0], "negative": ["76", 0]}},
        "61": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "62": {"class_type": "Flux2Scheduler", "inputs": {"steps": m["pas"], "width": l, "height": h}},
        "73": {"class_type": "RandomNoise", "inputs": {"noise_seed": graine}},
        "66": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": l, "height": h, "batch_size": 1}},
        "64": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["73", 0], "guider": ["63", 0], "sampler": ["61", 0],
                                                                 "sigmas": ["62", 0], "latent_image": ["66", 0]}},
        "65": {"class_type": "VAEDecode", "inputs": {"samples": ["64", 0], "vae": ["72", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "openwebui/klein9b", "images": ["65", 0]}},
    }


def graphe_sd15(m, invite, l, h, graine):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m["fichier"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": invite, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIF, "clip": ["4", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": l, "height": h, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                                                   "latent_image": ["5", 0], "seed": graine, "steps": m["pas"], "cfg": 7.0,
                                                   "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "openwebui/sd15", "images": ["8", 0]}},
    }


def graphe_video(m, invite, l, h, graine, duree, fps=24):
    # Repris de ~/.local/bin/generer-media (graphe_video) : Wan 2.2 TI2V-5B, longueur = multiple de 4 plus 1.
    longueur = max(9, int(duree * fps) // 4 * 4 + 1)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["fichier"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": invite, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIF, "clip": ["2", 0]}},
        "6": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": l, "height": h,
                                                                  "length": longueur, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0],
                                                   "latent_image": ["6", 0], "seed": graine, "steps": m["pas"], "cfg": 5.0,
                                                   "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["9", 0], "frame_rate": fps, "loop_count": 0,
                                                            "filename_prefix": "openwebui/wan22", "format": "video/h264-mp4",
                                                            "pingpong": False, "save_output": True, "pix_fmt": "yuv420p",
                                                            "crf": 19, "save_metadata": True}},
    }


def longueur_images(duree, fps, multiple):
    """Nombre d'images multiple·k + 1 le plus proche par défaut de `duree` s à `fps` (au moins multiple + 1)."""
    return max(multiple + 1, int(duree * fps) // multiple * multiple + 1)


def appliquer(gabarit, valeurs):
    """Copie du workflow `gabarit["graphe"]` avec `valeurs` ({type: valeur}) posées aux nœuds de `gabarit["noeuds"]`.
    Un nœud ou une entrée absents du workflow sont une ERREUR nommée, jamais une valeur ignorée en silence."""
    g = copy.deepcopy(gabarit["graphe"])
    for n in gabarit["noeuds"]:
        if n["type"] not in valeurs:
            continue
        for nid in n["node_ids"]:
            if nid not in g or n["key"] not in g[nid].get("inputs", {}):
                raise KeyError(f"gabarit : {n['type']} → nœud {nid}.{n['key']} absent du workflow")
            g[nid]["inputs"][n["key"]] = valeurs[n["type"]]
    return g


def types_du_gabarit(gabarit):
    return {n["type"] for n in gabarit["noeuds"]}


def construire(cle, texte, graine, entrees=()):
    """(graphe, description) pour le modèle `cle` du manifold ; `entrees` = [(type, nom ComfyUI)] des pièces jointes
    téléversées ("image" ou "video")."""
    m = MODELES[cle]
    invite, l, h, duree = lire_options(texte, m["taille"], m.get("duree", 2.0))
    if m["type"] == "klein":
        g = graphe_klein(m, invite, l, h, graine)
    elif m["type"] == "sd15":
        g = graphe_sd15(m, invite, l, h, graine)
    elif m["type"] == "video":
        g = graphe_video(m, invite, l, h, graine, duree)
    else:
        gab = GABARITS.get(m["gabarit"])
        if gab is None:
            raise KeyError(f"gabarit {m['gabarit']} absent : réinstaller par openwebui-medias")
        valeurs = {"prompt": invite, "width": l, "height": h, "seed": graine, "fps": m["fps"],
                   "length": longueur_images(duree, m["fps"], m["multiple"])}
        for genre, nom in entrees:
            if genre not in types_du_gabarit(gab):
                raise KeyError(f"{m['nom']} ne prend pas de {genre} en entrée")
            valeurs[genre] = nom
        g = appliquer(gab, valeurs)
    video = m["type"] in ("video", "gabarit")
    pas = f" · {m['pas']} pas" if "pas" in m else ""
    desc = f"{m['nom']} · {l}×{h}{pas} · graine {graine}" + (f" · {duree:g} s" if video else "")
    return g, desc


_FICHIER_OWUI = re.compile(r"/api/v1/files/([0-9a-fA-F-]{8,})")


def references_jointes(messages, fichiers):
    """Pièces jointes du DERNIER message utilisateur : [{"mime", "data"} | {"mime", "id"}], images d'abord.
    OWUI met les images dans le contenu (image_url : data: base64 ou /api/v1/files/<id>/content) et les autres
    fichiers (vidéo) dans __files__ ({"id", "content_type"|"file.meta.content_type"})."""
    refs = []
    dernier = next((m for m in reversed(messages or []) if m.get("role") == "user"), {})
    contenu = dernier.get("content")
    for p in contenu if isinstance(contenu, list) else []:
        if p.get("type") != "image_url":
            continue
        url = (p.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            tete, _, donnees = url.partition(",")
            refs.append({"mime": tete[5:].split(";")[0] or "image/png", "data": base64.b64decode(donnees)})
        elif _FICHIER_OWUI.search(url):
            refs.append({"mime": "image/png", "id": _FICHIER_OWUI.search(url).group(1)})
    for f in fichiers or []:
        fid = f.get("id") or (f.get("file") or {}).get("id")
        mime = (f.get("content_type") or ((f.get("file") or {}).get("meta") or {}).get("content_type")
                or (f.get("meta") or {}).get("content_type") or "")
        if fid and mime.startswith(("video/", "image/")) and not any(r.get("id") == fid for r in refs):
            refs.append({"mime": mime, "id": fid})
    return refs


def choisir_entree(m, refs):
    """[(type, référence)] que le modèle `m` consomme parmi `refs` (la première de chaque type exigé) ; ValueError
    portant la ligne d'usage si une pièce exigée manque."""
    voulu = (m.get("entree") or "").split("+") if m.get("entree") else []
    choix = []
    for genre in voulu:
        r = next((r for r in refs if r["mime"].startswith(genre + "/")), None)
        if r is None:
            raise ValueError(m["usage"])
        choix.append((genre, r))
    return choix


def corps_multipart(nom, octets, mime):
    """(corps, content-type) d'un POST /upload/image de ComfyUI (vidéos comprises : même dossier input/)."""
    borne = uuid.uuid4().hex
    tete = (f'--{borne}\r\nContent-Disposition: form-data; name="image"; filename="{nom}"\r\n'
            f"Content-Type: {mime}\r\n\r\n").encode()
    champs = "".join(f'\r\n--{borne}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}'
                     for k, v in (("type", "input"), ("overwrite", "true")))
    return tete + octets + (champs + f"\r\n--{borne}--\r\n").encode(), f"multipart/form-data; boundary={borne}"


_MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif", ".webp": "image/webp", ".png": "image/png"}


def premier_fichier(sorties):
    """Premier fichier de sortie (type output) de l'historique ComfyUI : dict filename/subfolder/type, ou None."""
    for sortie in sorties.values():
        for cle in ("gifs", "videos", "video", "images"):
            for f in sortie.get(cle) or []:
                if f.get("type") == "output":
                    return f
    return None


class Pipe:
    class Valves(BaseModel):
        COMFYUI_URL: str = Field(default="http://127.0.0.1:8188")
        DEMARRER_COMFYUI: bool = Field(default=True, description="lancer « openwebui comfyui » si ComfyUI ne répond pas")
        DELAI_IMAGE_S: int = Field(default=600)
        DELAI_VIDEO_S: int = Field(default=1800)

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": k, "name": m["nom"]} for k, m in MODELES.items()]

    def _requete(self, chemin, donnees=None, timeout=30):
        req = urllib.request.Request(self.valves.COMFYUI_URL + chemin,
                                     data=json.dumps(donnees).encode() if donnees is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read() if chemin.startswith("/view") else json.loads(r.read() or b"{}")

    def _vivant(self):
        try:
            self._requete("/system_stats", timeout=3)
            return True
        except Exception:
            return False

    def _televerser(self, nom, octets, mime):
        corps, ct = corps_multipart(nom, octets, mime)
        req = urllib.request.Request(self.valves.COMFYUI_URL + "/upload/image", data=corps, headers={"Content-Type": ct})
        with urllib.request.urlopen(req, timeout=120) as r:
            rep = json.loads(r.read())
        return f"{rep['subfolder']}/{rep['name']}" if rep.get("subfolder") else rep["name"]

    async def _octets(self, ref, user):
        """Octets d'une référence jointe : inline (data:) ou fichier du stockage d'OWUI (droit de lecture vérifié)."""
        if "data" in ref:
            return ref["data"], ref["mime"]
        from open_webui.models.files import Files
        from open_webui.storage.provider import Storage
        f = await Files.get_file_by_id(ref["id"])
        if f is None or (f.user_id != user.id and user.role != "admin"):
            raise ValueError("pièce jointe illisible")
        chemin = await asyncio.to_thread(Storage.get_file, f.path)
        mime = ((f.meta or {}).get("content_type") or ref["mime"])
        with open(chemin, "rb") as h:
            return h.read(), mime

    async def pipe(self, body, __user__=None, __request__=None, __metadata__=None, __event_emitter__=None, __task__=None):
        # OWUI appelle aussi le modèle choisi pour ses tâches (titre, suggestions, tags) : jamais de génération là.
        if __task__:
            return "Génération ComfyUI"
        cle = body.get("model", "").split(".", 1)[-1]
        if cle not in MODELES:
            return f"Modèle inconnu : {cle}"
        texte = next((m["content"] for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
        if isinstance(texte, list):
            texte = " ".join(p.get("text", "") for p in texte if p.get("type") == "text")
        if not texte.strip():
            return "Décrivez l'image ou la vidéo à produire."

        async def etat(msg, fini=False):
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": msg, "done": fini}})

        if not self._vivant():
            if not self.valves.DEMARRER_COMFYUI:
                return "ComfyUI ne répond pas sur " + self.valves.COMFYUI_URL
            await etat("Démarrage de ComfyUI…")
            p = await asyncio.to_thread(subprocess.run, ["openwebui", "comfyui"], capture_output=True, text=True, timeout=180)
            if not self._vivant():
                return "ComfyUI n'a pas démarré : " + ((p.stdout + p.stderr).strip()[-400:] or f"code {p.returncode}")

        from open_webui.models.users import Users
        user = await Users.get_user_by_id(__user__["id"])
        m = MODELES[cle]
        try:
            choix = choisir_entree(m, references_jointes(body.get("messages"), (__metadata__ or {}).get("files")))
        except ValueError as e:
            return f"{m['nom']} : {e}"
        entrees = []
        for genre, ref in choix:
            octets, mime = await self._octets(ref, user)
            ext = {"video/webm": ".webm", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(
                mime, ".mp4" if genre == "video" else ".png")
            nom = await asyncio.to_thread(self._televerser, f"owui-{uuid.uuid4().hex[:12]}{ext}", octets, mime)
            entrees.append((genre, nom))
        graine = random.randint(0, 2**48)
        try:
            graphe, desc = construire(cle, texte, graine, entrees)
        except KeyError as e:
            return f"{m['nom']} : {e.args[0]}"
        video = m["type"] in ("video", "gabarit")
        await etat(desc + " — en file ComfyUI")
        t0 = time.time()
        envoi = await asyncio.to_thread(self._requete, "/prompt", {"prompt": graphe, "client_id": str(uuid.uuid4())})
        if "prompt_id" not in envoi:
            return "Refusé par ComfyUI : " + json.dumps(envoi, ensure_ascii=False)[:600]
        pid = envoi["prompt_id"]
        delai = self.valves.DELAI_VIDEO_S if video else self.valves.DELAI_IMAGE_S
        sorties = None
        while time.time() - t0 < delai:
            await asyncio.sleep(2)
            try:
                h = (await asyncio.to_thread(self._requete, "/history/" + pid)).get(pid)
            except Exception:
                continue
            if not h:
                continue
            st = h.get("status", {})
            if st.get("status_str") == "error":
                msgs = [x for x in st.get("messages", []) if x[0] == "execution_error"]
                return "Erreur ComfyUI : " + (msgs[-1][1].get("exception_message", "?")[:500] if msgs else "?")
            if h.get("outputs"):
                sorties = h["outputs"]
                break
            await etat(f"{desc} — {time.time() - t0:.0f} s")
        if sorties is None:
            return f"Délai dépassé ({delai} s) ; la génération continue peut-être dans ComfyUI (prompt {pid})."
        f = premier_fichier(sorties)
        if f is None:
            return "ComfyUI n'a rendu aucun fichier."
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"})
        donnees = await asyncio.to_thread(self._requete, "/view?" + q, None, 120)
        duree = time.time() - t0
        type_mime = _MIME.get(f["filename"][f["filename"].rfind("."):].lower(), "image/png")
        video = type_mime.startswith("video/")

        from open_webui.routers.images import upload_image
        meta = {k: (__metadata__ or {}).get(k) for k in ("chat_id", "message_id")}
        _, fichier = await upload_image(__request__, donnees, type_mime, {**meta, "prompt": texte, "graine": graine}, user)
        await etat(f"{desc} — {duree:.1f} s", fini=True)
        pied = f"\n\n*{desc} · {duree:.1f} s*"
        if video:
            if __event_emitter__:
                await __event_emitter__({"type": "embeds", "data": {"embeds": [
                    f'<video controls autoplay loop muted playsinline style="max-width:100%" src="{fichier["url"]}"></video>']}})
            return f"[Vidéo ({len(donnees) // 1024} Ko)]({fichier['url']})" + pied
        return f"![image]({fichier['url']})" + pied
