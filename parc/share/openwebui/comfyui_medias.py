"""
title: ComfyUI médias
description: Images (Flux.2 Klein 9B, DreamShaper 8, SD 1.5) et vidéo (Wan 2.2 TI2V-5B) par ComfyUI, comme modèles du sélecteur.
version: 0.1.0
licence: MIT
"""
# Pipe « manifold » d'Open WebUI (pièce 53j-b, 27/09) : le réglage Images natif ne sert qu'UN workflow, et Klein
# (modèle de diffusion + encodeur Qwen3) ne se charge pas comme un checkpoint SD 1.5 — d'où un modèle du
# sélecteur par graphe. Le message de l'utilisateur est l'invite ; le fichier rendu par ComfyUI est rangé dans le
# stockage d'OWUI (lien /api/v1/files/<id>/content, survit à l'arrêt de ComfyUI) et affiché dans la réponse.
# Installé par parc/bin/openwebui-medias ; graphes testés à sec par tests/test_openwebui_medias.py.
import asyncio
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
}

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


def construire(cle, texte, graine):
    """(graphe, description) pour le modèle `cle` du manifold."""
    m = MODELES[cle]
    invite, l, h, duree = lire_options(texte, m["taille"])
    if m["type"] == "klein":
        g = graphe_klein(m, invite, l, h, graine)
    elif m["type"] == "sd15":
        g = graphe_sd15(m, invite, l, h, graine)
    else:
        g = graphe_video(m, invite, l, h, graine, duree)
    desc = f"{m['nom']} · {l}×{h} · {m['pas']} pas · graine {graine}" + (f" · {duree:g} s" if m["type"] == "video" else "")
    return g, desc


def premier_fichier(sorties):
    """Premier fichier de sortie (type output) de l'historique ComfyUI : dict filename/subfolder/type, ou None."""
    for sortie in sorties.values():
        for cle in ("images", "gifs", "videos", "video"):
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

        graine = random.randint(0, 2**48)
        graphe, desc = construire(cle, texte, graine)
        video = MODELES[cle]["type"] == "video"
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
        type_mime = "video/mp4" if f["filename"].endswith(".mp4") else "image/png"

        from open_webui.models.users import Users
        from open_webui.routers.images import upload_image
        user = await Users.get_user_by_id(__user__["id"])
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
