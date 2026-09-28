#!/usr/bin/env python3
"""296 (poste6) : rendu court de chaque graphe vidéo de parc/share/openwebui/videos/ sur un ComfyUI déjà lancé.

Reprenable : un graphe déjà rendu (ligne OK dans le TSV) est sauté. La vidéo de contrôle de wan-vace est le rendu
de wan22-14b-i2v (téléversée dans input/). Lancer par outils/videos-owui-rendu.sh sous carte.sh, jamais à nu.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request
import uuid

ORDRE = ["wan22-14b-i2v", "wan22-14b-t2v", "wan-vace", "ltx23", "ltx23-t2v"]
# Rendu court : 2 s. Wan 4k+1 à 16 i/s, LTX 8k+1 à 24 i/s ; dimensions du défaut réduites pour LTX (768×512).
COURT = {"wan": {"length": 33, "width": 832, "height": 480}, "ltx": {"length": 49, "width": 768, "height": 512}}


def requete(url, chemin, donnees=None, timeout=60):
    req = urllib.request.Request(url + chemin, data=json.dumps(donnees).encode() if donnees is not None else None,
                                 headers={"Content-Type": "application/json"} if donnees is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def televerser(url, fichier):
    frontiere = uuid.uuid4().hex
    corps = (f"--{frontiere}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fichier.name}\"\r\n"
             f"Content-Type: application/octet-stream\r\n\r\n").encode() + fichier.read_bytes() + f"\r\n--{frontiere}--\r\n".encode()
    req = urllib.request.Request(url + "/upload/image", data=corps, headers={"Content-Type": f"multipart/form-data; boundary={frontiere}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["name"]


def patcher(g, noeuds, valeurs):
    for e in noeuds:
        if e["type"] in valeurs:
            for nid in e["node_ids"]:
                g[nid]["inputs"][e["key"]] = valeurs[e["type"]]


def sonde(fichier):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames,r_frame_rate:format=duration", "-of", "json", str(fichier)],
                         capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    s = (j.get("streams") or [{}])[0]
    return s.get("width"), s.get("height"), s.get("nb_read_frames"), s.get("r_frame_rate"), float((j.get("format") or {}).get("duration", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphes", default=pathlib.Path(__file__).resolve().parent.parent / "parc" / "share" / "openwebui" / "videos", type=pathlib.Path)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--sortie-comfy", default=pathlib.Path("/mnt/2TO_SSD_2025_IA"), type=pathlib.Path)
    ap.add_argument("--tsv", required=True, type=pathlib.Path)
    ap.add_argument("--image", default=pathlib.Path("/mnt/AI_GENERATOR/Comfyuirtx5090/ComfyUI312/ComfyUI/input/example.png"), type=pathlib.Path)
    ap.add_argument("--seulement", nargs="*", default=ORDRE)
    a = ap.parse_args()

    deja = {l.split("\t")[0] for l in a.tsv.read_text().splitlines() if "\tOK\t" in l} if a.tsv.exists() else set()
    if not a.tsv.exists():
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=a.graphes).stdout.strip()
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        a.tsv.write_text(f"# 296 rendu court · {time.strftime('%d/%m %H:%M')} · HEAD {head} · ComfyUI {requete(a.url, '/system_stats')['system']['comfyui_version']} · {gpu}\n"
                         "graphe\tetat\tduree_s\tfichier\ttaille_o\tlargeur\thauteur\timages\tips\tduree_video_s\tdetail\n")
    nom_image = televerser(a.url, a.image)
    controle = None
    for nom in ORDRE:
        if nom not in a.seulement or nom in deja:
            continue
        g = json.loads((a.graphes / f"{nom}.api.json").read_text())
        noeuds = json.loads((a.graphes / f"{nom}.noeuds.json").read_text())
        court = COURT["ltx" if nom.startswith("ltx") else "wan"]
        patcher(g, noeuds, {**court, "image": nom_image, "seed": 42})
        if nom == "wan-vace":
            if controle is None:
                a.tsv.open("a").write(f"{nom}\tSAUTE\t\t\t\t\t\t\t\t\tpas de rendu i2v à téléverser en contrôle\n")
                continue
            patcher(g, noeuds, {"video": televerser(a.url, controle)})
        t0 = time.time()
        try:
            pid = requete(a.url, "/prompt", {"prompt": g, "client_id": "poste6-296"})["prompt_id"]
        except urllib.error.HTTPError as e:
            a.tsv.open("a").write(f"{nom}\tREFUSE\t\t\t\t\t\t\t\t\t{e.read().decode()[:400].replace(chr(10), ' ')}\n")
            continue
        while True:
            time.sleep(3)
            h = requete(a.url, f"/history/{pid}").get(pid)
            if h and (h.get("status", {}).get("completed") or h.get("status", {}).get("status_str") == "error"):
                break
        duree = time.time() - t0
        st = h["status"]
        if st.get("status_str") == "error":
            msg = " | ".join(str(m[1].get("exception_message", m[1]))[:200] for m in st.get("messages", []) if m[0] == "execution_error")
            a.tsv.open("a").write(f"{nom}\tECHEC\t{duree:.0f}\t\t\t\t\t\t\t\t{msg.replace(chr(10), ' ')}\n")
            print(nom, "ECHEC", f"{duree:.0f} s", msg, flush=True)
            continue
        fichier = None
        for out in h["outputs"].values():
            for cle in ("gifs", "videos", "images"):
                for f in out.get(cle, []):
                    fichier = a.sortie_comfy / f.get("subfolder", "") / f["filename"]
        w, hh, n, ips, dv = sonde(fichier)
        a.tsv.open("a").write(f"{nom}\tOK\t{duree:.0f}\t{fichier}\t{fichier.stat().st_size}\t{w}\t{hh}\t{n}\t{ips}\t{dv:.2f}\t\n")
        print(nom, "OK", f"{duree:.0f} s", fichier, f"{w}x{hh}", n, "images", flush=True)
        if nom == "wan22-14b-i2v":
            controle = fichier
    return 0


if __name__ == "__main__":
    sys.exit(main())
