# poste5 — pièce 296 : vidéo dans OWUI par la Pipe ComfyUI médias (avec poste6) — verdict

* instrument : `scratchpad/poste5-p296-28-09/controle-videos.py` (OWUI /api/chat/completions, image jointe en image_url,
  vidéo téléversée dans OWUI puis passée en files), sous `prise-videos.sh` ; journal `prise-videos.log` (ignoré par git, gardé dans le worktree)
* commit : 31a127696 (poste5-296) ; workflows d'poste6 0b541da98 (fusionnés)
* régime : carte 0 sous carte.sh (mesure, fenêtre d'poste1), compute-apps identiques avant/après (llama-server seul) ;
  durée de la vidéo : 2 s (défaut 5 s non chronométré)
* scellé : TENU par bras si lien /api/v1/files, mp4 non vide servi par OWUI, une entrée ComfyUI nouvelle ; plus un refus nommé
  de VACE sans pièce jointe
* mesuré (chargement compris, à froid sauf LTX t2v) : Wan 2.2 14B I2V 250,4 s · T2V 302,4 s · VACE 411,0 s ·
  LTX-2.3 I2V 272,5 s · LTX-2.3 T2V 64,1 s (modèles LTX déjà chargés) ; refus sans pièce TENU
* verdict : **6/6 TENU**. Installé : Pipe 0.2.0 + 5 fiches modèles (nom, ligne d'usage, vision/fichiers si entrée).
  Sauvegardes : `/mnt/AI_GENERATOR/openwebui/sauvegardes/webui-avant-296-20260928-204247.db` (base) et `medias-*`.
* durée : prévue ≤ 40 min ; tenue 20:44:02 → 21:06:01 (22 min)

Remarques : `openwebui comfyui` refuse sous une prise « mesure » (garde bd kmb) → ComfyUI lancé dans la prise, env
COMFY_ENV passé à l'appel (hors dépôt). OWUI remis à l'arrêt, comme trouvé ; drapeau de pause d'poste1 retiré.
