# Q36 — duck.ai, 27/09 (Open WebUI 0.11.4 + ComfyUI : multi-modèle image et vidéo, question utilisateur, pour poste5 et chef)

Sources primaires lues avant duck.ai : `docs.openwebui.com` (page ComfyUI, page Generate/Edit Images,
Rich UI Embedding), discussion GitHub `open-webui/open-webui#24154` (mai 2026), DeepWiki
`open-webui/docs` § Media Generation, discussion `#18677` (affichage vidéo en chat). 3 modèles duck.ai
(GPT-5.6 Luna, gpt-oss 120B, Gemma 4 31B, web activé pour les deux derniers) — même question, convergence
totale cette fois, pas de désaccord entre modèles.

## (1) Plusieurs modèles d'image dans une conversation

**Non, pas nativement.** Le réglage `Admin > Experience > Images` ne configure qu'**un seul** moteur,
une seule URL, un seul modèle/workflow de base pour toute l'instance — ce champ n'a rien à voir avec le
sélecteur de modèle en haut du chat, qui reste réservé au LLM texte. Aucune release depuis mai 2026
(discussion `#24154`) n'a ajouté de sélecteur multi-modèle natif pour l'image ; rien de plus récent trouvé
côté 0.11.4.

**Solution recommandée (communauté, confirmée par les 3 modèles)** : une **Pipe Function**, qui
enregistre plusieurs "pipes" — chacune pointant vers un backend ou un workflow ComfyUI différent. Chaque
pipe apparaît alors comme un modèle séparé dans le sélecteur de conversation normal, avec son propre
usage tracké dans les stats admin (comme un appel de modèle, pas de token compté).

Pour ton cas (Flux.2 Klein 9B, dreamshaper_8 SD1.5, SD1.5 de base) : une pipe par modèle, chacune
injectant son propre workflow ComfyUI exporté en API Format et ses propres paramètres (checkpoint,
résolution, steps). Recherche "image gen pipe" sur le site communautaire Open WebUI comme point de
départ (une implémentation de matthewh citée dans la discussion #24154).

## (2) Génération vidéo

**Pas de support natif.** La page Media Generation ne documente qu'image generation/edition (OpenAI,
Gemini, ComfyUI, Automatic1111, proxies unifiés) — aucun moteur vidéo listé. Une demande de fonctionnalité
("feat: video generation support", `open-webui/open-webui#28313`, ouverte août 2026) confirme que ce
n'est toujours pas supporté nativement, alors que ComfyUI (et vLLM Omni) supportent déjà la vidéo côté
backend.

**Patron recommandé et à jour (confirmé par les 3 modèles + doc primaire)** : un **Tool Python** qui :
1. envoie le workflow au `/prompt` de ComfyUI ;
2. attend la fin (polling `/history/{prompt_id}` ou WebSocket) ;
3. récupère le fichier vidéo produit (le rendre accessible par une URL HTTP servie par Open WebUI ou par
   ComfyUI lui-même — jamais un chemin local `file://`, invisible au navigateur) ;
4. retourne soit un simple lien de téléchargement, soit — pour un affichage inline persistant dans
   l'historique — un `HTMLResponse` avec `Content-Disposition: inline` contenant une balise
   `<video controls>` (mécanisme **Rich UI Element Embedding**, qui cite explicitement les "media
   players (video, audio)" comme cas d'usage documenté).

Squelette minimal (Luna) :
```python
from fastapi.responses import HTMLResponse
html = f"""<!doctype html><html><body>
<video controls preload="metadata" style="max-width:100%;">
<source src="{video_url}" type="video/mp4"></video>
<p><a href="{video_url}" download>Télécharger la vidéo</a></p>
</body></html>"""
return HTMLResponse(content=html, headers={"Content-Disposition": "inline"})
```
Point pratique signalé par Luna : l'iframe Rich UI est sandboxé, `video_url` doit être une URL HTTP
réellement accessible au navigateur (pas un chemin de conteneur ComfyUI/Open WebUI non exposé).

## Ce qui n'est pas sourcé

- Le détail exact du polling ComfyUI (`/history/{id}` vs WebSocket) est du fonctionnement standard
  ComfyUI, pas une doc Open WebUI spécifique — signalé par gpt-oss et Gemma.
- Aucune matrice de compatibilité officielle par build 0.11.4 précise pour le point (1) ; c'est une
  lecture cohérente de la doc générale + de la discussion #24154, pas une release note dédiée.

**RESTE** : rien en attente de ma part sur Q36 — livré à chef et poste5.
