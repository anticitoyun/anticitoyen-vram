# Q35 — duck.ai, 27/09 (Open WebUI 0.11.4 / ComfyUI, pour poste5 et poste6)

Sources primaires lues avant duck.ai : notes de version GitHub `open-webui/open-webui` v0.11.3 et
v0.11.4, `docs.comfy.org/changelog`, `Comfy-Org/ComfyUI` releases (v0.37.0-0.37.2, sept 2026).
3 modèles duck.ai (raisonnement) : GPT-5.6 Luna, gpt-oss 120B, Gemma 4 31B — même question aux trois.

## Ce qui est établi par une source primaire (retenu directement)

- **0.11.3** corrige un bug de migration qui laissait la base à moitié mise à jour (colonne
  `chat.timer_at` manquante) après un passage par 0.11.0-0.11.2 : l'appli s'arrête désormais sur
  l'erreur de migration au lieu de démarrer avec une base incomplète. (release note officielle)
- **0.11.4** : l'endpoint qui teste une connexion de génération d'image (ComfyUI, A1111, OpenAI
  images) a changé de forme — il prend maintenant la connexion à tester dans le corps de la requête,
  au lieu de la déduire de la config globale. Rupture pratique confirmée par les trois modèles.
- **0.11.4** ajoute `ENABLE_DIRECT_INTEGRATIONS` (nouvelle variable, pas un renommage) : sans elle,
  l'onglet Integrations personnel (connexions directes utilisateur, dont ComfyUI) reste masqué ; une
  connexion déjà créée continue de fonctionner.
- **0.11.4** allège fortement les images Docker (slim ≈ 175 Mo, standard ≈ -170 Mo) : paquets, Python
  en double, polices retirés. Une installation qui dépendait implicitement d'un paquet présent
  auparavant peut ne plus démarrer. Documenté dans la release, mais absent des sections "connexions".

## Désaccord entre modèles — à ne pas prendre tel quel

- **gpt-oss** affirme un renommage de l'endpoint ComfyUI `/prompt` → `/api/prompt` (400 sur l'ancien
  chemin) entre les versions 0.33-0.34, et un renommage `WEBUI_JWT_SECRET_KEY` → `WEBUI_SECRET_KEY`
  côté Open WebUI. **Aucun lien vérifiable fourni** (mentions "GitHub" sans URL), et rien de tel dans
  les changelogs officiels `Comfy-Org/ComfyUI` ni `open-webui/open-webui` lus avant la question. GPT-5.6
  Luna et Gemma, interrogés sur le même point, ne corroborent pas ce renommage et Luna dit explicitement
  ne pas trouver de rupture générale de `/prompt`. **Tranché : ce point de gpt-oss n'est pas retenu**,
  probable confabulation (comportement connu de ce modèle sur des identifiants précis type nom de
  variable/route sans lien).
- **Gemma** cite `ENABLE_SUBAGENTS`, un changement de `USER_AGENT` en `OpenWebUI/0.11.3`, et le retrait
  de `langchain-community` de l'image slim, sourcés uniquement via `olares.com` (miroir tiers, pas la
  release GitHub) — plausibles mais non vérifiés sur source primaire par moi. À vérifier avant de les
  citer comme fait si ça devient bloquant pour un déploiement réel.

## Ce qui reste non établi (les trois modèles s'accordent)

- Pas de rupture générale et intentionnelle documentée de la route `/prompt` de ComfyUI dans les
  versions 0.3x-0.37.x.
- Pas de suppression documentée du mécanisme `custom_nodes/<dossier>/__init__.py`.
- Les ruptures réelles de custom nodes viennent de la validation des entrées (types, valeurs par
  défaut désormais requises, listes de choix dynamiques), de dépendances Python/CUDA incompatibles,
  ou de comportements de nœuds spécifiques changés en amont (ex. SeC-Nodes retire FP8 pour instabilité
  numérique, oct. 2025) — jamais un changement de contrat HTTP de `/prompt` lui-même.
- Pour Open WebUI : pas de suppression documentée de champ de connexion OpenAI standard, ni de
  changement silencieux du format `/v1/chat/completions`.

## Pour poste5 / poste6

- Avant de monter en 0.11.4 depuis ≤ 0.11.2 : sauvegarder `webui.db` / le volume data, passer par
  0.11.3 pour laisser la migration échouer proprement si elle doit échouer, vérifier les colonnes.
- Un outil/fonction personnalisé qui suppose des paquets présents dans l'ancienne image standard/slim
  peut casser au démarrage en 0.11.4 — à tester après mise à jour, pas juste lire le changelog.
- ComfyUI : workflow à exporter en *API Format* (Save API Format), pas l'export JSON standard ;
  `COMFYUI_BASE_URL` doit correspondre exactement schéma/hôte/port réels (garde SSRF) ; un custom node
  qui casse après une mise à jour ComfyUI est d'abord à regarder côté validation d'entrée du nœud, pas
  côté route HTTP.

**RESTE** : rien en attente de ma part sur Q35 — livré à chef, poste5, poste6.
