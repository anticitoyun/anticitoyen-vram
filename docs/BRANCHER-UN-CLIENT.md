# Brancher un client sur acvram

acvram sert **deux** interfaces sur le même port : celle d'OpenAI et celle
d'Anthropic. La plupart des clients parlent l'une ou l'autre, et n'ont donc rien
à savoir d'acvram — il suffit de leur donner une adresse.

## Lancer le serveur

```bash
acvram serve <modele> --port 8000 --served-name mon-modele
```

* `--host` vaut `127.0.0.1` par défaut : le serveur n'écoute que la machine
  locale. Mettre `0.0.0.0` pour l'ouvrir au réseau — à ne faire qu'en connaissant
  son réseau, il n'y a pas d'authentification.
* `--port` vaut `8000` par défaut.
* **`--served-name` est le nom que le client devra employer.** Sans lui, le
  modèle est annoncé sous le nom de son dossier, qui est rarement commode. Le
  nom réellement annoncé se lit toujours par `GET /v1/models`.

## Routes

```
/v1/models             la liste, et le nom a employer
/v1/chat/completions   OpenAI, flux et hors flux
/v1/completions        OpenAI, complement brut
/v1/messages           Anthropic, flux et hors flux
/v1/embeddings         plongements
/health  /metrics      etat et compteurs
```

## Un client OpenAI

La plupart des clients — bibliothèque `openai`, interfaces web, extensions
d'éditeur — se contentent de deux variables :

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=peu-importe        # acvram ne verifie rien
```

Puis le nom du modèle, celui rendu par `/v1/models`.

## Claude Code

Claude Code parle l'interface d'Anthropic, servie par `/v1/messages` :

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=peu-importe
export ANTHROPIC_MODEL=mon-modele        # le nom de --served-name
```

Le flux est complet : `message_start`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`.

## Ce qui est vérifié, et ce qui ne l'est pas

**Vérifié le 10/09/2026** sur `Agents-A1-4B-kimi-nvfp4`, par requêtes directes :
les deux interfaces répondent avec la structure attendue, en flux comme hors
flux, et la séquence d'événements d'Anthropic est complète, aucun champ manquant.

**Non vérifié** : le branchement d'un vrai client Claude Code de bout en bout.
Les essais ont porté sur le protocole, pas sur la négociation qui l'entoure —
un client peut exiger un en-tête, une version ou une réponse à une requête
préalable que ces essais n'ont pas simulés. L'en-tête `anthropic-version:
2023-06-01` a suffi ; rien ne dit qu'il suffira à tous.

## Le modèle parle avant de répondre

Certains modèles — les Kimi en particulier — écrivent leur raisonnement en clair
avant leur réponse, souvent introduit par « Thinking Process: ». **C'est le
modèle, pas le serveur** : acvram ne transforme pas le texte. Un client qui
attend une réponse directe verra ce préambule ; c'est à lui, ou à l'invite, de
le traiter.
