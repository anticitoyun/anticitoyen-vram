# Politique de sécurité

## Versions suivies

Seule la dernière version publiée reçoit des correctifs de sécurité.

| Version | Suivie |
|---|---|
| dernière publiée (0.7.x) | oui |
| antérieures | non — mettre à jour |

## Signaler une vulnérabilité

**Ne pas ouvrir de ticket public.** Utiliser le signalement privé de GitHub :
onglet **Security** du dépôt → **Report a vulnerability**.

Indiquer :

* la version (`acvram --version`) et le commit ;
* la commande ou la requête HTTP qui reproduit le problème ;
* l'effet observé (fuite de données entre requêtes, exécution de code, déni de service…).

Un accusé de réception suit sous 7 jours. Le correctif est publié dans une version dédiée, avec
une entrée au `CHANGELOG.md` qui crédite l'auteur du signalement s'il le souhaite.

## Périmètre et hypothèses

* `acvram serve` écoute par défaut sur `127.0.0.1` et **n'a pas d'authentification** : l'API
  compatible OpenAI est prévue pour un usage local. L'exposer sur un réseau (`--host 0.0.0.0`)
  sans mandataire authentifiant (nginx, Caddy…) revient à donner l'usage du GPU à quiconque
  atteint le port ; ce n'est pas une vulnérabilité d'acvram.
* Les poids sont lus au format **safetensors** ; aucun chargement par `pickle` ni code
  distant de modèle (`trust_remote_code`) n'est exécuté.
* Les noyaux CUDA sont compilés localement depuis les sources du paquet ; un noyau
  précompilé n'est chargé que si son empreinte (source, versions de torch et de CUDA,
  architecture) concorde.
* Sont dans le périmètre : fuite de contenu d'une requête vers une autre (cache KV, lots),
  lecture ou écriture de fichiers hors des répertoires de modèles demandés, exécution de
  code par une requête HTTP ou un fichier de modèle.
