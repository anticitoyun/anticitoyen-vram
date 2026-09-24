# Porte KL rejouée sur MAX stable et nightly du jour (ordre chef) — persiste

* nightly du jour : `pixi add "modular=*"` sur le canal `max-nightly` résout à **26.7.0.dev2026092305**,
  identique à la version déjà testée (aucun nouveau build publié depuis) — pas rejouée, sans objet.
* stable : **MAX 26.6.0** (canal `https://conda.modular.com/max`), venv isolé distinct
  (`projet-mojo-stable`), mêmes invites, mêmes ids, même seuil (0,74), mêmes flags requis
  (`--no-enable-overlap-scheduler --force --enable-echo`, `KL_TOPK=7`)
* mesuré : **KL_max_global = 20,01123, IDENTIQUE au bit près à la nightly** (mêmes 5 valeurs par invite,
  même position — après « Okay », `，` contre `,`)
* verdict : **persiste sur la version stable** — l'hypothèse « défaut de la nightly 26.7 » est **réfutée**.
  Mojo s'arrête là cette semaine (ordre chef, décision utilisateur en amont).
