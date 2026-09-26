# Pièce 150 — scellé (poste5, 24/09, écrit AVANT la prise) : les ~5 ms/pas de service à b=8

Ordre chef. Défaut `Qwen3.8-27B-nvfp4` (tout nvfp4, régime par défaut) ; contrôle `Qwen3.8-27B-unsloth-mixte-i8c` sous
PROJ_MARLIN (celui de la 148 bis : banc 27,0 ms/pas, processus 21,5).

## À sec : ce que fait le banc et d'où peut venir le surcoût

Banc chat (`banc-chat-openai.py`) : par LOT, 8 requêtes `/v1/chat/completions` NON-stream, même invite (~40 jetons avec
gabarit), `max_tokens` 256, jusqu'au bout. Le lot suivant attend que les 8 soient finies. Débit = jetons / fenêtre.
Sur l'alias mixte, un lot dure 2 048 / 296,3 = **6,91 s**, dont 256 × 21,5 ms = 5,50 s de décodage pur : **1,41 s de
surcoût par lot**, soit 5,5 ms/pas. NInfer : 4,42 s contre 4,09 s, soit 0,33 s.

Sources possibles (lecture du code) :
1. **Par pas, en service seulement** : détokenisation incrémentale dans `_emit` (le banc de processus
   `frontiere-pas.py` tourne SANS tokeniseur : `Engine(loaded, None, …)`), livraison `call_soon_threadsafe` par jeton
   (`app.py:137`), concurrence pour le GIL avec la boucle asyncio et les fils HTTP.
2. **Par lot** : arrivée étalée des 8 requêtes (gabarit Jinja et tokenisation côté serveur, `add_request`), donc 1 à 3
   pas de préfill au lieu d'un groupé (`runner.py:1562` groupe ce qui est admis dans le MÊME pas) ; chaque préfill
   interrompt le pipeline de graphes ; queue de lot à b < 8 ; carte oisive entre la fin d'un lot et l'arrivée du
   suivant (réponse JSON, fils du client, nouvelles requêtes) ; attente de 2 ms du fil moteur au repos (`app.py:111`).
3. Préfill lui-même : 8 × ~40 jetons, éventuellement répété s'il y a plusieurs vagues d'arrivées.

## Instrument (une prise par alias, ≤ 10 min chacune)

1. `frontiere-pas.py` b=8, 300 pas, sur le DÉFAUT (le pas en processus manque pour ce modèle).
2. Serveur (`--max-batch 8 --max-model-len 4096 --speculative none`, `ACVRAM_TRACE_STEPS=1`) + banc chat b=8, fenêtre 20 s.
   `/metrics` relu avant et après la fenêtre : Δ `decode_seconds`, `prefill_seconds`, `decode_tokens`, pas, pas avec
   préfill. Médiane et p90 des pas EN SERVICE (lignes `[pas]`).
3. nsys EN SERVICE (`-t cuda,osrt --cuda-graph-trace=node`, collecte de 10 s sous le même banc) : trous GPU entre pas et
   entre lots, durée des préfills. Parts hôte lues seulement comme trous de carte, jamais comme durées d'appel (p69).

## Prédiction (alias mixte ; le défaut dans le même rapport)

* Médiane du pas de décodage EN SERVICE = pas en processus **+ 0,3 à 1,5 ms** (détokenisation et livraison de 8 jetons,
  GIL) ; p90 plus haut (pas avec préfill).
* Par lot : **préfill et arrivées 0,15-0,5 s**, **carte oisive entre lots 0,05-0,3 s**, **queue à b < 8 ≤ 0,1 s**.
* Total prédit par lot : 0,3-1,3 s, contre 1,41 mesuré. Ce qui manquerait au-delà de 1,3 s serait une source non listée.
* Défaut `Qwen3.8-27B-nvfp4` : même ordre en secondes par lot (les mêmes causes ne dépendent pas du format), donc un
  surcoût relatif du même ordre (banc 26,7 ms/pas en 102).

**Issues nommées.** (a) Pas de service > processus + 3 ms → le coût est PAR PAS (détokenisation, GIL), levier dans la
boucle de service pour tous les modèles. (b) Préfills > 0,5 s/lot ou > 2 pas de préfill par lot → admission : grouper
les arrivées d'une fenêtre courte. (c) Carte oisive entre lots > 0,3 s → le coût est entre les lots, en partie un
artefact du banc (le client attend tout le lot) : je le dirais, et un banc à lots chevauchants le rendrait réel ou non.

## Addendum 24/09 12 h 3x (après la prise, avant le verdict) : deux chaînes, un pytest du chef

* **Faute (poste5)** : en arrêtant ma première chaîne (ordre d'attendre poste1-p146e/p152), j'ai tué la coquille `bash -c`,
  pas le script enfant. Il a couru : `poste5-p150-defaut` 11:58:37-12:01:33 (avant l'ordre) et `poste5-p150-mixte`
  12:13:55-12:16:44 (après les « rendue » d'poste1-p146e 12:13:12 et d'poste1-p152 12:13:52 : ordre tenu à la lettre,
  par hasard). La seconde chaîne a refait `defaut` à 12:24:51 et `mixte` à 12:28:06. Quatre prises au lieu de deux.
* **pytest du chef** 12:28:05-12:28:37 pendant `mixte` de 12:28. Contrôle : ses 3 lots mesurés couvrent **12:29:00-12:29:21**
  (20,85 s finis à 12:29:21, mtime de `service-mixte.json`), la chauffe les précède de ~4 s : **aucun lot ni chauffe dans
  la fenêtre**, seul le chargement du modèle l'a recouvert. Lots gardés. La trace nsys du mixte vient de la prise de 12:13
  (propre, fichiers `1213-*`), celle de 12:28 est écartée par prudence.
* Sorties : `service-defaut*.json` = prise de 12:24 ; `service-mixte.json` = 12:28 ; `1213-service-mixte-nsys.json` = 12:13.
