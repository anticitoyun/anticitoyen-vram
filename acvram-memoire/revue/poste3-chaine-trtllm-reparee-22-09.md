# Chaîne TRT-LLM : diagnostic et réparation (poste3, 22/09, à sec)

poste2 (05 h 5x) : 3/18 fenêtres valides, chiffres suspects (TRT-LLM b=1 46 J,
prefill 70 963 j/s ≈ ×2 la prédiction) et fenêtre acvram restée en boucle
d'attente sans jamais prendre le verrou ni allouer (3ᵉ défaut de la chaîne).
Deux causes distinctes, trouvées à sec, corrigées ; **non commitées** tant que la
calibration acvram-seul n'a pas validé le patch.

## Défaut 1 — fenêtre acvram en boucle d'attente (jamais de verrou ni d'alloc)

- `scratchpad/trtllm-cellules-22-09/cellule.sh:20-23` : la cellule s'auto-enveloppe
  sous `carte.sh` en mode **MESURE** (`flock` fd 9 `LOCK_EX`), verrou tenu pour
  toute la durée.
- `cellule.sh:33` lançait `parc/bin/acvram-serveur`, qui à
  **`parc/bin/acvram-serveur:83`** relance `carte.sh` en mode **SERVICE** →
  `flock` exclusif sur le **même** fichier verrou déjà tenu par la cellule mère →
  attente indéfinie. Serveur jamais démarré, port 8090 jamais ouvert,
  `cellule.sh:42-45` boucle 300 × 2 s puis `return 1` (fenêtre perdue).

**Correctif** : le verrou MESURE est déjà tenu ; on lance le serveur sans repasser
par `carte.sh`. Le wrapper offre la porte `ACVRAM_EXEC` (`acvram-serveur:79`,
`exec` **avant** le bloc `carte.sh`) : `ACVRAM_EXEC=setsid parc/bin/acvram-serveur
<alias>` exec directement `acvram serve --port 8090 …` détaché, sous le verrou déjà
acquis. Arrêt de la fenêtre A par le port (`arreter_port 8090`) et non par le nom
`acvram-serveur` : le process exec'é s'appelle désormais `acvram serve`.

## Défaut 2 — prefill compté ×2 (70 963 j/s), b=1 énergie basse

- `charge.py:20` : `PROMPT_LONG` **constant**, rejoué en boucle par `charge.py:52`.
- `charge.py:53` comptait `n_invite = 2048` **supposé** à chaque requête.
- TRT-LLM (et acvram) réutilisent le cache KV d'un préfixe identique : dès la 2ᵉ
  requête le prefill n'est plus calculé, mais les 2048 jetons sont comptés → débit
  gonflé (≈ ×2). Même racine pour le b=1 46 J (réponses quasi gratuites via cache
  → énergie mesurée basse).

**Correctif** : (a) chaque invite est préfixée d'un identifiant unique
(`// requete <n> <horodatage>`, `charge.py:_uid`) → cache de préfixe défait, chaque
prefill réellement calculé ; (b) le débit prefill compte les **vrais**
`usage.prompt_tokens` renvoyés par le serveur (repli 2048 seulement si l'usage est
absent), au lieu de la constante.

## Calibration (EN ATTENTE de la fenêtre carte)

≤ 10 min de carte après la cellule A/V de poste2 (≈ 45 min). `charge.py` sur acvram
seul (5090, port 8090) doit retrouver **1 625 ± 40 t/s à b=12** et **380 ± 10 à
b=1**. Tant que ce n'est pas le cas : aucun chiffre TRT-LLM publié, et le patch
n'est pas commité.

### Résultat 06:28-06:38 — ROUGE (les deux hors cible)

| mesure | charge.py | cible | verdict |
|---|---|---|---|
| b=12 | **1 704,5 t/s** (2445 MHz, 36864 j/21,6 s) | 1 625 ± 40 [1585,1665] | hors, +39,5 trop haut |
| b=1 | **286,1 t/s** (2662 MHz, 5888 j/20,6 s) | 380 ± 10 [370,390] | hors, −84 trop bas |

Défauts 1 et 2 réparés (fenêtre acvram démarre bien : POST 200 OK dans le log ;
plus de boucle d'attente). Mais **charge.py ne reproduit pas la cellule
officielle** : il mesure des **requêtes courtes répétées** (`--mode` boucle de
requêtes de 256 jetons, non-streaming, aller-retour HTTP synchrone entre chacune).
Chaque itération porte le prefill de l'invite + la latence d'un aller-retour :
- **b=1** : ce coût fixe domine (peu de jetons par requête, un seul flux) →
  débit sous-estimé de ~25 % (286 vs 380).
- **b=12** : les 12 threads concurrents recouvrent l'overhead et l'agrégat
  sur-estime de ~5 % (1704 vs 1625).

Carte prise 06:28, rendue à ~06:38 (verrou repris par qvl30b-reconv-identite).

## 3e correctif (chef 22/09) — client officiel, charge.py abandonné

Tranché : on ne réinvente pas le client. `charge.py` est **abandonné** ; la chaîne
utilise le client de la cellule officielle, `banc-llamacpp-16-09.py decode` (harnais
de la cellule 1 625 t/s). Fait à sec :

- **cellule.sh réécrite** : `decode` (b=1 ET b=12 en un appel via `BANC_SLOTS=1,12`,
  `BANC_JETONS=1024`, `BANC_FENETRE_S=20`, invites en ids, `/v1/completions` SSE,
  `ignore_eos`) contre acvram (fenêtre A) puis trtllm-serve (fenêtre B), intercalé
  A B B A A B, verrou MESURE tenu en continu. `BANC_MOTEUR=acvram` = le protocole
  OpenAI-completions-ids-SSE, valable aussi pour trtllm (OpenAI-compatible).
- **acvram démarré avec les args EXACTS de la cellule officielle** :
  `acvram.cli serve … --max-batch 12 --max-model-len 2304 --served-name coder`,
  **sans** le wrapper `acvram-serveur` (qui ajoutait `--speculative ngram` et
  `--max-model-len 32768` → régime différent). C'est aussi ce qui règle le défaut 1
  (le serveur tourne sous le verrou déjà tenu, sans reprendre de verrou).
- **prefill** : le mode `prefill` du client lance `llama-bench` en LOCAL, pas via
  l'URL → inapplicable à un serveur OpenAI. Décodage b=1/b=12 seulement ici ; le
  prefill des cellules relève d'un instrument dédié — écart nommé.
- **calibrer.sh** : recalibration acvram seul par ce même client (une fenêtre
  `decode` `BANC_SLOTS=1,12`). Attendu par construction : 1 625 ± 40 / 380 ± 10.

**Test à sec (faux serveur HTTP OpenAI-SSE, sans carte)** : le client atteint
`BANC_URL`, parse le SSE, sort deux `RESULTAT` (slots 1 et 12) avec `jetons_s` et
`fenetre_valide=true`. Paramétrage de l'URL validé. `bash -n` OK sur cellule.sh et
calibrer.sh, `httpx`/venv/`trtllm-serve` présents.

### Recalibration 06:48-06:54 (client officiel) — b=12 vert, b=1 au bord

| mesure | client officiel | cible | verdict |
|---|---|---|---|
| b=12 | **1 615,8 t/s** (2449 MHz, 389 W, 22,8 s) | 1 625 ± 40 [1585,1665] | DEDANS ✓ |
| b=1 | **390,7 t/s** (2661 MHz, 247 W, 21,0 s) | 380 ± 10 [370,390] | +0,7 au-dessus de la borne |

Régime lu identique à l'officiel (sampler=graphe, rapatriement=epingle, kv=int8,
cache_prefixe=0). Le harnais reproduit la cellule officielle (b=1 = +2,6 % de
l'officiel 380,8 ; b=12 colle à 1625) — la sous-estimation de charge.py (286) a
disparu. Le +0,7 de b=1 est dans le bruit d'une fenêtre de 21 s, et va vers le
HAUT (pas de sous-estimation résiduelle). Choix de seuil renvoyé à la chef
(vert / arrêt strict). Cellules TRT-LLM à enchaîner sur feu.

Borne b=1 : initiale ±10 (écrite pour attraper l'erreur de méthode 286 = −25 %),
mesure 390,7, borne retenue **±3 % (≈ ±11)** — le +0,7 est le bruit d'une fenêtre,
pas une sous-estimation (amendement chef 22/09). Calibration VERTE.

### Cellules 06:56-07:09 — fenêtres acvram OK, fenêtres trtllm crashées (4e défaut)

Fenêtres A (acvram, args cellule officielle) mesurées, cohérentes avec la calibration :

| cellule | b=1 t/s | b=12 t/s | horloge b=12 |
|---|---|---|---|
| 1A | 390,4 | 1 661,7 | 2438 |
| 4A | 390,4 | 1 628,7 | 2459 |
| 5A | 392,4 | 1 590,2 | 2477 |

Fenêtres B (trtllm) : **AUCUN résultat**. trtllm-serve démarre bien (startup
complete, /v1/models + /metrics 200) mais le client **crashe** à
`banc-llamacpp-16-09.py:107` : `client.get("/metrics").json().get("cartes")` →
`AttributeError` car trtllm rend `/metrics` en **LISTE** Prometheus, pas un dict
`{"cartes": …}` acvram, et le `except` ne couvrait que `(httpx.HTTPError,
ValueError)`. 4e défaut de la chaîne.

**Correctif (sur la copie locale du client)** : `except` élargi à
`(httpx.HTTPError, ValueError, AttributeError, TypeError)` → un `/metrics`
non-acvram est ignoré, `CUDA_VISIBLE_DEVICES` reste 0. Re-testé à sec (faux
serveur rendant `/metrics` en liste) : le client sort b=1 et b=12 sans crash.

**Reste** : un créneau court (≤ 15 min) pour refaire les 6 fenêtres intercalées
A B B A A B d'un coup (refaire seulement B casserait l'intercalation anti-dérive).
Puis verdict 7 lignes (KV FP8/INT8, exclusions, graphes des deux côtés) + commit
(cellule.sh, calibrer.sh, banc-llamacpp-16-09.py, retrait charge.py). Rien commité :
comparatif incomplet. Scellé poste3 : TRT-LLM b=1 420 [380-470], b=12 1 700 [1 550-1 850].
