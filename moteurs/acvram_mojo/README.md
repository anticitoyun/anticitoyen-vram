# acvram_mojo

Moteur de comparaison d'acvram : question (b) du contrat (`acvram-memoire/revue/moteurs-rust-mojo-23-09.md`)
— le langage des NOYAUX. Pas de FFI vers nos `.cu`/Triton (option A de `acvram_rust`) : le moteur MAX/Mojo
sert le modèle avec **ses propres noyaux**, écrits en Mojo. Ce qui est isolé ici, c'est l'instruction émise
par un compilateur différent sur la même carte, pas le coût de l'hôte.

## État (étape 1, squelette)

| partie | état |
|---|---|
| faisabilité sm_120 | tenue (`acvram-memoire/revue/poste4-mojo-etape0-23-09.md`) |
| chargement + décodage glouton b=1 + `/v1/chat/completions` | **MAX le fournit nativement** (`max serve`), pas de code moteur à écrire — c'est le point (b) : MAX EST le moteur Mojo |
| porte (sha256 du texte, 5 invites, glouton, `tests/invites.json`) | **2/5 au bit** contre acvram (mêmes poids bf16 source, `outils/comparer.py`) |
| KL de repli (contrat §3, `outils/kl_reference.py`, dump HF `outils/dump_hf_reference.py`) | **écrite et jouée** — acvram KL_max=0,00149 (tenu, ≤0,74) ; MAX KL_max=20,01 (**NON tenu**) |
| cellule débit/énergie (contrat étape 2, conditionnée à « si tenu ») | **ne se lance pas** — porte KL non tenue |

## Contre quoi la porte est jouée

Même source : `/mnt/AI_GENERATOR/models_acvram/Qwen3-4B-bf16-hf` (Qwen/Qwen3-4B, sha256 des trois shards dans
`acvram-memoire/revue/poste4-mojo-etape0-23-09.md`), convertie sans quantification (`acvram convert --format
bf16`, x1.00, SNR 0 dB) pour le côté acvram. Deux serveurs `/v1/chat/completions`, glouton (`temperature=0,
top_p=1, seed=0`), interrogés **séquentiellement** (la 5090 tient les deux poids mais pas les deux caches KV
en même temps sans les borner), comparés par sha256 du texte de réponse.

## Porte NON tenue au bit (2/5) — lecture

Les deux moitiés divergent tôt (dès la deuxième phrase de la pensée `<think>`), pas seulement au bruit
d'arrondi final : c'est cohérent avec des noyaux différents (MAX/Mojo contre nos noyaux Triton/CUDA — même
alerte que pièce 119/120 sur Rust : c'est l'instruction émise qui compte).

## Porte KL — jouée, NON tenue pour MAX (`acvram-memoire/revue/poste4-mojo-kl-verdict-24-09.md`)

acvram KL_max = 0,00149 (tenu). MAX KL_max = 20,01 (non tenu), mais **concentré** : 7/8 jetons par invite
sont exacts (KL≈0) contre la référence HF bf16, une seule position (après « Okay », jeton attendu `,`) porte
tout l'écart — MAX y rend la virgule pleine chasse `，` au lieu de `,`, 5/5 invites, même position. Pas un
bruit de noyau diffus : un point de divergence net (sampler ou table de sortie MAX sur ce jeton précis),
à diagnostiquer avant de relancer la porte. La cellule débit/énergie (étape 2, conditionnée) ne se lance
pas tant que cette porte n'est pas tenue.

## Environnement (épinglé, contrat §2)

Mojo 1.2.0.dev2026092305 (eea89a4d), MAX 26.7.0.dev2026092305 (canal `max-nightly` + `modular-community`,
pixi 0.81.0), venv isolé hors `.venv` du dépôt (chemin non commité, `PATH` exporté par commande).

## Commandes

```bash
# venv isolé (une fois) : voir acvram-memoire/revue/poste4-mojo-etape0-23-09.md
pixi run --manifest-path <projet-mojo>/pixi.toml max serve --model-path <dossier-hf> \
  --quantization-encoding bfloat16 --devices gpu:0 --port <port> --served-model-name defaut

python outils/interroger.py tests/invites.json <port> sortie.json   # glouton, 5 invites
python outils/comparer.py sortie-acvram.json sortie-max.json resultat.json
```

Sous `outils/carte.sh` (type `mesure`), jamais les deux serveurs chargés en même temps sur la 5090 (OOM
constaté à la première tentative).

## Formats disponibles / natifs pour Qwen3 (rappel étape 0)

Qwen3 dense (MAX, `SUPPORTED_ENCODINGS`) : bfloat16 (défaut), float32, float8_e4m3fn. Pas de nvfp4/mxfp4 pour
cette architecture (réservé à Gemma4, Kimi-K2.5, Qwen3.5-MoE) — d'où la comparaison bf16 contre bf16, jamais
contre le nvfp4 du parc, sur ordre du chef après l'étape 0.

## Reste

Diagnostiquer la virgule pleine chasse (MAX) avant de rejouer la porte KL ; étape 2 (question a, si
demandée) : FFI Mojo vers nos noyaux — non commencée, pas dans ce contrat pour l'instant.
