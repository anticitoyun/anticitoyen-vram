# Recensement des bancs publics d'inférence LLM sur RTX 5090

**Résultat le plus important, en tête : nous n'avons aucun point de
comparaison public fiable en jetons/kJ pour notre régime (mono-flux, une
carte, nos formats et tailles de modèle).** Une seule source publiée
(`arXiv:2601.09527`, lue en entier depuis, voir plus bas) donne un chiffre
d'énergie absolu convertible en jetons/kJ, et il porte sur un régime
(concurrence 8, Qwen3-8B) différent du nôtre. **Conséquence sérieuse** :
nous ne pourrons jamais affirmer « meilleur que X en jetons/kJ » sur la foi
d'une source publiée existante — cette comparaison exigera de mesurer les
concurrents nous-mêmes, avec notre propre wattmètre et notre propre
protocole, comme on le fait déjà pour llama.cpp. « Rien ne nous dépasse de
façon sûre » plus bas dans ce fichier est une **absence de preuve, pas une
preuve d'absence** : aucune source publiée ne mesure notre régime, donc nous
ne savons pas où nous sommes situés, ni devant ni derrière.

Demandé par Jérôme le 10/09/2026. Recherche web (WebSearch + lecture directe
des sources primaires quand elles existent), jugée avec les mêmes critères
que nos propres bancs cette semaine : contexte nommé, format nommé, lot
nommé, méthode de chronométrage décrite, sinon **non comparable**, avec la
raison écrite plutôt que le chiffre recopié seul.

## Tableau

| source | moteur | modèle | format | contexte | lot | débit | énergie | méthode | comparable ? |
|---|---|---|---|---|---|---|---|---|---|
| [adrienbrault/qwen3.8-27b-rtx5090](https://github.com/adrienbrault/qwen3.8-27b-rtx5090) (dépôt GitHub, réel — vérifié par l'API GitHub, 10 étoiles, poussé le 09/09/2026) | vLLM 0.29.0rc2, fortement patché (chaîne 0101-0138, sm120 NVFP4 KV, MTP maison) | Qwen3.8-27B-NVFP4 (checkpoint NVIDIA/RedHatAI) | NVFP4 poids + NVFP4 KV | 262 K max déclaré ; pool KV 1 391 795 jetons sur **deux cartes** | **32 et 64 flux concurrents** (config « servie ») : code 3 809 t/s agrégé (119/flux) à 32, 4 497 t/s (70/flux) à 64 | puissance plafonnée à 400 W/carte (mesurée à 350 W à la concurrence servie) ; **pas de jetons/kJ** | scripts nommés et datés (`decode_ss.py`, `kv_capacity_probe.py`), dossiers de résultats horodatés par exécution, fidélité vs bf16 mesurée séparément — rigueur de méthode comparable à la nôtre | **Partiellement.** Format et architecture (Blackwell sm120, NVFP4) comparables ; **régime non comparable** — leur chiffre-phare tourne sur **deux 5090** à 32-64 flux simultanés, pas une carte en décodage mono-flux comme notre duel. Un chiffre mono-flux existe probablement dans le dépôt (single-stream mentionné dans les gotchas) mais je ne l'ai pas extrait avec confiance — à revérifier avant de le citer. |
| [ggml-org/llama.cpp discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) (fil communautaire officiel, protocole imposé) | llama.cpp, `llama-bench` | **Llama 2 7B Q4_0** (modèle de référence du fil, pas un modèle actuel) | Q4_0 GGUF | non précisé par le fil (protocole `pp512`/`tg128`, contexte court implicite) | 1 (mono-flux, `llama-bench` par défaut) | Une entrée « 5090 à 600 W » existe dans le fil (repérée, valeur `tg128` non extraite avec certitude avant la limite d'appels de cette recherche) | aucune | `llama-bench`, sortie standardisée (modèle, taille, params, backend, ngl, fa, test, t/s ± écart-type sur plusieurs passages) — **méthode plus rigoureuse que notre propre chronomètre mural** : moyenne + écart-type, pas un seul passage | **Méthode oui, modèle non.** Protocole exemplaire (mesure répétée, écart-type publié) mais sur un **Llama 2 7B**, bien plus petit que tout ce qu'on compare (14B-70B chez nous) — la vitesse mémoire-bornée n'extrapole pas linéairement, un chiffre 7B ne borne rien pour un 14B-70B. |
| Blogs agrégateurs (markaicode.com, gigagpu.com, presenc.ai, mayhemcode.com, spheron.network, runpod.io, kunalganglani.com, localaimaster.com, lyceum.technology) | vLLM / Ollama / llama.cpp / TensorRT-LLM, mélangés sans distinction claire | « modèles 8B », « Llama 3.3 70B », souvent non précisé | souvent non précisé (aucun format cité) | jamais précisé | jamais précisé | jamais mesurée | Un seul chiffre relayé sans protocole propre (« NVIDIA affirme jusqu'à 1,9× », vérifié : c'est une annonce marketing NVIDIA/IFA reprise telle quelle par le blog, pas une mesure du blog) | **Non comparable, systématiquement.** Exactement le type de source déjà écarté ce matin (« *expected speed* ») : pas de format, pas de contexte, pas de lot, pas de méthode, souvent un chiffre marketing « up to X× » repris sans vérification. |
| TensorRT-LLM sur 5090 | TensorRT-LLM | — | — | — | — | — | — | **Aucun banc public avec protocole nommé trouvé.** Toutes les mentions trouvées sont des affirmations de blog sans source primaire (« ~45 t/s sur un 70B », sans format ni contexte). Cohérent avec le carnet : TensorRT-LLM n'a pas de communauté de bancs publics sur cartes grand public comparable à celle de llama.cpp ou vLLM — il est benchmarké en datacenter (H100/B200/GB200), pas sur 5090. |
| MLPerf Inference | — | — | — | — | — | — | — | **Sans objet.** Aucune soumission MLPerf pour une carte grand public (5090) trouvée — la v6.0 (Nebius et autres) couvre Blackwell **datacenter** (B200/GB200 Ultra), pas GeForce. MLPerf n'a pas de catégorie consumer-Blackwell active à ce jour. |
| [arXiv:2601.09527](https://arxiv.org/abs/2601.09527) — Knoop & Holtmann, *Private LLM Inference on Consumer Blackwell GPUs* (14/01/2026, CC-BY, code+Docker publiés) **— lu en entier** | vLLM (le seul moteur testé, cf. limites) | Qwen3-8B (Table 7, celle qui isole le format) | **BF16 / W4A16 / NVFP4**, même moteur, même modèle, même contexte, même lot — voir plus bas | 8K (Table 7) ; 8K-64K ailleurs dans le papier (RAG/multi-LoRA/API) | **concurrence 8** (Table 7) — PAS mono-flux | 260 t/s (bf16) → 314 (W4A16) → 411 (NVFP4), RTX 5090 seule | **Wh/MTok mesuré, convertible en absolu** : bf16 403, W4A16 325, NVFP4 239 Wh/MTok → **689 / 855 / 1162 jetons par kJ** (converti par moi, 1 Wh = 3,6 kJ) | Énergie = « Joules per output token and Wh/MTok, dérivés de la télémétrie de puissance **DCGM** » (GPU seul, pas la prise murale — le papier le dit lui-même : « exclut 20-40% de CPU/mémoire/refroidissement, changerait les valeurs absolues mais pas l'ordre relatif ») ; débit = AIPerf 0.3.0, harnais de charge à concurrence contrôlée | **Format isolé, régime non comparable.** Répond à la question qui décide (point 3 demandé) : **oui, le NVFP4 est comparé au bf16 sur le MÊME moteur (vLLM), MÊME modèle, MÊME contexte, MÊME concurrence — seul le format change.** Le ×1,6/−41% mesure bien le format, pas un chemin de code différent. Mais la concurrence 8 (pas mono-flux) et DCGM-GPU-seul (pas notre wattmètre) restent deux écarts de régime avec notre protocole — **premier ancrage externe pour l'axe énergie qu'on ait, mais pas transposable tel quel.** |

## Ce que « personne ne publie », corrigé après lecture complète de l'arXiv

**Première version de ce fichier disait « personne ne publie de jetons/kJ
absolu » — faux, corrigé ici plutôt que laissé tel quel.** `arXiv:2601.09527`
publie du Wh/MTok (403/325/239 pour bf16/W4A16/NVFP4), directement
convertible en jetons/kJ (689/855/1162). Ce n'est PAS qu'un pourcentage
relatif comme je l'avais écrit en ne lisant que le résumé.

**Ce qui reste vrai, reformulé plus précisément** : sur sept sources
examinées, **une seule** publie une énergie absolue et convertible, et son
régime (vLLM, concurrence 8, télémétrie DCGM GPU-seule) diffère du nôtre
(mono-flux, notre propre wattmètre). Le dépôt vLLM/RTX5090 donne des watts de
plafond de carte, jamais de jetons/kJ. Tout le reste : rien.

**Conséquence pour notre axe** : nous avons maintenant un premier point
d'ancrage externe (NVFP4 vLLM ≈ 1162 jetons/kJ, Qwen3-8B, concurrence 8) —
mais aucun dans notre régime précis. Nous ne pourrons établir « meilleur que
X en jetons/kJ » qu'en mesurant nous-mêmes les concurrents, avec notre
propre wattmètre et notre propre protocole — ce qu'on fait déjà pour
llama.cpp.

## Les trois chiffres les plus solides

1. **Le dépôt `adrienbrault/qwen3.8-27b-rtx5090`** — le plus proche de nos
   propres standards de rigueur (scripts nommés, dates, dossiers de résultats
   horodatés, fidélité mesurée séparément), mais son régime (32-64 flux,
   deux cartes) n'est pas le nôtre (un flux, une carte). Format et
   architecture comparables ; le nombre affiché ne l'est pas sans conversion.
2. **Le fil `llama.cpp` #15013** — la méthode (moyenne + écart-type publiés
   par `llama-bench`) est plus rigoureuse que notre propre protocole ce
   soir ; le modèle de référence (Llama 2 7B) est trop petit pour éclairer
   nos 14B-70B.
3. **L'article arXiv 2601.09527** — le seul à isoler l'effet du FORMAT
   (NVFP4 vs bf16) sur l'énergie en gardant le reste égal, exactement le
   type de comparaison qu'on cherche à faire nous-mêmes. Non entièrement
   vérifié faute d'avoir lu le corps du papier.

## Ce qui nous dépasse, à confirmer

Rien dans ce recensement n'établit avec certitude un chiffre **mono-flux,
une carte** qui dépasse ce qu'on a mesuré cette semaine — mais aucun des
concurrents examinés n'a été mesuré dans ce régime précis non plus, sauf le
fil `llama.cpp` sur un modèle trop petit pour trancher. **La bonne réponse
honnête est : je n'ai pas trouvé de chiffre public, mono-flux, sur 5090
seule, sur un modèle comparable en taille aux nôtres, avec un protocole
publié — ni au-dessus ni en dessous.** C'est en soi une information : soit
personne ne publie ce régime précis, soit il faudrait creuser plus large
(Reddit r/LocalLLaMA, forums Discord de vLLM/llama.cpp) que ce que permettait
cette recherche.

## Périmètre non couvert, par manque de temps de recherche

* Contenu complet de l'article arXiv (tableaux, méthode de mesure d'énergie
  exacte) — lu au résumé seul.
* Chiffre mono-flux exact du dépôt `adrienbrault` (mentionné dans les
  fichiers `docs/`, pas extrait).
* Reddit r/LocalLLaMA, forums NVIDIA développeur, Discord llama.cpp/vLLM —
  non fouillés, probable gisement de chiffres communautaires bruts.
* SGLang spécifiquement — trouvé seulement comme second serveur dans le
  dépôt `adrienbrault` (161,7 t/s sur son propre OpenAI server, même réserve
  de régime multi-flux que vLLM ci-dessus), aucune source SGLang-native
  indépendante trouvée.
