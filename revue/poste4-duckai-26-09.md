# Synthèse duck.ai — lot du 25-26/09 (chef, ordre utilisateur)

Trois modèles, mode raisonnement, `https://duck.ai/` par Chromium : GPT-5.6 Luna, gpt-oss 120B (recherche web),
Gemma 4 31B. Désaccords signalés explicitement.

## Q1 — RoPE : tables cos/sin en fp32 ou bf16 ? (contexte : 213, 1er préfill fp32 non arrondi puis bf16)

**Consensus des trois modèles** : le calcul des angles/fréquences se fait en fp32 (référence Llama :
`freqs_cis` en `complex64`, deux fp32) ; le point de divergence entre implémentations est SI/QUAND ce
résultat est ensuite casté vers le dtype du modèle (bf16).

* **HF Transformers** : le calcul est protégé de l'autocast (fp32), le résultat est ensuite casté
  `cos.to(x.dtype)` — donc bf16 si les activations le sont. Une régression documentée (issue HF
  `#29301`, citée par Luna et Gemma) a fait passer certaines versions au calcul direct des `freqs` en
  bf16 (pas juste le cast final), avec un effet mesuré sur les embeddings de position.
* **vLLM** — mécanisme le plus précis, donné par Luna (source `vllm.ai`, code réel) :
  `_compute_cos_sin_cache()` calcule tout en fp32 ; à l'exécution `_match_cos_sin_cache_dtype(query)`
  compare le dtype du cache à celui de la requête et **convertit + remplace le cache en bf16 dès le
  premier appel compatible**. **Ceci correspond exactement au symptôme de la pièce 213** (1er préfill
  fp32, puis bf16) : le cache est construit fp32 puis mémorisé bf16 après le premier passage.
* **SGLang** : Luna cite un rapport d'exécution montrant `cos_sin_cache.dtype = torch.float32` alors
  que Q/K sont bf16 — le cache resterait fp32 même après le premier passage, contrairement à vLLM.

**Désaccord réel entre les modèles** : gpt-oss et Gemma citent une issue vLLM (`#863`, « RoPE should be
applied with float32 ») comme une **demande** (feature request) que vLLM garde fp32 de bout en bout —
lue comme suggérant que vLLM ne le fait pas nativement. Luna, elle, décrit le comportement **actuel** du
code (`_match_cos_sin_cache_dtype`) qui caste bien vers bf16 après le 1er appel. Les deux ne se
contredisent pas forcément (l'issue peut dater d'avant le mécanisme actuel), mais aucun des trois n'a
croisé les deux sources entre elles — à vérifier au code vLLM installé si le doute compte pour nous.

Point complémentaire (gpt-oss + Gemma, source arXiv « Give Me FP32 or Give Me Death? ») : bf16 (7 bits
de mantisse contre 23 en fp32) peut provoquer des « token flips » quand deux logits sont proches —
sensible pour un modèle de raisonnement, moins pour un contexte court à marge confortable.

**Sources citées** : github.com (issues HF #29301, vLLM #863, SGLang `torch_native_llama.py`),
`vllm.model_executor.layers.rotary_embedding.base` (vllm.ai), arXiv « Give Me FP32 or Give Me Death? ».

## Q2 — cause connue d'un 1er lot différent des suivants (latence/valeurs), bonnes pratiques d'échauffement

**Consensus fort, aucun désaccord réel entre les trois.** Distinction faite par tous : latence (attendue,
normale) vs valeurs différentes (anormal, signale un vrai bug si le cache/contexte est correct).

Causes de latence, dans l'ordre de fréquence citée : (1) chargement paresseux des modules CUDA
(`cuModuleLoad*`, kernels FlashAttention/FlashInfer/NCCL chargés au 1er usage — `CUDA_MODULE_LOADING=EAGER`
déplace ce coût au démarrage) ; (2) création/allocation du handle et du workspace cuBLAS/cuBLASLt (la
taille de workspace influence même l'algorithme choisi par les heuristiques, pas seulement la mémoire) ;
(3) autotune Triton (`@triton.autotune`, coût par nouveau bucket de forme, pas seulement au 1er appel
processus) ; (4) `torch.compile`/Inductor (tracing+compilation au 1er passage, guards → recompilation sur
nouvelle forme) ; (5) capture des CUDA Graphs (vLLM : 1er forward d'un mode graphe non nul déclenche la
capture, après un warmup eager préalable — SGLang : formes de prefill plus dures à capturer que le decode ;
TensorRT-LLM : `trtexec` capture le graphe de l'enqueue par défaut, coût de capture à exclure du régime
steady-state).

Valeurs différentes (si observées) : PAS une conséquence normale du warmup — chercher plutôt un changement
de kernel/chemin numérique (accumulation FP16/BF16 au lieu de FP32, kernel avec atomics non reproductibles
selon cuBLAS), une différence de composition de batch (continuous batching : seul puis fusionné), ou un bug
de prefix/KV cache (mauvais offset, collision entre requêtes). Protocole de diagnostic donné par Luna :
mesurer séparément T0 (contexte CUDA) → T5 (steady-state), comparer `CUDA_MODULE_LOADING=EAGER` vs défaut,
CUDA Graph on/off, torch.compile on/off, prefix caching on/off — la nature de l'écart (latence seule,
latence+logits proches, logits francs différents) pointe vers une cause différente à chaque fois.

Bonnes pratiques d'échauffement : couvrir les FORMES réellement rencontrées (longueurs de prefill,
batch sizes de decode, MoE, spéculatif, multimodal), pas juste répéter un seul warmup batch=1 ; toujours
`torch.cuda.synchronize()` avant de chronométrer ; pour un CUDA Graph, distinguer warmup eager (avant
capture) et warmup des replays (après capture, régime différent) ; publier séparément cold-start latency
et warm steady-state latency.

**Sources** : docs NVIDIA (CUDA Programming Guide §4.8 lazy loading, cuBLAS/cuBLASLt docs, TensorRT
Performance Benchmarking), triton-lang.org (tutoriel matmul/autotune), vllm.ai (CUDA Graphs), lmsys.org
(SGLang CUDA Graph avancé), GitHub (vLLM benchmarking tutorial #7181).

## Q3 — MoE NVFP4 Blackwell : Marlin W4A16 contre W4A4 groupé, décodage b=1-16 et préfill

**Luna et gpt-oss (web) convergent, très nuancés** : le point de bascule dépend de M_e (tokens réellement
reçus PAR EXPERT après routage top-k), pas du batch global b. Décodage b=1-16, beaucoup d'experts →
M_e souvent ≤4-8 → tuile Tensor Core NVFP4 (M=128 chez vLLM/CUTLASS) très sous-remplie (6,25 % de lignes
utiles à M=8, 3,13 % à M=4) → **Marlin W4A16 gagne largement** dans ce régime, y compris sur Blackwell.
Crossover vers W4A4 seulement si M_e médian dépasse ~16-64 selon les sources. Préfill : petit préfill
(32-256 tokens) PAS GARANTI en faveur de W4A4 si le routage disperse encore les tokens par expert ou si
le chemin FP4 natif retombe en fallback Marlin (**issue GitHub RTX 5090/SM120 citée par Luna et gpt-oss** :
certaines configs ModelOpt mixtes NVFP4 sont dirigées vers Marlin W4A16 avec un avertissement ambigu, alors
que la carte sait faire du FP4 natif — donc un test « NVFP4 W4A4 » peut mesurer un fallback sans le savoir).
Préfill moyen/grand (M≥512) : W4A4 natif reprend l'avantage, plus nettement sur B200 (chemin sm_100 mieux
supporté) que sur RTX 5090 (sm_120, maturité logicielle vLLM/CUTLASS/FlashInfer variable) — un benchmark
RAG public cité montre même W4A16 battant NVFP4 sur RTX 5090 dans certains cas.

**Désaccord net avec Gemma** : Gemma donne une règle binaire et absolue (« décodage → toujours memory-bound
→ Marlin gagne » / « préfill → toujours compute-bound → W4A4 écrase »), sans mention de M_e par expert, sans
zone de crossover, et surtout **sans la réserve RTX 5090/fallback logiciel** que Luna et gpt-oss citent avec
source (GitHub) — réserve directement pertinente pour notre carte. La règle de Gemma serait trompeuse
appliquée telle quelle sur notre matériel : elle ne permettrait pas de détecter qu'un « test W4A4 » mesure
en fait un fallback Marlin.

**Retenu pour acvram** : mesurer M_e (médian/p95 après routage), pas seulement b, avant de choisir un
chemin MoE NVFP4 ; sur RTX 5090 vérifier explicitement qu'un chemin dit « W4A4 » n'est pas un fallback
Marlin silencieux.

**Sources** : arXiv (Marlin original, kernels NVFP4/occupation de tuile, "ReSET", "Private LLM Inference
on Consumer Blackwell GPUs"), nota.ai (NVFP4 Explained), Hugging Face (kernel engineering FP4 Blackwell,
benchmark GPT-OSS b=1 vs grand batch), vllm.ai (WideEP/serving large-scale), GitHub (issue RTX5090/SM120
ModelOpt mixed checkpoint → fallback Marlin).

## Q4 — pièges des bancs MoE en génération libre à ids aléatoires ; ce que font vLLM/SGLang

**Consensus fort, confirme directement notre propre constat (piège déjà vécu : 209/217, régression −8,9 %
révélée comme un artefact du banc à ids aléatoires).** Les trois modèles (Luna en détail, gpt-oss et Gemma
convergents sur les mêmes risques) :

* **Sorties dégénérées** : des ids uniformes forment une séquence hors distribution d'entraînement —
  probabilité d'EOS anormale, boucles/répétitions, longueur de sortie très différente de `max_tokens` ;
  le débit mesuré peut refléter un arrêt précoce artificiel, pas un comportement représentatif.
* **Routage MoE non représentatif** : le routeur voit les états cachés (pas les ids bruts) — une entrée hors
  distribution change tout l'histogramme d'experts, peut sur/sous-charger certains experts, changer la
  taille des GEMM par expert et le coût d'all-to-all/EP. Un débit global identique peut cacher des profils
  de routage très différents — **Luna recommande de logger tokens/expert (moyenne, p95, max) systématiquement,
  pas seulement tokens/s**.
* **Divergence entre bras comparés** dès que le premier token généré diffère (génération libre, pas teacher
  forcing) : chemin numérique, cache de préfixe, tokenizer, template, arrêt précoce — toute la trajectoire
  diverge ensuite, y compris le routage. Pour une comparaison contrôlée : mêmes ids en entrée, greedy ou
  sampling identique, longueur de sortie fixée, `ignore_eos` si on veut comparer un coût à N tokens fixé.

**Ce que font les bancs publics** : vLLM `benchmark_serving`/`vllm bench serve` expose `random` (ids
synthétiques, longueurs contrôlées — **reste explicitement un test de charge, pas un proxy de texte
naturel**, selon la doc vLLM elle-même) ET des jeux réalistes (ShareGPT, Sonnet, HF, traces) ; option
`--prompt-token-ids` pour transporter les ids exacts sans re-tokeniser (ne règle pas le problème de
distribution). SGLang `bench_serving --dataset-name random` : selon version, peut échantillonner des
tokens RÉELS depuis ShareGPT puis tronquer/répéter (cas intermédiaire, distribution lexicale plus proche
du naturel qu'un tirage uniforme) — SGLang est en plus sensible aux préfixes partagés (cache radix), absent
avec des ids purement aléatoires.

**Recommandation retenue (Luna)** : publier au moins deux chiffres — débit à ids synthétiques (stress
système/capacité) ET débit à texte réel (routage/état caché réalistes) — jamais un seul, et toujours
accompagné des statistiques de charge par expert.

**Sources** : vllm.ai (`vllm bench serve`, `vllm.benchmarks.datasets`), GitHub (issue vLLM #4133 « Random
dataset in benchmark_serving is not random », vllm-bench, discussion SGLang), arXiv (« Zero Redundancy
Overheads in MoE Prefill Serving », déséquilibre expert mesuré), alphaxiv.org (SPEED-Bench).

## Q5 — GEMV int8 petit M (≤80) sur RTX 5090 : tranche en indice de grille vs boucle hôte de lancements

**Consensus Luna/gpt-oss, pas de désaccord matériel, mais Luna est franche sur l'absence de source
définitive** : « il n'existe pas, à ma connaissance, de benchmark public de référence comparant précisément
ces deux mappings sur RTX 5090 pour un GEMV INT8 M≤80 » — la réponse reste un raisonnement de premiers
principes + specs matérielles, pas une mesure publiée à citer telle quelle.

**Conclusion des deux** : pour M≤80, la boucle hôte de petits lancements a presque toujours un mauvais
profil (coût de lancement ~5 µs répété × S lancements, réduit encore le nombre de blocs déjà faible) — un
noyau fusionné unique, tranche en dimension de grille, est le meilleur point de départ. **Nuance
importante des deux** : M≤80 seul ne remplit de toute façon pas la RTX 5090 (170 SM, gpt-oss ; Luna:
"environ 170 SM actifs") — un bloc par ligne ne donne au mieux que 80 blocs actifs sur 170. **La vraie
question d'occupation est donc la dimension supplémentaire (batch/requête) à exposer dans la grille**,
pas seulement fusionné-vs-boucle : `grid = (ligne, tranche, requête)` plutôt que juste `(tranche)`.

Sur le cache L2 (~96-98,3 Mio) : **les deux mettent en garde contre l'hypothèse naïve `|W| ≤ C_L2 ⇒ résidence
garantie`** — la règle utile est `|W réutilisé| ≪ C_L2`, pas juste `|W| ≤ C_L2` (éviction par le vecteur x,
les échelles, les écritures de sortie, autres kernels). gpt-oss ajoute un point concret non mentionné par
Luna : `cudaAccessPropertyPersisting` pour forcer explicitement la résidence L2 d'une région — piste à
tester si la résidence naturelle est insuffisante.

Point de divergence mineur (pas un désaccord de fond) : Luna privilégie mapper M (les lignes/tranche) à la
grille avec une dimension batch/requête en plus ; gpt-oss suggère aussi d'envisager de mapper K (les
colonnes) plutôt que M si M reste trop petit pour remplir les SM — piste alternative à tester, pas
contradictoire.

**Retenu pour acvram** : ne jamais juger l'occupation sur M seul — vérifier `sm__ctas_active`/
`sm__warps_active` (Nsight Compute) et la présence d'une dimension batch/requête dans la grille avant de
conclure qu'un GEMV M≤80 est bien occupé ; tester `cudaAccessPropertyPersisting` si la résidence L2 des
poids est incertaine.

**Sources** : docs NVIDIA (specs RTX 5090/GB202, kernel launch latency, profilage L2 hit-rate/bande
passante), christianjmills.com (GPU MODE Lecture 8, checklist performance CUDA), spheron.network (specs
RTX 5090). Aucun benchmark public spécifique à ce mapping précis (M≤80, GEMV int8, RTX 5090) trouvé par
aucun des deux modèles — à valider nous-mêmes par microbenchmark/Nsight, pas par transposition d'un
résultat GEMM/LLM générique.

## Q6 — la mémoire du pool de graphes CUDA est-elle comptée dans la planification KV (vLLM/SGLang) ?

**Validation directe et généralisée de notre propre pièce 212** (le pool de graphes CUDA d'acvram dépassait
la marge fixe de 1 536 Mio, jusqu'à +1 198 Mio sur le mixte — jamais compté par conception, seulement
absorbé par accident). Les trois modèles confirment que c'est un coût réel et significatif ; **désaccord
net entre Luna (sourcée au code réel) et Gemma sur l'ORDRE DES OPÉRATIONS chez SGLang**, désaccord qui
compte pour trancher notre propre correctif.

**vLLM V1 (Luna, code cité `vllm/v1/worker/gpu_worker.py`)** : le pool CUDA Graph EST déduit dans le chemin
d'allocation réel — `available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory -
cudagraph_memory_estimate_applied`, une fonction `profile_cudagraph_memory()` existe explicitement dans le
profiling V1. **Mais une régression documentée** (issue GitHub citée) a fait DOUBLE-COMPTER cette pool dans
le calcul de la valeur *suggérée* `--kv-cache-memory` (bug distinct du chemin d'allocation réel, qui restait
correct) — leçon : le calcul affiché/suggéré et le calcul réellement appliqué peuvent diverger, à vérifier
séparément.

**SGLang (Luna, issue GitHub citée avec chiffres concrets)** : **le budget KV est calculé À PARTIR de
`available_gpu_memory`/`mem_fraction_static` AVANT la capture des graphes CUDA** ; l'ordre réel est
poids → profiling+budget KV → allocation des pools KV → **capture des graphes CUDA (ensuite)** → pool
résidente conservée toute la durée de vie du serveur. Un cas documenté : 51 formes de préfill capturées (4096
→ 4 jetons) retiennent **~1,81 Gio résidents**, désactiver le graphe de préfill libère cette VRAM (0,54 →
2,35 Gio libres après démarrage) — **« aucune règle automatique ne désactive le graphe en fonction de la
VRAM libre restante »** (citation directe du rapport SGLang). C'est EXACTEMENT le défaut que la pièce 212
a débusqué chez nous (mécanisme différent, même classe de bug : un coût réel après le calcul de marge,
jamais compté par conception).

**Désaccord avec Gemma** : Gemma affirme que vLLM ET SGLang suivent le MÊME ordre (« poids → capture CUDA
Graph → vérifier VRAM libre → allocation KV cache ») pour LES DEUX moteurs — c'est-à-dire que la capture
aurait lieu AVANT l'allocation KV dans SGLang aussi, ce qui **contredit directement** le rapport de bug
GitHub cité par Luna (capture APRÈS le calcul du budget KV chez SGLang, donc le nombre de tokens KV annoncé
n'a jamais pu tenir compte du pool). Gemma ne cite aucun code ni issue pour SGLang, contrairement à Luna —
son affirmation est probablement une généralisation erronée du comportement vLLM appliquée à SGLang sans
vérification. **À retenir : ne pas faire confiance à l'affirmation de Gemma sur l'ordre SGLang sans
revérifier au code SGLang installé.**

**Retenu pour acvram** : notre correctif 212 (marge GDN à 3 072 Mio, mesurée par famille comme poste5 l'a
fait pour la 172) est cohérent avec la pratique reconnue comme la plus sûre par Luna : mesurer la pool
résidente APRÈS toutes les captures (`max_t(pool résidente(t))`), jamais seulement le workspace du
`profile_run()` initial — et comparer VRAM libre avec/sans graphes activés est la mesure de contrôle la
plus fiable, méthode que nous pourrions reproduire (équivalent d'un `--disable-prefill-cuda-graph` chez
nous) pour vérifier que 3 072 Mio couvre bien tous nos modèles GDN dans le temps, pas seulement les 5
mesurés en 212.

**Sources** : GitHub (issue vLLM double-comptage `--kv-cache-memory`/CUDAGraph, `vllm/v1/worker/gpu/
model_runner.py`, issue SGLang « Prefill CUDA graph reserves ~1.8 GB and starves… »), lmsys.org (SGLang
Advanced CUDA Graph Techniques).

## Q7 — bonnes pratiques README d'un moteur d'inférence open source (modèle : animematrix)

**Consensus fort Luna/Gemma sur la structure** : logo/bandeau centré sobre → badges (CI, version, licence,
docs, Docker) → proposition de valeur en 3-4 lignes → tableau de compatibilité (modèles/formats/matériel) →
**benchmarks chiffrés placés haut, jamais un chiffre isolé** → Quickstart exécutable en <5 min (install →
lancement → premier appel curl/API) → support matériel avec statuts explicites (Stable/Experimental/Planned/
Not supported) → fonctionnalités (5-8 max, chacune reliée à une doc/exemple) → exemples d'usage → architecture
(diagramme simple préféré à la prose) → dépannage/FAQ → contribuer/licence/crédits en fin.

**Ce qu'il faut transformer du modèle animematrix (grand public) pour un moteur technique** :

| Élément du modèle | Pour acvram |
|---|---|
| Logo centré | conserver, sobre |
| Badges | conserver (CI, release, licence, docs) |
| Barre de langues à drapeaux | Luna : remplacer par liens texte (`[English](README.md) · [Français](README.fr.md)`) — Gemma : garder une ligne drapeaux discrète mais toujours avec lien texte vers le fichier. Les deux d'accord : jamais drapeaux SEULS comme seule navigation |
| Visuel principal (photo/rendu) | remplacer par diagramme d'architecture, capture de terminal fonctionnelle, ou graphique de débit — jamais un visuel esthétique seul |
| Sommaire à ancres | conserver si le README dépasse ~300 lignes |
| Séparateurs `---` | Luna : à utiliser peu (les titres Markdown suffisent) — Gemma : à conserver pour segmenter doc rapide/approfondie. Différence de goût mineure, pas un désaccord de fond |
| Tableaux | très utiles : benchmarks, compatibilité, configuration — TOUJOURS avec conditions du test (GPU, contexte, concurrence, commande de reproduction) sous le tableau |
| Section « soutenir » | remplacer par « Contributing »/« Sponsor » |
| Ton promotionnel | remplacer par affirmations mesurables et vérifiables (jamais « the fastest ever » sans protocole) |

**Benchmarks — point le plus détaillé (Luna)** : distinguer systématiquement TTFT, ITL, throughput
(tok/s ou req/s), latence p50/p99, taux d'erreur, consommation mémoire — un chiffre isolé type « 200 tok/s »
est ambigu (batch ? préfill ou génération ?). Documenter sous CHAQUE tableau : commit/version testée, modèle
et révision exacte, format/quantification, GPU/CPU/RAM/CUDA, longueur prompt, concurrence, commande exacte
de reproduction.

**Nuance/désaccord mineur** : Gemma recommande d'inclure une ligne de comparaison à un concurrent (vLLM,
TGI) dans le tableau de benchmarks pour un point de référence immédiat ; Luna ne le mentionne pas
explicitement (mais rien n'exclut cette pratique dans son cadre). Pas contradictoire — à évaluer selon si
acvram veut se positionner contre vLLM/TensorRT-LLM dans son propre README (ce que le dépôt fait déjà en
partie, cf. README actuel section « Résultats mesurés »).

**Retenu pour acvram** : structure Luna en 12 sections (Overview → Performance → Quickstart → Installation
→ Supported Models → Supported Hardware → Features → Examples → Architecture → Documentation →
Troubleshooting → Contributing → License), conserver logo centré + sommaire à ancres du modèle animematrix
mais remplacer le visuel principal esthétique par diagramme/terminal, et la barre de langues par des liens
texte explicites vers `README.fr.md` etc. (README.md anglais = référence).

**Sources** : vllm.ai, GitHub (llama.cpp, SGLang, ExLlamaV2 — noté archivé, développement continué dans
ExLlamaV3, à ne reprendre que pour la présentation des résultats).

---

# Lot 2 (26/09, chef — questions 8 à 12, releases GitHub AUR/COPR/Flathub/Weblate, pièces 236a/236b/236c/213b/235)

## Q8 — Flathub et CUDA : embarquer torch cu12x ou télécharger post-install ?

**Consensus fort Luna/Gemma** : embarquer la roue torch cu12x dans le manifeste flatpak (limiter à
`only-arches: x86_64` si pas de roue aarch64) ; `org.freedesktop.Platform.GL.nvidia` sert UNIQUEMENT à
exposer les bibliothèques du pilote NVIDIA de l'hôte dans le sandbox — **ce n'est PAS un fournisseur de
runtime CUDA ni un précédent pour PyTorch**, les deux modèles insistent sur cette distinction. Le pilote
NVIDIA complet, `libcuda.so`, le module noyau et un toolkit CUDA système complet ne doivent jamais être
embarqués — seule la roue torch (qui embarque déjà cuBLAS/cuDNN/cuFFT/NCCL etc.) suffit, avec un pilote
hôte compatible. Téléchargement post-install déconseillé pour la dépendance fondamentale (non
reproductible, réseau requis, taille invisible côté centre logiciel) — acceptable seulement pour des
modèles/checkpoints optionnels dans `~/.var/app/<appid>/data/`.

**Précédent concret cité (Luna)** : `io.github.tntwise.REAL-Video-EnhancerV2` télécharge PyTorch après
installation (>5 Gio), avec un bug documenté « no space left on device » malgré de l'espace disponible —
preuve que ce modèle marche mais reste fragile.

**Taille** : aucune limite officielle simple trouvée par aucun des deux modèles — seule limite documentée
et citée par Luna : 25 Mio pour le **dépôt Git du manifeste** (pas l'app construite). Seuils pratiques
donnés par Luna : <1 Gio confortable, 1-2 Gio défendable, 2-4 Gio réaliste mais à justifier, >5 Gio
possible mais mauvaise UX. Gemma ajoute une donnée non sourcée précisément : des bundles >10 Gio « peuvent
être signalés lors de la revue Flathub » — à vérifier soi-même avant de s'y fier telle quelle.

**Nuance de Gemma absente chez Luna** : beaucoup d'applications ML grand public préfèrent Docker/Distrobox
à Flatpak pour CUDA à cause du problème de compatibilité pilote/toolkit (source citée : Red Hat) — piste
alternative à considérer si Flathub s'avère trop contraignant pour acvram-gui.

**Retenu** : embarquer torch cu12x dans le manifeste (une seule variante CUDA, `only-arches: x86_64`),
`GL.nvidia` en extension GL classique, `torch.cuda.is_available()` vérifié au lancement avec message
explicite si absent, modèles optionnels téléchargés à part et annoncés en taille.

**Sources** : flatpak.org (docs Extensions), igalia.com (Digging further into Flatpak with NVIDIA),
flathub.org (docs Maintenance, fil 2022 packaging PyTorch), pytorch.org (Get Started, TORCH_CUDA_ARCH_LIST),
GitHub (issue REAL-Video-EnhancerV2 « no space left on device »).

## Q9 — AUR : dépendre de python-pytorch-cuda, ou venv/pip cu12x dans le paquet ?

**Consensus fort Luna/gpt-oss, aucun désaccord** : `depends=('python-pytorch-cuda' ...)` est la pratique
communautaire normale pour un paquet AUR intégré au système Arch. python-pytorch-cuda `Provides:
python-pytorch`, entre en conflit avec la variante CPU, dépendances déjà cohérentes (cuda, cudnn, nccl).
**Installer un venv + `pip install torch` pendant `prepare()`/`package()` est déconseillé** : pacman ne
connaît alors ni les fichiers ni les dépendances réelles, mises à jour torch/numpy/triton découplées des
mises à jour Arch, doublonnage des runtimes CUDA déjà empaquetés, build dépendant du réseau (contraire aux
règles makepkg/clean chroot), venv cassable par une mise à jour Python système. Règle générale citée
(discussions ArchWiki) : les dépendances Python doivent être exprimées dans `depends` du PKGBUILD via des
paquets pacman/AUR, pas dans un requirements.txt installé par pip.

**Exception légitime au venv** (les deux modèles d'accord) : seulement si l'application exige une version
précise/un index CUDA (cu124/cu126) absent des dépôts Arch — dans ce cas, le venv doit être créé HORS du
PKGBUILD (au premier lancement, dans `~/.local/share/...`), jamais silencieusement embarqué par le paquet,
et le PKGBUILD n'empaquette alors que le lanceur.

**Retenu pour acvram** : `depends=('python-pytorch-cuda' ...)` en pratique par défaut ; si acvram a besoin
d'une version torch/CUDA précise non fournie par Arch, documenter explicitement le modèle « environnement
Python privé » plutôt que de le déguiser en paquet Arch classique intégré.

**Sources** : archlinux.org (fiche python-pytorch-cuda, discussions forum sur les Python package
guidelines, template PKGBUILD wheel-only), manjaro.org (PKGBUILD UnstableFusion), pytorch.org.

## Q10 — COPR/Fedora : pas de torch CUDA empaqueté, quelle stratégie ?

**Consensus total Luna/Gemma, aucun désaccord.** Classement identique des trois options :

1. **RPM de l'application + venv CUDA documenté (utilisateur)** — recommandé, seule solution vraiment
   conforme à l'esprit Fedora pour une pile IA.
2. RPM tiers reconditionnant les wheels — possible techniquement dans un COPR indépendant (COPR n'exige
   PAS le respect des Fedora Packaging Guidelines, seulement les droits sur le contenu et des licences
   acceptables), mais PAS un paquet Fedora conforme par défaut : audit de licence par composant embarqué
   (CUDA/cuDNN/NCCL peuvent avoir des conditions de redistribution particulières), gestion CVE
   indépendante, ABI/architecture/version Python/variante CUDA multiplient les builds — à étiqueter
   explicitement comme paquet tiers expérimental si fait, jamais présenté comme du Fedora standard.
3. **`pip install torch` dans un scriptlet `%post` : interdit dans les faits** — hors du contenu
   contrôlé/vérifiable du paquet, nécessite réseau à l'installation (contraire à l'esprit des règles
   Fedora sur les scriptlets « sains », citées par Luna), casse dnf/résolution de dépendances, échoue en
   offline/images minimales/installations transactionnelles.

**Règle Fedora citée par les deux** : éviter les bibliothèques « bundlées » — un binaire précompilé ne doit
pas être simplement copié dans `%{buildroot}` sans justification ; les revues de paquets vérifient
explicitement l'absence de bibliothèques embarquées sans exception du Fedora Packaging Committee.

**Retenu pour acvram** : RPM COPR = code de l'application + dépendances Python standard des dépôts Fedora
uniquement ; torch CUDA en venv documenté (commande pip + index cu12x explicite dans la doc), détection
`torch.cuda.is_available()` au démarrage avec message clair si absent, jamais de téléchargement silencieux
dans `%post`.

**Sources** : fedoraproject.org (Packaging Guidelines wiki), copr.fedorainfracloud.org (User Documentation),
discussion.fedoraproject.org (fil « PyTorch to Fedora Introduction »), lwn.net (vendoring/bundled
libraries), pytorch.org.

## Q11 — Weblate : JSON monolingue + README Markdown dans le même projet

**Consensus total Luna/Gemma.** Un projet Weblate, deux composants distincts (JSON et Markdown ne
partagent jamais un composant) :

* **Interface** : `Format: i18next JSON file v4` (ou v3 selon la version de pluriels côté app),
  `Base file: locales/_source.json`, `File mask: locales/*.json`. Format générique JSON acceptable si
  `_source.json` est un simple objet clé-valeur sans particularité i18next, mais i18next JSON est plus
  cohérent (interpolations et pluriels CLDR gérés nativement).
* **README** : **il existe un format Markdown dédié** (`Format: Markdown file`) — ne jamais utiliser texte
  libre/JSON générique pour du Markdown. `File mask: README*.md` ou `docs/**/*.md`, `Base file:
  README.md`. Monolingue : Weblate extrait le contenu traduisible (titres, paragraphes, liens) en
  préservant la syntaxe structurelle (`#`, `**`, `[]()`), segmente par bloc/paragraphe.

**Points opérationnels critiques (Luna, non mentionnés par Gemma)** :
* Pour du Markdown avec syntaxe JSX, utiliser `MDX file`, pas `Markdown file`.
* **Weblate devient la source de vérité des traductions Markdown** : modifier directement un
  `README.fr.md` dans le dépôt risque d'être ignoré à la prochaine synchro — flux Git à prévoir en
  conséquence (export/génération depuis Weblate, pas d'édition manuelle des fichiers traduits).
* Chaînes identiques répétées (tableaux Markdown) : par défaut chaque occurrence = unité distincte
  (contexte ligne conservé) ; `markdown_merge_duplicates=True` regroupe au prix de ce contexte.
* Front matter YAML pris en charge, avec option pour traduire séparément les valeurs scalaires en
  préservant clés/structure.

**Retenu pour acvram** : projet Weblate unique, composant JSON i18next v4 pour l'interface + composant
Markdown dédié pour les README (masque à ajuster selon la convention `README.xx.md` déjà en place dans le
dépôt, cf. REGLES §langue) — vérifier le comportement `markdown_merge_duplicates` sur nos tableaux de
benchmarks avant de l'activer.

**Sources** : weblate.org (docs i18next JSON files, Markdown files — version 2026.9, Components,
Supported Formats).

## Q12 — RoPE : cache cos/sin à dtype fixe par module, ou indexé par dtype ?

**Réponse de fond (Luna, code réel cité) : ni un dtype fixe immuable, ni un dict[dtype] — un cache
mutable unique qui se recaste et se REMPLACE à chaque changement de dtype demandé, avec UNE exception
notable.**

* **vLLM** (`vllm/model_executor/layers/rotary_embedding/base.py`, réel) : `RotaryEmbeddingBase.__init__`
  calcule `cos_sin_cache` une fois, casté au dtype du module. À l'exécution,
  `_match_cos_sin_cache_dtype(query)` compare le dtype du cache à celui de `query` : s'ils diffèrent, le
  cache est casté vers `query.dtype` **et remplace le buffer principal** (`self.cos_sin_cache = ...`) —
  donc si l'usage alterne fp32/bf16 d'un appel à l'autre, **le cache est recasté et remplacé à chaque
  appel** (ping-pong), jamais deux versions gardées en mémoire simultanément dans le cas général.
  **Exception explicite documentée dans le code** : le chemin **AITER compilé** garde un second buffer
  persistant `cos_sin_cache_bf16` en PARALLÈLE du cache principal (`register_buffer`, jamais écrasé) —
  c'est la seule vraie forme de « cache indexé par dtype » (2 entrées fixes, pas un dict général). Le
  chemin **FlashInfer** est un cas à part : il consomme directement `self.cos_sin_cache` SANS passer par
  `_match_cos_sin_cache_dtype` — suppose que le dtype stocké est déjà compatible avec son propre contrat.
* **HF Transformers actuel** (`LlamaRotaryEmbedding`) : **aucun cache persistant** — les fréquences sont
  RECALCULÉES en fp32 à CHAQUE appel (`inv_freq_expanded @ position_ids_expanded`, sous
  `maybe_autocast(enabled=False)`), puis `cos.to(x.dtype)`/`sin.to(x.dtype)` à la fin seulement. Donc un
  appelant fp32 et un appelant bf16 obtiennent chacun un calcul frais, jamais de cache à invalider/remplacer.
* **Anciennes versions HF** : `_cos_cached`/`_sin_cached`, reconstruits (pas indexés) si `dtype`/`device`/
  `seq_len` changent — même famille de bug que vLLM standard (ping-pong), documenté dans une issue HF
  citée (cache reconstruit à précision insuffisante si le module était converti en bf16/fp16 avant
  l'extension de longueur).

**Désaccord net avec gpt-oss** : gpt-oss (répondu SANS recherche web cette fois, raisonnement à vide)
affirme catégoriquement que « vLLM ne maintient PAS de cache multi-dtype » — **contredit directement par
le code réel cité par Luna** (l'exception AITER `cos_sin_cache_bf16`). gpt-oss cite aussi un chemin de
fichier obsolète (`rotary_embedding.py` en fichier unique, plus la structure actuelle en sous-module) et
navigue à vue sans jamais confirmer par une source — **à ignorer sur ce point précis, faire confiance au
code cité par Luna**.

**Ce que ça donne pour la 213b/235 (arbitrage RoPE)** : si acvram fait alterner un chemin fusionné fp32
et un repli eager bf16 sur le MÊME module RoPE, le comportement par défaut (hors AITER) est un
cast-et-remplacement à CHAQUE appel — un coût caché (recast) à chaque bascule, PAS une conservation à la
demande. Si cette alternance est fréquente dans notre pipeline, envisager le motif AITER (second buffer
persistant nommé) plutôt que de compter sur `_match_cos_sin_cache_dtype`-like pour éviter le ping-pong.

**Sources** : GitHub (code réel `vllm/model_executor/layers/rotary_embedding/base.py`,
`modeling_llama.py` transformers actuel, ancienne implémentation `_cos_cached`, issue HF citée sur le
cache recasté bf16/fp16).

---

# Lot 3 (26/09, chef — questions 13 à 15, pièces 209/213b/235, verdict 221 d'poste1)

## Q13 — Marlin MoE nvfp4 échelles par ligne : inversion de signe salve (+12,8%) vs soutenu (−15,3%, −25% J/jeton)

**Consensus fort Luna/gpt-oss** : le bridage puissance/thermique est l'explication « la plus directe »
d'une inversion de signe après quelques secondes (les deux modèles le disent en ces termes, cohérent avec
la mémoire du groupe : « les invalidations de poste3 sont toutes dues au bridage »). Mécanisme : la salve
tourne au boost clock max (transitoire) ; le régime soutenu chauffe la carte, maintient la consommation
près de la limite, le pilote réduit la fréquence SM — **si le noyau row-wise est plus gourmand en
puissance par FLOP que la référence** (accès mémoire différents, plus d'instructions), il déclenche le
throttling plus tôt/plus fort, inversant un gain brut en perte nette. gpt-oss ajoute cette nuance
(spécifique au coût énergétique du noyau lui-même) sans la contredire.

**Quatre hypothèses, chacune avec sa signature distincte (tableau de Luna)** :

| Hypothèse | Signature salve | Signature soutenu (20s) | Mesure discriminante |
|---|---|---|---|
| Pression L2/DRAM (échelles row-wise) | échelles/poids encore en L2, coût faible | hit rate L2 baisse progressivement, secteurs L2/DRAM et latence montent, noyau devient memory-bound | `lts__t_sector_hit_rate.pct`, secteurs L2, `dram__bytes_read/write.sum`, sections NCU Memory Workload Analysis/Speed of Light/Roofline |
| Bridage puissance/thermique | boost clock élevé, transitoire | fréquence SM moyenne baisse, puissance proche limite, température/throttle reasons montent | nsys `--gpu-metrics-device` + `nvidia-smi dmon -s pucm -d 1`, `nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE` ; raisons NVML `SwPowerCap`/`SwThermalSlowdown`/`HwThermalSlowdown`/`HwSlowdown`/`HwPowerBrakeSlowdown` |
| Alternance prefill/decode (M variable) | salve dominée par une forme GEMM favorable | agrège plusieurs formes, petit M (decode) où chargement des échelles/lancement dominent | marqueurs NVTX par phase prefill/decode, NCU séparément à M grand/moyen/petit, `débit(M) = tokens/temps_à_M` |
| Warmup/premier passage | cache froid, compilation, allocations | ne devrait PAS expliquer une perte persistante après exclusion du warmup | comparer fenêtres initiale/médiane/finale sur la même mesure |

**Protocole discriminant recommandé (Luna)** : 4 runs NCU ciblés (référence/row-wise × cache froid/workload
répété, à petit M) avec `SpeedOfLight_RooflineChart`, `MemoryWorkloadAnalysis`, `LaunchStats`, `Occupancy`,
`SchedulerStats` + compteurs `lts__t_sector_hit_rate.pct`, `dram__throughput.avg.pct_of_peak_sustained_active`,
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`. Interprétation : fréquence SM qui baisse +
throttle reason = bridage ; fréquence stable + hits L2 qui chutent + octets DRAM qui montent = pression
mémoire ; perte concentrée à petit M = effet prefill/decode ; seules les premières fenêtres diffèrent =
warmup. **Les effets peuvent se cumuler** (plus de trafic mémoire → plus de puissance → throttling
déclenché plus tôt) — ne pas chercher une cause unique si plusieurs signatures apparaissent ensemble.

**Point méthodologique NCU signalé par Luna** : NCU peut rejouer le kernel plusieurs fois pour ses
métriques, ce qui peut vider ou modifier l'état du cache — utiliser `--cache-control none` et le replay
d'application si l'état persistant du cache fait partie de ce qu'on mesure.

**Sources** : docs NVIDIA (Nsight Compute Profiling Guide, Nsight Systems, NVML Clocks Event Reasons API
Reference), arXiv (« Measuring GPU utilization one level deeper »).

## Q14 — GEMV int8 (n=78-80) : seuil de bascule vers GEMM tensor cores (verdict 221 d'poste1)

**Pas de seuil universel** (Luna, explicite) : dépend de N, K, format des échelles, tuile disponible —
**attention à ne pas transposer un seuil SM90/H100 sur RTX 5090 (SM120)**, la plupart des chemins int8
documentés (vLLM CUTLASS notamment) ciblent SM80/89/90.

**Ordres de grandeur convergents (Luna + littérature citée)** : M=1-16 → GEMV CUDA/DP4A ; M≈16-32 → zone
de transition (comparer GEMV, split-K, petite GEMM tensor core) ; **M≈32-64 → la GEMM int8 MMA devient
généralement préférable** ; M≥64 → tenter la GEMM tensor core en premier. **À M=78-80, ne plus choisir un
GEMV par défaut** — tester une petite GEMM int8/MMA (M arrondi à une tuile de 128, ou permutation A↔B).
La raison n'est PAS la bande passante DRAM (poids déjà en L2, confirmé par la question) mais le manque de
parallélisme dans la dimension M pour un GEMV, alors qu'une GEMM réutilise mieux les activations et
expose plus de travail aux tensor cores.

**Code réel cité (vLLM)** : `scaled_mm_sm90_int8_dispatch.cuh` route par buckets `M∈[1,32]` /
`(32,64]` / `(64,128]` / `>128` vers des configs CUTLASS différentes (M32_NSmall/NBig, M64, M128,
défaut) — **M=80 tombe dans le bucket (64,128], config M128** — mais c'est un choix de TAILLE DE TUILE
GEMM, pas un choix GEMV-vs-GEMM (vLLM ne fait pas ce choix explicitement à ce niveau). **Point critique
pour nous** : ce code est spécifique SM90, et la doc vLLM actuelle signale une limitation INT8 sur
Blackwell, recommandant **FP8 plutôt qu'INT8 pour compute capability ≥10.0** — à vérifier si ça nous
concerne.

**SGLang/GemLite** : politique plus explicite documentée — batch=1 → GEMV ; 2-64 → GEMM split-K (matrices
« skinny »), retombant en GEMM classique `SPLIT_K=1` « à partir de 32 ou 64 selon la forme/le device » ;
>64 → GEMM généralement privilégiée.

**TensorRT-LLM** : aucun seuil global fixe documenté publiquement pour un GEMV W8A8 per-channel — seule
une mesure expérimentale publiée (H800, comparaison W4A16/FP8 sur Mixtral, pas notre cas) montre un
crossover autour de M≈32.

**Retenu pour acvram** : tester nous-mêmes M∈{1,2,4,8,16,24,32,40,48,64,80,96,128} avec GEMV CUDA dp4a
vs GEMM CUTLASS/cuBLASLt (accumulation int32) vs GEMM transposée (swap A/B) vs split-K — aucune des
sources ne donne de seuil validé sur RTX 5090/SM120 pour notre cas précis (int8 per-channel), seulement
des ordres de grandeur 32-64 à vérifier empiriquement chez nous ; vérifier aussi si le conseil « FP8 au
lieu d'INT8 sur Blackwell » de vLLM s'applique à notre chemin.

**Sources** : GitHub (`scaled_mm_sm90_int8_dispatch.cuh` réel), vllm.ai (doc INT8 W8A8, limitation
Blackwell), pytorch.org (GemLite/TorchAO/SGLang), arXiv (mesure Mixtral H800), GitHub (discussion
TensorRT-LLM CUDA graph batch sizes).

## Q15 — invites de texte répété biaisent-elles le routage MoE (Qwen3-Coder-30B-A3B) ?

**Réponse directe, papier réel trouvé et directement pertinent (Luna)** : « RepetitionCurse: Measuring
and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress » (arXiv) — mesure
exactement ce biais sur **Mixtral-8x7B ET Qwen3-30B-A3B** (même famille que notre Qwen3-Coder-30B-A3B),
longueurs 100 à 16k jetons. Invites quasi-monotones → entropie de routage basse et stable, déséquilibre
>90% pour certains experts (concentré dans les couches intermédiaires, moins dans les premières/dernières
couches). **Sur Qwen3-30B-A3B précisément : facteur de latence ×2,14 mesuré sur leur déploiement** —
confirme que le biais atteint le débit/latence système, pas seulement l'histogramme de routage.

**Nuance importante pour notre protocole (256 jetons, texte répété pas token unique)** : le papier utilise
des invites quasi-identiques token par token ; un passage de plusieurs jetons répété est moins extrême
(le modèle est causal, les états cachés dépendent du contexte précédent, les embeddings positionnels
cassent une partie de la symétrie) — **reste potentiellement biaisé mais pas garanti aussi sévère que
le cas extrême du papier**. Mesures recommandées par couche (pas seulement en moyenne, le déséquilibre
peut être localisé) : entropie normalisée `H_l = -Σ p_l,e log(p_l,e) / log(E_l)` et coefficient de
variation `CV_l = std_e(n_l,e) / mean_e(n_l,e)` — un prompt répété donne typiquement H plus bas et CV
plus élevé qu'un prompt naturel de même longueur.

**Effet dépend du matériel** : TP seul (pas d'EP) → déséquilibre modéré si les kernels sont bien groupés ;
**EP multi-GPU → effet plus sérieux** (le GPU le plus chargé détermine le temps de la couche, straggler) ;
petit batch decode → le hasard de quelques tokens domine ; gros préfill → la répétition sature
systématiquement les mêmes experts.

**Ce que font les bancs publics (aucune correction MoE spécifique appliquée par défaut)** :
* **vLLM** `vllm bench serve` : `sharegpt` (conversations réelles, référence la plus défendable),
  `sonnet` (longueur contrôlée mais contenu lexical varié — meilleur choix qu'une chaîne répétée pour un
  préfill contrôlé), `random` (**PIÈGE** : dans certaines versions échantillonne des séquences de
  ShareGPT puis les répète/tronque pour la longueur demandée — **PAS un texte naturel indépendant**,
  documenté dans l'issue GitHub #4133 déjà croisée en Q4), `prefix_repetition` (mesure le prefix caching,
  pas le débit MoE représentatif).
* **SGLang** `sglang.bench_serving` : mêmes familles (`sharegpt`, `random`) — privilégier `sharegpt` réel
  pour un routage proche du trafic naturel.

**Recommandation retenue (Luna)** : au minimum 3 charges à même longueur/concurrence — ShareGPT (référence
naturelle), Sonnet (contrôle reproductible), passage répété (stress de routage, jamais présenté comme
débit nominal) — publier `débit naturel vs débit répété` + ΔH et ΔCV, jamais un seul chiffre isolé.

**Sources** : arXiv (« RepetitionCurse », mesure directe Qwen3-30B-A3B), nvidia.com (routage MoE et
distribution sémantique du corpus), vllm.ai (`vllm bench serve` datasets), GitHub (issue #4133 « random
n'est pas random »).

---

# Lot 4 (26/09, chef/poste5 — question 16, suite de Q(19) d'poste1 sur le préfill GDN groupé)

## Q16 — chunk_gated_delta_rule accepte-t-il cu_seqlens pour grouper le préfill GDN en un seul appel ?

**Réponse directe et positive, code réel cité pour fla, vLLM ET SGLang — répond exactement au problème
d'poste1 (Q19 du lot 1 : préfill GDN qui boucle par séquence, ~1,6 ms/couche/séquence, ~600 ms Python).**

**fla** (`fla/ops/gated_delta_rule/chunk.py`, réel) : `chunk_gated_delta_rule(q,k,v,g,beta,...,
initial_state=None, output_final_state=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None)`
— `cu_seqlens` PROPAGÉ aux trois étapes internes (`chunk_local_cumsum`, `chunk_gated_delta_rule_fwd_intra`,
`chunk_gated_delta_rule_fwd_h`). C'est une récurrence CHUNKWISE réinitialisée à chaque frontière indiquée
par `cu_seqlens` (pas une seule récurrence continue qui déborderait d'une séquence à l'autre) — **au bit
du par-séquence par construction**, contrairement à ce que craignait notre question initiale (Q4 du lot 1,
« réfuterait : varlen fla non au bit du par-séquence »).

**État par séquence** : `initial_state` de forme `[N, HV, K, V]` (N = nombre de séquences, PAS déduit de
B mais de `len(cu_seqlens)-1`), `output_final_state=True` rend un `final_state` de même forme — un état
par séquence en entrée ET en sortie, indexé par position dans `cu_seqlens` (`initial_state[i]` ↔ tokens
`[cu_seqlens[i]:cu_seqlens[i+1]]`). Pour un vrai préfill sans préfixe, `initial_state` est nul ; pour un
extend après préfixe déjà en cache, état distinct par requête fourni.

**vLLM V1 récent (`vllm/v1/attention/backends/gdn_attn.py`, réel)** : le prefill Qwen3-Next/Qwen3.5 N'EST
PAS bouclé par séquence — `_build_chunk_metadata` + `prepare_chunk_indices(prefill_query_start_loc_cpu,
FLA_CHUNK_SIZE)` construisent les métadonnées varlen, `prefill_query_start_loc` = l'équivalent vLLM de
`cu_seqlens`. Champs distincts : `prefill_state_indices` (emplacement de l'état récurrent par requête),
`prefill_has_initial_state` (reprise depuis un préfixe en cache). `qwen3_next.py` appelle le chemin chunké
avec ces métadonnées. **Mélange prefill/decode géré séparément** : les decodes passent par le noyau
récurrent single-step, les prefills restants par le chemin chunké varlen — le rebasage de
`prefill_query_start_loc` exclut les tokens decode sans reboucler par séquence.

**SGLang** : le chemin CUDA récent suit le même modèle groupé (support `chunk_gated_delta_rule` ajouté
pour Qwen3-Next, notes de release citées). **Exception notée** : SGLang-JAX historique utilisait un
`lax.scan` séquentiel PAR TOKEN (`ragged_gated_delta_rule_ref`) même avec cu_seqlens propagé pour
réinitialiser aux frontières — un ticket d'optimisation SGLang-JAX propose de le remplacer par une forme
chunkwise. Cette exception ne concerne PAS le chemin CUDA principal.

**Conclusion pratique pour acvram** : l'implémentation efficace est `8 appels de projections groupées →
1 construction de cu_seqlens/chunk_indices → 1 appel Python à chunk_gated_delta_rule (plusieurs kernels
CUDA internes) → 8 états finaux indépendants` — PAS `8 × N_couches appels Python GDN`. Le support existe
déjà dans fla (dont acvram dépend, cf. `.venv/lib/.../fla/ops/gated_delta_rule/chunk.py` vu en session) ;
vLLM et SGLang CUDA l'exploitent tous les deux. **À vérifier** : notre `fla` vendored a-t-il bien
`cu_seqlens` dans sa signature `chunk_gated_delta_rule` (probable, même paquet), et notre `couches.py`
utilise-t-il actuellement cette voie ou la boucle Python par séquence qu'poste1 a mesurée ?

**Sources** : GitHub (code réel `fla/ops/gated_delta_rule/chunk.py`, `fla/ops/kda/chunk_fwd.py`,
`vllm/v1/attention/backends/gdn_attn.py`, issue vLLM Qwen3.5/Qwen3-Next GDN, ticket SGLang-JAX
optimisation), newreleases.io (notes de release SGLang v0.5.8, support Qwen3-Next GDN).

## Q17 — FP8 natif vs int8 par canal sur RTX 5090 (SM120) : le conseil vLLM « FP8 sur CC≥10.0 » tient-il ?

**Découverte critique (Luna, bugs GitHub réels)** : le conseil générique de vLLM (« FP8 plutôt qu'INT8 sur
compute capability ≥10.0 », cité en Q14) **ne s'applique PAS forcément à notre carte** — plusieurs chemins
FP8 de vLLM et SGLang REJETTENT explicitement SM120 en pratique :

* **vLLM CUTLASS MoE FP8** : `cutlass_group_gemm_supported()` contient littéralement
  `if cuda_device_capability < 90 or cuda_device_capability >= 110: return False` — **120 tombe dans la
  zone rejetée** ; sources compilées `grouped_mm_c3x_sm90.cu`/`grouped_mm_c3x_sm100.cu` mais **aucune
  `grouped_mm_c3x_sm120.cu`**. Erreur runtime documentée et citée : `No compiled cutlass_scaled_mm for
  CUDA device capability: 120. Required capability: 90 or 100`.
* **SGLang FP8 blockwise** : ticket RTX 5090 réel, SGLang 0.5.3/CUDA 12.8/PyTorch 2.8.0, erreur `No
  implemented fp8_blockwise_scaled_mm for current compute capability: 120` — un ticket dédié confirme
  explicitement le format non supporté sur RTX 5090/RTX PRO 6000 à cette date.
* **GEMM groupé MoE FP8 pur** : un rapport plus récent indique un repli sur des tactiques SM89 (faute de
  noyaux TMA FP8 natifs SM120), alors que **NVFP4 a une couverture SM120 plus avancée** — cohérent avec
  notre propre choix historique du NVFP4 comme format principal.

**Réponse théorique/matérielle (pas garantie logicielle)** : SM120 possède bien des instructions MMA FP8
natives (`mma.sync.aligned.kind::f8f6f4`, `mxf8f6f4.block_scale`, documentées CUTLASS), mais **le débit
FP8 non block-scaled SM120 est comparable à Ada, PAS au ×2 annoncé pour SM100/B200** (`tcgen05.mma`) — ne
pas transposer les chiffres B200 à la RTX 5090. SM120 ne supporte que le layout TN, cluster figé à 1×1×1
(pas de multicast GeForce). `sm_120`/`sm_120a` = même puce, `a` = fonctionnalités compilateur
supplémentaires, pas une puce différente.

**Décodage n=1-8** : FP8 et INT8 transfèrent ~1 octet/élément chacun — **gain de bande passante brute
quasi nul**, ne pas présumer un avantage FP8 sans benchmark de nos formes exactes ; un INT8 par canal bien
optimisé peut être aussi rapide ou plus rapide. **Préfill n=64-128** : un GEMM FP8 natif A PLUS DE
CHANCES d'être meilleur (M plus grand amortit lancement/remplit les tensor cores), mais pas garanti —
dépend du mode de scaling (bloc/tensor-wide vs notre par-canal, qui ne correspond pas forcément au
scaling matériel optimisé).

**Point le plus important pour nous (chaîne FP8→int8)** : nos tenseurs int8 viennent d'une
déquantification+requantification FP8→int8 par canal — **chemin jugé sous-optimal** par Luna face à
utiliser directement `FP8 E4M3 + échelles d'origine → GEMM FP8 natif` (évite la perte de précision et le
travail de préparation de la requantification) — **mais uniquement SI un vrai noyau FP8 SM120 existe pour
notre chemin exact** (dense scaled_mm ≠ MoE grouped GEMM, niveaux de support différents).

**Protocole de mesure recommandé** : 4 variantes (FP8 natif échelles d'origine / INT8 par canal actuel /
FP8 sur poids reconvertis depuis l'int8 pour isoler le noyau / chaîne complète FP8 originale vs FP8→INT8)
à M=1,2,4,8 (decode) et M=64,128 (préfill), vérifier `smsp__inst_executed_pipe_tensor` (Nsight Compute) et
**le kernel RÉELLEMENT lancé, pas seulement le nom du backend** (une build `TORCH_CUDA_ARCH_LIST="12.0"`
ne crée pas magiquement les kernels manquants si le fichier de dispatch n'a pas de spécialisation SM120).

**Retenu pour acvram** : ne pas migrer vers FP8 sur la seule foi du conseil générique vLLM — vérifier
D'ABORD si notre chemin exact (dense vs MoE groupé) a un kernel FP8 SM120 compilé et fonctionnel avant
toute mesure comparative ; le risque concret est de mesurer un FALLBACK (SM89 ou erreur) en pensant tester
du FP8 natif Blackwell.

**Sources** : nvidia.com (docs CUTLASS SM120/SM100 GEMMs, discussion SM120/SM120a, ldmatrix sm120),
GitHub (issues réelles : CUTLASS MoE backend unavailable SM_120, Blackwell SM120 FP8 MoE fails GLM-4.7,
SM89 tactics fail SM120 pure FP8 MoE, SGLang 0.5.3 ne peut pas lancer FP8 sur RTX 5090, feature request
FP8 blockwise SM120), nvidia.com (SGLang Release 26.02 notes).

---

## Q18 — préfill GDN groupé par cu_seqlens (FLA) : au bit en fp32 (8/8), pas en bf16 (0/7) — normal ?

**PRÉMISSE INVALIDÉE (poste1, 245, 26/09)** : les 0/7 bf16 venaient d'une ERREUR du test (poids de
convolution bf16 comparés à une convolution de référence en fp32, boucle de référence qui plantait), pas
d'une dérive numérique réelle. Test corrigé : 35 passed, dont le bf16 au bit ; diagnostic séparé confirme
fla bf16 isolé = varlen = groupé, à 0 ulp (poste1-245 58cc9dd23, `scratchpad/poste1-p245-26-09/diag.txt`). La réponse ci-dessous
reste une synthèse duck.ai VALIDE sur ce que garantissent FLA/vLLM/SGLang en général, mais **ne pas la
citer comme une mesure sur notre cas** : notre 0/7 n'était pas une mesure de dérive bf16, c'était un bogue
de banc.

**3 avis convergents (Luna, gpt-oss, Gemma 4)** : NON, l'égalité au bit en bf16 entre un appel groupé
(cu_seqlens) et des appels séparés par séquence n'est PAS un contrat raisonnable — seul le fp32 s'en
approche, et encore sous réserve (voir plus bas).

**Ce que garantit réellement `chunk_gated_delta_rule` de FLA** : les bornes de chunk sont bien relatives à
chaque séquence, pas à l'offset aplati du batch — `fla/ops/utils/index.py::prepare_chunk_indices()` calcule
`chunk_counts` par séquence puis `_segmented_arange()` donne `(seg_id, intra_chunk_idx)` : une séquence qui
commence au jeton aplati 37 a bien son 1er chunk en `chunk_id=0`, jamais `37 % chunk_size`. La segmentation
mathématique est donc un choix délibéré, pas un accident de la mise à plat (Luna, code cité).

**D'où vient la dérive bf16 quand même** :
- bf16 n'est pas associatif : tout changement d'ordre des réductions change l'arrondi.
- Triton autotune/sélection de bloc dépend de la FORME groupée (taille de batch, num_warps, config) —
  un appel séparé et un appel groupé peuvent ne pas utiliser la même config compilée, même sur la même
  séquence. FLA issue #734 documente des soucis d'autotune (`chunk_local_cumsum_scalar_kernel`) liés à la
  forme, spécifiques à H100.
- Les GEMM de reconstruction tournent sur tensor cores (TF32/bf16) et RÉASSOCIENT la récurrence — cause
  citée explicitement par SGLang (RFC #28511, GDN/KDA) : « not bit-exact in floating point … reassociates
  the recurrence ».
- Même le fp32 n'est pas automatiquement garanti au bit : seulement plus plausible, à condition que
  l'ordonnancement complet des noyaux et l'ordre de réduction soient tenus identiques.

**Comment vLLM et SGLang qualifient ce chemin** : aucun des deux ne revendique le bit-exact.
- **SGLang RFC #28511** donne des tolérances numériques explicites, alignées sur le noyau vLLM amont :
  **fp32 atol 1e-4, bf16 atol 2e-2**.
- **SGLang-JAX issue #1416** (préfill GDN) : « numerical parity is the hard part » — référence fp32 =
  `ragged_gated_delta_rule_ref` (JAX pur), vérification VALEUR PAR VALEUR (pas seulement la forme), critère
  = tolérance + métriques modèle (GPQA-Diamond, MMLU-Pro) préservées, jamais un chiffre isolé.
- **vLLM** : `tests/ops/test_gated_delta.py` compare le noyau FLA à une référence PyTorch avec
  `torch.testing.assert_close` (tolérance, pas égalité). vLLM issue « GDN_ATTN does not support
  batch-invariant mode » traite le déterminisme solo/batché comme un TRAVAIL FUTUR, pas une propriété
  acquise — cohérent avec un modèle de correction par tolérance, pas par garantie bit à bit.
- FlashInfer issue #3329 : hang CUDA-graph shape-dépendant sur le préfill GDN groupé, contourné par le
  backend Triton/FLA — un problème de robustesse, pas une question de bit-exactness.

**Pour notre cas (8/8 fp32, 0/7 bf16)** : cohérent avec l'attendu de la littérature — le résultat fp32
tenu au bit n'est probablement pas un hasard (moins de tensor cores impliqués, ordre plus stable), mais
n'est pas non plus une garantie contractuelle de FLA ; le résultat bf16 non tenu au bit ne signale PAS en
soi un bogue. Protocole recommandé pour 245 : comparer groupé/dégroupé en fp32 avec atol/rtol explicites
(inspirés SGLang : fp32 1e-4, bf16 2e-2) plutôt qu'une égalité au bit en bf16 ; sur le service, préférer un
critère KL/logits plutôt qu'une comparaison d'octets.

**Sources** : GitHub (fla/ops/utils/index.py, fla/ops/gated_delta_rule/chunk.py ; FLA issue #734 autotune
H100 ; FLA issue #640 GDN precision triton3.5/H20 ; vLLM issue GDN_ATTN batch-invariant ; vLLM
`vllm.model_executor.layers.fla.ops.chunk` ; vLLM/FlashInfer issue #3329 hang préfill groupé ; SGLang RFC
#28511 précision GDN/KDA ; SGLang-JAX issue #1416 préfill GDN ; SGLang issue Qwen3.5 multi-item scoring),
vllm.ai (doc API varlen GDN).

---

## Q19 — 233 tenseurs W8A8 FP8 (e4m3, échelles canal × jeton) : int8 par canal → FP8 natif SM120, quel critère d'acceptation si la KL actuelle ne tient pas ?

**3 avis convergents (Gemma 4, Luna, gpt-oss)** : la pratique s'est déplacée des métriques de distribution
(KL brute) vers des métriques de comportement/tâche — la KL seule ne suffit pas et n'est PAS le critère de
release dans l'écosystème vLLM/llm-compressor/Red Hat. Le seuil « ≥ 99 % de récupération » est une
convention de rapport très répandue, PAS une norme industrielle formelle unique.

**Hiérarchie des critères observés en pratique** :
1. **Benchmarks de tâches (le vrai gate de production)** : MMLU, GSM8K, ARC-Challenge, HellaSwag,
   Winogrande, via `lm-evaluation-harness`. Chiffres exacts publiés par Red Hat/llm-compressor pour un
   Llama 3.1 70B W8A8 : MMLU 83,88→83,65 (99,7 % récup.), MMLU CoT 85,74→85,41 (99,6 %), ARC-C
   93,26→93,26 (100,0 %), GSM8K 93,10→93,25 (100,2 %), HellaSwag 86,40→86,28 (99,9 %), Winogrande
   85,00→85,00 (100,0 %), **moyenne 83,89→83,96 (100,2 %)**.
2. **Taux de récupération moyen** (score quantifié / score référence × 100), PAS une exigence de ≥ 99 %
   tâche par tâche — l'étude Red Hat « Give Me BF16 or Give Me Death? » (>500 000 évaluations, famille
   Llama 3.1) trouve **99,75 % moyen en 8-bit**, **99,36 % en W4A16**, mais un pire cas à **≈ 96,88 %**
   (TruthfulQA, W4A16 8B) — une moyenne à 99 % peut cacher un échec spécifique à une tâche (ex. AIME25
   à ≈ 86 % sur une carte modèle Red Hat NVFP4/FP8, malgré GSM8K à 99,76 %).
3. **Perplexité (PPL)** : filtre rapide/diagnostic (change les échelles ont-elles saturé, un noyau a-t-il
   dérivé), PAS un gate de production suffisant seul — reconnu explicitement insuffisant pour
   l'instruction-following/sécurité (Red Hat, arXiv llama.cpp eval paper).
4. **Accord top-1/logits** : outil de régression et de localisation de bug, pas de seuil universel reconnu.

**Seuils KL trouvés (utiles pour vous, qui partez d'un critère KL)** : deux cartes modèles
compressed-tensors distinctes citées par Luna donnent des seuils DIFFÉRENTS et non canoniques — l'une
`KL < 0,005` + MMLU ≥ 99,7 % + RULER@128k ≥ 99 % ; l'autre `KL < 0,014` + MMLU ≥ 99 % + RULER@128k ≥ 97 %.
Un papier (« Statistically-Lossless Quantization ») propose `KL ≤ 0,01` comme prédicteur théorique de
« task-lossless », avec un score EAR (chevauchement top-10) ≥ 0,99 comme critère complémentaire. **Ce sont
des seuils définis par leurs auteurs pour LEUR recette, pas des standards vLLM/industrie.**

**vLLM en pratique** : la doc FP8 traite l'évaluation comme une étape séparée après quantification/service
— quantifier, servir, `lm_eval` (ex. GSM8K sur 250 échantillons, score 0,768 ± 0,0268), comparer au modèle
original SOUS LES MÊMES réglages (tokenizer, template, few-shot, échantillonnage, **BOS-token** — un
mismatch d'évaluation peut se faire passer pour une régression de quantification). Aucun seuil universel
imposé dans la doc elle-même.

**Recommandation pour notre cas (233 tenseurs W8A8→FP8 natif)** : ne pas chercher un seuil KL de
remplacement unique et théorique. Remplacer/compléter le critère KL par : (a) PPL sur corpus de référence
comme filtre rapide (proche BF16, pas un seuil absolu isolé) ; (b) recovery moyen ≥ 99 % sur un panel de
tâches (MMLU + GSM8K minimum, HellaSwag/ARC-C si possible) via lm-eval, **jamais une moyenne seule** — noter
aussi le pire score par tâche (le cas TruthfulQA/AIME25 montre qu'une moyenne à 99 % peut masquer un score
à 87-96 % sur une tâche spécifique) ; (c) vérifier BOS-token et réglages identiques entre les deux bras
avant de conclure à une régression.

**Sources** : Red Hat (« Optimize a model with LLM Compressor », doc FP8 llm-compressor, exemples Llama
3.1 70B W8A8), arXiv (« Give Me BF16 or Give Me Death? », Statistically-Lossless Quantization, papier
d'évaluation llama.cpp), Oracle Cloud (blog FP8 dynamique, recovery Llama 3.3-70B), GMI Cloud (blog
FP8/FP4), NVIDIA (blog Transformer Engine/Blackwell, MXFP8 vs BF16), vllm.ai (doc FP8, doc KV-cache FP8
état de l'art), Hugging Face (cartes modèles compressed-tensors, carte Red Hat NVFP4/FP8 GSM8K/AIME25).

---

## Q20 — préfill MoE solo (Qwen3-Coder-30B-A3B, 128 experts top-8, NVFP4) : ≈29 jetons/expert, borné par les lancements/petites tuiles — que font vLLM/SGLang/TensorRT-LLM, quel gain typique ?

**3 avis (gpt-oss, Luna très détaillé, Gemma 4 générique)** : convergents sur le fond, avec un avertissement
important de Luna spécifique à notre carte (SM120).

**Ce qui compte le plus à M≈29, par ordre (Luna)** :
1. **Passer d'une boucle par expert à UN lancement groupé/scheduled** — le plus gros gain, avant même la
   taille de tuile. Une boucle naïve (128 lancements séparés × 3 GEMM) est catastrophique en surcoût de
   lancement ; un noyau groupé aplati (`moe_align_block_size` côté vLLM : tri des jetons, table
   `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded`, code exact dans
   `vllm/model_executor/layers/fused_moe/{fused_moe,moe_align_block_size}.py`) traite tous les experts en
   quelques lancements, indépendamment de leur nombre.
2. **Éviter le sur-remplissage (padding) des tuiles** : à M=29, `BLOCK_M=32` ne gaspille que 3 lignes ;
   `BLOCK_M=64` en gaspille plus de la moitié. Mais `BLOCK_M=16` double le nombre de tuiles et peut coûter
   plus en métadonnées — pas un choix automatique, dépend de N/K/registres/nombre d'experts non-vides.
   Balayage utile : `BLOCK_M={16,32,64} × BLOCK_N={64,128,256} × GROUP_SIZE_M={1,4,8} × split-K`.
3. **Ordonnancement persistant/grille adaptative** — utile si la grille initiale sous-occupe le GPU
   (peu de tuiles par expert). SGLang documente une grille adaptative (divise BLOCK_SIZE, double la grille
   si trop petite) — bénéfice relatif MOINDRE sur RTX 5090 (moins de SM) que sur B200 où c'est mesuré.
4. **Fusion gate+up** : remplace 2 petits GEMM par 1 plus grand — optimisation de second ordre APRÈS le
   groupement, pas le levier principal.
5. **Tri/permutation (`moe_align_block_size`)** : nécessaire mais coûte un lancement + tampon séparé ; à
   M≈29 son coût peut être comparable aux GEMM eux-mêmes — les meilleures implémentations fusionnent
   l'indexation avec le premier GEMM plutôt qu'un tri générique séparé.
6. **Split-K** : utile seulement si le premier GEMM est sous-occupé en parallélisme N ; sinon coûte plus
   qu'il ne rapporte (réduction partielle supplémentaire) — à autotuner, jamais activer sans mesure.

**AVERTISSEMENT SPÉCIFIQUE SM120 (Luna, à vérifier avant toute conclusion)** : TensorRT-LLM issue #11932
rapporte qu'au moins une release candidate 1.3.0 voit ÉCHOUER les deux chemins NVFP4 MoE (TRTLLMGen ET
CUTLASS) sur SM120 pendant le profilage de tactique, y compris l'échec d'initialisation d'un GEMM groupé
TMA warp-specialized. Les comparatifs publiés B200/SM100 (SGLang 1,78× vLLM à b=1, 1,40× à b=128, GPT-OSS-20B
NVFP4) NE SE TRANSPOSENT PAS forcément à SM120 — vérifier la version exacte de tout moteur tiers avant
d'utiliser ses chiffres comme référence pour nous.

**Gain typique attendu (engineering expectations, PAS de mesure publiée exacte pour notre cas précis)** :
- Contre une boucle naïve par expert : **2-6×** (GEMM groupé seul) → **3-8×** (+ gate/up empaqueté + petites
  tuiles) → **5-10× ou plus** si la boucle de référence est particulièrement mauvaise (lancement séparé par
  expert/projection/activation/réduction, synchronisation après chaque expert).
- Contre un GEMM groupé DÉJÀ compétent (notre cas probable, puisqu'on a déjà un chemin groupé) : plutôt
  **1,2-1,5×** de mieux fusion/ordonnancement/tuiles, **1,5-2×** si l'implémentation de référence a un mauvais
  ordonnancement à petit M. Repère mesuré (H20, CUTLASS) : 42,0→74,5 µs à b=4 (**1,77×**), 85,7→209,2 µs à
  b=16 (**2,44×**) pour un chemin fusionné routage+gate-up+quant+down+réduction en pipeline persistant bas
  latence — plus proche de notre régime que les chiffres B200 top-4/32-experts.

**Recommandation pour la 270 (poste1)** : ne pas comparer notre débit à un chiffre publié SGLang/vLLM/TRT-LLM
tel quel (GPU, nombre d'experts, top-k et modèle différents à chaque fois cité). Mesurer d'abord si notre
chemin actuel est déjà un GEMM groupé à une passe ou une boucle par expert — c'est ce qui détermine si le
levier est 2-6× (boucle→groupé) ou 1,2-2× (groupé→mieux optimisé). Vérifier aussi la version exacte de tout
moteur tiers utilisé en comparaison sur SM120, vu l'issue #11932.

**Sources** : GitHub (vLLM `fused_moe.py`, `moe_align_block_size.py` + tests `test_moe_align_block_size.py` ;
SGLang issue #7994 NVFP4 MoE pipeline ; TensorRT-LLM `cutlass_extensions/gemm/kernel/splitk_gemm_grouped.h` ;
CUTLASS discussion #1536 split-K petit-M ; TensorRT-LLM issue #11932 échec NVFP4 SM120), vllm.ai (doc
DeepGemmFP4Experts, Blackwell SM100/SM120), Hugging Face (comparatif B200 NVFP4 SGLang/vLLM GPT-OSS-20B),
arXiv (Cross-Platform Fused MoE Dispatch in Triton), mufeezamjad.com (worklog GEMM groupé persistant,
238→23,8 µs).

---

## Q22 — décodage spéculatif RTX 5090 (SM120) pour Qwen3/Qwen3-Coder : MTP natif, EAGLE-3, Medusa

**3 avis (Luna très prudente et sourcée, gpt-oss chiffré mais incohérent en interne, Gemma sans réponse
exploitable)**. Écart notable entre Luna et gpt-oss : à traiter comme un signal d'alerte, pas comme deux
sources équivalentes — voir mise en garde en fin de section.

**Point de départ important (Luna)** : les Qwen3 originaux (4B/8B/14B/32B/30B-A3B) N'ONT PAS de tête MTP
native en général. Seuls Qwen3-Next, Qwen3.5, Qwen3.6/3.8 et Qwen3-Coder-Next en ont une (ou un format de
checkpoint compatible MTP), selon le checkpoint exact. Vérifier lequel de nos modèles Qwen3 a réellement
une tête MTP avant de planifier autour de ce chiffre.

**Taux d'acceptation** (Luna, plages larges, PAS de mesure RTX 5090/Qwen3 publiée en tableau croisé) :
- **MTP natif** : ~50-90 % par jeton proposé, souvent 65-85 %, peut chuter fortement en long contexte.
  Repère réel opposé : mesure utilisateur Qwen3.8-27B sur DGX Spark (proche Blackwell) à **47,5 %**
  d'acceptation, longueur moyenne acceptée 2,42 — l'acceptation native n'est PAS automatiquement haute.
  Une issue vLLM Qwen3.5 MTP documente un effondrement vers 0 % quand les configs positionnelles
  cible/brouillon divergent en long contexte.
- **EAGLE-3** : ~70-90 % sur trafic code/instruction bien apparié, MAIS dépend fortement de l'appariement
  brouillon/cible — une tête EAGLE-3 entraînée pour Qwen3-8B n'est pas automatiquement adaptée à
  Qwen3-30B-A3B, Qwen3-Next ou Qwen3-Coder-Next ; un brouillon mal apparié peut être PIRE que le MTP natif
  tout en coûtant plus de VRAM.
- **Medusa** : aucun checkpoint Medusa Qwen3/Qwen3-Coder de production identifié — pas mesurable pour du
  Qwen3 de série, à traiter comme un projet de recherche/entraînement personnalisé, pas une option prête.

**Gain mesuré/attendu à b=1 et b=4** : AUCUNE mesure publique RTX 5090 + Qwen3 + b=1/b=4 pour les 3
méthodes n'existe (Luna). Repères de proxy (autre matériel/modèle) : MTP natif Qwen3.6-27B sur DGX Spark
~1,8-1,94× ; Qwen3.8-27B 11,4→24,7 t/s (~2,17×, note explicite de sensibilité au prompt par l'auteur) ;
EAGLE-3 générique 2,7-3,3× (papier, pas RTX 5090). Plage de planification (engineering expectations, pas
des mesures) : MTP 1,4-2,0× à b=1 / 0,9-1,4× à b=4 (peut devenir négatif) ; EAGLE-3 1,7-3,0× à b=1 /
1,0-1,6× à b=4. **Pour les modèles MoE (notre cas Qwen3-Coder-30B-A3B)** : ne pas présumer qu'un bon taux
d'acceptation donne un gain proportionnel — le dispatch d'experts et le coût de vérifier K+1 jetons peuvent
dominer ; à plus grand lot, le décodage spéculatif peut être plus lent même avec une bonne acceptation.

**Coût VRAM** : MTP natif le moins cher (partage embeddings/tête LM/poids cible, coût additionnel ≈ poids
du bloc MTP + KV brouillon B×K×taille_KV). EAGLE-3 ajoute un modèle brouillon complet + KV + tampons
d'arbre de candidats — peut consommer plusieurs Go même pour un brouillon dit « petit » (repère cité : ~3 Go
pour un déploiement Llama 70B, spécifique au modèle). Medusa : pas de modèle brouillon séparé mais plusieurs
têtes avec projection de vocabulaire potentiellement grande chacune si non partagée.

**Ce que servent vLLM/SGLang aujourd'hui (Luna, avec fichiers exacts)** :
- **vLLM** : `vllm/config/speculative.py` liste les méthodes (`ngram`, `medusa`, `mlp_speculator`,
  `draft_model`, `eagle`, `eagle3`...) et les types de modèles MTP (`qwen3_next_mtp`, `qwen3_5_mtp`, `mtp`).
  Fichiers pertinents : `vllm/v1/spec_decode/`, `vllm/model_executor/models/{llama_eagle3,qwen3_next_mtp}.py`,
  `vllm/v1/spec_decode/medusa.py`. Bogues connus : problème de forme de poids dans `qwen3_next_mtp.py`
  (à vérifier contre le commit exact déployé) ; une issue documente que le RoPE/YaRN du brouillon n'hérite
  pas toujours des overrides du modèle cible en long contexte (acceptation qui s'effondre). Une issue
  récente (model-runner-V2) rapporte que l'autorégresseur ignore parfois le K dynamique du planificateur,
  reproduite sur RTX 5090 double carte — pertinent à b≥4.
- **SGLang** : support EAGLE/EAGLE3/NEXTN selon modèle et version (`--speculative-algorithm EAGLE3`,
  `--speculative-draft-model-path`, `--speculative-num-steps`, etc.). NEXTN vient historiquement de
  DeepSeek, compatibilité Qwen3 à vérifier au cas par cas. Un format de checkpoint EAGLE-3 SpecForge
  n'est PAS forcément interposable avec le format vLLM/référence — vérifier le format avant de réutiliser
  un brouillon d'un moteur à l'autre.

**MISE EN GARDE sur gpt-oss** : sa réponse donne des chiffres très précis (EAGLE-3 4,3-5×, tableau VRAM en
dixièmes de Go) mais est **incohérente en interne** — elle affirme d'abord une acceptation EAGLE-3 de
75-85 %, puis cite dans la MÊME réponse une issue SGLang « confirmant » une acceptation de ~0,5 % et un
gain de seulement 1,x× sur RTX 5090. Ne pas retenir les chiffres précis de gpt-oss sans re-vérification
directe des sources qu'il cite (issue GitHub #11948, arXiv, DFlash blog) — possible confabulation de
précision. Gemma n'a pas produit de réponse exploitable (recherche web non aboutie).

**Recommandation pour la 279/poste1** : pour Qwen3-Coder-Next, commencer par MTP natif à K=2 ou 3 si le
checkpoint en a une vraie tête (vérifier d'abord) ; mesurer l'acceptation ET la longueur moyenne acceptée
par tour, pas seulement un taux nominal ; ne comparer à EAGLE-3 que si un checkpoint brouillon RÉELLEMENT
apparié à notre famille Qwen3-Coder existe ; désactiver la spéculation si le gain mesuré disparaît à
notre concurrence cible ; prévoir la VRAM du KV brouillon et des tampons de graphe CUDA séparément du KV
cible. Medusa hors périmètre sans entraînement dédié.

**Sources** : GitHub (`vllm/config/speculative.py`, `vllm/v1/spec_decode/`,
`model_executor/models/{llama_eagle3,qwen3_next_mtp}.py`, issue poids qwen3_next_mtp, issue RoPE/YaRN long
contexte, issue model-runner-V2 K dynamique RTX 5090, SGLang issue Eagle3 RTX 5090 inconsistante), nvidia.com
(rapports utilisateurs Qwen3.6-27B/Qwen3.8-27B/Qwen3-Coder-Next sur DGX Spark), arXiv (Medusa original,
benchmark Qwen3 générique EAGLE-3, TriSpec), vllm.ai (doc spec-decode, tailles de capture CUDA graph),
spheron.network (guide déploiement EAGLE-3/Medusa, DFlash), amd.com (MTP DeepSeek V3/SGLang),
consciousengines.com (compatibilité formats checkpoint EAGLE-3).

---

## Q23 — MTP k>1 : rattrapage KV de la tête après vérification, ou cache écrit pendant le brouillon ? (poste5, 277)

**CORRECTION PAR LA SOURCE (poste5, 26/09, prime sur l'avis duck.ai ci-dessous)** : Luna se trompe pour
vLLM. Code réel, vLLM 0.29.0 installé (`/opt/ia/vLLM/.venv`),
`vllm/v1/spec_decode/llm_base_proposer.py::propose` (:510-560) : un PREMIER passage du brouillon
(MTP/EAGLE) tourne sur TOUS les jetons de la passe de vérification (`target_token_ids` décalés d'un cran,
:850-866) avec `target_hidden_states` — `self.hidden_states[:num_tokens] = target_hidden_states` (:866) —
et écrit le KV du brouillon À CES POSITIONS. **C'est exactement le rattrapage** : les positions acceptées
sont recalculées avec l'état caché de la CIBLE à chaque pas, pas conservées telles qu'écrites pendant la
proposition. **vLLM fait donc le rattrapage ; acvram (qui rogne seulement) diverge de vLLM sur ce point
précis.** Le reste de la synthèse Luna ci-dessous (SGLang, causes alternatives, protocole
d'instrumentation) n'a pas été re-vérifié au code et reste à prendre avec cette réserve. Point à vérifier
côté acvram signalé par poste5 : leur compte d'acceptation utilise accepté/proposé SANS le jeton bonus —
vérifier que la comparaison à 0,475-0,489 (DGX Spark/NInfer) utilise la même définition avant de conclure
à un écart réel.

**Réponse duck.ai d'origine (Luna, 1 avis — voir correction ci-dessus, INEXACTE pour vLLM)** : affirmait que
vLLM et SGLang ne rejouent PAS de passage de rattrapage de la tête MTP sur les jetons acceptés avec les
états cachés de la cible, et que le comportement d'acvram (rogner la longueur, garder le cache MTP écrit
par la tête elle-même) serait celui des moteurs de référence. **Ceci est faux pour vLLM d'après le code
réel (voir ci-dessus).**

**Deux caches distincts, à ne pas confondre** :
- **KV de la cible** : le passage de vérification écrit les entrées K/V de la cible pour les positions
  spéculatives ; seul le préfixe accepté est retenu, le reste est ignoré/annulé.
- **KV de la tête MTP/du brouillon** : écrit par le passage de proposition, en utilisant les états cachés
  PRODUITS PAR LA TÊTE ELLE-MÊME (récursion). L'opération d'acceptation tronque normalement ce cache ; elle
  ne rejoue PAS chaque jeton accepté à travers la tête avec les états cachés de la cible.

**Fichiers vLLM cités** : `vllm/v1/spec_decode/mtp_proposer.py`, `vllm/v1/spec_decode/eagle_proposer.py`,
`vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/sample/rejection_sampler.py` (fait juste
accepter/rejeter, ne relance jamais le modèle MTP). Doc vLLM-Ascend (organisation équivalente) :
`load_model`, `dummy_run`, `generate_token_ids`, `_prepare_inputs`, `_propose` — le rejet est décrit comme
« jeter le jeton rejeté et tout ce qui en dérive », jamais comme « rejouer les jetons acceptés dans la
tête ». **SGLang** : `sglang/srt/speculative/`, `eagle_worker.py`, `spec_utils.py` — même logique NEXTN/MTP,
committer le plus long préfixe accepté, rien décrit comme rejeu par état caché de la cible.

**Pourquoi garder l'état de proposition n'est PAS intrinsèquement une erreur** : la récursion de la tête MTP
est *voulue* comme `ĥ_{i+1} = M(ĥ_i, e(x̂_{i+1}))` — un jeton accepté ne dit QUE qu'il a passé la
vérification cible, PAS que son état caché de proposition égale l'état caché cible correspondant. Remplacer
le cache MTP par des états cachés cible définirait un AUTRE brouillon, pas une exigence de correction.

**Ce qui explique VRAIMENT un écart d'acceptation (Luna, avec repères chiffrés)** :
- **Dégradation par position, effet d'entraînement connu (FastMTP, papier arXiv)** : MTP « vanille » :
  ~70 % à la position 1, ~10 % à la position 2, quasi 0 % à la position 3 ; un entraînement récursif
  (fine-tuning spécifique) améliore ces positions à ~80 %/56 %/36 %. C'est un DÉCALAGE
  ENTRAÎNEMENT/SERVICE : beaucoup de têtes MTP sont entraînées en teacher forcing (chaque couche MTP voit
  le VRAI jeton), mais le service les nourrit récursivement de leur propre jeton échantillonné — la cause
  documentée de l'erreur qui s'accumule.
- **Repère SGLang réel de la même famille de problème que vous** : une issue SGLang compare le MÊME modèle
  MTP sous SGLang (**~0,33** d'acceptation) contre vLLM (**~0,63**) — écart attribué à un comportement
  spécifique SGLang/model-runner, PAS à une différence d'algorithme spéculatif fondamental. Une autre issue
  NEXTN/MTP montre une acceptation ~2,4-3,3 (longueur moyenne) juste après démarrage, qui chute vers 1 après
  accumulation d'un bogue d'état de cache — un redémarrage restaure le taux d'origine.
- **Aucune ablation publique** « cache de proposition conservé » vs « recalculé depuis les états cachés
  cible » n'existe pour Qwen3-Next/3.5/3.8 ou DeepSeek-V3 — votre écart (0,38 vs 0,475-0,489) ne peut PAS
  être attribué en toute sécurité à ce seul choix.

**Causes plausibles alternatives à l'écart 0,38 vs 0,475-0,489 (liste Luna)** : définition différente du
taux d'acceptation (jetons acceptés/proposés vs longueur moyenne acceptée vs jeton bonus inclus ou non vs
par position) ; `num_speculative_tokens` différent ; greedy vs rejection sampling ; température/top-p
différents ; quantification cible ou tête différente ; désaccord de poids cible/tête ; partage
embedding/tête LM incorrect ; mauvais position_ids ou masque d'attention ; KV de brouillon tronqué à tort ;
gestion d'état hybride (DeltaNet/Mamba) ; utilisation récursive d'une seule tête alors que le checkpoint a
des couches MTP dédiées par position ; désalignement cache/backend CUDA graph ; distribution du prompt ;
longueur de contexte.

**Protocole d'instrumentation recommandé** : logger séparément par tour `draft_len`, `accepted_draft_len`,
`bonus_token_accepted`, `first_rejection_position`, `per_position_acceptance[k]`, `target_argmax[k]`,
`draft_token[k]`, `draft_probability[k]`, `target_probability[k]`. Puis comparer l'acceptation PAR POSITION
(k=1,2,3...) à l'agrégat. **Test diagnostique décisif** : faire tourner 2 variantes — (a) jeter le cache MTP
à chaque tour et régénérer la proposition suivante depuis l'état caché cible ; (b) tronquer le cache MTP au
préfixe accepté et continuer récursivement (= comportement actuel acvram). Si (a) améliore nettement
l'acceptation en position 2/3, l'état retenu est incohérent avec la récurrence de référence. Si NON,
l'écart est probablement ailleurs — le plus souvent alignement jeton/état caché, indexation de position,
gestion du rejet, ou différence de définition de métrique.

**Sources** : GitHub (vLLM `mtp_proposer.py`, `eagle_proposer.py`, `deepseek_mtp.py`,
`rejection_sampler.py`, issue « how does vllm handle wrong tokens », issue NEXTN/MTP acceptance decays to 0,
issue MTP always rejects draft tokens, issue Step-3.5-Flash MTP), vllm.ai (doc MTP vllm-ascend, proposer
`_prepare_inputs`/`_propose`), lmsys.org (Accelerating SGLang with Multiple Token Prediction), arXiv
(FastMTP), redhat.com (Optimize vLLM speculative decoding with FastMTP heads).

---

## Q24 — décodage spéculatif greedy annoncé « identique » au décodage sans spéculation : vrai au bit ? (poste5/chef, 277a)

**Réponse (Luna, 1 avis très sourcé)** : la revendication est « sans perte algorithmiquement », PAS
« identique au bit ». Votre inquiétude (la vérification à q_len=k+1 passe par d'autres noyaux GEMM que le
décodage à q_len=1, pouvant faire basculer un argmax sur une quasi-égalité) est explicitement reconnue par
au moins un des trois moteurs, et démontrée réelle par des issues ouvertes sur les trois.

**Ce que « sans perte » veut dire mathématiquement** : avec une arithmétique exacte, un jeton brouillon
n'est accepté QUE s'il correspond à la règle de décodage de la cible — la preuve suppose que les logits de
la cible utilisés pour la vérification SONT les mêmes logits mathématiques que le décodage ordinaire
produirait. Elle ne dit RIEN sur deux exécutions en précision finie avec des formes de séquence, algorithmes
GEMM, ordres d'accumulation, CUDA graphs ou noyaux de quantification différents.

**vLLM (le plus explicite des trois)** : sa doc dit littéralement que la « sans-perte » théorique tient
« jusqu'aux limites de précision numérique du matériel » — erreurs flottantes pouvant produire de légères
différences de distribution ; l'égalité en décodage greedy est testée comme une propriété ALGORITHMIQUE,
pas bit à bit. `tests/spec_decode/e2e` utilise une assertion d'égalité EXACTE (jeton par jeton), pas une
tolérance ni une KL. **Issue vLLM #7627** (« Add documentation on lossless guarantees ») : un utilisateur
attendait une sortie spéculative à température 0 identique au greedy ordinaire, a observé une divergence à
quelques dizaines de jetons dans la génération — reproduite.

**SGLang** : moins explicite sur la mise en garde GEMM/near-tie que vLLM, mais preuve pratique par les
issues. **Issue #13123** : EAGLE-3 ignore le drapeau d'inférence déterministe — le chemin non-spéculatif est
stable d'un run à l'autre, EAGLE3 spéculatif varie, scores GSM8K qui varient avec EAGLE3 alors que le
non-spéculatif reste constant. Étude « Batch Speculative Decoding Done Right » (arXiv, 2026) : exact-match
ET partial-match utilisés (PAS de KL) ; le mode déterministe SGLang-EAGLE améliore l'exact-match de
**69,8 % à 85,0 %** sur un montage Vicuna, SANS atteindre 100 % — rendre l'exécution plus déterministe
RÉDUIT mais n'élimine PAS toute divergence. **Issue #33985** : la vérification spéculative arrive avec une
forme d'attention différente (topk=192) du décodage normal (topk=128), un chemin de noyau différent est
sélectionné — confirme que le chemin spéculatif n'est PAS simplement le noyau de décodage à un jeton
rappelé en boucle.

**TensorRT-LLM** : test le plus révélateur de la bonne séparation à faire. Dans
`tests/unittest/_torch/modeling/test_modeling_llama.py` : les logits spéculatif/référence sont comparés
avec `torch.testing.assert_close(..., atol=1.0, rtol=1.0)` (**tolérance TRÈS large sur les logits**), mais
les IDs de jeton greedy sont vérifiés par égalité EXACTE :
`assert token_id_ref == token_id_gen, "Greedy sampling token id not match"`. **C'est presque exactement la
bonne séparation pour ce problème** : les logits n'ont pas besoin d'être numériquement identiques, mais le
jeton greedy sélectionné doit correspondre. **Issue TensorRT-LLM #10309** (« Qwen3 + Eagle3 generates
different result with greedy decoding compared to no eagle version ») : à température 0, sortie différente
avec/sans EAGLE3 — preuve directe contre une garantie bit-à-bit inconditionnelle.

**Réponse directe à l'inquiétude technique** : le scénario est réel — décodage ordinaire (forme
[batch, 1, hidden]) et vérification spéculative (forme [batch, k+1, hidden]) peuvent sélectionner des
noyaux GEMM int8/fp8 différents, des configurations split-K/tuile différentes, des précisions
d'accumulation différentes, des épilogues fusionnés ou non, des noyaux d'attention différents, des captures
CUDA graph différentes — assez pour faire basculer un argmax sur une quasi-égalité (`ℓ_a=10.000000,
ℓ_b=9.999999` vs `ℓ̃_a=9.999998, ℓ̃_b=10.000000` → argmax bascule de a à b, aucune preuve de rejection
sampling ne peut réparer ça). L'affirmation correcte est « préserve la distribution/décision greedy de la
cible EN SUPPOSANT des logits/probabilités cible cohérents » — PAS « identique au bit à travers tous les
noyaux, formes, tailles de lot, modes de quantification et configurations matérielles ».

**Comment les 3 moteurs testent (aucun n'utilise la KL comme critère principal de production)** :
égalité exacte jeton par jeton (les 3, détecte toute divergence greedy visible, n'établit PAS des logits
identiques au bit) ; TensorRT-LLM ajoute une tolérance sur les logits (`allclose`) EN PLUS de l'égalité
exacte des jetons — une tolérance sur les logits SEULE est insuffisante (un `allclose` peut passer alors
que l'argmax bascule si la marge top-1/top-2 est plus petite que l'erreur tolérée).

**Protocole de test recommandé (Luna)** : décoder greedy ordinaire et spéculatif (même modèle, prompt,
graine, échantillonnage) ; comparer les IDs de jeton EXACTEMENT ; à chaque pas de vérification, comparer
les logits cible aux positions correspondantes ; noter la marge top-1/top-2 `m_t = ℓ_t(1) − ℓ_t(2)` ;
signaler les cas où l'écart absolu de logit entre les deux chemins est comparable à `m_t` ; répéter sur
k=1..K, tailles de lot 1/2/4/…, chemins CUDA-graph et ordinaires, FP16/BF16/FP8/INT8, tous les
noyaux/backends GEMM pertinents ; tester séparément le mode déterministe.

**Sources** : vllm.ai (doc Speculative Decoding, `docs/features/speculative_decoding/README.md`), GitHub
(vLLM issue #7627 lossless guarantees, issue scoring model prepare_inputs GPU ; SGLang issue #13123 EAGLE-3
ignore deterministic flag, issue #33985 topk=192 vs 128 dispatch différent ; TensorRT-LLM
`speculative-decoding.md`, `test_modeling_llama.py` (assert_close atol/rtol=1.0 + égalité exacte des
jetons), issue #10309 Qwen3+Eagle3 greedy différent, release notes précision EAGLE3 multi-GPU), arXiv
(« Batch Speculative Decoding Done Right », exact/partial match, 69,8 %→85,0 % en mode déterministe),
nvidia.com (tutoriel Speculative Decoding with TensorRT-LLM, cohérence de distribution).

---

## Q25 — hybride GDN+attention : GDN et int8 au bit, sortie finale non — suspects attention pleine et GEMM nvfp4 (poste5, 277a-bis)

**Réponse (Luna, 1 avis, exceptionnellement complète et sourcée)** : les deux suspects restants sont EXACTEMENT
les catégories d'opérateurs visées par les travaux « batch-invariant kernels ». « Récurrence GDN au bit » ne
garantit PAS une sortie de bout en bout au bit si l'attention ou le GEMM MLP change d'ordre de réduction.

**Distinction utile** : invariant au LOT (résultat inchangé si on ajoute/retire d'autres jetons dans le même
appel de noyau) ≠ invariant en POSITION (inchangé si la position change dans un appel de taille fixe) ≠
invariant en FORME (inchangé entre M=1 et M=k+1 — c'est votre cas précis). La plupart des travaux publiés
traitent la 1re propriété ; votre cas de vérification spéculative sollicite en plus la forme et la
disposition du cache.

**GEMM (le suspect nvfp4)** : un GEMM cuBLAS/CUTLASS/tensor-core n'est PAS invariant en forme en général —
`[1,K]@[K,N]` peut différer de `[M,K]@[K,N]` puis sélection d'une ligne, car le noyau change de forme de
tuile de sortie, d'instruction tensor-core, de split-K ou non, d'ordre d'accumulation. **Thinking Machines
identifie explicitement split-K, stream-K et le changement d'instruction tensor-core comme sources de
non-invariance** ; leur solution : UNE SEULE configuration de noyau (y compris stratégie de réduction fixe)
pour toutes les valeurs de M, même en perdant en performance. **Un GEMM quantifié a des sources
supplémentaires** : ordre de chargement des échelles, parcours des groupes d'échelles, placement de la
déquantification, gestion du zéro-point, arrondi de l'épilogue. La bibliothèque Thinking Machines publique
est étroite (remplace seulement `torch.mm`/`addmm`/`mean`/`log_softmax`) — **PAS une preuve que tout GEMM
FP8/FP4/Marlin/CUTLASS est invariant**. vLLM a un item de suivi SÉPARÉ pour NVFP4 GEMM/MoE — pas couvert
automatiquement par le chemin `torch.mm` ordinaire ; une issue vLLM rapporte même que
`VLLM_BATCH_INVARIANT=1` peut FORCER les modèles NVFP4 sur un chemin d'émulation plutôt que leurs noyaux
matériels normaux (bogue de support, pas preuve directe d'un écart numérique, mais signe que NVFP4 exige un
traitement spécifique).

**Attention pleine (l'autre suspect)** : non-invariante quand la réduction sur la dimension KV est
partitionnée différemment. Coupables principaux : split-KV/FlashDecoding, nombre de splits KV variable,
tailles de split différentes, **traitement SÉPARÉ du KV en cache et du KV du jeton courant** (le même
intervalle logique peut donner 5 blocs de réduction au lieu de 4 selon la frontière cache/courant — change
nécessairement l'ordre de réduction flottant), frontières de bloc softmax différentes. Correctif Thinking
Machines : mettre à jour le cache KV et la table de pages AVANT le noyau d'attention pour que toute la
séquence KV logique ait UNE disposition physique cohérente ; **taille de split FIXE** (pas seulement un
nombre de splits fixe — avec une taille fixe, le nombre de splits peut varier mais l'ordre de réduction de
chaque split complet reste stable).

**SGLang documente les mêmes solutions par backend** : FlashInfer (`fixed_split_size`, désactiver le split
KV dynamique), FlashAttention-3 (`num_splits=1`), Triton (taille de split de décodage fixe, troncature du
préfill par blocs alignée sur la taille de split). Issue de suivi SGLang liste séparément : FlashInfer FA2
invariant au lot, changement de taille de tuile Triton, support FA3 déterministe, **RMSNorm invariant** (item
séparé — un autre suspect possible, la RMSNorm bascule aussi entre réduction sur une ligne/plusieurs
lignes/split selon la taille de lot), quantification FP8/NVFP4 GEMM/MoE invariante, all-reduce déterministe.
**Chaque opérateur est un problème d'invariance SÉPARÉ — corriger l'un ne corrige pas les autres.**

**vLLM `VLLM_BATCH_INVARIANT=1`** : bêta, route via des alternatives déterministes ; fichiers pertinents
`vllm/model_executor/layers/batch_invariant.py` (chemin matmul persistant), `.../quantization/fp8.py`
(chemin batch-invariant FP8), `vllm/v1/attention/` (sélection de backend), `tests/v1/determinism/
test_batch_invariance.py`, `docs/features/batch_invariance.md`. **Réserve documentée par vLLM lui-même** :
la garantie porte sur la taille/l'ordre du LOT, PAS une garantie générale couvrant chaque backend entre
q_len=1 et q_len=k+1, spécialement pour backends de quantification non supportés, noyaux fusionnés
personnalisés, MoE, GDN/attention linéaire, formes de capture CUDA graph différentes.

**Protocole de localisation décisif recommandé** : comparer les deux exécutions à CHAQUE frontière — projections
Q/K/V, sortie attention pleine, sortie projection attention, sortie GDN, résidu post-attention, sortie
RMSNorm, entrée MLP, **sortie GEMM MLP nvfp4**, résidu post-MLP, RMSNorm finale, logits de tête. Pour
l'attention : forcer la config la plus conservatrice (désactiver le split KV si possible, `num_splits=1`
FA3, taille de split fixe FlashInfer/Triton, matérialiser/mettre à jour le KV cache AVANT l'attention,
vérifier que la vérification spéculative utilise la MÊME disposition physique du KV et les mêmes frontières
de bloc que le décodage, désactiver les changements de frontière du préfill par blocs pendant l'expérience).
Pour le MLP : forcer M=1 ET M=k+1 à travers LA MÊME implémentation et config de tuile ; si le backend NVFP4
ne peut pas, déquantifier temporairement et faire tourner un GEMM BF16 commun — si l'écart disparaît, le
noyau NVFP4 est confirmé coupable.

**Classement de plausibilité pour votre cas (GDN et int8 déjà au bit)** : 1) frontière split-KV/cache-courant
de l'attention pleine ; 2) sélection de noyau/tuile du GEMM MLP nvfp4 ; 3) changement de tuile
softmax/réduction de l'attention ; 4) une normalisation ou réduction de résidu cachée ; 5) spécialisation de
forme CUDA-graph/épilogue fusionné ; 6) échantillonnage/log-softmax, SI l'écart n'apparaît que sur les IDs de
jetons (pas sur les logits bruts).

**Sources** : thinkingmachines.ai (« Defeating Nondeterminism in LLM Inference »), GitHub
(thinking-machines-lab/batch_invariant_ops ; vLLM `model_executor/layers/batch_invariant.py`,
`quantization/fp8.py` ; issue #27059 batch-invariant VLMs ; issue VLLM_BATCH_INVARIANT crashes NVFP4 ; issue
tracking SGLang déterministe), lmsys.org (« Towards Deterministic Inference in SGLang »), vllm.ai (doc Batch
Invariance).

---

## Q26 — recouvrement pipeliné (async scheduling/overlap scheduler) + spéculation ngram : alternance amorce/vidage coûteuse (chef/poste5, 277)

**Réponse (Luna, 1 avis, très rigoureux — marque explicitement ce qui n'est PAS sourcé, comme demandé)**.

**Résumé court** :
- **vLLM** : recouvrement (« async scheduling ») et spéculatif sont CONÇUS pour tourner ensemble, activés par
  défaut par défaut pour les méthodes supportées — y compris un travail DÉDIÉ pour le ngram GPU (pas hérité
  automatiquement du spéculatif à base de modèle).
- **SGLang** : le « overlap scheduler » et le spéculatif coexistent pour EAGLE/Spec-V2, **mais
  l'implémentation ngram actuelle DÉSACTIVE explicitement le overlap** (sglang-jax issue #927 : « overlap
  scheduling remains disabled on this path »).
- **Hystérésis** : **SGLang EN A UNE, réelle et documentée**, mais **UNIQUEMENT pour EAGLE/EAGLE3, PAS pour
  le ngram** — le support adaptatif ngram est un chantier séparé (PR #23629, feuille de route). **Aucune
  hystérésis ngram trouvée ni chez vLLM ni chez SGLang.**
- **Changement de mode en vol** : **NI L'UN NI L'AUTRE ne documente un « drain/flush » général du pipeline
  au changement de mode**. Le mécanisme normal est différent : compter explicitement les jetons ACCEPTÉS par
  pas, corriger l'état hôte optimiste après le résultat de vérification côté GPU, ne committer que le
  préfixe accepté — pas de vidage global nécessaire dans le cas ordinaire « pas de correspondance
  ngram/correspondance trouvée ».

**vLLM, détail** : issue de suivi #28947 (« Legacy Effort: Asynchronous Scheduling Support ») dit
littéralement « Asynchronous Scheduling is now enabled by default in vLLM even with Speculative Decoding » ;
travail de compatibilité listé : PR #24799 (compatibilité de base), #30495 (pénalités), #29223 (logprobs),
#31336, #29821 (sortie structurée). **Le ngram GPU compatible async est un item SÉPARÉ, PR/issue #29184** —
preuve que la compatibilité n'est pas automatique pour ngram spécifiquement. Code : `vllm/config/scheduler.py`
(`SchedulerConfig.async_scheduling`, `get_scheduler_cls()` → `AsyncScheduler` ou `Scheduler` classique).
Limite connue : issue #29134 documente des points de synchronisation hôte/device RESTANTS dans le
spéculatif async (`seq_lens_cpu`, comptage de jetons acceptés), fichiers
`vllm/v1/worker/gpu_model_runner.py::_get_valid_sampled_token_count` — une limite de PERFORMANCE, pas une
preuve que l'async est désactivé.

**Sur la disponibilité ngram (pas de correspondance)** : pour vLLM, « spéculatif activé » est normalement une
config STATIQUE de requête/moteur ; le proposeur ngram peut produire zéro jeton de brouillon utile sur un
pas donné SANS reconfigurer dynamiquement le planificateur ni changer le mode async du moteur. **Luna n'a
PAS trouvé de déclaration directement sourcée d'un contrôleur d'hystérésis basé sur le taux d'acceptation
pour le ngram vLLM** — deux mécanismes distincts existent (longueur de spéculation configurée fixe ;
comportement de proposition/vérification par pas, y compris accepter peu ou aucun jeton proposé), à ne pas
confondre avec le travail vLLM plus large de « dynamic speculative decoding »/« adaptive verification »
(capacités distinctes, documentées séparément).

**Comment vLLM évite les doublons sans vidage global** : les candidats spéculatifs ne sont JAMAIS committés
aveuglément — vérifiés, seul le préfixe accepté + le jeton de remplacement/bonus approprié est committé.
Ancien code : `vllm/vllm/engine/output_processor/multi_step.py` (positions de sortie spéculatives
invalides marquées -1, retirées avant append). Historique : `vllm/spec_decode/batch_expansion.py` (expansion
du lot pour les positions de vérification, contraction après rejet/acceptation). **Actuel (V1 async)** :
`vllm/v1/core/sched/{async_scheduler,scheduler}.py`, `vllm/v1/worker/gpu_model_runner.py`,
`vllm/v1/request.py` — le planificateur peut avancer l'état HÔTE de façon optimiste pendant qu'un pas GPU
précédent est en vol ; **le résultat côté device fait autorité, l'état hôte est corrigé APRÈS** (décrit
explicitement dans vLLM-Ascend issue #17479 : la vérification spéculative détermine le vrai
`num_computed_tokens`/positions/longueurs sur le device, l'état hôte optimiste est corrigé ensuite). **Luna
n'a PAS trouvé de source documentant une opération générale « changer de mode en vol, vider toute la file
async, puis redémarrer »** — seulement une correction par pas, pas un protocole de vidage global pour un
changement de mode dynamique arbitraire.

**SGLang, détail** : overlap scheduler = prépare le lot suivant côté CPU pendant que le GPU exécute le lot
courant (événements CUDA pour résoudre les dépendances), activé par défaut sauf `--disable-overlap`. Suivi
spéculatif+overlap : issue #11762 (« Overlap Spec Support ») — support EAGLE overlap déjà présent au moment
de l'issue, **ngram listé comme item de suivi SÉPARÉ**. Preuve la plus directe pour ngram : sglang-jax issue
#927, contrainte d'implémentation actuelle explicite « overlap scheduling remains disabled on this path »,
travail futur = cache thread-safe/overlap-safe AVANT de pouvoir activer l'overlap pour ngram.

**Hystérésis SGLang, code réel** : `python/sglang/srt/speculative/adaptive_spec_params.py`, classe
`AdaptiveStepSlot` — suit la longueur d'acceptation par EMA, ajuste `num_steps`. Mécanismes documentés :
`ema_alpha`, `update_interval`, `warmup_batches`, `up_hysteresis`, `down_hysteresis`, ensembles de pas
candidats (ex. `candidate_steps: [1,3,5,7]`, `up_hysteresis: 0.0`, `down_hysteresis: -0.25`). Intention
explicite dans le code : augmenter les pas spéculatifs quand les brouillons sont acceptés régulièrement,
diminuer en cas de rejet précoce, lisser par EMA pour éviter l'oscillation, mise à jour PÉRIODIQUE (pas à
chaque pas) pour la stabilité. **MAIS `adaptive_unsupported_reason()` limite actuellement l'adaptatif à
EAGLE/EAGLE3 et rejette les autres algorithmes** — le ngram adaptatif est un item de feuille de route séparé
(PR #23629 « feat: adaptive spec support ngram »).

**Séquence ngram SGLang (sglang-jax #927)** : `SpeculativeAlgorithm.NGRAM`, `NgramCache`, `NgramWorker`,
`NgramVerifyInput` — brouillon → allocation/préparation des créneaux de vérification → vérification cible →
conservation du préfixe accepté → libération IMMÉDIATE des créneaux rejetés → pas suivant construit depuis
l'état de requête corrigé. **Puisque l'overlap est désactivé sur ce chemin actuellement, la question du
vidage d'un pas overlap-ngram en vol ne se pose pas dans cette implémentation.** Pour l'EAGLE overlap-activé,
un bogue réel existe (issue #18168, accès mémoire illégal EAGLE+overlap scheduler) — un rapport de bogue,
PAS une documentation d'un protocole général de vidage.

**Recommandation pour la 277** : votre situation (vidage nécessaire au passage simple→spéculatif, alternance
coûteuse ngram) N'A PAS d'équivalent documenté directement sourcé chez vLLM ou SGLang pour le ngram — ni l'un
ni l'autre n'a d'hystérésis ngram publique à imiter telle quelle. Le patron d'hystérésis SGLang pour
EAGLE/EAGLE3 (EMA + `up_hysteresis`/`down_hysteresis` + mise à jour périodique, pas à chaque pas) est un bon
GABARIT à adapter au ngram vous-mêmes, mais ce n'est PAS un mécanisme existant à réutiliser directement pour
ngram. Vérifier d'abord si votre 55 pas pour 32 jetons vient bien de l'alternance amorce/vidage (comme
supposé) ou d'un défaut de bookkeeping similaire à celui documenté par vLLM issue #29134/#17479 (correction
d'état hôte optimiste mal faite) avant d'investir dans un contrôleur d'hystérésis complet.

**Sources** : GitHub (vLLM issue #28947 async+spéculatif, PR #24799/#30495/#29223/#31336/#29821, issue/PR
#29184 ngram GPU async, issue #29134 sync hôte/device restante, `vllm/config/scheduler.py`,
`vllm/v1/core/sched/{async_scheduler,scheduler}.py`, `vllm/v1/worker/gpu_model_runner.py`,
`vllm/spec_decode/batch_expansion.py`, `vllm/engine/output_processor/multi_step.py` ; vLLM-Ascend issue
#17479 ; SGLang issue #11762 Overlap Spec Support, issue #18168 EAGLE+overlap illegal memory access, PR
#23629 adaptive spec ngram, roadmap Further Ngram Speculative Decoding Support,
`python/sglang/srt/speculative/adaptive_spec_params.py` ; sglang-jax issue #927 ngram overlap désactivé),
lmsys.org (SGLang v0.4 Zero-Overhead Batch Scheduler), vllm.ai (doc Speculative Decoding, doc
`vllm.config.scheduler`).

## Q27 — Regroupement du préfill pour modèles hybrides GDN/Mamba (chef, veille pièce 274, pour poste1 284)

**Contexte posé** : préfill mixte sans gain de lot, TTFT 0,48 → 5,4 s de b=1 à b=12. Question à Luna (GPT-5.6,
un seul avis pour cette question — pas de second modèle interrogé faute de temps, à signaler à chef/poste1
si un second avis est voulu) : chunked prefill sur l'état récurrent, restrictions de mélange préfill/décodage,
stockage de l'état SSM par requête.

**Chunked prefill — vLLM** : `vllm/v1/attention/backends/mamba_attn.py`,
`BaseMambaAttentionMetadataBuilder._compute_chunk_metadata/_build_chunk_metadata_tensors/_prefill_cpu_metadata`.
Un chunk Mamba ne contient les jetons que d'UNE SEULE requête (contrainte documentée dans le code), état
récupérable seulement aux frontières `chunk_size` (tenseurs `cu_chunk_seqlens`, `seq_idx`,
`last_chunk_indices`). Différent de l'attention classique, qui peut aplatir plusieurs requêtes dans un même
`cu_seqlens`. Mode `--mamba-cache-mode align` : état disponible seulement aux frontières de bloc/chunk alignées,
pas au grain du jeton (doc LMCache hybride, GDN non supporté en mode `all`). Conséquence signalée : la taille
d'état Mamba peut forcer une taille de bloc d'attention PARTAGÉE plus grande que l'optimum du noyau d'attention
seul.

**Chunked prefill — SGLang** : `hybrid_linear_attn_backend.py`, `linear/gdn_backend.py`,
`linear/kernels/gdn_flashinfer.py`, `mem_cache/mamba_radix_cache.py`. Feuille de route SGLang elle-même liste
comme limitations actuelles : stockage d'état tous les k pas / aux points de branchement, stockage d'état
intermédiaire GDN/Mamba, tailles de page > 1, ordonnancement en chevauchement pour hybrides (non sourcé plus
précisément que la liste de roadmap — Luna cite "GitHub" sans numéro exact ici). Une proposition récente définit
`mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, page_size)` — l'état a bien sa propre granularité de chunk,
distincte du pagage de l'attention.

**Mélange préfill/décodage — vLLM** : PAS d'interdiction générale, mais les lignes doivent être classées et
réordonnées (`GPUModelRunner.calculate_reorder_batch_threshold`, `reorder_batch_to_split_decodes_and_prefills`,
`Mamba2AttentionMetadataBuilder.build`). Un bogue réel cité (vLLM, numéro d'issue non précisé par Luna au-delà
de la description) : si le seuil de réordonnancement global est abaissé par un autre backend, des lignes de
décodage peuvent traverser les noyaux de préfill Mamba, qui n'écrivent que le créneau d'état ORDINAIRE —
faute de correction pour le spéculatif (créneaux dédiés requis) → corruption, pas juste perte de perf.
`cu_chunk_seqlens`-style groupement confirmé, mais restreint : un chunk ne traverse jamais deux requêtes.

**Mélange préfill/décodage — SGLang** : chemin mixte explicite (`ScheduleBatch.prepare_for_extend`,
`Scheduler.get_new_batch_prefill`, `mix_with_running`, `ScheduleBatch.merge_batch`). Bogue cité en détail par
Luna (issue SGLang, numéro non précisé au-delà du contenu) : `merge_batch()` effaçait `mamba_track_indices`,
`mamba_track_mask`, `mamba_track_seqlens`, empêchant `_track_mamba_state_extend` de tourner sur le lot mixte
alors que le créneau d'état était déjà réservé → corruption de sortie ET dégradation de latence substantielle
rapportées. Défaut d'UNE version, pas preuve d'un défaut universel actuel.

**Stockage de l'état récurrent par requête — vLLM** : V0 (ancien) : tenseur par séquence active × couche,
dimensionné sur `max_num_seqs` — pas analogue à un cache KV paginé. V1 : allocateur hybride unifié
(`vllm/v1/kv_cache_interface.py`, `vllm/v1/core/kv_cache_manager.py`) qui groupe les couches par type et
présente des vues différentes sur une mémoire physique partagée ; pour un groupe Mamba, l'entrée contient
l'état récurrent (conv + SSM temporel), pas du K/V par jeton. Answer explicite à la question posée (« pool
dédié analogue au gestionnaire de blocs KV ? ») : **oui en fonction, pas forcément en allocateur totalement
indépendant** — l'état Mamba peut partager l'allocation physique sous-jacente avec les groupes d'attention
(padding/alignement pour compatibilité). Note importante de Luna : le travail de service désagrégé P/D cité
couvre Mamba2 en premier et liste GDN comme limité/roadmap — ne pas supposer que Qwen3.5/GDN suit exactement
le même mécanisme de transfert d'état que Mamba2 seulement parce que les deux sont "hybrides SSM".

**Stockage — SGLang** : allocateur séparé et plus visible : `HybridReqToTokenPool` (association
requête→créneau d'état Mamba), `HybridLinearKVPool` (mappage couche linéaire→tenseurs d'état), `MambaRadixCache`
(instantanés d'état réutilisables pour le partage de préfixe). Différence clé avec le KV classique : un hit
radix-cache sur un état Mamba force une COPIE dans une nouvelle région (la requête en cours modifiera son état
en place — partager le même tenseur entre deux requêtes actives créerait une interférence), alors qu'une page
KV peut simplement être référencée. Tenseurs de suivi du mélange préfill/décodage : `mamba_track_indices`,
`mamba_track_mask`, `mamba_track_seqlens`, `mamba_next_track_idx`, `mamba_last_track_seqlen`.

**Lecture de votre symptôme (0,48→5,4 s, b=1→12), explicitement qualifiée par Luna d'INFÉRENCE non sourcée par
le code, pas de diagnostic direct** : combinaison plausible de (a) budget de jetons du planificateur consommé
par le préfill croissant, (b) chunks Mamba non aplatissables entre requêtes → chaîne récurrente + métadonnées
de créneau par requête, (c) mélange préfill/décodage empruntant un chemin de noyau moins favorable
(classification de ligne + gestion d'index d'état), (d) état mis à jour EN PLACE limitant le batching/
checkpointing arbitraire, (e) alignement de bloc/chunk large imposé par la taille d'état pouvant sur-calculer,
(f) recouvrement de lancements de noyaux (préfill attention, préfill GDN, écritures d'état, copies d'état,
décodage) dominant le débit. Mesures recommandées par Luna pour isoler la cause chez vous : préfill pur b=1..N,
décodage pur b=1..N, mixte à jetons de décodage fixes, mixte à budget de jetons total fixe ; instrumenter
`num_prefills`, `num_decode_tokens`, jetons réellement planifiés, taille de chunk Mamba/GDN, taille de bloc
d'attention, nombre de lancements forward, nombre de lancements de copie/scatter d'état, si les lignes
préfill/décodage ont été réordonnées/scindées.

**Non sourcé / à vérifier** : les numéros exacts d'issue GitHub vLLM et SGLang cités ci-dessus n'ont pas été
donnés précisément par Luna dans le texte récupéré (citations `<citation src="N">` sans résolution du numéro
réel affiché) — à retrouver soi-même si une pièce en dépend. La liste de limitations de la roadmap SGLang
(état tous les k pas, page_size>1, etc.) n'est pas attachée à un numéro de PR/issue précis dans la réponse.
Le mécanisme de transfert d'état désagrégé P/D pour GDN spécifiquement (vs Mamba2) est signalé par Luna
lui-même comme non confirmé.

**Sources** : GitHub vLLM (`vllm/v1/attention/backends/mamba_attn.py`, `vllm/v1/kv_cache_interface.py`,
`vllm/v1/core/kv_cache_manager.py`, `GPUModelRunner.calculate_reorder_batch_threshold`,
`reorder_batch_to_split_decodes_and_prefills`, `Mamba2AttentionMetadataBuilder.build`, doc hybride P/D
désagrégé), GitHub SGLang (`hybrid_linear_attn_backend.py`, `linear/gdn_backend.py`,
`linear/kernels/gdn_flashinfer.py`, `mem_cache/mamba_radix_cache.py`, `managers/schedule_batch.py`,
`HybridReqToTokenPool`, `HybridLinearKVPool`), doc LMCache (mode `--mamba-cache-mode align`), pytorch.org (doc
hybride vLLM) — numéros d'issue/PR précis non résolus dans la réponse récupérée, cf. paragraphe « non sourcé »
ci-dessus.
