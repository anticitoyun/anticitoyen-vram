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
