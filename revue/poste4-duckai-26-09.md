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
