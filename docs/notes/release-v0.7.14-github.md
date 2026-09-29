# acvram v0.7.14 — English / français

## English

Claude Code and kimi model menus fixed across all engines; recovery after an out-of-memory prefill.

- **Real test of the model menus**: 163 aliases started and queried, then 80 aliases under Claude Code and kimi at
  their serving context. The failures found are fixed in this release and in the companion parc package 0.1.8.
- **kimi with llama.cpp, vLLM and the fast engine**: every such alias failed with 401. The parc installer had created
  new random API keys that the kimi configuration did not know; `kimi-modele` now uses the current keys, and the
  installer no longer creates keys when clients already exist (it reuses theirs or refuses with instructions).
- **vLLM on the RTX 5090**: all vLLM aliases died at startup. Triton attention is now the default (FlashInfer needs
  nvcc), and MLA models (GLM) use a bf16 KV cache (the fp8 kernel exceeded the shared memory of sm_120).
- **Claude Code on local models**: the hook text Claude Code inserts as a late system message is now placed before
  the question (0.7.13 put it after, and small models answered the hook instead); relaxed template for llama.cpp;
  Gemma 4 end-of-turn tokens.
- **Engine**: MoE expert stacks are built before the context warm-up (Qwen3-Coder-30B holds 4,096 tokens without CUDA
  graphs instead of 3,072); after an out-of-memory prefill the memory is returned and other requests keep being
  served, instead of failing one after another. The nominal path is unchanged.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.14/acvram-0.7.14.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.14/CHANGELOG.md)

---

## Français

Menus de modèles Claude Code et kimi réparés sur tous les moteurs ; reprise après un préfill en manque de mémoire.

- **Test réel des menus** : 163 alias démarrés et interrogés, puis 80 alias sous Claude Code et kimi à leur contexte
  de service. Les pannes trouvées sont corrigées dans cette version et dans le paquet parc 0.1.8.
- **kimi avec llama.cpp, vLLM et le moteur rapide** : tous ces alias échouaient en 401. L'installateur parc avait créé
  de nouvelles clés d'API aléatoires que la configuration kimi ignorait ; `kimi-modele` prend désormais les clés
  courantes, et l'installateur ne crée plus de clés quand des clients existent (il reprend les leurs ou refuse en
  disant quoi faire).
- **vLLM sur la RTX 5090** : tous les alias vLLM mouraient au démarrage. L'attention Triton est le défaut (FlashInfer
  exige nvcc) et les modèles MLA (GLM) utilisent un cache KV bf16 (le noyau fp8 dépassait la mémoire partagée de sm_120).
- **Claude Code sur les modèles locaux** : le texte de crochet que Claude Code glisse en message système tardif est
  placé avant la question (la 0.7.13 le mettait après, et les petits modèles répondaient au crochet) ; gabarit assoupli
  pour llama.cpp ; jetons de fin de tour de Gemma 4.
- **Moteur** : les piles d'experts MoE sont construites avant la chauffe du contexte (Qwen3-Coder-30B tient 4 096 jetons
  sans graphes CUDA au lieu de 3 072) ; après un préfill en manque de mémoire, la mémoire est rendue et les autres
  requêtes restent servies, au lieu d'échouer l'une après l'autre. Le chemin nominal ne change pas.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.14/acvram-0.7.14.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.14/CHANGELOG.md)
