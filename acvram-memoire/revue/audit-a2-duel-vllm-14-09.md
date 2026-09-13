# Audit A2 — premier duel vLLM/TabbyAPI depuis la réinstallation

poste2, 14/09/2026. Consigne de chef : valider vLLM et TabbyAPI sur
Qwen3-Coder-30B-A3B, régime pp2048, même NVML power.draw.instant (idle
soustrait), même fenêtre, contre le chiffre du jour d'acvram (16 938 j/s,
poste4, cp.async).

## Modèle vLLM

Aucun checkpoint NVFP4 local ne correspondait exactement à
Qwen3-Coder-30B-A3B pour vLLM (seul `Qwen3.6-27B-NVFP4`, un autre modèle,
existait dans `models_vllm/`). **Téléchargé avec accord explicite de
l'utilisateur** (18,1 Gio) : `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4`
(NVIDIA ModelOpt, base `Qwen/Qwen3-Coder-30B-A3B-Instruct`, le même modèle
de base que notre conversion acvram) → `models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4/`.

## Dénominateur commun (poste4, 14/09)

Ma première mesure était fausse (cache de préfixe : invite identique à
chaque répétition, `enable_prefix_cache` par défaut → un pas de ~27 ms
quelle que soit L, débit gonflé). poste4 a pointé son script canonique
(`outils/banc_prefill_chaud.py`, branche poste4 ea98f36, 16 938 j/s) et
son dénominateur : `L / durée d'un generate(max_tokens=1) complet`,
synchronisé aux deux bouts, invite **différente** à chaque répétition,
cache de préfixe désactivé, 2 passes de chauffe + 7 répétitions, médiane.
`outils/banc_prefill_vllm.py` (ce commit) reproduit ce protocole via
`vllm.LLM.generate()` en offline (pas de serveur HTTP — évite de mesurer
autre chose, comme `vllm bench latency` l'aurait fait selon poste4).

## Trois obstacles d'infrastructure (« depuis la réinstallation »)

1. **FlashInfer absent du venv `/opt/ia/vLLM/.venv`** — `RuntimeError:
   FlashInfer backend is not available` au premier appel réel (backend
   choisi automatiquement). Pas réparé (installation du paquet hors de
   mon périmètre décidé ici) — contourné.
2. **FLASH_ATTN refuse le cache KV FP8** sur notre 5090 (sm_120) :
   `ValueError: ... FP8 KV cache requires FA3 on SM90 or FA4 on SM100`.
   Contourné par `attention_config={"backend": "TRITON_ATTN"}`.
3. **`max_model_len` par défaut (262 144, contexte max du modèle)**
   réservait plus de cache KV (12 Gio) que de VRAM libre (6,43 Gio) pour
   un seul prefill de 2048 jetons. Fixé à 4096, comme
   `banc_prefill_chaud.py` côté acvram.

Aucun de ces trois n'est un défaut d'acvram — tous trois sont propres à
l'installation vLLM de ce poste, pas mesurés avant aujourd'hui (« premier
duel depuis la réinstallation », chef).

## Piège de puissance trouvé sur ma propre mesure

Envelopper tout le sous-processus vLLM (chargement du modèle + boucle de
mesure) dans un seul relevé NVML donnait 3 W nets (67 W fenêtre, 64 W
idle) — implausible pour du calcul réel sur une 5090. Cause : le
chargement (~30-60 s, quasi-idle côté GPU pendant la lecture des poids)
domine la fenêtre et dilue la médiane vers l'idle. Corrigé en déplaçant
l'échantillonnage NVML **à l'intérieur** de `banc_prefill_vllm.py`,
autour de la seule boucle de mesure (2 chauffes + 7 répétitions) — la
puissance de charge, elle, reste correcte pour `banc_prefill_chaud.py`
côté acvram (mesurée de l'extérieur, chargement apparemment plus court
avec les noyaux déjà en cache).

## Résultat

| moteur | pp2048 (j/s) | σ | ms/pas | puissance idle | puissance nette |
|---|---|---|---|---|---|
| acvram (MMA, S=4) | 17 111 | 37 | 119,7 | 17 W | 78 W |
| vLLM (TRITON_ATTN) | 34 788 | 1 441 | 58,9 | 64 W | 99 W |

**vLLM ≈ 2,03× plus rapide en prefill brut sur ce régime**, avec une
dispersion (σ=1441, ~4 % du débit) nettement plus large que la nôtre
(σ=37, ~0,2 %) — cohérent avec l'observation de poste4 du 13/09 sur le
duel llama.cpp (notre dispersion 22-31 % contre 1,5 % pour l'adversaire) :
un adversaire mieux industrialisé peut être plus rapide ET plus stable,
ou l'un des deux seulement — ici, plus rapide mais moins stable.

**Réserve sur l'idle** : 17 W (acvram) contre 64 W (vLLM) pour la même
carte, mesurés à des instants différents — la 5090 partage un bus PCIe
x8/x8 avec la 3080 Ti et d'autres sessions tournaient en parallèle
(carte.sh attendue avant chaque mesure) ; l'écart d'idle n'est peut-être
pas structurel, à vérifier si la comparaison doit trancher plus finement
que ×2.

## Ce qui reste

TabbyAPI (EXL3 4.0bpw, `models_exl3/Qwen3-Coder-30B-A3B-4.0bpw-EXL3`) —
pas d'API offline `generate()` aussi directe que vLLM/acvram dans ce
dépôt : exllamav3 n'expose qu'un usage serveur (TabbyAPI lui-même). Un
client HTTP avec le même dénominateur (measurer TTFT du premier jeton
d'un `generate(max_tokens=1)`, invite différente par répétition) reste à
écrire.
