# poste7 — Après la campagne : la prochaine priorité est les hybrides (GDN, KDA, Mamba2), le seul régime où acvram est 6-10× derrière ; puis deux profils avant tout nouveau chantier de noyau (17/09)

Entrée : chef — TRT-LLM clos sans cellule (poste3 f0163a9 : `gdn_mixer.py:441`, `linear.py:1653`, rc15 ne suit pas ces architectures) ; paliers 0-2 clos ; menus 97/142 (poste8) ; 70B +89 % ouvert.

## 1. Où en est l'objectif (classés seulement, chiffres des verdicts)

| régime | acvram | meilleur classé | tout moteur | écart acvram |
|---|---|---|---|---|
| Coder b=12 t/s / J | **997,8 / 0,400** | EXL3 855,7 / 0,362 | vLLM Marlin 2 031 / 0,197 | +17 % t/s, −10 % J vs EXL3 ; ÷2 vs Marlin |
| Coder b=1 | 287,1 / 1,185 | llama.cpp 341,4 / 1,108 | — | −16 % (poste F terminé) |
| Coder prefill j/s | 9 913 | llama.cpp 15 717 | Marlin 20 988 | −37 % ; ÷2,1 |
| GLM b=12 | 568,1 / 0,700 | acvram seul classé sous 1,02 | vLLM Marlin classé qualité 1,0164 | qualité gagnée, vitesse non comparée |
| **Qwen3.8-27B b=12 (GDN torch)** | **97** | llama.cpp 323 | vLLM 621 | **÷3,3 ; ÷6,4** |
| Nemotron-3.5-30B-A3B (Mamba2 torch) | 266 b=1, non classé (srcexl3 1,0795) | vLLM 401 b=1 / 1 799 b=12 / 0,221 J | — | non classé + ÷1,5 b=1 |
| hybrides, prefill (palier 1) | Falcon-H1R 597, Nemotron-Nano 998 j/s | — | — | **÷10** |

Lecture : sur denses et MoE ordinaires, acvram tient (b=12 en tête des classés, qualité GLM). Sur les hybrides — quatre familles du parc : Qwen3.5/3.8 (GDN), Kimi-Linear (KDA), Nemotron-H (Mamba2), Falcon-H1 — la récurrence tourne dans la référence torch de `transformers` (`gdn.py:31-32, 117` : `torch_chunk_gated_delta_rule` au prefill, `torch_recurrent_gated_delta_rule` au décodage) et c'est là que la table des menus va imprimer un ÷6 à ÷10. **C'est le plus gros multiplicateur disponible, et le moins cher.**

## 2. Priorité 1 — hybrides : les noyaux Triton de `fla`, pas un noyau maison

Fait vérifié ce soir : `.venv/bin/python` (torch 2.14.0+cu130, triton 3.8.0) **n'a pas `flash-linear-attention` (`fla`)**, ni `mamba_ssm`, ni `causal_conv1d`. Or `kda.py:32-37` et `mamba2.py:9` sont déjà écrits pour prendre `fla.ops.kda.chunk_kda` et `chunk_simple_gla` « si disponible » — ils tournent donc en torch aujourd'hui, silencieusement. `gdn.py` n'a même pas la branche. C'est ce que vLLM utilise pour ses 621 t/s : même mathématique, noyau Triton fusionné (un lancement par couche au lieu de dizaines).

Chantier (poste4 à sec, poste3 une fenêtre) :
1. `pip install flash-linear-attention` dans `.venv` ; contrôle à sec : `fla.ops.gated_delta_rule.{chunk_gated_delta_rule, fused_recurrent_gated_delta_rule}` s'importent et compilent sous triton 3.8 / sm_120 sur un tenseur jouet (faux si l'import ou le JIT échoue : alors version de fla à épingler, ou noyau maison — décision séparée).
2. `gdn.py` : branche `fla` au prefill (`chunk_gated_delta_rule`) et au décodage (`fused_recurrent_gated_delta_rule`, batch = séquences actives, état `S` par séquence conservé), gardée par une variable de régime `ACVRAM_GDN=fla|torch` **portée par `regime_ligne()`** (REGLES § 4 : le régime dans la signature). Même chose pour le décodage de `kda.py` et `mamba2.py` (`fused_recurrent_kda`, `fused_recurrent_simple_gla` ou `mamba_ssm.selective_state_update`).
3. **Test d'équivalence dans le même commit** (règle 9) : logits torch vs fla sur 64 jetons × b=3, `max |Δ|` ≤ 2⁻⁷ relatif en bf16, PPL 1 tranche ± 0,002 ; bras cassant : inverser l'axe [K,V] de l'état (le piège que `kda.py:137` décrit) doit le rendre rouge.
4. Fenêtre carte (poste3, ≤ 40 min) : Qwen3.8-27B calibA, mêmes instruments que palier 2.

Scellés, écrits avant : Qwen3.8 b=12 **97 → ≥ 400 t/s** (faux si < 250 : la récurrence n'est pas le goulot, profiler avant de continuer) ; prefill 2048 hybrides **≥ 5× la voie torch** ; PPL inchangée ± 0,002 ; b=1 : **dénominateur = calibA 63,8 t/s (le converti de la fenêtre), pas srcexl3 67** (poste1, `avis-poste1-protocole-hybrides-17-09`) — et ce 63,8 est déjà bridé en puissance, donc le juge b=1 est double : **t/s ≥ 85** et **J/jeton ≤ 0,75 × le JSON calibA du même instrument** ; colonnes W et horloge SM dans le JSON. Si W plafonne sur les deux bras, seul le J/jeton juge (un noyau fusionné fait moins de travail par jeton : il doit se voir en joules même sous plafond) ; faux si J/jeton > 0,9 ×. Avant la fenêtre : `regime_ligne()` doit porter `ACVRAM_GDN` — contrôle par le JSON réel d'un chargement à sec (`[régime]` contient `ACVRAM_GDN=fla`), pas par lecture du code. Issue qui me gênerait : fla à peine 1,5× plus rapide parce que les 124 ms/pas sont ailleurs (projections, conv causale, normalisation gated — alors le profil dit lequel).

Prérequis à sec, poste2 (1 h) : **Nemotron-3.5-30B-A3B srcbf16** (bf16 sur disque depuis le palier 2) → entrée classable ; le srcexl3 (double quantification, 1,0795) était condamné d'avance. Corpus bras A. Puis poste3 : PPL 3 tranches dans la même fenêtre que Qwen3.8.

## 3. Priorité 2 — deux profils avant tout nouveau noyau (poste3, une fenêtre de 30 min, torch.profiler, pas ncu)

* **prefill Coder 2048** : part dequant NVFP4→bf16 / GEMM bf16 / attention / MoE routage. C'est −37 % vs llama.cpp et ÷2,1 vs Marlin, sur *tous* les modèles. Décision après : si dequant + GEMM ≥ 60 %, le candidat est W4A8 sur les experts routés avec expert partagé en bf16 (le 1 % venait du partagé, `fp4_gemm.py:257` ; l'ancien w8a8 requantifiait les poids à chaque appel, +56 % — c'était un défaut d'implémentation, pas du format) ; sinon, l'attention ou le routage.
* **décodage Coder b=12** : les trois premiers noyaux par temps. ÷2 vs Marlin sur la cellule phare ; je ne propose rien tant que le profil n'est pas écrit — les postes E/C/B0/F ont pris tout ce que la conjecture pouvait donner.

KV 4 bits (conçu 8e7a78e, poste4 débloquée) : **après** la priorité 1 — gain = capacité (slots, contexte), pas t/s sur les cellules du comparatif (à b=12/2048 le pas lit 17 Go de poids MoE contre < 1 Go de KV).

## 4. À sec, en parallèle, sans décision de ma part

* poste2 : corpus de calibration par défaut = bras A dans `collect.py` (verdict calibration, « deux correctifs à sec »), avec son test. 30 min.
* poste2, 70B +89 % (pas prioritaire, un seul contrôle) : même pipeline `srcQ4_K_M → nvfp4` sur un Llama-3.1-8B, PPL vs sa source GGUF ; +89 % aussi ⇒ architecture llama dans acvram (rope `llama3`, `layers.py:646-666`, ou GQA 8 têtes kv) ; ≈ 1,01 ⇒ propre au 70B (i1 imatrix : types de tenseurs) et on ferme.
* poste8 : menus, contrôle chiffre↔verdict (déjà commandé).
* poste1 (session neuve) : revue du protocole hybrides ci-dessus avant la fenêtre — c'est son poste.

## 5. Décisions qui appartiennent à l'utilisateur (une ligne chacune, pas de mesure engagée avant)

1. Installer le `.deb` (0.6.x, 340 Ko, `sudo dpkg -i`) — produit, prêt depuis le 11/09.
2. GUI `/convertir` (poste2, après hybrides) — produit, pas mesure.
3. gpt-oss-120b → lecteur MXFP4 (`poste7-convertisseur-formats` § 3.2) — chantier de format, ~2 j.
4. Gemma-4-31B-it bf16 téléchargé ? → `gemma-4-31B-it-nvfp4` srcbf16 (poste2) + témoin bf16 (poste3), une ligne classable de plus ; TRT-LLM n'y reviendra pas avant une version qui suit `gemma4`.
