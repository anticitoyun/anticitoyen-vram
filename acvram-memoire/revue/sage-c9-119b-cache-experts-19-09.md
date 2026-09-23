# Sage — C9 ouvert (oui utilisateur 17 h 35) : Mistral-Small-4-119B et le cache d'experts ; horizon = fin d'abonnement dimanche 20/09 08 h 00 ; GLM ne décode pas sous P1 (19/09, 17 h 55)

Source : utilisateur 17 h 35 (« le cache d'experts/119B c'est OUI ») et 17 h 40 (« pas de clôture 23 h 45, le groupe continue jusqu'à la fin de l'abonnement dimanche 20 septembre ») ; heure retrouvée : **08 h 00** (`sage-menus-cloture-19-09` § 3 et `vibe.md` du 19/09 : « abonnement clos demain 08 h ») ; `sage-cache-experts-13-09` (plan M0-M5 scellé) ; `verdict-glm-b12-19-09` (Manon, b92d231) ; Hugging Face, `config.json` de `mistralai/Mistral-Small-4-119B-2603`.

## 0. Horizon et deux urgences

* **Plus de clôture** : la file de carte tourne jusqu'à 08 h 00 ; bilan Sage 07 h 00, dernier push Jérôme 07 h 45. Tout ce qui est dans `sage-cloture-23h59` et `sage-poursuite-chantiers` reste, sans l'arrêt de 22 h 30.
* **GLM-4.7-Flash ne décode pas sous le défaut livré** (P1, 0.6.13 et 0.6.14) : `MoEBlock._grouped` (`model.py:1070`) appelle `nvfp4_gemv_grouped` avec la pile naturelle rendue par la disposition unique → TypeError à la capture, serveur mort au premier pas ; jamais couvert par une capture (18/09 : prefill seul). Un mécanisme arrêté à une dimension (MECANISMES) : le GEMV Marlin couvre la forme MoE de Coder, la disposition unique rend la pile pour tous. Décision : (b) **la disposition unique ne rend la pile que si le GEMV Marlin couvre la forme** (refus nommé dans `_construire_marlin`, chemin d'avant sinon), test qui casse, avant tout chantier ; (a) GEMV Marlin pour 64 experts top-4 + expert partagé = chantier **C10**. Puis **passe de capture godets {1, 2, 8, 16} sur chaque MoE du parc servi** (Jérôme, 5 min chacun, `verdict-capture-parc-19-09`) — la règle « capture avant tout défaut de décodage » vaut par architecture, pas par modèle vedette.
* **Devstral** : `devstral-24b-srcawq-nvfp4` est déjà converti (dense 24B, tient en VRAM) — il n'est pas un cas de cache ; il lui manque ses cellules et sa PPL classée (Manon, 45 min quand la file le permet, scellé ≤ 1,02 contre HF bf16).

## 1. Le modèle (chiffres de `config.json`, pas d'une note)

`Mistral3ForConditionalGeneration`, texte `mistral4` : 36 couches, hidden 4 096, **128 experts routés top-4 + 1 expert partagé** (moe_intermediate 2 048, partagé 12 288), **MLA** (q_lora 1 024, kv_lora 256, nope 64 + rope 64, v 128, 32 têtes), vocab 131 072, YaRN jusqu'à 1 M, tour de vision Pixtral (hors périmètre : texte seul). 119,4 G paramètres.

```
experts routés / couche   128 × 3 × 4096 × 2048 = 3,22 G   → × 36 = 116 G params  ≈ 61 Go en NVFP4
expert partagé            36 × 3 × 4096 × 12288 = 5,4 G   ≈ 2,9 Go
MLA (bf16, REGLES § 9)    ≈ 1 G                            ≈ 2 Go
embeddings + tête         2 × 131072 × 4096               ≈ 1 Go bf16
total texte                                               ≈ 67 Go   contre 32 Go de VRAM (5090) + 12 (3080 Ti) + 76 Go de RAM libre
actifs / jeton            4 × 3 × 4096 × 2048 × 36 = 3,6 G  ≈ 1,9 Go d'experts par jeton en NVFP4
```

**Arithmétique qui cadre tout** : sans aucun succès de cache, 1,9 Go/jeton à 18,7 Go/s (PCIe x8 mesuré 13/09) = **102 ms → 10 j/s à b=1**. Avec un tiers des experts résidents (20 Go) et un taux de succès h, octets = 1,9 × (1 − h) : h = 0,5 → 51 ms → ~18 j/s ; h = 0,7 → 31 ms → ~28 j/s. Le concurrent réel sur cette machine est **llama.cpp avec les experts sur processeur** (`-ot exps=CPU`, RAM ~70-80 Go/s) : ~25-40 ms/jeton théoriques, 10-20 j/s pratiqués. **Le cache d'experts ne se justifie que par h, et h se mesure avant d'écrire une ligne de cache** (M1 du 13/09).

Sources à télécharger (Jérôme, en cours, réseau seul) : `mistralai/Mistral-Small-4-119B-2603-NVFP4` (13 shards `consolidated-*.safetensors`, **70,8 Go**, format natif Mistral + `params.json` : les noms de tenseurs ne sont pas ceux de HF — la « lecture directe » du 17/09 (GLM, dépôt en disposition HF) exige ici une table de correspondance) ; repli prouvé : `unsloth/Mistral-Small-4-119B-2603-GGUF` Q4_K_M (~70 Go, chemin `srcQ4_K_M → nvfp4` déjà utilisé pour Coder) ; `mistralai/Mistral-Small-4-119B-2603-eagle` (392 Mo, tête EAGLE : sur un MoE borné par le PCIe, k jetons vérifiés par pas partagent leurs experts — la spéculation y vaut plus que partout ailleurs, C9-bis).

## 2. Ordre des mesures — prédictions et seuils écrits ici, avant

| étape | qui | quoi | prédiction Sage | seuil (rend « faux ») |
|---|---|---|---|---|
| M0 | Manon, 10 min carte | bande PCIe dans le moteur (`acvram bench --what bandwidth`, copie plate 21 Mo) — jamais tranchée (18,7 contre 52,8) | **18,7 Go/s** (x8) | > 40 → tout coût ci-dessous ÷ 2,8 ; écrire la valeur dans les tiers (1792 → 1050) |
| M1 | Océane à sec (1 h), Manon 1 h carte | taux de succès d'experts sur trace réelle (`ACVRAM_TRACE_ROUTAGE`, ≥ 20 requêtes, ≥ 50 000 jetons) — **sur Coder d'abord** (disponible), 119B dès qu'il charge ; `h_pin(C)`, `h_lru(C)`, C ∈ {16, 32, 48, 64, 96}, apprises sur la première moitié, jugées sur la seconde | Coder b=1 `h_lru(64)` = 0,78 ± 0,08 (13/09) ; 119B `h_lru(43 = E/3)` **0,55 ± 0,10** | Δh = h(E/2) − 0,5 **< 0,10** → pas de cache apprenant : exil par couche seul + 3080 Ti ; ≥ 0,25 → cache engagé |
| M2 | Manon 30 min | lecture zéro-copie (noyau groupé sur pointeurs hôte) contre `cudaMemcpyAsync` | 70-85 % de la copie | < 60 % → copie par noyau de rassemblement, graphes par segment (M4) |
| M3 | Océane 2 h + Manon 30 min | prototype une couche : identité `torch.equal` pile contiguë ↔ pile à table (tout écart = bogue) ; temps 12 couches à moitié résidentes, `pin` contre `aveugle` | pas 0,56 × en octets PCIe | gain < 15 % du pas à VRAM égale → cache fermé, exil par couche |
| C9-charge | Manon | 119B : `ModelSpec` `mistral4` (MLA + MoE + expert partagé + YaRN ; vision ignorée), table de correspondance consolidated → HF, chargement avec exil (VRAM 5090 : MLA + partagé + tête + 20 Go d'experts ; reste en RAM), PPL 3 tranches contre HF bf16 **non mesurable ici** (238 Go) → contre la PPL publique du GGUF Q4_K_M de llama.cpp sur le même corpus, colonne à part | PPL ratio NVFP4 officiel / Q4_K_M ≤ 1,00 | > 1,02 → source Q4_K_M |
| C9-cellules | Manon | b=1 t/s et J, prefill 2 047, contre llama.cpp `-ot exps=CPU` même GGUF, harnais égal | acvram b=1 **≥ 15 j/s** avec h ≥ 0,5 ; llama.cpp 10-20 | acvram < llama.cpp → la 3080 Ti calculante (§ 6 du 13/09) ou rien |

## Ordre

* **Océane** — (0) correctif GLM (b) + test, pointeur à Manon pour G1 ; (1) C9-M1 à sec : `taux_de_succes` étendu (`pin` par couche, capacité par couche, distincts par pas), trace Coder à faire prendre par Manon ; (2) C9-M3 prototype une couche (placement par expert, `_tables_adresses` existe) après M1 ; (3) C10 GEMV Marlin top-4 + partagé ; le reste de `sage-poursuite-chantiers` inchangé, C2 et C1 restent devant C10.
* **Manon** — file du soir inchangée ; M0 (10 min) dans le premier trou ; trace M1 Coder (1 h, b=1 puis b=12, `ACVRAM_DISABLE_CUDA_GRAPHS=1`, nom de fichier portant le régime) ; C9-charge dès la fin du téléchargement (`ModelSpec` `mistral4`, table consolidated → HF, exil) ; Devstral cellules + PPL quand la file le permet.
* **Jérôme** — téléchargements (NVFP4 officiel, puis GGUF Q4_K_M unsloth, puis EAGLE) vers `/mnt/2TO_2023_980PRO/Modeles/models_acvram/`, progression par `du -sh` dans `ETAT` ; `llamacpp-serveur` du GGUF 119B avec `-ot exps=CPU` prêt pour la cellule concurrente ; capture-parc après le correctif GLM ; `ETAT` : horizon 08 h 00, C9 et C10 dans la file.
* **Sage** — relit M0/M1 dès leur sortie (M1 décide si le cache existe) ; bilan 07 h 00.
