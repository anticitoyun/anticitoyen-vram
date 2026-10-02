# evp — plan chiffré gpt-oss-120b (exil), à sec

poste1, 02/10, sans carte. Le code est celui du 20b, justesse tenue (`poste1-evp-justesse-02-10.md`) : le 120b n'a
pas d'architecture nouvelle (même GptOss : puits, fenêtre 128 alternée, YaRN, SwiGLU bornée, biais d'experts). Seuls
changent les nombres : 36 couches, 128 experts, top-4.

## 1. Tailles (calculées à la main, méthode écrite)

Un expert : gate_up 2 880 × 5 760 + down 2 880 × 2 880 = 24,9 M paramètres. En NVFP4, cela fait 4,5 bits par
paramètre (4 bits + une échelle fp8 pour 16), soit 14,0 Mo.

| bloc | 20b (contrôle) | 120b |
|---|---|---|
| experts (couches × experts × 14,0 Mo) | 24 × 32 → 10,7 Go | 36 × 128 → **64,5 Go** |
| attention bf16 (26,5 M par couche) | 1,27 Go | 1,91 Go |
| embed + lm_head bf16 | 2,32 Go | 2,32 Go |
| biais, normes, routeurs | ≈ 0,03 Go | ≈ 0,1 Go |
| **total** | **14,3 Go** (fichier réel 14,4 Go = 13,4 Gio : la méthode tient à 1 %) | **≈ 68,8 Go** |

Octets lus par jeton à b=1 : experts actifs 4 × 36 × 14,0 Mo = **2,02 Go**, plus 3,07 Go hors experts
(attention + lm_head).

### Défaut du plan trouvé au passage

`acvram plan <source HF>` sous-compte un MoE non converti. Il rend 7,56 Go pour la source du 20b, contre 12,6 Go
pour le converti, et **36,3 Go pour le 120b**. Avec `--gpus all`, il annonce même que le 120b tient sur les deux
cartes sans rien en RAM hôte, ce qui est faux de 32 Go. Le plan ne vaut donc qu'après la conversion, lu dans le
manifeste. Bead anticitoyen-vram-gu1 ouverte pour le défaut (estimation du MoE depuis config.json).

## 2. Placement (5090 32 Gio + 3080 Ti 12 Gio + 93 Gio de RAM, 67 disponibles)

* 5090 : hors experts 4,3 Go, KV int8 à 8 192 jetons ≈ 0,3 Go (36 couches × 8 × 64 × 2 o), espace de travail et
  contexte CUDA ≈ 3 Go. Restent **≈ 26,5 Go d'experts, soit 41 %**.
* 3080 Ti, si utilisée : ≈ 10,5 Go d'experts (16 %).
* RAM hôte épinglée : le reste, soit 37,5 Go (A) ou 27 Go (B). Cela tient dans les 67 Gio disponibles, mais
  **pas pendant une phase HF du 20b** (42 Go) : pas de chevauchement.

## 3. Débit prédit à b=1 (avant toute mesure)

Ce qui manque de la carte passe par le PCIe en zéro-copie UVA, mesuré à 23,6 Go/s (C9, 22/09). La 5090 lit à
1,5 To/s. Le taux de présence h est la part des octets d'experts actifs déjà sur une carte.

| placement | h uniforme | h avec placement par fréquence (≈ 0,73, mesuré sur le 119B en C9) | t/s, chemin d'aujourd'hui (+ ≈ 30 ms de lancements eager, 36 couches) | t/s, chemin groupé capturable |
|---|---|---|---|---|
| A : 5090 + RAM | 0,41 → 1,19 Go hôte → 50 ms | 0,73 → 0,55 Go → 23 ms | **12-19** | 19-38 |
| B : 5090 + 3080 Ti + RAM | 0,57 → 0,87 Go → 37 ms | ≈ 0,82 → 0,36 Go → 15 ms | **14-22** | 25-50 |

Le mur, c'est le PCIe : même un chemin parfait plafonne vers 50 t/s, tant que les experts froids ne sont pas
calculés sur l'hôte. Il faudrait pour cela le noyau AVX-512 de la pièce 22 (le noyau scalaire fait 16,6 Go/s,
il ne gagne rien).

Référence llama.cpp (même GGUF MXFP4 officiel, `--n-cpu-moe`, experts calculés par le processeur) : 25-45 t/s,
de mémoire, sans source vérifiée. **Sans le chemin groupé, acvram est attendu derrière llama.cpp sur le 120b** :
c'est la mesure evp (2) du 20b qui dira si ce chemin passe en premier.

## 4. Justesse du 120b

La référence HF sur processeur est impossible : le MXFP4 déquantifié en bf16 fait ≈ 234 Go, pour 93 Gio de RAM.
Le déchargement sur disque coûterait environ 45 min par invite (6 × le 20b, qui prenait 7 min 40). Deux pièces :

1. **Le code est déjà jugé** sur le 20b (F tenu, KL 0,0012) : même classe, mêmes fonctions.
2. **Contrôle propre au 120b** : écarts de chargement (indices d'experts, 128 au lieu de 32, placement en exil).
   On compare aux log-probabilités top-20 de llama.cpp sur le GGUF MXFP4 officiel 120b (≈ 63 Go à télécharger, taille à vérifier),
   invites de evp, 32 pas en argmax forcé. Seuils posés avant : top-1 ≥ 95 %, KL sur le top-20 renormalisé ≤ 0,05,
   aucune invite > 0,20. Ils sont plus lâches que F, parce que la référence est un autre moteur. Contrôle qui peut
   rendre faux : un expert mal indexé doit faire tomber le top-1 bien sous 90 %. C'est à éprouver avant la carte,
   sur le jouet, en permutant deux experts.

## 5. Étapes et coûts

| étape | carte | durée | condition |
|---|---|---|---|
| a. conversion (passage direct MXFP4 → NVFP4, sans calibration ; le 20b a pris 18 s) | non | ≈ 3-5 min (≈ 69 Go écrits sur AI_GENERATOR, 531 Go libres) | processeur chargé quelques minutes : hors des mesures de temps d'un autre poste |
| b. `acvram plan` sur le converti, A et B | non | 1 min | — |
| c. GGUF 120b officiel (référence et comparatif) | non | téléchargement ≈ 63 Go | outil de comparaison, à conserver |
| d. justesse (§ 4) | oui | ≈ 15 min (chargement en exil ≈ 3 min, deux fois) | après evp (2) |
| e. débit b=1, A puis B, contre llama.cpp `--n-cpu-moe` (duel-moteurs) | oui | ≈ 25 min | mesure par poste2, scellé à écrire après (d) |

Ordre proposé : a → b à la prochaine fenêtre sans mesure de temps ; c en fond ; d et e après le verdict de evp (2).
Si evp (2) rend le chemin groupé prioritaire, il passe avant e : mesurer le 120b sur la boucle par expert ne
chiffrerait que les lancements.
