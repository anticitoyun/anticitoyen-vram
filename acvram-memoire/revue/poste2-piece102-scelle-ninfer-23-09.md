# Pièce 102 — préparation à sec, cellule acvram contre NInfer (poste2, 23/09, ordre chef)

## 1. Compilation NInfer — BLOQUÉE (dépendance manquante, pas de sudo)

Clone superficiel dans `/mnt/AI_GENERATOR/ninfer` (hors dépôt, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`).
`cmake --preset release` (Ninja, Release, `NINFER_BUILD_APPS=ON`) échoue à la configuration :
`cmake/Dependencies.cmake:4` exige `pkg_check_modules(FFMPEG REQUIRED … libavformat libavcodec
libavutil libswscale)`, dépendance dure (pas de flag pour la désactiver dans le CMakeLists lu).
Les `.so` runtime existent (`/usr/lib/x86_64-linux-gnu/libavformat.so.62`) mais pas les en-têtes
`-dev` ni les `.pc` de pkg-config sur le système ; des `.pc` existent dans des environnements conda
d'AUTRES comptes (anticitoyenlm, anticitoyenpos) — non utilisés : lier contre l'environnement d'un
autre utilisateur serait un build non reproductible et hors du périmètre de cette session.
`sudo apt-get install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev` refusé
(authentification interactive requise, pas de sudo sans mot de passe pour apt sur ce compte).
**Reste : installation des quatre paquets `-dev` par qui a le mot de passe, puis
`cmake --preset release && cmake --build build --parallel 4` (nice 19, sans carte).**

## 2. Poids NInfer — téléchargé

`hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir
/mnt/AI_GENERATOR/ninfer/models` : fichier récupéré (poids du fichier au sha256 du manifeste HF, non
vérifié bit à bit ici faute de checksum publié dans le README lu).

## 3. Scellé de la cellule — Qwen3.8-27B-nvfp4, b=1 et b=8, sans spéculation des deux côtés

Instrument : même banc HTTP que les pièces 89/94 bis/94 ter (`banc-llamacpp-16-09.py`, mode
`BANC_MOTEUR=` selon le côté — `acvram` pour nous, à écrire côté NInfer une fois son serveur HTTP
identifié dans `apps/` : `/completion`-compatible ou OpenAI selon `docs/`), ABBA ≥ 5 lots/bras,
`-lgc 2700` posé des deux côtés (le NInfer refuse toute architecture hors sm_120a — même carte,
pas de choix d'horloge à faire), J/jeton net, seuil 2σ (Welch, n=10/moteur = 2×5), écart déclaré
seulement au-delà — méthode identique aux pièces 89/94 ter.

Alias acvram : `Qwen3.8-27B-nvfp4` (`/mnt/AI_GENERATOR/models_acvram/Qwen3.8-27B-nvfp4`, dense,
sans spéculation → serve avec `--speculative none`, comme nos bras acvram habituels).

### Prédictions chiffrées (avant toute mesure)

* **Base connue** : cellule du 17/09 (`inventaire-cellules-certifie-18-09.md`, avant marlin, avant
  réduction déroulée, avant 8-warps par défaut) : b=1 **67,2 t/s** (5,940 ms/pas) ; b=12 **96,9 t/s**
  agrégé (deux paliers proches, 4,115-4,125 ms/pas). Depuis : GEMV_LAYOUT=marlin (P1, gain décodage
  b=1 mesuré +5 % sur Coder 18/09), réduction déroulée (−1,9 % au bit, pièce 92), défaut 8 warps
  confirmé devant à b=12 (pièce 94 bis). Ces gains sont mesurés sur Coder (MoE), pas sur un dense
  27B : reportés par analogie, avec marge d'erreur large.
* **Prédiction b=1** (sans spéculation) : **70-85 t/s** (5,9-8,3 ms/pas au décodage), soit +5 à +25 %
  sur la cellule 17/09 — le gain marlin/réduction touche surtout le MoE et l'attention, un dense
  27B (une seule matrice de poids par couche, pas de routage) devrait profiter surtout de marlin.
  Falsificateur : < 65 t/s (régression) ou > 95 t/s (gain injustifié par les leviers connus,
  suspect).
* **Prédiction b=8** (sans spéculation, extrapolé du b=12 du 17/09 à l'échelle du batch, MoE→dense
  non directement comparable) : **300-420 t/s**, plafond puissance probable à ce batch (comme le
  Coder b=12 aujourd'hui, bridages="puissance" systématiques en capture). Falsificateur : < 260 t/s
  ou > 480 t/s.
* **NInfer** (leur table publiée, AVEC spéculation MTP3, horloge libre, une vague) : C=1 143,8 t/s
  (acceptance 48,9 %), C=8 766,6 t/s. **Non comparable tel quel** à notre bras sans spéculation :
  aucun chiffre NInfer sans MTP n'est publié dans le dépôt lu — le nombre qui sortira de LEUR bras
  ABBA sans spéculation ici est la première mesure de ce point, pas une reprise d'un chiffre connu.
  Attente qualitative seulement : à b=1 sans spéculation, NInfer (moteur natif, noyaux maison,
  aucune dépendance Marlin/CUTLASS) devrait rester au-dessus d'acvram (même logique qui les fait
  gagner en MoE contre nous ailleurs) — mais rien de chiffré n'est scellé ici pour leur bras faute
  de donnée de référence publiée à comparer.

## Ordre respecté

Aucune prise de carte GPU effectuée ni tentée (clone/build/téléchargement tous à sec ou réseau,
`CUDA_VISIBLE_DEVICES=""` posé à la compilation). File de carte (poste6 puis poste1) non doublée.
Compilation bloquée : reste à qui peut poser les paquets `-dev` avant toute prise.
