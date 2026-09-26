# poste7 — C9 : la barre est llama.cpp à 24 j/s avec les experts en RAM (ma prédiction 10-20 dépassée) ; le cache d'experts par PCIe × 8 atteint la parité au mieux (h ≥ 0,55 = le h prédit) — C9 en pause après M0 et cette cellule, la question « 3080 Ti calculante ou experts sur processeur » va à l'utilisateur au bilan (20/09, 00 h 36, horloge machine)

Source : `verdict-c9-llamacpp-119b-19-09` (poste2 c337520c : UD-Q4_K_M, `-ot exps=CPU`, 5090 seule sous 2 700 — b=1 **23,1 j/s à 8 fils, 24,2 à 16, 20,3 à 24** ; passes de 128 jetons 16,0-16,7 ; prefill 2 048 = 488 j/s ; carte 89 W = 0,69 J/jeton GPU seul, processeur non mesuré ; première prise fausse gardée : ids hors vocabulaire tekken 131 072 → `BANC_VOCAB`) ; `verdict-c9-m0-19-09` (22,6 Go/s, 84 ms/jeton sans cache = 11,9 j/s) ; `poste7-c9-119b-cache-experts-19-09` (h_lru(E/3) prédit 0,55 ± 0,10 ; Δh < 0,10 → pas de cache apprenant).

## 1. L'arithmétique qui ferme la question cette nuit
| chemin 119B b=1 | j/s | ce qu'il faut |
|---|---|---|
| llama.cpp, experts en RAM (~70-80 Go/s) | **24,2** | rien : il existe, mesuré, 89 W carte |
| acvram, experts sur PCIe × 8 sans cache | 11,9 | C9-charge (`ModelSpec` mistral4, absent de main) |
| acvram, cache d'experts résidents (≈ 20 Go sur 32) | 24 à **h = 0,55**, 32 à h = 0,7 | C9-charge + cache + épinglage ; h réel inconnu (prédit 0,55 ± 0,10) : **parité au mieux, à pile ou face** |
| acvram, 3080 Ti calculante (12 Go d'experts int8 sm_86, activations seules sur le lien) ou experts sur processeur | > 24 possible | plusieurs jours : second appareil dans le pas, ou noyaux MoE processeur — ce que llama.cpp a mis des années à faire |

Ma prédiction « 10-20 j/s pratiqués » est **fausse** (24,2) : la RAM hôte sert des experts Q4 à un débit que PCIe × 8 n'atteint pas, et c'est le concurrent réel. Conclusion, écrite pour le bilan et pour l'utilisateur : **C9 tel que conçu (cache PCIe) ne dépasse pas llama.cpp sur cette machine** ; ce qui le dépasserait engage une architecture à deux appareils ou un chemin processeur — une décision d'objectif, pas une fenêtre. **C9 en pause** : pas de M1 cette nuit (la trace Coder ne dit rien du 119B), pas de C9-charge sans un oui de l'utilisateur informé de la barre ; la cellule llama.cpp 119B entre au comparatif comme référence de la machine.

## 2. Suite immédiate
Le bras bf16 de structure de C13-c **n'existe pas** (poste1, 00 h 25 : 147 Ko > 99 Ko même en bf16) — poste2 ne l'attend pas ; sa file est celle de `poste7-suite-nuit-00h30` § 3 : **niveau 2** (test carte 1 min) → **C5-b** (20 min) → niveau 3 Coder à son commit ; M1 retiré.

## Ordre
* **poste2** — niveau 2 test carte (1 min) maintenant, puis C5-b ; pas de sonde bf16, pas de M1.
* **poste1** — inchangé (niveau 2 correctif, niveau 3 Coder en code).
* **chef** — ETAT : C9 en pause (barre 24,2, parité au mieux), cellule llama.cpp 119B au comparatif ; question utilisateur pour le bilan : « 119B : llama.cpp fait 24 j/s avec les experts en RAM ; acvram par PCIe plafonne à la parité ; dépasser = 3080 Ti calculante ou experts processeur, plusieurs jours — on y va ? » ; INDEX ; commit + push.
