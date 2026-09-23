# Sage — Chantier GEMM dense petit M : CLOS après trois tours (349 → 392 → 412 t/s) ; réduction fusionnée en défaut par le J/jeton ; CUDA chiffré = parité vLLM, pas plus, donc derrière la GEMV experts (17/09)

Entrée : Laure 43d025b — nu 441 t/s (ma bande « ≥ 455 / < 435 » laissait 435-455 orphelin : **quatrième fois aujourd'hui, une ligne après avoir écrit la règle** — carnet, et cette fois la règle est mécanique : je n'écris plus « faux si < Y » avec Y ≠ X) ; rondes **412 t/s, 0,967 J** (≤ 0,97 tenu) ; `_reduire_kernel` disparu, `_dense_etroit` 18,2 → 19,8 ms (l'épilogue se paie dedans), net −1,7 ms ; GEMM dense = 75 % du pas à **1,03 To/s effectifs** (58 % de la bande ; plancher 11 ms).

## Verdict

* **Réduction fusionnée : défaut.** Le juge nu se lit contre (441 < 455) ; le juge J/jeton tient (0,967 ≤ 0,97) ; le mécanisme est compris au dixième (−1,7 ms net, épilogue déplacé, pas gratuit). Équivalence tenue. Gain petit mais réel et expliqué : il reste.
* **Chantier clos, pas de 4e tour.** Trois tours, +18 % cumulé depuis le défaut GEMV (349 → 412 ; ×3,2 depuis 128 en comptant le palier 1). Il ne reste plus de colle : 75 % du pas est *un* noyau à 58 % de la bande. Ce que Triton donne sur cette forme est donné.
* **Cellule finale aux menus** : « Qwen3.8-27B acvram b=12 **412 t/s · 0,97 J** (bridé 400 W ; nu 441) · b=1 70,0 · PPL 1,0253 géo non classé » ; vLLM 621 / — : ×1,5 devant, écrit tel quel dans la table.

## CUDA, chiffré une fois pour toutes sur cette cellule

Pas actuel 29 ms dont GEMM 22 ms à 1,03 To/s. Un noyau CUDA façon Marlin à 1,45 To/s : 20 Go → 13,8 ms, pas ≈ 21 ms ⇒ **≈ 570 t/s : la parité avec vLLM (19 ms), pas l'avance.** Pour 3-4 jours de Laurine sur un modèle non classé. Il passe derrière la GEMV experts ≥ 85 % (Coder b=12, classée, cellule phare) et derrière la porte migration d'échelles (préfill) ; il ne revient que si l'utilisateur demande la parité vLLM sur les denses à b > 1 comme objectif propre — une ligne à lui porter.

## Ordre

Laurine → GEMV experts ≥ 85 % (`sage-lecture-profils` § 2 : micro-banc octets/temps à sec, puis Laure 20 min ; scellé b=12 Coder ≥ 1 300 t/s nu, deux bandes : ≥ 1 300 tenu, < 1 300 faux ; J/jeton en second juge). Manon → calibration Nemotron bras A (en cours). Katy → cellules Qwen3.8 et Nemotron aux menus.
