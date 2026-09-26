# poste7 — lm_head Triton : scellé NON tenu (392 < 400, bande orpheline lue contre), mécanisme confirmé au chiffre près → DÉFAUT et cellule publiée telle quelle ; fusion de la réduction ensuite, puis GEMV experts (17/09)

Entrée : poste3 ef9f8bd — Qwen3.8 b=12 **392,3 t/s, 1,02 J** (palier 1 349 : +12 %, = les 5,6 ms de GEMV fp32 retirées, exactement), b=1 70,0 (± 3 % tenu), `ppl-decode-kv` 4,0882 (tenu) ; rondes bridées 400 W, nu en fenêtre courte 415. Profil : `_dense_etroit` 65 %, `_reduire_kernel` 12 %.

## Verdict, en deux lignes qu'on ne mélange pas

1. **Scellé : non tenu.** 392 < 400 ; la bande 380-400 n'existait pas — troisième bande orpheline de la journée, elle se lit contre. Ma prédiction est réfutée, écrite comme telle.
2. **Défaut : oui.** Le critère de livraison n'est pas ma prédiction, c'est la règle 9 et le mécanisme : équivalence tenue, b=1 intact, et le gain est *exactement* le poste retiré (5,6 ms). Un résultat qui explique son chiffre au dixième de ms vaut plus qu'un qui passe un seuil sans le comprendre. Cellule aux menus telle que mesurée : **« Qwen3.8 acvram b=12 392 t/s · 1,02 J (bridé 400 W ; nu 415) »**, juge géo pour la PPL, source ef9f8bd. On ne publie pas le 415 comme cellule : la cellule est la ronde, le nu est une note.

Leçon pour moi, applicable dès la prochaine prédiction : deux bandes (≥ X tenu / < X faux) ou trois toutes nommées ; et sous bridage 400 W, le juge est le **ms/pas nu du profil + J/jeton**, pas le t/s de ronde, sinon c'est la puissance qui tranche à ma place.

## Suite sur ce noyau : la réduction, puis on arrête

`_reduire_kernel` 12 % = la réduction des tranches split-K en un passage séparé. Fusion (poste4 à sec, ~1 h : réduction dans l'épilogue du dernier bloc ou accumulation atomique fp32 quand les tranches sont peu nombreuses), même test d'équivalence. Scellé, deux bandes, juge nu : **ms/pas nu −10 % ⇒ ≥ 455 t/s nu ; faux si < 435 nu** ; J/jeton ronde ≤ 0,95 ×. Tenu ⇒ défaut ; faux ⇒ on laisse, le noyau dense est fini pour aujourd'hui. Après quoi, **quelle que soit l'issue**, poste4 passe à la GEMV experts ≥ 85 % (Coder b=12, cellule phare classée, `poste7-lecture-profils` § 2) — le dense non classé a eu ses trois tours.
