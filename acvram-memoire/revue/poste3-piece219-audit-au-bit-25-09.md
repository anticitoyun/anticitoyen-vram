# Pièce 219 (poste3, 25/09, ordre chef, à sec) : audit du périmètre « même processus » pour les
preuves au bit / KL du 24-25/09

Constat de poste5 : dans un même processus, le 1er moteur calcule autrement que les suivants (mixte
b=8, s4 jeton 27, s7 jeton 64, reproductible). Objectif : pour chaque verdict listé, l'instrument
en fichier:ligne et si la comparaison A/B tourne dans un même processus (et si oui, à quelle
granularité — moteur complet chargé une fois puis basculé, ou fonction/couche isolée) ou dans des
processus séparés.

| pièce | instrument (fichier:ligne) | granularité | verdict |
|---|---|---|---|
| **172** | `scratchpad/poste5-p172-25-09/diag172.py:33` (`load_model`, une fois) / `:77` (`kernels._DEPAQ_PARTAGE` basculé à chaud) | **MOTEUR COMPLET, MÊME PROCESSUS** | logits fp32 « égaux au bit » A/B' — exposé direct au mécanisme de poste5 |
| **175** | `scratchpad/poste6-p175-25-09/kl-decode-lot.py:22` (`load_model`) / `:32` (`gdn._GDN_AB` rebâti par couche, bascule à chaud) | **MOTEUR COMPLET, MÊME PROCESSUS** | KL_max(A‖·), qualité TENUE — exposé direct |
| **176** | `tests/test_gdn_qkv_gate_176.py:31-48` (`torch.equal`, `monkeypatch.setattr`, deux appels de fonction) | même processus, granularité NOYAU (aucun moteur chargé) | 54 verts — risque moindre : compare deux appels directs, pas un moteur qui tourne |
| **179** | `tests/test_depaq_int8_179.py:41-55` (`monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", …)`) | même processus, granularité NOYAU | au bit, 1+7 — risque moindre, même mécanisme que 176 |
| **182** | `scratchpad/poste1-p182-25-09/profil-ctx.py` (« en processus », graphes) + `tests/test_gdn_z_bf16_182.py`, `tests/test_attn_gqa_182.py` | **explicitement « en processus » par l'autrice elle-même** (titre du verdict : « au bit, en processus TENU ») ; granularité intermédiaire (profil sous graphe, pas un chargement de moteur complet type 172/175) | verdict composite : au bit/en-processus TENU, banc (processus séparés, serveur neuf) NEUTRE — la partie « en processus » est la plus exposée |
| **187** | `tests/test_int8_tranche_187.py:65-72` (`_empreintes`, `subprocess.run`, un `python -c` par réglage `ACVRAM_INT8_TRANCHE`) | **PROCESSUS SÉPARÉS** (un sous-processus par valeur testée) | au bit TENU (6/6 contre 16) — hors du mécanisme de poste5 par construction |
| **194 (b2)** | `tests/test_gdn_ab_flux_194.py:46-55` (`_couche(monkeypatch, flux)`, deux instances de couche construites dans le même test, `torch.equal`) | même processus, granularité COUCHE (deux modules `GatedDeltaNet` instanciés côte à côte, pas un moteur servi) | au bit TENU — risque intermédiaire, pas un moteur complet mais deux instances vivent en même temps dans le même interprète |
| **195** (kl2, 195b) | `scratchpad/poste6-p195-25-09/kl-decode-195-kl2.py:26` (`load_model`, une fois) / `:34` (`ACVRAM_ETROIT_CANAL` basculé à chaud) | **MOTEUR COMPLET, MÊME PROCESSUS** | KL A‖B max 0,0049 ≤ seuil 0,0104, TENU — exposé direct, MÊME PATRON que 172/175 |
| **201** | `tests/test_i8c_transitoire_201.py:62-70` (`monkeypatch.setattr(kernels, "_i8c_poids", …)`) — granularité NOYAU ; **mais** l'attribution modèle entier (`scratchpad/poste5-p201-25-09/diag201.py:33`, sha256 C1/C3 entre commits F/L/201) charge le modèle une fois PAR SCRIPT/COMMIT | tests unitaires : même processus, granularité NOYAU ; comparaison inter-commits : **PROCESSUS SÉPARÉS** (un `load_model` par exécution, jamais deux commits dans le même interprète) | tests verts + sha256 cohérents — le sha256 inter-commits est hors du mécanisme (rejoué à froid à chaque fois) |
| **209** (a/b/c) | (a/b) `tests/test_marlin_pile_par_ligne_209.py:35-62`, `tests/test_marlin_moe_par_colonne_209b.py:96-149` (`monkeypatch`, fonctions isolées) ; (c) `scratchpad/poste6-p209-25-09/kl-decode-209.py` — **l'autrice l'écrit noir sur blanc : « en DEUX processus » (A = par-ligne 0, B = 1)** | (a/b) même processus, granularité NOYAU ; (c) **PROCESSUS SÉPARÉS**, explicite | KL 0,324 ≤ 0,572 TENU, ABBA serveur neuf par passe — (c) hors du mécanisme par construction et par intention affichée |
| **210** | `outils/gpu/mesure/ttft-service-p145.py:40`, prises 1-8 contre un serveur unique | **pas une comparaison A/B de code** — diagnostic d'un seul chemin servi (pourquoi un jeton à texte vide comptait comme « rien reçu ») | N/A : aucune preuve au bit/KL à deux bras dans ce verdict |

## Lecture

Deux verdicts sont exposés SANS ATTÉNUATION au mécanisme que poste5 a trouvé : **172** et **175**
partagent le patron exact (`load_model` une fois, bascule d'un flag global à chaud, comparaison des
logits/KL entre les deux états DU MÊME moteur chargé) — et **195** (kl2, la mesure qui a fait passer
195b au défaut) suit LE MÊME patron avec `ACVRAM_ETROIT_CANAL`. Si le 1er moteur d'un processus
calcule autrement que les suivants, ces trois verdicts n'ont peut-être jamais comparé A et B : ils
ont comparé « moteur froid (A, premier chargé) » contre « moteur chaud (B, même instance, flag
changé) » — la différence mesurée pourrait être en partie ou en totalité l'artefact de poste5, pas
l'effet du flag.

Risque intermédiaire : **182** (l'autrice nomme elle-même la partie « en processus » de son verdict,
distincte du banc HTTP à serveur neuf qui lui est immunisé) et **194 b2** (deux instances de couche
vivent dans le même interprète, pas un moteur servi complet, mais assez proche du patron pour
mériter un rejeu).

Hors mécanisme par construction : **187** (sous-processus dédié par réglage), la partie « KL » de
**209 (c)** (deux processus explicites, l'autrice le documente), la comparaison inter-commits de
**201** (un `load_model` par script, jamais deux commits dans le même interprète), et les tests
unitaires à granularité noyau/fonction de **176, 179, 194(test), 201(test), 209(a/b)** qui ne
chargent jamais de moteur — plus proches de purs appels de fonction que du patron « 1er moteur vs
suivants » que décrit poste5, mais je ne peux pas exclure formellement que le mécanisme (quel qu'il
soit — JIT, cache d'autotune, état global d'un kernel) descende jusqu'à ce niveau ; à chef de
trancher si le rejeu doit aussi les couvrir.

**À rejouer en A, A', B une fois la cause trouvée (ordre chef), par priorité** :
1. 172, 175, 195(kl2/195b) — patron exact, moteur complet, risque le plus direct.
2. 182 (partie « en processus »), 194 b2 — granularité intermédiaire.
3. 176, 179, 194(test), 201(test), 209(a/b) — granularité noyau/couche, risque résiduel à trancher par chef.

## Tri demandé par chef : le biais du 1er moteur ne peut créer qu'une DIFFÉRENCE parasite, jamais
une ÉGALITÉ parasite — seuls les verdicts FAUX/NON TENUS obtenus en même processus sont en danger

| pièce | comparaison en même processus | verdict de CETTE comparaison | en danger ? |
|---|---|---|---|
| 172 (`diag172.py`) | logits fp32 A/B' | **TENU** (égal au bit) | non — une égalité ne peut pas être un artefact du biais |
| 175 (`kl-decode-lot.py`) | KL(A‖·) qualité | **TENU** | non |
| 175 (`frontiere-pas.py`) | vitesse A/C/T (b=1 triton NON TENU) | **NON TENU** (triton b=1) | **processus séparés en réalité** (`prise-vitesse.sh:15` relance `frontiere-pas.py` par bras, `ACVRAM_GDN_AB=$E` posé avant chaque sous-processus) — je m'étais trompée de colonne dans le tableau du dessus en le classant tel quel : à corriger, cette vitesse-là n'est PAS en même processus, hors du tri demandé ici |
| 176 (tests) | au bit | TENU | non |
| 179 (`test_depaq_int8_179.py`) | au bit | TENU | non |
| 179 (`eng179.py`, en processus, L=92/120/78) | diagnostic (pas de TENU/FAUX noté) — explique le **FAUX du banc mixte B/A −0,8 % (prédit +5 à +8)** | le FAUX lui-même vient du banc HTTP, **processus séparés** (serveur neuf par passe) — hors du tri ; MAIS l'explication du FAUX s'appuie sur un chiffre en-processus (905→439 ms, moteur complet, même patron que 172/175) | **zone grise, signalée** : le FAUX du banc n'est pas en danger (processus séparés), mais sa CAUSE alléguée (le gain 92→ mesuré en processus) mériterait un rejeu avant de la considérer acquise |
| 182 (en processus) | z sans cast + GQA au bit | **TENU** (bas des fourchettes) | non |
| 187 | au bit | TENU, processus séparés de toute façon | non (hors mécanisme) |
| 194 b2 (tests) | au bit | TENU | non |
| 195 kl2 | KL A‖B | **TENU** (0,0049 ≤ 0,0104) | non — biais en plus la rendait plus dure à tenir, elle tient quand même |
| 201 (tests) | au bit | TENU | non |
| 209 a/b (tests) | au bit | TENU (avec une nuance connue et expliquée : w13 tensoriel à 2⁻⁷, pas au bit — cause identifiée, bf16, pas le biais du 1er moteur) | non |

**Réponse directe** : sur les onze pièces couvertes, **aucun verdict FAUX/NON TENU n'a été obtenu par
une comparaison en même processus** — le seul NON TENU en même processus apparent (175, vitesse
triton b=1) est en réalité en processus séparés (correction du tableau ci-dessus) ; le seul FAUX
recensé (179, banc mixte) vient d'un banc HTTP à processus séparés, mais sa cause alléguée repose
sur un chiffre en-processus (`eng179.py`) du même patron que 172/175 — la seule zone grise à
vérifier avant de tenir cette explication pour acquise, pas un verdict à rouvrir.
