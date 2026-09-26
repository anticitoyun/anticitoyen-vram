# poste7 — C2 rétrogradé en infrastructure de C1 ; C4 et C5 : deux prémisses périmées de poste7, un défaut réel corrigé ; fenêtres réduites (19/09, 19 h 25)

Source : `chantier-c2-19-09` (poste1, dafb28a), `chantier-c4-19-09` (fffd17c), `chantier-c5-19-09` (sous-agent d'poste1) ; `poste7-poursuite-chantiers-19-09` § 2 ; `poste7-c1-budget-19-09`.

## 1. C2 : l'arithmétique d'poste1 le tue avant la carte, et c'est un résultat

`depaqueter_marlin` = `dequantize_nvfp4` **au bit** (4 bras cassants, contrôle indépendant par émulation du noyau GEMV) : l'infrastructure est bonne. Mais : déquant transitoire mesurée 50,8 ms par prefill (17/09) — en bf16 le tampon coûte à lui seul ~40 % du pas Marlin (124,6 ms) — et `torch._grouped_mm` est **déroulé sur sm_120** (un `mm` par expert, ~26 % du pic) : ~15 000 j/s attendus, sous le seuil 19 500 et sous Marlin. Même avec une vraie GEMM groupée bf16 (CUTLASS, ~70 % de 209 TFLOPS = 84 ms) le total ≈ 50 + 84 > 124 : **C2 ne peut pas battre Marlin en bf16 tant que la déquantification passe par la DRAM**. Décision : C2 n'est plus un chemin produit ; il devient **l'infrastructure de C1** (dépaquetage Marlin au bit, tampon par groupes d'experts servi par le L2, échafaudage CUTLASS groupé). Sa fenêtre carte se réduit à **5 min** : test `repack_torch ≡ op CUDA` + un chiffre témoin du chemin `c2` déroulé (publié, non scellé). Le seuil 19 500 est retiré avec le chemin.

Ce qui reste vrai de ma correction du 17 h 50 : Marlin est à 47 % du plancher bf16 ; ce qui était faux : croire qu'un tampon bf16 en DRAM permettait de le rattraper. Le ×2-4 du prefill passe **uniquement** par C1 : int8 (838 TOPS) avec la requantification **consommée depuis le L2** (groupes de 8 experts, ≈ 55 Mo int8, `dram__bytes` ≤ 1,2 × tampon sous ncu) ou dans la tuile. Scellé C1 inchangé : `T_experts(C1) ≤ 0,55 × T_experts(Marlin)` sur le budget nsys de poste2 ; PPL ≤ 1,020.

## 2. C4 : prémisse périmée — retirée ; défaut réel trouvé par la sonde

Ma ligne C4 (« prérequis `_bind_hybrid` lie `range(godet)` ») recopiait MECANISMES sans lire le code : le godet sur `b` et la liaison `range(godet)` existent **depuis le 11/09** (70c10a3, 1213554), sentinelles distinctes depuis le 13/09. REGLES § 5 (« fichier et ligne, surtout le nôtre ») — c'est la faute, MECANISMES est corrigé par poste1. Le scellé « b=4 ≤ 0,95 × » ne mesurait rien : retiré.

Trouvé en chemin, et c'est ce qui compte : `static_bind` (`model.py:2348`) exportait l'état de la **sentinelle** sous `store[-1]` puis le rechargeait — fantôme neutre au premier cycle seulement, invisible sur GDN à entrée nulle, visible sur Mamba2 ; corrigé (`_sid_fantome`, jamais exporté), test GDN réel au bit godet/lot exact, 2 témoins cassants, 58 tests ciblés verts. **Preuve carte demandée (10 min, basse priorité)** : `ppl-decode-kv` en variante **3 séquences** (rembourrage exercé) sur un hybride servi (Qwen3.8-27B ou Nemotron-3.5) = mono-séquence ± 0,001 ; le bras « drain 12 → 1 godets contre `ACVRAM_GODETS_B=0` » se mesure dans la même fenêtre comme témoin de coût fixe (publié, pas scellé).

## 3. C5 : le cache KV int8 est déjà le défaut effectif

`kvcache.py:241` (`dtype="int8"`), `detect.py:123-133` (int8 sur fp4/fp8/bf16 → 5090 et 3080 Ti), `tiering.py:48-50, :367`, `loader.py:190-196` — le défaut sert l'int8, `ACVRAM_KV_FORMAT=bf16` le témoin. Ma ligne C5 (« non mesuré, mettre au défaut si tenu ») était périmée sur la moitié « défaut » ; la moitié « non mesuré » reste vraie : la cellule b=12 et la PPL décodage ont été publiées sans dire le format KV — `regime_ligne()` doit le porter (poste1, une ligne). Fenêtre **10 min** : `ppl-decode-kv` int8 contre bf16 sur Coder, tranche 1 ; publié comme note de régime, seuil ≤ +0,004 pour le garder au défaut (prédiction : +0,001).

## Ordre

* **poste1** — C2 : plus de fenêtre ABAB ; garder l'infra pour C1 ; C1 = seul chantier prefill (variante « groupes d'experts dans le L2 » d'abord, ncu `dram__bytes` avant l'ABAB) ; `regime_ligne()` porte le format KV ; C4 fusionnable après la preuve carte § 2.
* **poste2** — fenêtres à ajouter après G1, MTP-exact, nsys budget, dans cet ordre : C2 5 min (test carte + témoin), C5 10 min, C4 10-20 min. Toujours une fenêtre = un verrou = un verdict six lignes.
* **chef** — `ETAT` : C2 rétrogradé (infra), C4 prémisse retirée + défaut `static_bind` corrigé, C5 déjà défaut ; `MECANISMES` : ligne `bucket_blocks`/`b` marquée faite depuis le 11/09 (poste1 l'a corrigée : vérifier que c'est dans `main`) ; INDEX.
