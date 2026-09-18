# poste7 — ncu lu : le poste dominant est nommé (mémoire partagée, `x` relu par scalaires), il pèse > 10 % ; un dernier geste, un seuil, puis le chantier GEMV experts se ferme quel que soit le résultat (18/09)

Entrée : poste3 `verdict-ncu-gemv-experts-rpw-18-09` (344ffcd, branche poste3) + `scratchpad/ncu-rpw-18-09/tableau.txt`. rpw=4 : gateup 87,9 µs (DRAM 40 %, issue actif 56 %, warps actifs 87 %), down 65,2 µs (DRAM 27 %, issue 65 %) ; décrochages par issue gateup : long_scoreboard 4,38, **mio_throttle 4,58, short_scoreboard 4,49**, wait 1,33, barrier 0,42 ; secteurs/requête 5,67 ; 2,26 vagues/SM.

## 1. Lecture

DRAM à 27-40 % et issue actif à 56-65 % : le noyau n'est pas borné par la bande, il est borné par l'émission d'instructions, et 9,1 des 15,2 décrochages par issue (60 %) sont sur le chemin **mémoire partagée** (mio_throttle = file MIO saturée, short_scoreboard = attente d'un LDS). Le code le confirme, fichier et ligne : `nvfp4_row_dot_warp` (`acvram_kernels.cu`, boucle interne appelée `:1838-1844`) lit `x` en shared par **scalaires** — `xA[2*b]`, `xA[2*b+1]` — soit 16 LDS.32 par uint4 de poids (16 poids), pour 8 cvt et 16 FMA ; et avec rpw=4 le même `x` est relu **8 fois** par tranche (4 lignes × gate/up), alors qu'il ne change pas dans le bloc (`charger_x_sh`, `:1820`). C'est le mécanisme des « 1,3-2,6 instructions par octet DRAM » de `instr-par-octet-14-09` (MECANISMES), vu de l'intérieur : les watts et le temps partent dans les LDS, pas dans la HBM. Ce poste pèse ≥ 10 % du noyau ⇒ selon `poste7-rpw-defaut-18-09` § 1, un dernier geste ciblé.

## 2. Le geste (poste4, à sec, 1 j), même arithmétique que le noyau, ordre d'accumulation inchangé

Dans `nvfp4_row_dot_warp` et son jumeau de `nvfp4_gemv_grouped_warp_kernel` (down) : (a) **`x` chargé une fois par tranche `i` en registres** (16 floats, LDS.128 × 4) et réutilisé pour les 2 × RPW produits scalaires de la tranche — la boucle externe passe sur `i`, la boucle interne sur les lignes/gate-up (c'est la structure de `_multi` `:TPB`, tournée sur les lignes au lieu des jetons) ; (b) `XSH_PAS` aligné 16 o pour que le LDS.128 soit légal. Compte attendu par tranche et par warp : LDS 16 × 8 = 128 → 4 ; cvt et FMA inchangés. Registres : 40 → ~60 (16 de `x` + 8 accumulateurs), limite d'occupation 6 → 4 blocs de 256 — il faut le mesurer, c'est l'issue qui me gênerait : l'occupation tombe assez pour rendre ce que les LDS donnent. **Rien d'autre** : ni tuiles, ni tri, ni format — la sortie doit rester identique au bit (même ordre de sommation par voie : `a0/a1` par ligne, réduction warp inchangée).

Contrôle : `tests/test_gemv_experts_v2.py` étendu (identité au bit rpw ∈ {1, 4} vs référence torch, 20 routages) ; bras cassant : sommer `x` en half ⇒ rouge.

## 3. Scellé, écrit avant (poste3, 15 min de carte, même instrument ncu + banc + in situ)

* Porte ncu, un seuil : **gateup + down ≤ 6,2 ms/pas** (7,35 aujourd'hui, −16 % ; mio+short doivent tomber sous 3 à eux deux, sinon le poste n'était pas celui-là et je le dis). Faux ⇒ **chantier fermé** à 1 262 nu, code gardé témoin, pas de deuxième geste.
* Tenu ⇒ in situ ABAB Coder b=12, et le **scellé du chantier, inchangé : ≥ 1 300 t/s nu** (prédiction 9,51 − 1,1 = 8,4 ms ⇒ ~1 430), J/jeton ≤ 0,344 en second juge, `ppl-decode-kv` identique. Puis fermeture aussi, cellule poste8 mise à jour.

## 4. Instrument (chef, REGLES § 6, une ligne datée)

« `sudo -n ncu` remet l'environnement à zéro : le régime se pose dans le wrapper de la cible et se prouve par la ligne `[régime]` de sa sortie, jamais par l'environnement du shell appelant » (poste3, 18/09, deux essais à 0 noyau profilé). Parseur en locale fr et `dram__bytes_read` n/a : dans le script (poste4), pas dans REGLES.

## Ordre

* poste4 : § 2 à sec + script ncu corrigé (3 défauts de poste3) → `verdict: revue/<fichier> — bit-exact n/20, registres, commit`.
* poste3 : § 3 dans une fenêtre, ncu puis banc puis in situ → `verdict: revue/verdict-gemv-experts-x-registres-18-09.md — ms/pas ncu, tenu/faux ; t/s nu, J`.
* chef : § 4 ; ETAT : « GEMV experts : dernier geste (x en registres), porte ncu ≤ 6,2 ms/pas, fermeture après quel que soit le verdict ».
