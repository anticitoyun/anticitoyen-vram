# poste7 — P1 bloqué par la VRAM : la « seconde disposition » pèse la pile entière (14,5 Gio, pas 1,7 Go — réfuté sur moi) ; la voie retenue est UNE disposition, Marlin, servie aussi au décodage, tranchée par un banc de 20 min à trois bandes ; sinon GEMV relisant la disposition Marlin ; l'hôte et le repack à la volée sont écartés par l'arithmétique (18/09)

Entrée : poste3 1ee2da9 — 11/11 tests verts sur carte, T 9 775 j/s σ 102 ; bras marlin **RÉGIME DÉGRADÉ, mesure refusée** par `exiger_regime_nominal` (30/48 couches exilées, 3 840 experts en RAM hôte, `experts_layout` retombe à simple, branche Marlin jamais prise) ; cause `loader.py:1706-1707` : `_reserve_prefill` ajoute Σ `mlp_bytes` MoE = **14,5 Gio**. 17 Gio + 14,5 + KV > 31,36.

## 1. Réfuté sur moi, et la garde a fait son travail

J'ai écrit « +1,7 Go » (`poste7-p1-porte-marlin` § 2) en reprenant le chiffre de la note de poste4 sans l'arithmétique : une seconde disposition des mêmes codes 4 bits pèse exactement la pile — Σ `mlp_bytes` du manifeste, 14,5 Gio sur Coder, lisible à sec. Leçon carnet : **un chiffre de VRAM entre dans un plan après lecture du manifeste, jamais depuis une note.** Deux gardes ont tenu : le `Plan` a compté (il a exilé au lieu de planter), `exiger_regime_nominal` a refusé la mesure (rien de dégradé n'est publié). C'est REGLES § 3 « rendre le contrôle impossible à sauter » qui a joué, pas la vigilance.

## 2. Les trois voies de poste3, par l'arithmétique

| voie | coût | verdict |
|---|---|---|
| 3. disposition en RAM hôte, préchargée par prefill | 14,5 Gio / ~25 Go/s PCIe ≈ **0,6 s par prefill** contre 0,13 s de scellé | écartée |
| 2. repack par couche à la volée | 0,1 s × 48 = 4,8 s en torch ; en noyau borné par la bande : 14,5 Gio lus + écrits ≈ **30 ms par prefill** sur 48,8 de GEMM — mange 60 % du gain (16 400 → ~13 000) | écartée, sauf si un jour le repack devient gratuit |
| 1. **une seule disposition** (Marlin), lue aussi par le décodage | 0 octet ; le prix est dans le noyau de décodage | **retenue**, en deux formes ci-dessous |

## 3. Voie 1, forme (a) d'abord : Marlin MoE au décodage aussi — c'est le chemin de vLLM (2 031 t/s à b=12 sur cette carte)

Marlin est conçu pour M petit ; vLLM décode Coder avec `fused_marlin_moe` sur la même disposition que son prefill. Le port de poste4 a le noyau ; ce qui manque est son appel à T = b × top-k sous graphe (aligneur capturable : pas de `.item()`).
**Banc à sec puis 20 min de carte (poste4 écrit, poste3 lance)** : formes décodage Coder — b=12 (96 paires, ≤ 96 experts distincts par pas, routages réels rejoués) et b=1 (8 paires) — ms par pas des experts, Marlin contre GEMV actuel (rpw=4, xreg=down : ≈ 5,2 ms nu à b=12, 0,93 à b=1), rejeu de graphe, sortie = GEMV ± 2⁻⁷ contre fp32 (chaque chemin contre fp32, critère du 18/09).
Trois bandes, écrites avant : b=12 experts **≤ 5,5 ms** ET b=1 **≤ 1,0 ms** → disposition unique = Marlin, P1 continue à 0 octet, cellules décodage re-tamponnées au harnais égal (scellé : b=12 ≥ 0,97 × 1 126 et J ≤ 1,03 × 0,347 ; b=1 ≥ 0,97 × 366) ; **5,5-6,5 ms à b=12 ou 1,0-1,2 à b=1** → P1 se paie sur le décodage : le juge est la cellule complète (t/s ET J des deux lots), on ne le devine pas, une passe ; **> 6,5 ou > 1,2** → forme (b). Prédiction : Marlin à b=12 **4,6-5,4 ms** (77 % de bande sur 8,1 Go, comme vLLM), b=1 **0,9-1,1** ; issue qui me gênerait : b=1 > 1,2 (Marlin à M = 8 lignes sur 8 experts distincts paie ses tuiles 16 × 64 à vide) — alors (b).

## 4. Forme (b) si (a) échoue : le GEMV lit la disposition Marlin

poste4, 2 j : `nvfp4_gemv_grouped_*` lit les quartets dans l'ordre des fragments Marlin (table de permutation constante, 16 valeurs par tuile) ; le GEMV vient d'être réglé (rpw/xreg) sur l'ordre naturel, donc re-porte ncu-borné + ABAB ; scellé : b=12 nu ≥ 0,97 × 1 312, b=1 inchangé ± 3 %, bit-exact contre le GEMV actuel (mêmes codes, même ordre de somme). Si (b) échoue aussi : **P1 fermé pour VRAM sur Coder**, verdict daté, et la ligne utilisateur le dit — pas de disposition hôte, pas de repack.

## Ordre

* poste4 : banc § 3 à sec (formes décodage, routages réels, aligneur capturable) ; prédiction écrite ; puis poste3 20 min. Rien d'autre sur P1 avant ce verdict.
* chef : ETAT — P1 « bloqué VRAM, voie disposition unique, banc décodage en cours » ; carnet poste7 réfuté (+1,7 Go) ; REGLES § 3 ligne « chiffre de VRAM = manifeste, pas une note ».
* poste2 : P2 converti inchangé (indépendant). poste1, poste8 : inchangés.

## 5. Verdict banc décodage (poste3 ee7f76e) : b=12 Marlin 6,81 contre GEMV 6,92 (parité, 0 hors 2⁻⁷) ; b=1 Marlin 1,41 contre 0,82 (×1,7, structurel : tuile m16n8k16 pour une ligne) — forme (a) morte à b=1 ; forme (b) lancée, avec une variante (b') à chiffrer d'abord

* Mes bandes absolues (≤ 5,5 / ≤ 1,0) étaient calées sur un GEMV « 5,2 ms nu » in situ ; le témoin du même banc dit 6,92 : **seuil posé sans le témoin dans le même instrument** (REGLES § 3, faute déjà écrite le 16/09, refaite). Le rapport, lui, est sans ambiguïté : b=12 parité (0,98×), b=1 ×1,7 — Marlin ne sert pas le décodage à b=1, donc pas de disposition unique « Marlin partout ».
* **Forme (b), poste4, 2 j bornés, avec un choix à chiffrer à sec en ½ j avant d'écrire une ligne** : (b) le GEMV lit la disposition Marlin (permutation de quartets dans un noyau borné par la bande : chaque instruction de plus se paie en temps ET en watts, `instr-par-octet`) — ou (b') **Marlin lit la disposition naturelle** (cp.async en ordre naturel, permutation à l'écriture en shared ou adressage `ldmatrix` permuté) : le GEMM prefill est borné par le calcul à 152 TFLOPS, il a du mou pour des shuffles que le GEMV n'a pas. poste4 chiffre les deux (registres/shared par `-Xptxas -v`, instructions ajoutées par octet) et prend celle qui a la marge ; porte micro-banc à sec avant la carte : (b) GEMV-marlin ≥ 0,97 × GEMV à b=1 ET b=12, bit-exact ; (b') Marlin-naturel ≥ **137 TFLOPS** (bande haute conservée) et 0 hors 2⁻⁷. Porte fausse sur les deux → **P1 fermé VRAM**, verdict daté, ligne utilisateur.
* Instrument : `.view(torch.uint8)` sur bg/bu/bd dans le vrai banc (poste4, même commit).
