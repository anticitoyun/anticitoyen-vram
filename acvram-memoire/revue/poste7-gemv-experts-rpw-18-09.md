# poste7 — GEMV experts après v2 : oui à la mesure RPW, 2 min de carte par poste3, un seuil, et pas une ligne de noyau avant (18/09)

Entrée : chef — porte v2 FAUX (9,66 vs 7,88 ms/pas, +23 %, bit-exact, `verdict-gemv-experts-v2-porte-18-09`), v1 défaut, v2 témoin. « 54 % relus » = octets demandés, L2 servait ×1,37 (quatrième retour de MECANISMES « une relecture servie par le L2 n'est pas un levier » — la règle existait, je l'ai laissée passer dans `poste7-lecture-profils` § 2). Piste poste4 : latence par ligne (2 uint4 par voie en vol, étage + `__syncthreads` par bloc de 8 lignes), `ACVRAM_GROUPED_RPW=2/4` exposé, jamais mesuré.

## 1. Pourquoi la piste vaut 2 minutes, et pas plus avant le chiffre

`rpw` = lignes par warp (`acvram_kernels.cu:1864-1868`, `GW_WARPS` 8, `:1757`) : à `rpw=1` chaque bloc de 8 warps met en scène `x` + échelles en shared (`shm = K + K/32` floats, 8,4 Ko à K=2048) pour 8 lignes de sortie, puis chaque warp n'a que 2 chargements de 16 o en vol. À `rpw=4` : 4× moins de blocs, 4× moins de mises en scène de `x`, 4 lignes de chargements indépendants par warp — les deux effets de poste4 vont dans le même sens, et le prix est l'occupation (4 accumulateurs, 4× moins de CTA : à M=768, 24 blocs par expert au lieu de 96, sur 170 SM avec ~70 experts c'est encore 1 680 CTA, suffisant). Le mécanisme se tient ; la relecture ne se tenait pas non plus moins bien sur le papier — d'où la mesure d'abord.

## 2. Mesure (poste3, même instrument que la porte v2 : `outils/banc-gemv-experts-18-09.py`, même commit 13ebdc9, ABAB)

Prérequis à sec, poste4, 20 min : `GROUPED_RPW` dans `regime.py` (défaut 1, comme `MOE_GEMV` `:64`) et la ligne de régime imprimée par le banc — aujourd'hui ni l'un ni l'autre (grep) ; le `.cu` lit la variable en `static const` au premier appel, donc **un processus par valeur**, variable posée avant l'import, et le JSON de chaque bras porte `ACVRAM_GROUPED_RPW=n` (REGLES § 3 : prouver que la configuration a pris, dans le processus qui mesure).

Trois bras `rpw ∈ {1, 2, 4}`, gate/up et down chacun, médianes ms/couche → ms/pas, témoin rpw=1 relancé en fin (doit rendre 7,88 ± 0,10, sinon la fenêtre est polluée). Identité au bit avec rpw=1 exigée sur les 20 routages (les lignes sont indépendantes : un écart est un bogue, pas du bruit).

**Scellé, un seul seuil : min(rpw=2, rpw=4) ≤ 6,7 ms/pas** (−15 %, 1,33 To/s, 74 % de 1,79). Tenu ⇒ la latence par ligne est un levier réel, valeur retenue en défaut après in situ (le scellé Coder b=12 ≥ 1 300 t/s nu reste : 6,6 → 5,6 ms d'experts + 3,4 ms de reste = 9,0 ms = 1 330, cohérent). Faux ⇒ le levier n'est pas la latence par ligne ; alors **une passe ncu** (poste3, `--launch-count` borné, REGLES § 6) sur `nvfp4_gemv_grouped_gateup` rpw=1 : `smsp__warp_issue_stalled_long_scoreboard`, `sm__warps_active`, `dram__bytes_read` — et poste4 lit avant d'écrire. Issue qui me gênerait : rpw=2 tenu et rpw=4 pire (occupation), le levier plafonne à −15 % et les 1 300 restent hors d'atteinte par ce seul poste.

## Ordre

* poste4 : `GROUPED_RPW` dans `regime.py` + ligne de régime dans le banc, test que la ligne porte la valeur posée, à sec, commit → `verdict: revue/<fichier> — prêt`. Pas de noyau.
* poste3 : trois bras + témoin, 5 min de carte, `verdict: revue/verdict-gemv-experts-rpw-18-09.md — min(2,4) = n,nn ms/pas, tenu/faux, bit-exact oui/non`.
* chef : rien à fusionner avant le verdict.
