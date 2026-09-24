# Verdict — pièce 125 bis : le W8A8 du préfill est-il coupable ? (poste6, 24/09)

instrument : `outils/gpu/mesure/kl-chemins-p125.py` mode `prefill`, deux bras dans la même prise (`P125_SUFFIXE=-cublas` / `-gemv`), `scratchpad/poste6-p125-24-09/prise-bis.sh` → `prise-bis.txt`, `coder/compare-gemv-<alias>.json`, preuves `coder/prefill-{cublas,gemv}-<alias>-preuve.json`
commit : 96c3638a (branche poste6), asserté rc 65 ; venv de mesure, PYTHONPATH de l'arbre
régime : b=1, Coder i8c ; bras cublas = défaut (`CHEMINS_INT8` cublas 192, gemv 293) ; bras gemv = `ACVRAM_INT8_GEMV_MAX=4096` sur la ligne de régime (`CHEMINS_INT8` gemv 485, cublas 0) ; dumps HF du 02 h 03 ; forcé du 02 h 04 en repère ; cpu-safe=off (max_perf_pct 100 début et fin) ; compute-apps début = fin = llama-server 4627
scellé : `revue/poste6-piece125bis-scelle-24-09.md` (b2eb09d7 + addendum f7332dfb, témoin GEMV écrit avant la remesure) — FAUX si KL_moy(cublas) ≤ 1,25 × KL_moy(gemv) sur ≥ 2/5
mesuré : **les deux bras sont identiques au bit sur 3/5 invites** (L2 cublas‖gemv = 0,0 à toutes les couches, invites 1, 2, 4 : L = 73, 62, 77 jetons ≤ 80) — le W8A8 ne s'engage qu'au-delà de `ACVRAM_INT8_GEMV_MAX = 80` lignes. Sur les 2 invites où il joue (L = 86, 87) : KL 0,103 → 0,089 (×1,16) et 0,057 → 0,045 (×1,26) ; L2 d'invite 0,149 → 0,133 et 0,187 → 0,166 (−11 %) ; le préfill sans W8A8 rejoint le forcé (0,089 / 0,089 ; 0,045 / 0,040)
verdict : **FAUX** au scellé (ratio ≤ 1,25 sur 4/5, dont 3 identiques) — le W8A8 n'est pas la cause de l'excédent mesuré en 125 ; il coûte +0,013 nat (+16-26 %) et +11 % de L2 là où il s'applique, et l'excédent des invites 1-2 (préfill ×2,1 et ×3,35 le forcé) existe **à projections identiques** : la cause est ailleurs
durée : prévu ≤ 5 min ; tenu 30 s (`tenue=30s`, deux chargements de 14 s) ; deux prises nulles avant (02:11 témoin bf16 inexistant, 02:13 `$R` non lié — aucune carte prise pour la seconde)

## Chiffres (KL(HF ‖ préfill) moyenne, 32 positions ; L2 relative à la dernière couche, positions d'invite)

| invite | L | W8A8 engagé | KL cublas | KL gemv | KL forcé (02 h 04) | L2 cublas | L2 gemv | L2 forcé |
|---|---|---|---|---|---|---|---|---|
| 0 | 86 | oui | 0,103 | 0,089 | 0,089 | 0,149 | 0,133 | 0,137 |
| 1 | 73 | non (au bit) | 0,0076 | 0,0076 | 0,0036 | 0,127 | 0,127 | 0,127 |
| 2 | 62 | non (au bit) | 0,077 | 0,077 | 0,023 | 0,132 | 0,132 | 0,132 |
| 3 | 87 | oui | 0,057 | 0,045 | 0,040 | 0,187 | 0,166 | 0,173 |
| 4 | 77 | non (au bit) | 0,026 | 0,026 | 0,034 | 0,136 | 0,136 | 0,137 |

## Lecture
* **Le seuil GEMV décide qui paie le W8A8** : sous 81 lignes de préfill le Coder i8c ne le paie jamais (chemin GEMV W8A16,
  au bit avec le témoin) ; au-dessus, ce que la 125 a mesuré : +0,013 nat et +11 % de L2 sur 2/2 invites, et le préfill
  revient au niveau du forcé une fois le W8A8 retiré. Les cinq invites de kl-gabarit sont COURTES (62-87 jetons) : une
  invite réelle (gabarit + question de 200-2 000 jetons) est presque toujours au-dessus de 80 — **la 125 a sous-estimé la
  part du W8A8 sur les prompts servis**, sans le savoir. Ce point n'invalide pas le verdict 125 (équivalents à 25 %),
  il borne sa portée aux prompts ≤ 80 jetons pour ce qui est des projections.
* **L'excédent des invites 1-2 n'est pas le W8A8** (projections identiques au bit, préfill ×2,1 et ×3,35 le forcé) :
  candidats, non mesurés, à nommer par ordre de soupçon — (a) le MoE de préfill (Marlin groupé W4A16, `ACVRAM_PREFILL_GROUPED=marlin`,
  moe.py:1749) contre le GEMV Marlin du décodage : même poids, autre ordre de sommation et autre lot par expert ;
  (b) l'attention de préfill (flash bf16 sur K/V bf16, `attention.py`) contre la paginée int8 ; (c) la glue C15 (résidu
  différé, épilogue fusionné). Un bras « préfill par le décodage » existe déjà : c'est `force` — il donne le plancher ;
  il manque un bras par famille (MoE seul en décodage, attention seule), ce que l'instrument ne sait pas faire aujourd'hui.
* Sur les 3 invites courtes, le témoin GEMV a rendu **0 ulp** sans qu'on le lui demande : c'est le contrôle que le bras
  cublas n'a rien mesuré là — un scellé écrit sur « 5 invites » aurait dû prévoir la longueur ; je l'ai appris ici.

## Ce que je ne décide pas (à chef)
Le W8A8 coûte 16-26 % de KL sur les prompts > 80 jetons, en échange du préfill C15 (× 1,29-1,34). Le défaut servi se juge
en débit ET en KL (ordre du 02 h). Si une pièce suit : (1) rejouer les deux bras sur 5 invites LONGUES (≥ 512 jetons, où le
W8A8 porte 100 % des lignes) pour chiffrer son vrai prix ; (2) la cause de l'excédent à projections identiques (candidats
a-b-c) avant tout correctif — un correctif du W8A8 ne fermerait pas la 125.
