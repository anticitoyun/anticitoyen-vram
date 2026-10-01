# Prise LeapQuant INT8 (01/10 07:06) : ÉCHEC C2 — rouge sur la sortie hors gel ; cause probable dans l'INSTRUMENT (bascules fp16 de la clé non bornées), à confirmer sur carte (poste5)
* instrument : `tests/test_gdn_etat_int8.py` (vérificateur resynchronisé) ; prise `scratchpad/poste5-int8-01-10/prise.sh ac15edc5f`, sortie `prise.txt`
* commit : ac15edc5f (HEAD asserté par la prise)
* régime : carte 0 (RTX 5090) sous `carte.sh` mesure ; avant/après : seul le llama-server permanent (PID 4436, 5 614 Mio)
* scellé : revue/poste5-leap-int8-tolerance-01-10.md, issues C1-C5
* mesuré : 8 passés, 2 rouges (`test_noyaux_contre_reference` [F1 on et off], borne de sortie, ligne 151) ; banc non lancé (arrêt au premier échec, comme prévu)
* verdict : **C2 tel que scellé** (sortie hors borne, hors gel) ; cause probable : une borne incomplète, pas le noyau ; aucun banc, aucun gain revendiqué
* durée : prévu ≤ 10 min / tenu 7 s (journal `tenue=7s`, 07:06:40 → 07:06:47)
## Ce qui est passé, ce qui a cassé
Sont passés sur carte : la roue libre du noyau contre la récurrence exacte (256 pas, TOL_INT8, sans dérive), la couche
GDN entière en INT8 contre fp32, l'export puis le chargement AU BIT, et les tests processeur. Ont cassé : les deux
`test_noyaux_contre_reference`, sur la borne élément par élément de la sortie. Les écarts affichés valent ~1e-9 pour des
sorties de ~4e-3, très sous la borne : le rouge vient donc d'éléments isolés. Le script ne gardait que 8 lignes de
pytest, si bien que le pas, la tête et le rapport fautifs sont perdus. C'est corrigé : la sortie complète est gardée.
## Diagnostic à sec
* Sous TRITON_INTERPRET=1 (les noyaux Triton exécutés sur processeur), aux têtes réelles (HV 48, H 16, GQA 3), le
  vérificateur tient sur 40 pas et 2 gels : l'algorithme et les indices du noyau sont justes (`repro_interpret.py`).
* **Trou de la dérivation** : la borne ne contenait aucun terme pour la CLÉ du pas. Les deux côtés normalisent k avec
  deux ordres de somme différents, puis l'arrondissent en fp16. À sec, deux ordres de somme font basculer **102 éléments
  sur 983 040 (1,0e-4) en 40 pas à b=12** d'une ulp fp16 (2⁻¹⁰). Une bascule déplace s_k, w et k·q de ~1e-3 relatif,
  loin au-dessus de γ_n ; le premier rouge en est la signature attendue.
* Je l'avais pourtant listé en écrivant les bornes, puis laissé de côté. Ce n'est pas une tolérance à élargir.
## Correctif (instrument) et ce qui le rendrait FAUX — scellé avant la prochaine prise
La borne prend l'écart de clé **mesuré**, Δk lu dans les deux enregistrements `bk` (Triton et référence) : les termes
b·α·Mᵀ|Δk| sur w et (Σ|Δk||q|)·|w| sur o, propagés à o. Le message de rouge dit maintenant combien de sorties hors borne
tombent sur des têtes **SANS** bascule de clé. **Prochaine prise** (≤ 10 min, même script) : **C2′ = un seul hors borne
sur une tête sans bascule → bogue du noyau**, arrêt. Tout vert → C1, puis le banc. À sec : la faute Z/126 reste rejetée,
la variante correcte tient ; 6 passés et 4 sautés, plus TRITON_INTERPRET aux têtes réelles.
## Écart à signaler
Mes vérifications processeur (~2 min, un cœur, nice 19) ont tourné pendant le début de la prise de poste2 (rejeu 275, 07:08:35) :
c'est contraire à « aucun pytest pendant une campagne ». Son rejeu mesure la qualité (MMLU) : charge négligeable, mais
l'écart est réel.
