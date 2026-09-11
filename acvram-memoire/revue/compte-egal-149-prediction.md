# Compte égal à 149 — prédiction scellée avant mesure

11/09/2026, **écrit avant de convertir**. À budget fixe, quatre ordres ont donné
quatre PPL et quatre comptes de promus différents (149–198) ; le compte
confond « quels tenseurs » et « combien ». Ce bras fixe le compte à **149**
(celui de B, qui existe déjà) et laisse chaque ordre choisir SES 149.

## Protocole

`ACVRAM_MAX_PROMUS=149` promeut exactement les 149 premiers de l'ordre, budget
ignoré. Trois conversions, un exemplaire chacune (PPL déterministe) :

```
ordre A (snr)            ACVRAM_ORDRE_SAC absent   (defaut, snr_par_octet_decroissant)
ordre erreur            ACVRAM_ORDRE_SAC=erreur
ordre base_croissant    ACVRAM_ORDRE_SAC=base_croissant
```

B (inverse, 149) sert de référence : PPL déjà mesurée **5,4482**.

## Prédiction — scellée

**B reste le meilleur à compte égal → l'ORDRE compte.** C'est ma prédiction.

Raisonnement : à budget fixe B (149) battait déjà A (198) et erreur (196) —
*moins mais mieux*. Si le seul facteur était le nombre, forcer A/erreur/
base_croissant à 149 devrait les amener au niveau de B. Je parie que non : je
prédis que les 149 de B ne sont pas les mêmes que les 149 de A, et que ceux de B
valent mieux.

Les deux issues, écrites d'avance :
- **PPL(A@149), PPL(erreur@149), PPL(base_croissant@149) > 5,4482** → l'ordre
  compte, B choisit de meilleurs 149. Résultat majeur : le critère de tri est le
  levier, pas le budget.
- **Les quatre convergent vers ~5,4482** → seul le NOMBRE comptait ; l'ordre est
  sans effet une fois le compte fixé, et tout notre travail sur les clés de tri
  visait un fantôme.

Une issue intermédiaire (certains convergent, d'autres non) départagerait les
ordres entre eux — mais ne renverserait pas la conclusion principale : si un
seul reste au-dessus de B, l'ordre a un effet.

## Réserve

Trois points + la référence B = quatre PPL à compte égal. n=4, mais ici on ne
cherche pas une corrélation : on compare quatre valeurs à une référence connue.
Un écart de 0,004+ (l'ordre de grandeur A↔B observé) est bien au-dessus du
déterminisme de la PPL, donc lisible sur un seul exemplaire.

---

**ISSUE (11/09) : RÉFUTÉE.** base_croissant@149 (5,4463) égale B@149 (5,4482) ; à compte égal la PPL suit les OCTETS dépensés (Pearson −0,992), pas l ordre. Voir `compte-egal-149-resultat.md`.
