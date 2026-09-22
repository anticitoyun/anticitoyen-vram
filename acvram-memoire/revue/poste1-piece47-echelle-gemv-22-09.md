# Pièce 47 — l'échelle AWQ des experts passée aux GEMV, au bit (conception + code, à sec) — 22/09 (poste1)

## Prédiction et seuils, écrits AVANT le code

* origine : pièce 42, verdict `poste1-piece42-glue-22-09.md` — 0,473 ms/pas et
  384 lancements/pas (8 par couche) de gathers, divisions et casts en torch
  devant les deux GEMV d'experts, payés par **tout** alias dont les experts ont
  des statistiques AWQ réelles (`moe.py:1186` et `1294`), quel que soit le
  format des projections d'attention.
* remède retenu (chef) : table d'échelles passée aux **deux** noyaux
  (`nvfp4_gemv_marlin_kernel<XT, 1>` down, `<XT, 2>` gate+up), division faite à
  la lecture de `x`, **au bit** contre le chemin torch actuel. Pas de repli à la
  conversion.
* **prédiction chiffrée** : −0,47 ms/pas (0,473 ± sa dispersion) sur tout alias
  calibré ; glue_torch de **591 → ≈ 207 lancements/pas** et de 0,964 à ≈ 0,49
  ms/pas ; aucun effet sur un alias sans échelles d'experts (l'officiel :
  exactement 0 lancement retiré).
* **seuils, et ce qu'ils rendraient si l'hypothèse était fausse** :
  1. **Équivalence au bit** : sur les formes réelles, `sha256` des sorties du
     GEMV identique entre le chemin torch et le chemin fusionné. Faux → la
     division dans le noyau n'a pas l'arrondi de PyTorch (ordre bf16 → fp32 →
     bf16) et le remède redevient « pas au bit », donc à juger par KL.
  2. **Le test doit pouvoir casser** : la faute réintroduite une fois (division
     retirée du noyau) doit le faire échouer. S'il passe encore, ce n'est pas un
     test d'équivalence et il ne compte pas.
  3. **Lancements** : `familles-noyaux` doit rendre ≈ 207 (± 10) sur l'alias
     alpha2. Au-dessus de 300, la branche torch est encore prise quelque part —
     le gain annoncé serait faux et il faudrait dire où.
  4. **Gain de temps** : ≥ 0,35 ms/pas mesuré. En dessous, les lancements
     retirés n'étaient pas le coût (le coût serait la bande, pas le lancement) —
     à publier tel quel, c'est un résultat.
* la compilation et la mesure attendent la carte (poste5, pièce 44) ; tout ce
  qui suit est écrit et relu à sec.
