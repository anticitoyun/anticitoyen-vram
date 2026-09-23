# Verdict — juge (5) : PPL décodage réel, deux bras (15/09)

Manon, 15/09. Protocole : `outils/ppl-decode-mma-coder30b.py`, teacher
forcing, KV continu, `Engine` sous graphes, Coder-30B-nvfp4, corpus
wiki-gptq.txt, 8191 jetons notés, chaque pas vérifié t≤32
(`pas_t_le_32=8191/8191` sur les deux bras — le chemin décodage est bien
emprunté, pas supposé). Juge de Jérôme (deux bras, pas de seuil absolu) :
B/A ≤ 1,010 → (5) s'ouvre ; > 1,015 → défaut décodage ; entre les deux →
Sage.

## Résultat

| bras | flag | PPL |
|---|---|---|
| A (GEMV W4A16) | `ACVRAM_MOE_DECODE_MMA=0` | 8,7660 |
| B (MMA W4A4) | `ACVRAM_MOE_DECODE_MMA=1` | 8,7614 |

**B/A = 0,999475.** Sous 1,010 : **(5) s'ouvre** par ce juge.

## Contrôle de cohérence : ÉCHOUÉ tel qu'annoncé, cause nommée

Jérôme demandait A proche de l'étalon du parc 9,12 ± 0,05. A = 8,7660 —
hors tolérance (écart −4,0 %, ~7× la marge). **Cause nommée, pas
soupçonnée** : ce script score le jeton 0 en quasi-absence de contexte
UNE SEULE fois sur 8191 positions (contexte continu, jamais réinitialisé).
L'étalon (`acvram eval --window 2048 --stride 2048 --min-context 0`)
réinitialise le contexte à CHAQUE fenêtre : ~4 fenêtres sur 8192 jetons,
donc ~4 positions à contexte quasi nul (les plus coûteuses — « un jeton
prédit sans contexte coûte une dizaine de nats », `evaluate.py`) au lieu
d'une seule. Cet écart était PRÉDIT avant la mesure, dans l'en-tête du
script (« cache KV continu... jamais moins de contexte que l'étalon »),
et va dans le sens attendu (PPL plus basse, pas plus haute).

**Pourquoi le ratio reste valide malgré cet échec** : le biais (contexte
continu) est IDENTIQUE sur les deux bras — seul le flag MMA change entre
A et B, tout le reste du montage est un bit-à-bit commun. Un biais
partagé qui décale les deux PPL dans le MÊME sens ne change pas leur
RAPPORT. Ce que le contrôle de cohérence dit vraiment, c'est que ce
montage ne peut PAS servir à comparer contre l'étalon absolu 9,1218 —
seulement A contre B, ce qui est exactement ce que demande le juge.

## Deux avertissements, non liés au verdict

- `CUDACachingAllocator` : deux avertissements OOM transitoires (12-100
  Mio, quelques secondes avant le premier pas noté) sur CHAQUE bras,
  texte identique — cohérent avec une réservation mémoire de capture de
  graphe CUDA sous pression passagère juste après la libération de la
  carte par Laure, PAS un signe de dégradation silencieuse : le régime
  affiche `graphes=on piles_ok=True` sur les deux bras, et les 8191/8191
  jetons sont notés sans erreur.

## Suite

Rendu à Jérôme : (5) s'ouvre par le ratio, mais la cohérence absolue
échoue pour une raison nommée — décision finale (ouverture, ou passage
par Sage pour trancher le poids du contrôle de cohérence) à lui/elle.
