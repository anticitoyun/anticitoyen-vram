# Verdict — A7 : alpha AWQ commun gate/up, Llama-2-7B

poste2, 15/09. Seuil scellé initial (`prediction-a7-alpha-commun-gateup-
14-09.md`) : PPL ≤ 5,4141 × 1,005. **Ce seuil ne s'applique pas comme
écrit — régime mal identifié, corrigé ci-dessous avant de conclure.**

## Ce qui a été mesuré, dans l'ordre

1. Conversion `Llama-2-7b-alphacommun-int8` (`--alpha-commun-gate-up
   --format int8 --snr-floor 25`, mêmes options que le témoin) : 32/32
   paires gate/up fusionnées, SNR sortie moyen 45,3 dB.
2. `acvram eval` (régime `--window 2048 --stride 2048 --min-context 0`,
   `wiki-gptq.txt`, le seul cadrage annoncé comparable à l'étalon
   extérieur) : **PPL 6,1287**.
3. Écart énorme avec 5,4141 (+13,2 %) — au lieu de conclure « réfuté »
   sur ce seul chiffre (règle 6 : un chiffre hors de son régime est
   faux comme décision), **contrôle avant conclusion** : même éval sur
   le témoin int8 SANS alpha commun (`Llama-2-7b-int8`, déjà existant) :
   **PPL 6,1273**. Puis sur le témoin bf16 pur (`Llama-2-7b-fp16pur`,
   PLEINE PRÉCISION) : **PPL 6,1301**.

## Constat

Les trois — bf16 (6,1301), int8 (6,1273), int8-alpha-commun (6,1287) —
sont **à moins de 0,05 % les uns des autres**, tous à ~13 % de 5,4141.
**5,4141 n'est pas la PPL de ce modèle sous `acvram eval`, dans AUCUN
format, pas même en pleine précision.** C'est l'étalon d'un OUTIL
EXTÉRIEUR (`transformers`/GPTQ, selon les notes de mémoire), pas
d'`acvram eval` — deux implémentations avec les mêmes window/stride/
min-context peuvent diverger sur des détails non alignés ici (tokenizer,
gestion du BOS, réduction de la cross-entropy) et donner deux nombres
absolus différents pour la MÊME qualité de modèle. Mon seuil du 14/09
comparait un chiffre `acvram eval` à un chiffre `transformers`/GPTQ sans
vérifier qu'ils partagent un régime — exactement l'erreur que la règle 6
existe pour empêcher, et je l'ai faite en écrivant la prédiction.

## Verdict corrigé

Contre la seule comparaison valide (même outil, même régime, même
corpus, même carte) : **alpha commun gate/up ne coûte RIEN de mesurable
en PPL** (6,1287 vs 6,1273 témoin, +0,023 % — bruit). Le mécanisme de
recherche jointe fonctionne comme prédit dans son volet qualité.

**Ce que je ne peux PAS dire** : si 6,1287 (ou 6,1273, ou 6,1301) est
« bon » en absolu — la seule référence externe fiable (5,4141) n'est
pas dans le même régime que ces trois chiffres, et je n'ai pas
reproduit son régime exact ce soir (pas le temps ni la carte demandée
pour ça). Ce que je PEUX dire : l'écart alpha-commun-contre-témoin est
correct et clos, l'écart-à-l'extérieur reste une question ouverte et
distincte, à ne pas mélanger avec ce verdict.

## Suite

Le volet débit (seuil ≥ +1,0 pt décodage, sealed le 14/09) reste à
mesurer si un chiffre juste peut être hors sujet vaut la peine :
puisque la PPL ne coûte rien, la question qui reste est purement le
débit, pas la qualité.

## Correction à faire

`prediction-a7-alpha-commun-gateup-14-09.md` cite 5,4141 comme seuil
direct pour un `acvram eval` — à corriger ou annoter pour la prochaine
fois : soit établir le régime EXACT de 5,4141 sous `acvram eval` (le
mesurer une fois, sceller CE chiffre comme référence acvram-native),
soit ne jamais comparer un chiffre `acvram eval` à un chiffre externe
sans ce pont.
