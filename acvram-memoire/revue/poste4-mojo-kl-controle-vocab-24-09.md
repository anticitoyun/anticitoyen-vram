# Contrôle vocabulaire — id 11 vs id 3837, `,` vs `，` (ordre chef)

Lu directement dans `tokenizer.json` du dossier servi par MAX (le même fichier, `tokenizers.Tokenizer`
bas niveau, pas le wrapper `transformers.AutoTokenizer` utilisé par `kl_reference.py` — élimine une
normalisation propre à HF Python) :

```
id 11   -> decode -> ','   id_to_token: ','
id 3837 -> decode -> '，'  id_to_token: 'ï¼Į'
','  -> encode -> [11]    ['，'] -> encode -> [3837]
```

Aucune collision, aucun alias : les deux ids sont distincts et sans ambiguïté dans les deux sens, dans le
fichier de vocabulaire que MAX lit lui-même (même dossier modèle). Le texte `，` reçu de `top_logprobs` ne
peut provenir que de l'id 3837 dans ce vocabulaire — pas d'un id 11 mal affiché par un détokéniseur bogué.

## Verdict confirmé

L'hypothèse « défaut de table/détokéniseur » est exclue par lecture directe du fichier, pas seulement par
un aller-retour applicatif. Le verdict `poste4-mojo-kl-diagnostic-virgule-24-09.md` **tient tel quel** :
à entrées identiques, MAX assigne réellement ≈56 % de masse à l'id 3837 (`，`) là où HF en bf16 assigne
&lt;1e-9 — écart de calcul réel dans MAX/Mojo. Fin de la porte KL pour cette étape : c'est un résultat.
