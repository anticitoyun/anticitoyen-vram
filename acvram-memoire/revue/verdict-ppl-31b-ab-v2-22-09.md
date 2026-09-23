# PPL relative A/B 31B après correctif eval — TENU sur le comparatif, valeurs absolues toujours pathologiques, tension avec le scellé E — 22/09 (Manon)

* instrument : `acvram eval <alias> --corpus wiki-gptq.txt --window 2048 --stride 2048 --min-context 256 --json`, UN modèle à la fois, correctif `ad32ba54` (1 séquence + plan KV serveur)
* commit : main à jour, worktree manon-w-21-09
* régime : A = gemma-4-31B-it-nvfp4-vision (max6), B = gemma-4-31B-it-nvfp4-4sur6-vision
* scellé (Maîtresse) : PPL_B ≤ PPL_A si Four Over Six vaut quelque chose ; prédit PPL(ctx512) 10-20 sans exil (Océane)
* mesuré : **A** : PPL(ctx128)=5445,9644, PPL(ctx512)=936,6544 — **identique au bit** aux 2 essais précédents corrompus par l'exil (le correctif a supprimé l'alarme d'exil cette fois, mais PAS changé le chiffre : l'exil n'était probablement pas la vraie cause). **B** : PPL(ctx128)=3889,8307, PPL(ctx512)=522,4622. Aucune alarme d'exil sur A ni B avec le correctif.
* verdict : **TENU sur le comparatif** — PPL_B < PPL_A sur les deux contextes (3889,8 < 5445,96 ; 522,5 < 936,65), conforme à la prédiction. **Mais les deux valeurs absolues restent à 25-45× la fourchette saine prédite (10-20)** — un défaut partagé touche les deux alias (probable gabarit/tokenisation de `acvram eval`, pas un défaut de quantification propre à l'un ou l'autre, hors mon domaine à diagnostiquer). **Tension nommée sans trancher** : ce résultat CONTREDIT le scellé E texte du matin (`verdict-scelle-e-31b-22-09`, decode-pas KL_A=0,867 < KL_B=1,393, B jugé PIRE que A) — les deux instruments donnent un ordre A/B opposé.
* durée : ~2× quelques minutes, aucune erreur/exil

## Suite
Le chiffre PPL absolu n'est pas exploitable tant que sa cause pathologique commune (probable bug de gabarit ou de découpage du corpus par `acvram eval`) n'est pas nommée — à Océane. Le comparatif A/B relatif (PPL) et le decode-pas (KL) se contredisent : ni l'un ni l'autre ne doit être pris seul pour trancher Four Over Six sans que cette contradiction soit résolue.
