# GLM b=1 A/B v2 (A=59533b29, B=d12b2a08) — 21/09

* instrument : `scratchpad/glm-b1-ab-21-09/chaine.sh` (fichier suivi)
* commit : A = worktree figé 59533b29, B = worktree figé d12b2a08
* régime : B (réussis) porte `kv=latent-bf16 pipeline=1 eco=2700`
* scellé : moyenne(B) ≥ 0,98 × moyenne(A)
* mesuré : A0-chauffe (hors mesure) ÉCHEC, A1 161,9 t/s, B1 ÉCHEC, A2 162,4 t/s, B2 164,4 t/s
* verdict : **INDÉCIDABLE** — un seul point B valide (B2), échantillon insuffisant pour une moyenne ; sur ce seul point, B2/moyA(162,15) = 1,014 (+1,4 %), sous le seuil de 2 % mais un point ne scelle rien. **Défaut découvert, symétrique A et B, pas spécifique au commit** : A0-chauffe ET B1 sont restés bloqués après « mémoire avant capture » (aucune erreur, aucun progrès) jusqu'au garde-temps de `outils/carte.sh` (900 s pile, confirmé par le journal du verrou : `tenue=900s` pour les deux) — mon curl-loop (120 s) abandonnait avant, affichant à tort « serveur jamais prêt » alors que le process continuait à tourner 13 min de plus sur la carte, non nettoyé par mon script (le port n'était pas encore bindé au moment de mon kill). Les rechargements suivants (A1/A2/B2, cache JIT chaud) ont tous réussi vite. Bug de mon script signalé pour mémoire : `v(nom)` plante en `FileNotFoundError` si `banc-<nom>.log` n'existe même pas (cas où le serveur n'a jamais répondu), pas seulement en cas d'absence de `jetons_s` dans le fichier.
* durée : 2007 s (13:47:18–14:20:46), dominée par les deux blocages à 900 s

nvidia-smi propre après (seul PID 4286).

## Suite
Rejeu reporté par la chef à l'accalmie de 15 h 10 (protocole 6 fenêtres, médiane, garde load1/W). Corriger `chaine.sh` avant relance : tuer le process serveur par PID réel (pas seulement par port) dès l'abandon du curl-loop, et fiabiliser `v()` contre un fichier absent.
