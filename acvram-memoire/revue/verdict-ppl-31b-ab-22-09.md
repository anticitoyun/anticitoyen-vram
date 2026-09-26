# PPL relative A/B 31B (`acvram eval`, wiki-gptq) — ÉCHEC, exil VRAM même seul, PPL non exploitable — 22/09 (poste2)

* instrument : `acvram eval <alias> --corpus wiki-gptq.txt --window 2048 --stride 2048 --min-context 256 --json` sous `carte.sh`, worktree poste2-w-21-09
* commit : main à jour
* régime : essai 1 = A (max6) et B (4sur6) passés ensemble en arguments positionnels ; essai 2 = A seul
* scellé (chef) : PPL_B ≤ PPL_A si Four Over Six vaut quelque chose
* mesuré : **les deux essais (A+B ensemble, puis A seul) donnent la MÊME alarme** : `[acvram] ATTENTION — exil : 11 MLP en RAM hôte = 97,1 ms/jeton de PCIe, soit 619 % du pas de décodage résident (15,7 ms) ; au-delà du seuil 20 % : falaise de l'exil` — et le MÊME résultat PPL aberrant, identique au bit près entre les deux essais : PPL(ctx=128)=5445,9644 (1024 jetons), PPL(ctx=512)=936,6544 (6140 jetons). Poids réels 23,8 Gio pour 29,5 Gio libres + préfill 3,44 Gio (27,24 Gio, sous la limite) — pourtant 11 couches MLP exilées quand même.
* verdict : **ÉCHEC, PPL non exploitable** — l'exil se produit MÊME avec UN SEUL alias 31B (nvfp4-vision) chargé pour `acvram eval` fenêtre 2048 : ce n'est pas un défaut de charger A et B ensemble (ma première hypothèse, réfutée par l'essai 2). La cause exacte (pourquoi 27,24 Gio de poids+préfill déclenche quand même un exil de 11 MLP sur 29,5 Gio libres) est hors mon domaine de diagnostic (planificateur mémoire, `acvram/memory/`), à nommer par poste1. **Je n'ai pas relancé B** : même cause garantie, même résultat corrompu, gaspillage de carte évité.
* durée : essai 1 ≈ quelques min, essai 2 ≈ quelques min (les deux corrompus)

## Suite
`acvram eval` sur ce 31B ne peut pas juger la qualité tant que l'exil n'est pas nommé/corrigé (probablement le planificateur `eval` réserve un budget KV disproportionné pour la fenêtre 2048, contrairement au chemin `serve`). PPL A/B non tranchée aujourd'hui ; le scellé E (decode-pas texte) reste la seule mesure de qualité valide disponible (B pire que A, KL 1,393 vs 0,867).
