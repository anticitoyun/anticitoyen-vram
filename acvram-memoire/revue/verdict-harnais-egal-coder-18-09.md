# Verdict — harnais égal Coder, première passe : **contrôle du harnais FAUX** (llama.cpp b=1 259,1 t/s, hors bande 331-352) — cause : machine à charge 70 (trois suites pytest de pairs à > 1 000 % CPU chacune) ; cellules acvram de la même fenêtre **indécidables**, à refaire sur machine calme

instrument : `scratchpad/harnais-egal-18-09/chaine.sh` — `banc-llamacpp-16-09.py` paramétré (d3c5bb2 : `BANC_URL`, `BANC_MOTEUR=acvram` → `/v1/completions` ids + SSE ; chemin llama.cpp inchangé, mêmes args qu'au 16/09 hors port) ; contrôle llama-server Q4_K_M b=1 20 s puis `acvram serve` b=1 / b=12 (ctx 2304/slot)
commit : d3c5bb2, 12:39-12:50
régime : `load average 70,8` à 12:52 (`pytest tests/ -q -k not gpu` et deux autres suites, 1 000-1 125 % CPU chacune, lancées par des pairs pendant ma fenêtre) ; carte : 0 processus étranger
scellé : contrôle 341,4 ± 3 % → tenu = on mesure ; hors bande = on s'arrête
mesuré : contrôle **259,1 t/s** (6 lots, 6 150 jetons, 23,7 s ; passes courtes 149 → 304, croissantes ; journal llama-server : décodage par requête 199 → 304 t/s, erratique ; 16/09 : 341,4, passes courtes 307-318, stables) · acvram (même fenêtre, contaminée) : b=1 320,6 t/s 1,011 J (7 lots, 7 168 jetons) ; b=12 966,1 t/s 0,405 J (2 lots, **20 829 jetons < 24 576 = 2 × 12 × 1 024** : lots finis sur EOS → indécidable même sur machine calme, attend l'`ignore_eos` d'Océane)
verdict : **FAUX (contrôle)** — le harnais n'est pas mis en cause (chemin llama.cpp inchangé, args identiques) ; la machine l'est : llama.cpp b=1 est sensible à l'hôte et trois suites de tests en parallèle le font varier de 200 à 304 t/s ; **aucun chiffre acvram publié** de cette passe ; à rejouer intégralement (contrôle puis acvram) sur machine calme, après accord des sessions (REGLES : accord avant un test) ; b=12 après l'`ignore_eos`

## Note
- La règle « hors bande = le script a changé » a bien arrêté la publication ; la cause réelle est ailleurs — le contrôle protège aussi de ça. À ajouter au protocole de tout harnais HTTP : relever `load average` et les processus > 100 % CPU en tête de fenêtre (comme la carte l'est déjà), et refuser au-dessus d'un seuil (proposé : load > nombre de cœurs / 2).
