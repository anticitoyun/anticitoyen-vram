# yals — kimi « muet 300 s » sur les alias YALS : ni YALS ni kimi, le lanceur kimi-yals : verdict (poste5, 30/09)

* instrument : lecture seule — lignes horodatées de /mnt/AI_GENERATOR/YALS/yals.log (log_prompt/log_requests à false : aucune génération), forme (clés, rôles, longueurs) des sessions ~/.kimi-code-local/sessions, /proc/<pid>/fd ; reproduction du motif à sec (`sleep 8`, aucune carte)
* commit : poste5-yals (voir git log) ; edz lu : ~/TSV/menus-reels-definitif.tsv colonnes 1-8, 10, 12, 13 (jamais 9)
* régime : à sec, aucune prise ; aucune réponse de modèle lue (REGLES § 6 : tailles et codes seulement)
* scellé : « kimi a fini » serait FAUX si une session de la fenêtre n'avait pas lastTurnReason=completed ou avait duré ≥ 300 s ; « le lanceur tient les tuyaux » serait FAUX si aucun processus vivant n'avait le fd 1/2 sur un pipe de l'edz
* mesuré : 7/7 sessions kimi terminées « completed » en 1,8-15,9 s ; PID 2008404 (bash kimi-yals, 10:21:55) vivant à 11:53, fd 1 et 2 = pipe, enfant = proxy 5011 ; motif reproduit : 8,0 s (avant) contre 0,0 s (après)
* verdict : VRAI — trois causes, aucune côté kimi
* durée : 0 (à sec)

## Cause 1 (7 bras : agents-4b, ariel-24b, deepseek-32b, falcon-h1r-7b, glm47, lfm25, 10:21 rapide) : tuyaux tenus
YALS charge, reçoit la requête de kimi (28-33 k jetons d'invite : système + 25 outils), la termine en 2-16 s ; la
session kimi se clôt « completed » (6 à 125 jetons, aucun appel d'outil). Mais `~/.local/bin/kimi-yals:63` et `:95`
lancent le proxy et YALS par `( cd "$YALS_DIR" && … setsid nohup X >> log 2>&1 < /dev/null & )` : la redirection ne
vaut que pour X ; le sous-shell de la liste `cd && …` reste vivant (il attend `setsid`) avec le stdout/stderr de
l'APPELANT. `subprocess.run(capture_output=True)` du harnais attend la fin de fichier de ses tuyaux, donc la mort de
YALS (au changement d'alias) ou du proxy (jamais : 2008404 tient encore le tuyau du bras de 10:21) → « aucune réponse
en 300 s ». Même motif dans `~/.local/bin/kimi-tabby:63` (proxy). En terminal, rien ne se voit (les fd sont le tty).
Correctif proposé (hors dépôt, `~/.local/bin`, décision chef — l'edz d'poste1 rejoue ces bras en fin de chaîne) :
`( cd "$YALS_DIR" && exec env … setsid nohup ./YALS ) >> "$YALS_DIR/yals.log" 2>&1 < /dev/null &` (idem proxy ×2).

## Cause 2 (4 bras : awaxis-31b, gemma4-12b, gemma4-26b, gemma4-26b-apex) : architecture inconnue
YALS 3610610 : « unknown model architecture: 'gemma4' », et `/v1/model/load` rend quand même 200 en ~0,5 s sans
modèle (11 chargements fantômes). kimi-yals attend 24 × 5 s puis réessaie à contexte moitié, 3 fois : > 300 s.
Ces alias ne sont pas servables par ce YALS : retrait du menu yals ou YALS plus récent (décision) ; kimi-yals devrait
lire « unknown model architecture » et échouer net.

## Cause 3 (granite41-30b ; deepseek-32b au 1er essai) : cache KV
131 072 à q8_0 : « cudaMalloc failed … 17 408 MiB », 422 « Model context not initialized ». Le 422 est définitif mais
kimi-yals attend encore 120 s « en arrière-plan » avant de réessayer à 65 536 : granite dépasse les 300 s ; deepseek-32b
charge à 65 536 puis tombe dans la cause 1. Correctif : pas d'attente après un 422.

## Pistes de l'ordre, écartées par la mesure
Fenêtre servie ≠ annoncée : réelle (deepseek-32b 65 536 servi, 131 072 annoncé) mais sans effet ici (proxy borne sur
ctx-actuel ; 1er tour) · compaction : non déclenchée (28,6 k + 8 192 < 0,75 × contexte) · outils : aucun appel ·
fin de tour : « completed » partout.

## Restes
PID 2008404 (bash orphelin, tient un tuyau mort ; parent du proxy 5011) laissé, inoffensif. Les 4 délais d'poste1
sont à rejouer après le correctif de la cause 1, sinon ils redonneront 300 s.

## Suite (décisions chef, 30/09 12 h)
* Cause 1 corrigée hors dépôt (aucun outil du dépôt ne génère ces scripts) : `~/.local/bin/kimi-yals` (proxy, YALS) et
  `kimi-tabby` (proxy, TabbyAPI), 4 sites : redirection sur le sous-shell + `exec setsid nohup` ; sauvegardes
  `*.avant-yals-20260930-120417`, écriture atomique, `bash -n` propre. Test à sec sur le TEXTE réel de chaque site
  (commande remplacée par `sleep 8`, sous capture) : avant 8,0 s ×4, après 0,0 s ×4, démon vivant après retour ×4.
  poste1 prévenue avant le rejeu de ses 4 délais.
* Cause 2 : les 4 alias gemma4 sont retirés des menus (même méthode que lfm25 : sauvegardes `*.avant-yals-gemma4-<date>`,
  écriture atomique). Comptes avant → après : notes-modeles 463 → 459, usage-sources 719 → 712, ~/.kimi-code et
  ~/.kimi-code-local/config.toml 349 → 345 modèles chacun (yals 35, TOML relu, aucun des 4). Les fichiers GGUF restent sur le disque.
* Cause 3 : anticitoyen-vram-zjf (P3), non corrigée.
* Preuve réelle (poste1, rejeu edz 30/09 12 h 1x) : yals-agents-4b-kimi OK avec les lanceurs corrigés (300 s → réponse) ; orphelin kimi-yals de 10:21 tué par poste1 à 12:05.
