# jd6 — carte.sh : un service d'un autre poste est refusé tant que le premier sert : verdict (poste5, 29/09)

* instrument : tests hermétiques `tests/test_carte_postes_jd6.py` (verrou isolé ACVRAM_VERROU, faux serveurs `sleep`, jamais les vrais /tmp/acvram-carte-*.lock)
* commit : poste5-jd6 (voir git log)
* régime : à sec, aucune prise réelle
* scellé : le refus doit casser sur l'ancien carte.sh (vérifié : 3/5 rouges, dont le refus) ; la famille carte existante reste verte
* mesuré : 6/6 verts ; tests/test_carte*.py + test_surveillance_groupe_carte + test_verrou : 57 verts
* verdict : VRAI
* durée : 0 (à sec)

## Cause
`flock -s` (pièce ComfyUI, 27/09) admet N services sur la carte ; le `.qui` n'avait pas de poste. Le 29/09 à
14:44-14:46, deux postes (poste6, poste1) ont servi en même temps sur la 5090.

## Correctif (outils/carte.sh, branche service)
* Poste d'un service : `ACVRAM_POSTE`, sinon `ACVRAM_SESSION` (posé par le lanceur de session, ex. « poste5 ») ; vide
  ou `-` = partage déclaré, sans poste.
* Tant qu'un service d'un poste vit, un service d'un AUTRE poste est REFUSÉ (code 4) : détenteur nommé, marche à
  suivre donnée, ligne `refus` au journal. Même poste (Open WebUI + ComfyUI d'une session), services sans poste
  (permanents lancés hors session, ou `ACVRAM_POSTE=-`) : partage inchangé. L'appoint 8081 vit sur la carte 1
  (autre VERROU) : non concerné.
* `.qui` : 5e champ `poste=<nom>` sur les lignes de service à poste ; absent sinon (contrat 4 champs au bit). La
  réécriture du `.qui` garde le 5e champ des autres services.
* Mutex court `$VERROU.postes` (fd 7, `flock -w 30`) autour de la lecture, du contrôle et de l'écriture du `.qui` :
  deux postes arrivés ensemble ne passent pas tous les deux. Le serveur ne l'hérite pas (`7>&-`).

## Trouvé en route (le 5e champ a réveillé deux lectures)
* carte.sh:285 `read -r _dp _ _dn _dt` : le type recevait « service poste=… » ; une MESURE face à un service à poste
  attendait au lieu d'être refusée. Corrigé (`_` final) + test cassant.
* carte.sh:338 : le 5e champ d'un service était lu comme un pgid ; il est maintenant exigé numérique.
* acvram/server/app.py (route /verrou) : `split(None, 3)[3]` donnait au type tout le reste du fichier (déjà faux avec
  plusieurs services depuis le 27/09) ; premier mot seul désormais.

## Limites
* Un script lancé hors session et sans ACVRAM_POSTE n'a pas de poste : il partage avec tous, comme avant. Pour
  compter comme poste, un tel script doit poser ACVRAM_POSTE.
