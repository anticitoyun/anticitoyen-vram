# Suite de anticitoyen-vram-jxm (poste3, 23/09) — écarté, ticket resté ouvert

Rappel de chef : le correctif du gardien de service (cecc4e29) n'explique pas les deux
cas signalés — .qui de A (pris 18:15:50) manquant à 18:48, .qui de B (pris 18:55:53)
manquant vers 19:19, **aucune prise ni rendue au journal entre-temps** dans les deux cas.
Pistes données : le trap générique l.301 (rm inconditionnel, pour une mesure ou une prise
refusée), un autre écrivain du chemin, un nettoyage de /tmp.

## Écarté, avec la preuve

- **`guet.sh`, `surveillance-groupe.sh`, `~/.local/bin/rafraichir-poste`,
  `~/.local/bin/openwebui`** : lisent `.qui` (existence, premier champ), n'écrivent ni ne
  suppriment jamais ce chemin (grep sur les quatre fichiers, aucune occurrence de `rm`/`>`
  sur `$INFO`/`$VERROU.qui`).
- **`systemd-tmpfiles-clean`** : actif sur cette machine (`systemctl list-timers`, dernier
  passage 18:30:56 — DANS la fenêtre du cas A), mais la règle `/tmp` par défaut
  (`/usr/lib/tmpfiles.d/tmp.conf:11`, `q /tmp 1777 root root 10d`) n'efface qu'après
  10 jours d'inactivité — un fichier écrit à 18:15:50 et disparu à 18:48 (33 min) ne peut
  pas être cette règle. Aucune règle locale plus agressive sur `/tmp` (`grep` sur
  `/usr/lib/tmpfiles.d/*.conf` et `/etc/tmpfiles.d/*.conf`).
- **Réentrée imbriquée de carte.sh** (la conversion appellerait carte.sh en son sein) :
  bloquée bien avant `$INFO` par la garde `ACVRAM_CARTE_TENUE` (exit 66, ligne ~157), qui
  est exportée et donc héritée par le service détaché — un appel imbriqué ne peut pas
  atteindre l'écriture de `$INFO`.
- **Une mesure/état qui écraserait `$INFO` d'un service vivant** : structurellement
  impossible tant que le service tient `flock -n 9` en exclusif — une mesure concurrente
  reste bloquée dans l'attente ou est refusée (REFUS MUTUEL nommé), jamais dans la section
  qui écrit `$INFO`. Vérifié en relisant l'ordre du script (flock avant toute écriture).

## Fait, en défense (ne résout pas les deux cas, mais ferme la même classe de défaut)

Le trap générique (l.301, TYPE=mesure/etat) portait le MÊME défaut de principe que le
gardien de service avant cecc4e29 : `rm -f "$INFO"` inconditionnel à la sortie. Rien ne
prouve qu'il est en cause ici (le flock devrait l'empêcher — voir ci-dessus), mais rien ne
prouve le contraire non plus, et la garde ne coûte rien. Ajoutée par symétrie, avec une
trace **ANOMALIE** au journal si jamais ce trap trouvait un `.qui` qui n'est plus le sien —
de quoi laisser une preuve la prochaine fois, au lieu du silence actuel.
Test : `tests/test_carte_trap_generique.py` (2 tests : cas normal inchangé, cas d'un `.qui`
substitué non effacé + ANOMALIE journalisée) ; cassé une fois (garde retirée, rouge pour la
bonne raison), restauré.

## Trouvé au passage, PAS corrigé (hors ticket, à ouvrir séparément)

En écrivant le test du cas normal, une mesure réelle (`sleep 1`) sous `carte.sh` avec
`ACVRAM_DUREE_MAX` par défaut (1800 s) a fait pendre `subprocess.communicate()` jusqu'à
son propre timeout : le garde-fou de durée (`outils/carte.sh`, bloc `_garde`, ligne ~336)
lance `( sleep "$DUREE_MAX"; ... ) &` puis, à la fin normale de la commande, fait
`kill "$_garde"` — mais ce `kill` ne cible que le PID de la SOUS-SHELL, pas le `sleep`
qu'elle a lancé : le `sleep 1800` devient orphelin (reparenté), et comme ses descripteurs
1/2 sont hérités de carte.sh (jamais fermés pour eux, seuls 8/9 le sont), il garde le
tuyau stdout/stderr ouvert jusqu'à 30 minutes après la fin réelle de la mesure — invisible
tant qu'on ne lit pas le pipe jusqu'à l'EOF (un simple `wait()` sur le code de retour ne le
voit pas). Contourné dans les nouveaux tests par `ACVRAM_DUREE_MAX=0`. Sans lien avec le
.qui qui disparaît (le garde ne touche jamais `$INFO`), mais un défaut réel, orthogonal :
à ouvrir en pièce séparée si personne ne le connaissait déjà.

## Reste

Ticket jxm resté OUVERT pour les deux cas signalés : aucune cause confirmée au-delà du
gardien de service déjà corrigé. Sans accès à la carte (poste6 en conversion), impossible
de reproduire en direct. Si ça se reproduit, la nouvelle trace ANOMALIE (si le trap
générique est en cause) ou un luxe de détail supplémentaire à ajouter au journal du
gardien de service (ex. le contenu de `.qui` juste avant un `rm` sauté par la garde
jxm) donnerait la prochaine preuve.
