# kwh — acvram-serveur attend que la carte ait rendu la VRAM avant de démarrer : verdict (poste5, 30/09)

* instrument : tests hermétiques `tests/test_attente_carte_kwh.py` (faux nvidia-smi dans PATH rejouant des instants, ACVRAM_EXEC qui relève l'instant du lancement) ; un passage à sec sur la vraie carte (lecture seule, port 1)
* commit : poste5-kwh (voir git log)
* régime : à sec, aucune prise ; carte à l'edz d'poste1 (service 88705 vivant pendant le passage réel)
* scellé : chaque test de défaut casse sur le lanceur d'avant (vérifié : 5/6 rouges ; le 6e, « un vivant ne fait pas attendre », est la garde de non-régression et passe des deux côtés)
* mesuré : 6/6 verts ; tous les tests qui lisent parc/bin : 112 verts ; passage réel : 0,29 s, aucune attente à tort (5090 : 523 Mio utilisés, compute-apps ≥ 500)
* verdict : VRAI
* durée : 0 (à sec)

## Cause
`parc/bin/acvram-serveur` tuait le serveur remplacé puis `sleep 4`, sans rien vérifier ; et rien du tout quand le
précédent avait été arrêté par un autre (harnais, menu). Un PID mort quitte compute-apps AVANT que le pilote ait
rendu sa VRAM (le harnais attendait déjà « mort et sans VRAM au PID », et l'OOM est venu quand même : poste1 a dû
ajouter memory.used ≤ 1 Gio, 249674e39). La garde VRAM lisait aussi cette VRAM orpheline comme occupée → refus à tort.

## Correctif (`attendre_carte_rendue`)
Sur la carte servie : attendre qu'aucun PID de calcul ne soit en sortie (disparu, zombie, PF_EXITING de
/proc/<pid>/stat), que le PID remplacé ne vive plus, et que utilisée − Σ compute-apps ≤ ACVRAM_ORPHELINS_MIO (1 024).
Un processus VIVANT (ComfyUI, 8081) ne fait pas attendre. Appelée avant la garde VRAM et après l'arrêt du remplacé.
Délai ACVRAM_ATTENTE_VRAM (60 s ; 0 = coupé), puis refus nommé rc 1 : « carte non rendue en N s : PID [..] en sortie ;
M Mio sans processus ». Remplacé sourd au SIGTERM : SIGKILL après ACVRAM_DELAI_TERM (30 s), signé au journal des arrêts.
nvidia-smi muet ou mémoire par PID illisible ([N/A]) : attente inactive / critère des orphelins coupé, dit.

## Restes
* vllm-serveur et llamacpp-serveur n'ont pas la garde (même défaut possible au changement de moteur) — hors pièce.
* Preuve en service réel (changement de modèle par le menu, deux gros alias) : à faire sur fenêtre de carte.
