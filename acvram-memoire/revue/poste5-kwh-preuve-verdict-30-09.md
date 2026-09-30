# kwh — preuve réelle sur la carte : aucun OOM au changement de modèle, mais la garde n'a pas été exercée : verdict (poste5, 30/09)

* instrument : `scratchpad/poste5-kwh-preuve-30-09/prise.sh` sous carte.sh mesure ; mémoire de la carte 0 toutes les 250 ms (`sorties/memoire.csv`) ; journaux des lanceurs (1-4.log), lus pour les lignes du lanceur seulement (aucune sortie de modèle)
* commit : f01d833d7 (branche poste5-kwh-preuve, depuis origin/main 85a64a920 : kwh + zjf + 7gb fusionnés) ; paquet acvram 0.7.15
* régime : RTX 5090 seule, ACVRAM_POSTE=poste5, lanceurs sans verrou service (ACVRAM_CARTE_SH inexistant) sous la prise mesure
* scellé : `revue/poste5-kwh-preuve-scelle-30-09.md` (f01d833d7, avant la prise)
* mesuré : 4/4 lancements prêts, 0 « mort au démarrage » / OOM / refus ; libération de la VRAM : A (27,3 Gio) → 15 Mio en ≤ 1,3 s, B (12,9 Gio) → 15 Mio en ≤ 0,35 s après SIGTERM, vLLM (28,9 Gio) → 15 Mio en ≤ 1,5 s ; aucune ligne « carte rendue » aux étapes 2-4 (vllm-serveur a vu 32 097 Mio libres)
* verdict : P1, P3, P4 TENUS ; P2 NON PROUVÉ — la garde n'a pas été exercée (seuil du scellé)
* durée : prévue ≤ 900 s, tenue=308 s (14:41:51 → 14:46:59) ; compute-apps début = fin (appoint 8081, carte 1)

## Lecture
Sur ce poste, un serveur tué rend sa VRAM en moins d'1,5 s. Les lanceurs mettent plus longtemps que ça avant d'appeler la
garde : vllm-serveur passe d'abord par le déchargement de TabbyAPI et par ses propres `kill && sleep 3-6`, et
acvram-serveur attend la mort du serveur qu'il remplace. Le changement de modèle est donc sûr aujourd'hui, mais cette
prise ne prouve pas que la garde protège : elle n'a jamais eu à attendre. La cascade de l'edz du 30/09 00:48 (10 OOM en
3 min, un démarrage dans la seconde) n'est pas reproduite par un changement de modèle « au calme ». Sa fenêtre
dépendait du harnais et de la charge de la nuit. La preuve de la garde reste celle des tests hermétiques
(test_attente_carte_kwh, 12).
Pour l'exercer en réel, il faudrait tuer un gros serveur et lancer le suivant dans les 0,3 s, sous charge. Ce n'est pas
le chemin de l'utilisatrice : je ne le propose pas sans demande.

## Constat annexe (7gb en réel)
Le trap EXIT de carte.sh a nommé et arrêté le relevé `nvidia-smi -lms 250` du script (« ORPHELIN … descendant pid
2771740 … (7gb) »). Le script l'avait bien tué (trap EXIT), mais nvidia-smi était encore en train de sortir : c'est
bénin, et c'est la première récolte réelle de _reaper_descendants. Cas limite : un processus déjà en sortie
(PF_EXITING) pourrait être attendu sans être nommé. Ce n'est pas une panne, seulement une ligne de journal en trop.
