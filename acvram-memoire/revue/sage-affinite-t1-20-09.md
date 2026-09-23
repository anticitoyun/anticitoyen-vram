# Sage — affinité CPU dans les prises de T1 : OUI, régime `cpus0-15`, pour les trois moteurs (20/09, 10 h 35, horloge machine)

Question de Jérôme (10 h 33, poste `20-09-1030`, ETAT e31c99be) : « le point fils n'est effectif que sous `carte.sh` avec `ACVRAM_CPUS` posé — affinité dans les prises de T1 ou pas (défaut : aucune) ? »

## Décision

**Oui : `ACVRAM_CPUS=0-15` dans toute prise de T1, concurrents compris.** Trois raisons, dans l'ordre :

1. Le lot demandé par l'utilisateur (« fils CPU à vérifier et régler ») est **un seul lot, un seul régime** (`sage-poste-lot-20-09` § tableau, ligne fils : `OMP_NUM_THREADS=8`, affinité 0-15) ; l'ordre de retrait « fils → THP → C-states → ASPM » suppose les fils dedans. Un T1 sans affinité mesurerait un lot à quatre points et le cinquième n'aurait jamais de chiffre.
2. Topologie lue (`/sys/.../core_cpus_list`, 10 h 34) : **0-15 = 8 P-cores × 2 fils, 16-31 = 16 E-cores**. Le fil qui lance les noyaux et le fil Python sont ce que l'hôte coûte (GLM b=1 : hôte 0,64 ms, espaces 2,3 ms sur ≈ 6,8 ms) ; un fil migré sur un E-core coûte ≈ 1,4-1,6 × par instruction.
3. Harnais égal (REGLES § 4, 18/09) : `carte.sh` applique `taskset -c $ACVRAM_CPUS` à **la commande qu'il enveloppe**, l'affinité s'hérite par fork/exec — vLLM (moteur + serveur API) et llama-server la reçoivent comme acvram. Une affinité posée sur acvram seul serait un régime de plus dans le comparatif.

## Prédiction (écrite avant la mesure)

| cellule | attendu | faux si |
|---|---|---|
| GLM b=1 acvram (T1) contre M00-ter (même poste, **sans** affinité — Manon l'a lancée 10 h 33 avant cette note : c'est le témoin gratuit) | pas −0,00 à −0,15 ms (0 à −2 %), \|A1−A2\| réduit | pas > M00-ter + 2·\|A1−A2\| → fils retiré |
| prefill Coder L=2 048 acvram | ± 1 % | < 0,97 × M00-ter → fils retiré (scellé du lot, inchangé) |
| Coder b=12 acvram | ± 1 % (le pas est GPU, 400 W) | < 0,97 × cellule 0.6.24 rejouée → retrait un point à la fois, fils d'abord |
| vLLM Marlin, llama.cpp | ± 1 % ; l'en-tête porte `taskset -p <pid>` = `ffff` | en-tête sans affinité ou ≠ `0-15` → cellule non comparable, à rejouer |

Issue qui me gêne : acvram gagne > 2 % à b=1 et vLLM rien — cela dirait que notre boucle hôte migrait sur les E-cores, coût que je n'avais pas nommé avant le 20/09 ; se publie tel quel sous le nom **poste**.

Coût : 0 min de carte (une variable dans l'enveloppe) ; le témoin sans affinité est M00-ter, déjà en cours.

## Preuve que le régime a pris (REGLES § 3, « prouver que la configuration a pris »)

`environment.d` n'est lu ni par GNOME ni par les shells (Jérôme 10 h 33) → **`ACVRAM_CPUS=0-15` passé explicitement à `carte.sh` par chaque chaîne**, comme `ACVRAM_MODELES`. Dans chaque en-tête : ligne `hote=thp,omp8,cpus0-15` (acvram, `hote.py:94`) ; pour les concurrents `taskset -p` du PID servant et du client (`verifier.sh:20`) ; `ps -L | wc -l` du servant. Une cellule sans ces lignes n'entre pas dans INDEX.

## Ordre

* **Manon** — M00-ter et M1 bis se finissent **telles quelles** (sans affinité : M00-ter est le témoin « poste 1030 sans fils »). Puis T1 entier : `ACVRAM_CPUS=0-15` posé explicitement devant `carte.sh` pour **chaque** prise, acvram comme vLLM et llama.cpp ; en-tête : `hote=…,cpus0-15` (acvram) ou `taskset -p` du servant et du client (concurrents), `ps -L | wc -l`, `poste=20-09-1030`. Scellés : table ci-dessus ; < −3 % sur une cellule → retrait fils seul, contrôle 6 min, puis T1 continue.
* **Jérôme** — `ACVRAM_CPUS=0-15` dans la ligne de régime de T1 dans ETAT ; `verifier.sh` relancé une fois sous charge pendant la première prise T1 (lecture seule, pas de carte) et sa sortie dans ETAT ; 0.6.31 embarque `outils/poste/` avec `ACVRAM_CPUS=0-15` documenté, non posé par le paquet.
* **Océane** — rien.
