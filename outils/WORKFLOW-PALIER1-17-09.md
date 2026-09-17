# Workflow — Intégration des verdicts palier 1 dans les menus

**État:** À sec (préparation complète). Prêt à exécuter après diagnostic 70B/TRT-LLM.

**Auteur:** poste8  
**Date:** 17/09/2026  
**Cible:** 142 modèles palier 1 (blocs 0-7) + 4 palier 2 (Kimi, Gemma, Nemotron)

---

## Résumé

Intégrer dans `claude-modeles.md` et `kimi-modeles.md` les PPL (avec juge géo/médiane) et débits b=1 extraits des fichiers verdict.

**Format attendu :**
```
- `Modèle-X-nvfp4` (17G) — PPL 1,0253 géo (méd 1,0152) — verdict-palier1-bloc0-17-09 | b=1 63,8 t/s — verdict-palier1-bloc0-17-09 | [description originale]
```

---

## Étapes

### 1. Proposer le CSV template (manuel : ~5 min)

```bash
python outils/proposer-palier1-csv-17-09.py > /tmp/palier1-template-17-09.csv
```

Cela produit un CSV avec :
- **Modèles extraits** automatiquement des verdicts
- **Colonnes à remplir** : PPL, juge, PPL_médiane, débit_b1, régime

**Travail manuel :**
- Ouvrir chaque verdict bloc par bloc
- Compléter les colonnes pour chaque modèle
- Notation : utiliser le même format que les entrées existantes (e.g. "1,0253 géo", "63,8 t/s")

**Résultat :** `/tmp/palier1-mapping-17-09.csv` (rempli)

---

### 2. Vérifier en mode dry-run

```bash
python outils/integrer-palier1-verdicts-17-09.py /tmp/palier1-mapping-17-09.csv --dry-run
```

Cela affiche tous les changements proposés **sans modifier les fichiers**.

```
[DRY-RUN] claude-modeles.md : 95 modèles prêts à être mis à jour
[DRY-RUN] kimi-modeles.md : 47 modèles prêts à être mis à jour
Total : 142 modèles intégrés
```

**Vérification :**
```bash
git diff claude-modeles.md  # Aucun changement (mode dry-run)
```

---

### 3. Exécuter l'intégration

```bash
python outils/integrer-palier1-verdicts-17-09.py /tmp/palier1-mapping-17-09.csv
```

Cela modifie :
- `acvram-memoire/revue/claude-modeles.md`
- `acvram-memoire/revue/kimi-modeles.md`

**Vérification :**
```bash
git diff --stat claude-modeles.md kimi-modeles.md
# Doit afficher ~140 lignes modifiées
```

---

### 4. Tester les verdicts

```bash
rtk pytest tests/test_menus.py::test_i_verdicts_cites_existent_et_contiennent_chiffres -v
```

Le test existant `test_i` doit passer (vérifie que chaque chiffre cité existe dans son verdict source).

**Résultat attendu :** `PASSED`

---

### 5. Commit et fusion

```bash
git add acvram-memoire/revue/claude-modeles.md acvram-memoire/revue/kimi-modeles.md
git commit -m "menus : intégration verdicts palier 1 + palier 2 (142+4 modèles)"
# Aucune ligne Co-Authored-By ni Claude-Session (voir REGLES.md §1)
git push
```

---

## Structure des fichiers préparés

### Scripts disponibles

- **`proposer-palier1-csv-17-09.py`** (57 lignes)
  - Entrée : verdicts bloc*.md
  - Sortie : CSV template avec modèles proposés
  - Usage : `python ... > /tmp/csv`

- **`integrer-palier1-verdicts-17-09.py`** (150 lignes)
  - Entrée : CSV rempli
  - Sortie : menus modifiés (ou dry-run)
  - Options : `--dry-run` pour tester sans modifier

- **`extraire-palier1-verdicts-17-09.py`** (ancienpréparation, peut être ignoré)
  - Tentative de parsing automatique (échouera sur format narratif)
  - Garder pour référence, ne pas utiliser

### Fichiers verdict disponibles

Tous les 8 blocs sont en place dans `acvram-memoire/revue/` :
- `verdict-palier1-bloc0-17-09.md` (10 modèles)
- `verdict-palier1-bloc1-17-09.md` (10 modèles)
- `verdict-palier1-bloc2-17-09.md` (20 modèles)
- `verdict-palier1-bloc3-17-09.md` (20 modèles)
- `verdict-palier1-bloc4-17-09.md` (20 modèles)
- `verdict-palier1-bloc5-17-09.md` (20 modèles)
- `verdict-palier1-bloc6-17-09.md` (20 modèles)
- `verdict-palier1-bloc7-17-09.md` (22 modèles, bloc réduit)

Plus les verdicts palier 2 (4 modèles) :
- `verdict-palier2-kimi-linear-17-09.md`
- `verdict-palier2-gemma-4-26b-17-09.md`
- `verdict-palier2-nemotron-17-09.md`
- `poste7-calibration-verdict-17-09.md` (GLM, Coder)

---

## Notes

1. **À sec = préparation :** Les scripts sont prêts, aucun ne s'est exécuté sur les vrais données. Première exécution sera après validation finale.

2. **CSV template :** Le CSVtemplate produit par `proposer-palier1-csv-17-09.py` extraira les modèles par regex, mais recopier/vérifier manuellement les PPL/débits depuis les verdicts est obligatoire.

3. **Format strict :** La validation dans `test_i_verdicts_cites_existent_et_contiennent_chiffres` accepte:
   - Virgule ou point décimal (1,0253 ou 1.0253)
   - Zéro final optionnel (1,015 ou 1,0150)
   - Ainsi : chaque chiffre dans les menus doit figurer dans le verdict cité

4. **Prédiction <5 lignes :** Selon poste7, la passe finale (142 modèles) doit modifier <5 lignes en moyenne par modèle. Si >15 lignes par modèle, revoir le format ou le merge.

5. **Diagnostic 70B/TRT-LLM :** En attente. poste4 a clôturé le 70B (3 bogues trouvés et corrigés, commit 410dd07). TRT-LLM pas encore lancé.

---

## Arrêt et relance

Si interrompue à mi-chemin :
- `git status` : voir les fichiers modifiés
- `git stash` : sauvegarder si besoin
- Relancer depuis l'étape 2 (dry-run) avec le CSV partiellement rempli

**Jamais forcer :** `git reset --hard` sans archiver le CSV — les données de verdicts sont précieuses.

---

## Références

- REGLES.md §4 : Conditions obligatoires (juge + source verdict)
- test_i dans tests/test_menus.py : Falsifiable (peut rendre FAIL)
- Commit précédent : 95675e0 (preuve de concept GLM + Qwen3.8)
