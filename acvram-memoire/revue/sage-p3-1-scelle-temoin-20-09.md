# Sage — P3 (1) Qwen3-VL-2B : lecture du +10,2 %, défaut candidat n° 1, scellé (b″) avec témoin (20/09, 18 h 56)

Source : `verdict-p3-1-qwen3vl-2b-20-09` (main 91b1e5bd) ; code lu sur main 0e058e97 et transformers 5.17.0 du `.venv`.

## 1. Lecture du +10,2 %
* **Résolution** : n = 20, sd 22,2 % → 2 SE = 9,9 % : géo +10,2 % ∈ [−0,2 ; +21,7] — non résolu comme biais (§ 4 : « non établi pire de > 1 % »). Le seuil « ± 1 % » de (b′) était sous 2 × la résolution du témoin (2 SE = 3,1 % à sd 7 %) : pas un scellé (§ 3), retiré.
* **Ce qui est résolu : la dispersion.** sd 22,2 contre 7,0 % du témoin, F(19,19) = 10,1, p < 10⁻⁵. Biais et dispersion sont un seul fait : un écart centré sur les logits élève la NLL en moyenne (logsumexp convexe) ; le témoin donne l'échelle (sd 7 % → +1,75 %), ×3,2 en sd donne +5,5 à +17 % selon que le biais suit la sd ou la variance — +10,2 est dedans. Lecture : « acvram porte sur le chemin image un écart ≈ 3 × le bruit bf16 », pas « acvram est biaisé de 10 % ».
* **Greedy 13/20 ne dit rien** (témoin 13/20 : quasi-égalités du corpus) ; premier jeton 17 contre 19 = 2 images, ± 2 binomial, pas résolu. Seule la marge top-2 fp32 au pas divergent tranche (12B bf16 : jamais un renversement ≥ 1,7 nat).
* **KV int8 hors de cause pour l'excès** : KV bf16 sur img00-02 reste à +23 % vs fp32, premier jeton 2/3 → défaut en amont du cache, au prefill (k = 0). Mais le bras porte un bruit que le témoin n'a pas (`mesure-c.py:29-40` force les cibles pas à pas : lp passe par le cache int8 ; HF n'a pas de KV quantifié) : asymétrie de format, retirée du bras CHEMIN ci-dessous.

## 2. Fait lu (fichier:ligne) — défaut candidat n° 1 : le masque image de Gemma appliqué à Qwen3-VL
* acvram `engine/layers.py:979-980` : `ouvert = masque_images(...) if images and causal` ouvre les cellules image→image (docstring :933-935, « Gemma 4 ») pour TOUTE séquence à plages ; `engine/model.py:593` `plages = batch.images_de(i)` sans condition de famille ; aucun drapeau (`grep -ri bidirectionnel acvram/` : un commentaire, runner.py:1256).
* transformers 5.17.0 `models/qwen3_vl/modeling_qwen3_vl.py` : LM causal — `create_causal_mask` :33/:821, `is_causal = True` :492 (les `is_causal=False` :260-319 sont la tour ViT) ; `mm_token_type_ids` ne sert qu'au M-RoPE (:993) ; zéro masque bidirectionnel (gemma3 : 7 occurrences).
* Donc chez HF les 256 jetons image de Qwen3-VL sont causaux ; acvram les laisse se voir vers l'avant : K/V des lignes image changés à chaque couche, sur les 20 images, déterministe, prefill seul (les requêtes de décodage sont du texte). Invisible aux preuves au bit de (b)/(c) (deepstack, M-RoPE ne passent pas par `attention()`) ; compatible avec k = 0, KV hors de cause, et le KL 0,57 d'img00 (Océane 17 h 49 ; bf16 du 12B : KL ≤ 0,012).
* **Prédiction écrite avant** : porte de famille posée → sd(acvram bf16 / fp32) ≤ 10 %, géo +1 à +4 %, premier jeton ≥ 18/20 (70 %). Si la porte ne change rien (sd > 14 %) : suspect 2 = arithmétique du prefill (`attention()` :998 = SDPA à masque explicite, chemin math/efficient), suspect 3 = dtype d'application du M-RoPE ; juge = diff-couches acvram-bf16 contre HF-bf16 (première couche divergente, lignes texte contre lignes image), aucun correctif avant le nom.

## 3. Scellé (b″) — remplace (b′) ; témoin du même passage, même format (bf16), même référence (HF fp32, TF32 coupé)
| clause | seuil | témoin mesuré (17 h 58 / 18 h 26) | aujourd'hui (kv=int8, 91b1e5bd) |
|---|---|---|---|
| 1 dispersion | sd_log(acvram **kv=bf16** / HF fp32) ≤ 2 × sd_log(HF bf16 / HF fp32) | 7,0 % → seuil 14,0 % | 22,2 % → **FAUX** |
| 2 biais (filet, une direction) | géo(acvram kv=bf16 / HF fp32) − 2 SE ≤ +3 %, résolution publiée | +1,75 %, 2 SE 3,1 % | +10,2 − 9,9 = +0,3 % → non établi pire (sans pouvoir à n = 20) |
| 3 jetons | premier jeton = fp32 sur ≥ 18/20 ET toute divergence greedy (k ≤ 8) à marge top-2 fp32 < max(1 nat, 2 × marge du témoin), marges publiées | 19/20, sa divergence à marge publiée | 17/20, marges non lues |
| 4 servi | kv=int8 contre kv=bf16, 20 images : géo ≤ +1 % → int8 reste ; > +3 % → `kv=bf16(vision)` ; entre : fiche ; greedy 8 égal ≥ 18/20 | — | −1,7 % à n = 3 : non résolu |
| 5 inchangés | godets {1,2,8,16} avec image servis ; texte au bit (Coder i8c) | — | tenus 17 h 58 |
Tenu = 1 ∧ 3 ∧ 5, 2 « non établi pire », 4 rendu ; **faux si 1 ou 3 tombe**. Régime : 2B bf16 sous graphes, godet 1, cache préfixe off, ordre [texte, image], cibles = description seule à comptes égaux, ligne de régime complète (mrope=/deepstack=/masque_images=). Références fp32/bf16 réutilisables tant que corpus, refs (sha) et transformers 5.17.0 ne bougent pas.

## Ordre
* **Océane (à sec, ≤ 30 min, pointeur)** : (1) porte de famille sur `masque_images` : ouverte seulement si le manifeste le dit (`architectures` Gemma4* → bidirectionnelle, Qwen3VL* → causale, inconnu → refus nommé, jamais un défaut de classe) ; `regime_ligne` porte `masque_images=bidir|causal` ; tests cassants « Qwen3-VL + plage → aucune cellule ouverte » et « Gemma + plage → ouvertes » ; (2) marge top-2 fp32 au premier jeton de img02/img08/img12 (logits de `ref-2b-fp32.pt` s'ils y sont, sinon HF CPU fp32, 1 min), publiée AVANT le correctif : < 0,3 nat = quasi-égalité, ≥ 1 = défaut.
* **Manon (trou de 8 min, après § 3b)** : rejeu `chaine-p3-1.sh` sur le correctif fusionné — refs réutilisées si sha égal, bras kv=bf16 ×20 (168 s) puis kv=int8 ×20, godets ; verdict aux cinq clauses, sept lignes, marges des divergences dans le verdict.
* **Jérôme** : ETAT — (b′) remplacé par (b″), « ± 1 % » retiré ; 0.6.34 sert Qwen3-VL-2B seulement si (b″) tenu ; P3 (3)/(4) 30B ensuite.

## Addendum 19 h 03 — marges d'Océane : signal d'instrument avant lecture
Trois marges top-2 « fp32 » identiques à 0,750 nat sur trois images (img02/08/12) et une à 0,000 (img18) ne sont pas des valeurs fp32 : 0,750 = 3 × 0,25 = 6 × 0,125, la grille bf16 des logits d'amplitude 16-32 ; un ex æquo exact est improbable en fp32, courant en bf16. Prédiction : les logits de `ref-2b-fp32.pt` (ou la tête qui les produit dans `references-transformers.py`) passent par un bf16 — à vérifier à sec (1 min) : dtype du tenseur sauvé et top-2 imprimés à 6 décimales ; si tous multiples de 0,125 → la référence « fp32 » a un étage bf16, à corriger AVANT le rejeu de Manon (sinon le témoin et le scellé (b″) jugent contre une référence bf16 déguisée, § 4 bis). Si les valeurs sont fp32 pleines : coïncidence publiée telle quelle, marges 0,75 = quasi-égalités → les 3 divergences restent compatibles avec le bruit, la clause 1 (dispersion) tranche.
