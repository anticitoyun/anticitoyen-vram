# Prédiction corrigée, écrite avant la mesure

La prédiction initiale composait trois gains en un seul nombre :

    decodage  1,026 x 1,069 x 1,0791 - 1  =  +18,4 %

**Les trois facteurs ne s'appliquent pas au même modèle.** Vérifié en chargeant
les deux :

    bf16-pur   scalers actifs = 0    tenseurs nvfp4 = 0   fusions = 44 MLP + 48 attn
    nvfp4      scalers actifs = 336  tout en nvfp4        fusions = 0 (96 refus)

* sur **bf16**, la fusion opère, mais le double tampon est dans `nvfp4_gemv`
  (aucun tenseur nvfp4) et le cache d'échelle n'a **aucun scaler** à mettre en
  cache ;
* sur **nvfp4**, le double tampon et le cache opèrent, mais la fusion est
  refusée sur les 96 groupes.

**Aucun modèle ne porte les trois.** Le +18,4 % ne décrit aucune construction
mesurable.

## Prédiction par modèle

    bf16-pur    decodage  +2,60 %                     fusion seule
    nvfp4       decodage  1,069 x 1,0791 = +15,3 %    tampon x cache
    bf16-pur    prefill   +4,23 % sur le TTFT         ligne separee

Le prefill porte sur le TTFT, les autres sur le débit : **les additionner
fabriquerait un nombre qui ne correspond à aucune grandeur mesurable.**

## Règle de lecture, posée d'avance

* **mesuré ≥ prédit − résolution** → chemins disjoints, gains confirmés ;
* **mesuré < prédit − résolution** → les chemins se recouvrent, et **on ne garde
  pas les chiffres avec un total plus petit**. La paire suspecte est double
  tampon + cache d'échelle, tous deux sur le décodage nvfp4 : elle se départage
  par un test à une variable, chacun seul contre la même base. À ne lancer que
  si le déficit dépasse la résolution.

## Conditions à déclarer dans le tableau

* le bf16 **refuse les graphes** sur cette carte à contexte plein — *poids en
  flux, `layers.44.mlp.gate_proj`*. Contexte réduit à 8192, et **déclaré** ;
* n ≥ 3 passages de régime, dispersion à deux décimales ;
* TTFT en deux lignes, froid et régime ;
* empreintes sur le même périmètre que le passage publié.

## Empreintes : ce qu'elles disent et ne disent pas

Une empreinte dit *« est-ce le même calcul »*, jamais *« est-ce le bon »*. Le
passage des échelles en float32 **fait délibérément changer les bits** —
`fp32 → fp16 → bf16` est un double arrondi et peut différer d'un ulp de
`fp32 → bf16`. Donc :

    empreinte CHANGEE sur un modele reconverti   attendu
    empreinte INCHANGEE sur un modele reconverti  le correctif n'a pas pris

Le contrôle vaut **dans les deux sens**, ce n'est pas une exception à tolérer.
