# La borne mémoire de la RTX 5090, et ce qu'elle ferme

Mesuré le 9/09/2026 par deux sessions, trois dispositifs indépendants.

## Le chiffre

| charge                                | Go/s | % du pic annoncé |
|---------------------------------------|------|------------------|
| lecture pure `sum()` bf16, 80 Mio      |  998 | 55,7 % |
| lecture pure `sum()` bf16, 200 Mio     | 1055 | 58,8 % |
| cuBLAS bf16, couche complète           | 1056 | 58,9 % |
| notre GEMV nvfp4, couche complète      |  991 | 55,3 % |
| décodage en service, 26,09 Gio/28,14 ms|  995 | 55,5 % |
| copie f32 (lit **et** écrit)           | 1280 | 71,5 % |

**Le pic annoncé de 1792 Go/s suppose un trafic mixte.** Une lecture seule ne
sature pas les contrôleurs mémoire et plafonne autour de **1050 Go/s**, soit
58 % du pic. La copie, qui lit et écrit, monte à 71 % — par un autre chemin,
même conclusion : la limite est structurelle, pas logicielle.

## Ce que cela ferme

Notre GEMV nvfp4 est à **94 % du plafond atteignable**, non à 55 % du pic.

| piste                     | état |
|---------------------------|------|
| lire plus vite            | ~6 % restants, contre une borne matérielle |
| ordonnancer mieux         | 2,1 % de temps mort mesuré sous `nsys` |
| fusionner davantage       | +2,6 % pris, dentelure au-delà |
| le noyau nvfp4 lui-même   | +6,9 % pris par le double tampon |

**Il ne reste que « lire moins d'octets ».**

## La prédiction que le format ne peut pas contredire

À 1050 Go/s en lecture, un modèle lisant *N* Gio par pas ne peut pas descendre
sous :

    t_min = N × 1,0737 / 1050   secondes

Vérification sur le cas connu : 26,09 Gio → 26,7 ms de plancher, 28,14 ms
mesurés, soit 95 % de la borne. **Seul le nombre d'octets entre dans cette
formule** — aucun choix de format, de noyau ou d'ordonnancement n'en sort.

## Deux instruments écartés en chemin

**`sum()` sur de l'uint8** rendait 66 à 84 Go/s, vingt fois trop bas :
`t.sum(dtype=torch.int64)` convertit et réduit en 64 bits, et c'est la
réduction qui domine. La mesure ne portait pas sur la lecture. Elle était
assez absurde pour se dénoncer ; à 900 Go/s elle serait passée.

**Un tenseur unique répété** rendait jusqu'à 2429 Go/s, soit 36 % **au-dessus**
du pic matériel : le L2 de la 5090 fait 96 Mio et le tenseur y restait d'un
appel à l'autre. Tout dispositif de bande passante doit faire tourner assez de
jeux pour déborder ce cache.
