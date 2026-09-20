#!/usr/bin/env python3
"""Juge statistique apparié et séquentiel (poste7-tests-30min-20-09 § 3.3 ; poste7-reprise-rapide addendum 07 h 43) :
moyenne géométrique des rapports PPL_B/PPL_A sur les tranches appariées, 2 SE sur les log-rapports avec
SE = max(SE échantillon, SD_PLANCHER/√n) (SD_PLANCHER = 1,97 % = écart-type par tranche mesuré sur β), décision
jamais avant NMIN = 5 paires, « indécidable » publié à NMAX.
    geo-sequentiel.py <dossier> [A] [B]   (JSON ppl-<A>-*.json / ppl-<B>-*.json, clé "ppl") ; env SEUIL NMIN NMAX SD_PLANCHER
Codes : 10 tenu (géo + 2 SE ≤ seuil) · 11 faux (géo − 2 SE > seuil) · 12 continuer · 13 indécidable à NMAX."""
import glob, json, math, os, sys
O = sys.argv[1]; SEUIL = float(os.environ.get("SEUIL", "0.005"))
NA = sys.argv[2] if len(sys.argv) > 2 else "A"; NB = sys.argv[3] if len(sys.argv) > 3 else "B"   # dénominateur, numérateur
paires = []
for fa in sorted(glob.glob(os.path.join(O, f"ppl-{NA}-*.json"))):
    fb = fa.replace(f"ppl-{NA}-", f"ppl-{NB}-")
    if os.path.exists(fb):
        a, b = json.load(open(fa))["ppl"], json.load(open(fb))["ppl"]
        paires.append((os.path.basename(fa)[len(NA) + 5:-5], a, b, math.log(b / a)))
n = len(paires); NMAX = int(os.environ.get("NMAX", "9"))
if n < 2:
    print(f"VERDICT NON MESURÉ : {n} paire(s)"); sys.exit(12 if n < NMAX else 1)
lr = [p[3] for p in paires]; m = sum(lr) / n
# poste7 07 h 43 (poste7-reprise-rapide addendum) : à n < 5 le SE de l'échantillon est sous-estimé (β : décidé à
# n = 3 à tort) → SE = max(SE échantillon, SD_PLANCHER / √n), SD_PLANCHER = 1,97 % = écart-type par tranche de β
SD_PLANCHER = float(os.environ.get("SD_PLANCHER", "0.0197"))
se_ech = (sum((x - m) ** 2 for x in lr) / (n - 1)) ** 0.5 / n ** 0.5
se = max(se_ech, SD_PLANCHER / n ** 0.5)
geo = math.exp(m) - 1; bas, haut = math.exp(m - 2 * se) - 1, math.exp(m + 2 * se) - 1
for nom, a, b, l in paires:
    print(f"  {nom:8s} {NA} {a:.4f} {NB} {b:.4f} {NB}/{NA} {100 * (math.exp(l) - 1):+.2f} %")
print(f"GÉO {NB}/{NA} sur {n} tranches : {100 * geo:+.2f} % [2 SE : {100 * bas:+.2f} ; {100 * haut:+.2f}] "
      f"(SE {100 * se:.2f} % = max(échantillon {100 * se_ech:.2f}, plancher {100 * SD_PLANCHER / n ** 0.5:.2f}))")
if haut <= SEUIL:
    v = f"TENU : géo + 2 SE ≤ +{100 * SEUIL:.1f} % → {NB} équivalent à {NA} (=2 défaut / feu vert 0.6.25)"
elif bas > SEUIL:
    v = f"FAUX : géo − 2 SE > +{100 * SEUIL:.1f} % → {NB} dégrade {NA} (=2 : opt-in ; arbres : bisect 6 pas sur les 9 tranches)"
else:
    v = f"NON TRANCHÉ à 2 SE (intervalle contient +{100 * SEUIL:.1f} %) → pas de défaut / pas de feu vert par la règle"
# 3.3 (poste7-tests-30min-20-09) : arrêt séquentiel — décidé dès que l'intervalle à 2 SE est entièrement d'un côté
# du seuil ; sinon on continue jusqu'à NMAX paires → « indécidable » publié. Codes : 10 tenu, 11 faux, 12 continuer,
# 13 indécidable à NMAX.
NMIN = int(os.environ.get("NMIN", "5"))     # poste7 07 h 43 : jamais décidé avant 5 paires (SE sous-estimé à n < 5)
if n >= NMIN and (haut <= SEUIL or bas > SEUIL):
    etat, code = "DÉCIDÉ", (10 if haut <= SEUIL else 11)
elif n >= NMAX:
    etat, code = f"INDÉCIDABLE à {NMAX}", 13
else:
    etat, code = f"CONTINUER ({n}/{NMAX})", 12
print(f"VERDICT {v} — {etat}")
json.dump({"n": n, "nmax": NMAX, "geo": geo, "bas_2se": bas, "haut_2se": haut, "seuil": SEUIL, "paires": paires,
           "verdict": v, "etat": etat}, open(os.path.join(O, "geo.json"), "w"), indent=1)
sys.exit(code)
