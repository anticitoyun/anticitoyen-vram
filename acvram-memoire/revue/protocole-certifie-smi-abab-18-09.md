# Protocole — contrôle de l'instrument certifie (poste7-profil-verdict-18-09) : ABAB Coder b=1 ancien instrument (nvidia-smi dans la fenêtre) contre nouvel instrument (poste1, NVML en processus ou hors fenêtre)

prérequis : le nouvel instrument d'poste1 ; à défaut, bras B = `certifie-b12-15-09.py` lancé avec la lecture `smi` neutralisée par variable (à demander à poste1 : `CERT_SANS_SMI=1` ou équivalent) — je ne modifie pas l'instrument moi-même.
instrument : ABAB Coder b=1 (`ACVRAM_HYBRID_SLOTS=1`, rondes 20 s, régime classé, défauts du jour), A = ancien, B = nouveau, ordre A B A B, même fenêtre ; juge = pas ms des rondes (et t/s) ; J/jeton informatif.
scellé (poste7) : écart A − B attendu 0,10-0,14 ms/pas (deux appels de 27 ms par ronde de 1 787 pas ≈ 0,135 ms) ; < 0,06 ms ⇒ autre cause (le smi n'explique pas l'écart pur/rondes) ; > 0,20 ⇒ l'instrument porte plus que le smi. Prédiction : 0,12-0,14.
inventaire préalable : `revue/inventaire-cellules-certifie-18-09.md` (197 passes, 34 verdicts ; biais 0,135/pas_ms : ≥ 2 % sur 48 passes, toutes b ≤ 2 ; max 3,9 % ; cellules étiquetées, non effacées).
