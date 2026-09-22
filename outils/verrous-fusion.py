#!/usr/bin/env python3
"""Tous les verrous de fusion, evalues EN UNE PASSE, pour chaque groupe.

Ce script existe parce que la recherche des verrous s'est faite un par un,
dans l'ordre ou ils bloquaient : biais, puis hadamard, puis format, puis
echelle. Une recherche qui progresse ainsi GARANTIT que l'erreur se repete,
parce qu'a chaque etape le verrou suivant est invisible par construction —
trois conclusions fausses en une journee, chacune tiree d'un verrou leve sur
quatre.

Le correctif n'est pas la vigilance, c'est la structure : la conjonction est
ecrite UNE FOIS et tous ses termes sont evalues ensemble. La regle devient une
structure de donnees au lieu d'une discipline, et elle ne peut plus etre
enfreinte.

Elle rend en prime l'assiette de CHAQUE voie, puisque chaque voie est le
sous-ensemble defini par les termes qu'elle leve.
"""
import collections
import json
import os
import re
import subprocess
import sys
from datetime import datetime

import torch
from safetensors import safe_open

from outils.parc import get_parc_models, _get_configured_roots
OCTETS = {"nvfp4": .5625, "int4_awq": .5625, "int8": 1.0625,
          "bf16": 2.0, "fp16": 2.0, "q3n": .40625}
GROUPES = {"qkv": ("self_attn", ["q", "k", "v"]),
           "gate_up": ("mlp", ["gate", "up"])}
# `fuse()` n essaie QUE trois empileurs : int8, nvfp4, plain (bf16/fp16).
# q3n n en a aucun — un groupe parfaitement homogene en q3n ne fusionne
# JAMAIS. Compter l homogeneite comme une liberte, c est lire la propriete
# voisine de celle qui decide.
#
# int4_awq en a un depuis `stack_int4_awq_linears` : ses `scales` sont deja
# `[out, in//group]`, une echelle par ligne de sortie, si bien que la
# concatenation sur l axe 0 est exacte sans rien introduire — la ou NVFP4,
# porteur d une echelle globale scalaire, avait eu besoin de
# `global_scale_rows`. Les 135 groupes releves fusionnent depuis.
EMPILABLES = {"nvfp4", "int8", "int4_awq", "bf16", "fp16"}


def _echelles_egales(dossier, wm, cles):
    """Les act_scale du groupe sont-elles identiques AU BIT PRES ?

    Rend (verdict, presentes). Une echelle absente n'est pas une echelle
    differente : les deux cas se distinguent et ne se traitent pas pareil.
    """
    kk = [c + ".act_scale" for c in cles]
    presentes = [k for k in kk if k in wm]
    if not presentes:
        return True, False                 # aucune echelle : rien a concilier
    if len(presentes) != len(kk):
        return False, True                 # certaines seulement : incomparable
    ref = None
    for k in kk:
        with safe_open(os.path.join(dossier, wm[k]), framework="pt") as f:
            t = f.get_tensor(k)
        if ref is None:
            ref = t
        elif not torch.equal(ref, t):
            return False, True
    return True, True


def verrous(dossier):
    """Un enregistrement par groupe, avec TOUS ses verrous."""
    p = os.path.join(dossier, "acvram_manifest.json")
    m = json.load(open(p))
    ts, wm = m["tensors"], m.get("weight_map", {})
    out = []
    for c in range(400):
        for genre, (sous, noms) in GROUPES.items():
            cles = [f"model.layers.{c}.{sous}.{n}_proj.weight" for n in noms]
            if not all(k in ts for k in cles):
                continue
            fmts_liste = [str(ts[k].get("format")) for k in cles]
            fmts = set(fmts_liste)
            hads = {ts[k].get("hadamard_block") or 0 for k in cles}
            biais = [k.replace(".weight", ".bias") in wm for k in cles]
            # Gardes communes aux DEUX empileurs, absentes de la premiere
            # version de ce script : meme largeur d entree et meme group_size.
            # Les omettre majore l assiette, exactement comme le biais la
            # majorait tant que `stack_int8_linears` le refusait (22f08b8).
            entrees = {tuple(ts[k].get("shape", [0, 0])[1:]) for k in cles}
            gsz = {ts[k].get("group_size") for k in cles}
            ech_ok, ech_presentes = _echelles_egales(dossier, wm, cles)
            out.append({
                "couche": c, "genre": genre,
                "format_ok": len(fmts) == 1 and fmts <= EMPILABLES,
                "format_sans_empileur": len(fmts) == 1 and not fmts <= EMPILABLES,
                "hadamard_ok": len(hads) == 1,
                "biais_ok": len(set(biais)) == 1,      # tous ou aucun
                "taille_ok": len(entrees) == 1 and len(gsz) == 1,
                "echelle_ok": ech_ok,
                "echelles_presentes": ech_presentes,
                "formats": sorted(fmts),
                # La voie 2+1 ne sauve un groupe heterogene que si DEUX de ses
                # trois membres partagent un format qui a un empileur. Un
                # gate_up n a que deux membres : il n a pas de voie partielle.
                "partiel_possible": (
                    len(cles) == 3
                    and any(fmts_liste.count(f) >= 2 and f in EMPILABLES
                            for f in fmts)),
            })
    # PAS d arret anticipe : une couche sans attention pleine ne signale pas
    # la fin du modele. Sur les architectures hybrides (attention lineaire ou
    # GDN alternee) la couche 0 en est depourvue, et l ancien `break` arretait
    # tout des la premiere : 1977 groupes manquants sur 42 modeles, les
    # Qwen3.x-27B comptes 1 groupe au lieu de 80. Trouve par Laurine.
    return out


def _experts_presents(ts, c, proj):
    """Indices d experts presents pour cette couche et cette projection,
    decouverts depuis les cles du manifeste -- jamais devines depuis un
    compte d experts lu ailleurs (config.json peut mentir ou differer du
    checkpoint reellement converti)."""
    motif = re.compile(rf"^model\.layers\.{c}\.mlp\.experts\.(\d+)\.{proj}_proj\.weight$")
    return sorted(int(mo.group(1)) for k in ts
                  for mo in (motif.match(k),) if mo)


def verrous_experts(dossier):
    """Un enregistrement par (couche, projection) pour le STACKING inter-experts.

    Ceci n est PAS le meme axe que `verrous()` : un MoE ne fusionne jamais
    gate+up d un meme expert. loader.py l exclut explicitement de
    Attention/MLP.fuse() -- "les experts d un MoE en sont exclus : ils
    passent par le chemin groupe, qui empile deja les 128 experts, et les
    fusionner un a un doublerait leurs poids sans rien accelerer". Le vrai
    mecanisme (MoEBlock._try_build_stacks) empile les E experts d UNE SEULE
    projection a la fois -- gate, up et down chacun leur pile, INDEPENDANTES :
    un `down` heterogene entre experts n empeche pas `gate` de s empiler.

    Les verrous de `_try_build_stacks`, lus dans le code, evalues EN UNE
    PASSE comme le reste de ce script :
      - format homogene entre experts ET empilable (memes empileurs que
        qkv/gate_up, voir EMPILABLES ci-dessus) ;
      - shape et group_size identiques entre experts (une pile est un
        tenseur [E, ...], elle n accepte pas des membres de tailles
        differentes) ;
      - scaler d entree identite : ni rotation Hadamard (`hadamard_block`)
        ni echelle de canal AWQ (`has_act_scale`) sur AUCUN expert -- le
        code refuse au moindre scaler actif, meme si tous les experts
        partageraient la meme echelle (contrairement a l echelle qkv/gate_up,
        qui peut etre COMMUNE) ;
      - aucun expert de la couche en flux (mlp_storage cpu dans le plan) :
        un expert exile vers l hote n a pas de pile GPU a rejoindre.
    """
    p = os.path.join(dossier, "acvram_manifest.json")
    m = json.load(open(p))
    ts = m["tensors"]
    plan = m.get("plan", m)
    layers_plan = {lp.get("index"): lp for lp in plan.get("layers", [])}
    out = []
    for c in range(400):
        for proj in ("gate", "up", "down"):
            experts = _experts_presents(ts, c, proj)
            if not experts:
                continue
            cles = [f"model.layers.{c}.mlp.experts.{e}.{proj}_proj.weight"
                    for e in experts]
            fmts = {str(ts[k].get("format")) for k in cles}
            shapes = {tuple(ts[k].get("shape", [])) for k in cles}
            gsz = {ts[k].get("group_size") for k in cles}
            hads = {ts[k].get("hadamard_block") or 0 for k in cles}
            has_scale = {bool(ts[k].get("has_act_scale")) for k in cles}
            lp = layers_plan.get(c, {})
            out.append({
                "couche": c, "genre": f"stack_{proj}", "n_experts": len(experts),
                "format_ok": len(fmts) == 1 and fmts <= EMPILABLES,
                "format_sans_empileur": len(fmts) == 1 and not fmts <= EMPILABLES,
                "taille_ok": len(shapes) == 1 and len(gsz) == 1,
                "scaler_identite": hads == {0} and has_scale == {False},
                "streamed": lp.get("mlp_storage") == "cpu",
                "formats": sorted(fmts),
            })
    return out


def _classifier_modele(nom):
    """Classe un modèle : prod / essai / ?
    Essai : contient 'temoin' (modèles de mesure), ou 'agents-a1-4b-kimi' (variantes kimi),
    ou 'qwen2.5-coder-14b-pur' (conversions de test)
    Sinon : ? (défaut, jamais prod par défaut)
    """
    lower_nom = nom.lower()
    if 'temoin' in lower_nom:
        return 'essai'
    if 'agents-a1-4b-kimi' in lower_nom:
        return 'essai'
    if 'qwen2.5-coder-14b-pur' in lower_nom:
        return 'essai'
    return '?'


def _ecrire_temoin(donnees_modeles, output_path='acvram-memoire/corpus/verrous-fusion.tsv'):
    """Écrit le fichier témoin TSV avec en-tête explicite."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                        stderr=subprocess.DEVNULL, text=True).strip()
    except:
        commit = 'unknown'

    roots = _get_configured_roots()

    with open(output_path, 'w') as f:
        f.write(f"# Témoin verrous-fusion — {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"# Commit : {commit}\n")
        f.write(f"# Modèles : {len(donnees_modeles)}\n")
        f.write(f"# Racines déclarées par outils/parc.py :\n")
        for root in roots:
            count = len([d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d))
                        and os.path.isfile(os.path.join(root, d, 'acvram_manifest.json'))])
            f.write(f"#   - {root} ({count} modèles)\n")
        f.write(f"#\n")
        f.write("nom_modele\tracine\ttotal\tfusionnent\tvoie_2plus1\tsans_recours\techelle\tmanifeste_mtime\ttype\n")

        for row in sorted(donnees_modeles, key=lambda x: x['nom']):
            f.write(f"{row['nom']}\t{row['racine']}\t{row['total']}\t{row['fusionnent']}\t"
                   f"{row['voie_2plus1']}\t{row['sans_recours']}\t{row['echelle']}\t"
                   f"{row['mtime']}\t{row['type']}\n")


def main(argv):
    models_with_roots = get_parc_models(single_root_only=False, include_location=True)
    cibles = argv[1:] or sorted(models_with_roots.keys())
    total = collections.Counter()
    par_voie = collections.Counter()
    donnees_modeles = []

    for nom in cibles:
        info = models_with_roots.get(nom)
        if not info:
            continue
        d, root = info
        if not os.path.isfile(os.path.join(d, "acvram_manifest.json")):
            continue
        try:
            gs = verrous(d)
        except Exception as e:                          # noqa: BLE001
            print(f"  {nom} : illisible ({str(e)[:40]})", file=sys.stderr)
            continue

        # Récupérer la mtime du manifeste
        manifest_path = os.path.join(d, "acvram_manifest.json")
        mtime = datetime.fromtimestamp(os.path.getmtime(manifest_path)).strftime('%Y-%m-%d')

        # Accumuler les stats pour ce modèle
        stats_modele = {
            'nom': nom,
            'racine': root,
            'total': 0,
            'fusionnent': 0,
            'voie_2plus1': 0,
            'sans_recours': 0,
            'echelle': 0,
            'mtime': mtime,
            'type': _classifier_modele(nom),
        }

        for g in gs:
            total["groupes"] += 1
            stats_modele['total'] += 1
            libres = (g["format_ok"] and g["hadamard_ok"]
                      and g["biais_ok"] and g["echelle_ok"]
                      and g["taille_ok"])
            if libres:
                par_voie["fusionnent deja"] += 1
                stats_modele['fusionnent'] += 1
                continue
            # quelle voie leverait CE groupe, sans rien approximer ?
            if g["format_sans_empileur"]:
                par_voie["homogene SANS empileur (int4_awq, q3n)"] += 1
            elif not g["taille_ok"]:
                par_voie["bloque par TAILLE (entree ou group_size)"] += 1
            elif not g["echelle_ok"]:
                par_voie["bloque par ECHELLE (table AWQ)"] += 1
                stats_modele['echelle'] += 1
            elif not g["format_ok"]:
                if g["partiel_possible"]:
                    par_voie["bloque par FORMAT, voie 2+1 possible"] += 1
                    stats_modele['voie_2plus1'] += 1
                else:
                    par_voie["bloque par FORMAT, rien a sauver"] += 1
                    stats_modele['sans_recours'] += 1
            elif not g["hadamard_ok"]:
                par_voie["bloque par HADAMARD"] += 1
            else:
                par_voie["bloque par BIAIS"] += 1

        donnees_modeles.append(stats_modele)
    print(f"{'etat':38s} {'groupes':>8s} {'part':>7s}")
    t = total["groupes"] or 1
    for k, n in par_voie.most_common():
        print(f"{k:38s} {n:8d} {100*n/t:6.1f} %")
    print(f"{'TOTAL':38s} {t:8d}")
    print("\nLa voie 2+1 ne s'applique qu'aux groupes bloques par le FORMAT SEUL,")
    print("dont DEUX membres partagent un format empilable : un groupe dont les")
    print("echelles different reste refuse quel que soit son format, et un gate_up")
    print("n'a que deux membres, donc pas de voie partielle.")

    # Écrire le fichier témoin explicable
    _ecrire_temoin(donnees_modeles)

    # Axe SEPARE : le stacking inter-experts d'un MoE. Ne s'ajoute ni ne
    # retranche rien au compte ci-dessus -- qkv et gate_up ne voient jamais
    # les experts (loader.py les exclut de fuse()), ce nouvel axe ne voit
    # qu'eux. Un modele sans MoE y contribue zero ligne, pas un zero compte.
    print("\n" + "=" * 55)
    print("STACKING INTER-EXPERTS (axe separe, n'entre pas dans le TOTAL ci-dessus)")
    print("=" * 55)
    total_exp = collections.Counter()
    par_voie_exp = collections.Counter()
    controle = collections.Counter()          # (modele, couche) -> nb de genres vus
    for nom in cibles:
        info = models_with_roots.get(nom)
        if not info:
            continue
        d, root = info
        if not d or not os.path.isfile(os.path.join(d, "acvram_manifest.json")):
            continue
        try:
            ge = verrous_experts(d)
        except Exception as e:                          # noqa: BLE001
            print(f"  {nom} : illisible pour les experts ({str(e)[:40]})", file=sys.stderr)
            continue
        vues = collections.Counter()
        for g in ge:
            total_exp["groupes"] += 1
            vues[g["couche"]] += 1
            libre = (g["format_ok"] and g["taille_ok"]
                      and g["scaler_identite"] and not g["streamed"])
            if libre:
                par_voie_exp["empilent deja"] += 1
                continue
            if g["format_sans_empileur"]:
                par_voie_exp["homogene SANS empileur (int4_awq, q3n)"] += 1
            elif not g["taille_ok"]:
                par_voie_exp["bloque par TAILLE (shape ou group_size)"] += 1
            elif g["streamed"]:
                par_voie_exp["bloque : experts en flux (mlp_storage cpu)"] += 1
            elif not g["scaler_identite"]:
                par_voie_exp["bloque par SCALER (hadamard ou echelle AWQ)"] += 1
            else:
                par_voie_exp["bloque par FORMAT"] += 1
        # Controle : chaque couche MoE doit rendre 3 genres (gate, up, down)
        # SUR UNE ARCHITECTURE SwiGLU, ou 2 (up, down) sur une architecture
        # sans porte separee (hidden_act relu2 : Nemotron/DeepSeek-style,
        # aucun gate_proj par expert -- verifie sur config.json, 6 modeles
        # du parc). Jamais E ni 3*E : ce serait le signe que les experts ont
        # ete remis en unite de groupe au lieu d'etre l'assiette d'un groupe.
        for couche, n in vues.items():
            if n not in (2, 3):
                controle[f"{nom} couche {couche} : {n} genres (attendu 2 ou 3)"] += 1
    if total_exp["groupes"]:
        te = total_exp["groupes"]
        for k, n in par_voie_exp.most_common():
            print(f"{k:45s} {n:8d} {100*n/te:6.1f} %")
        print(f"{'TOTAL (genres gate/up/down, PAS des experts)':45s} {te:8d}")
    else:
        print("Aucun MoE avec experts nommes model.layers.N.mlp.experts.E.* trouve.")
    if controle:
        print(f"\nCONTROLE VIOLE sur {len(controle)} (modele, couche) :", file=sys.stderr)
        for k in list(controle)[:20]:
            print(f"  {k}", file=sys.stderr)
    else:
        print("\nControle OK : chaque couche MoE rend exactement 3 genres (gate/up/down).")


if __name__ == "__main__":
    main(sys.argv)
