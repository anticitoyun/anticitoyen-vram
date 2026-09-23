"""Où les clés `genre` et `cout_decroissant` cessent d'être la même clé.

Sur Llama-2-7B elles désignent le même ensemble : toute projection d'attention
coûte 7,38 Mio, toute projection de MLP au moins 19,82. Le confondant est
STRUCTUREL — attention 4096x4096, MLP 11008x4096 — donc il se casse sur un
modèle où une projection d'attention est aussi grosse qu'une du MLP.

LA CONDITION EXACTE, et ce n'est PAS `intermediate_size < hidden_size` :

    num_attention_heads x head_dim  >=  intermediate_size

La reformulation par `hidden_size` suppose `nh x head_dim == hidden_size`, ce
qui est FAUX pour 42 des 92 denses du parc — jusqu'a un facteur 2,0
(Qwen3-0.6B : hidden 1024, nh x hd 2048). Il y a donc DEUX routes pour casser
le confondant, pas une : rétrécir le MLP, ou ÉLARGIR l'attention au-dela de
`hidden_size`. La seconde existe et elle est courante ; elle ne suffit
simplement pas — le meilleur dense du parc atteint 0,6667, soit un facteur 1,5
du seuil.

Le rapport indicatif reste intermediate_size / hidden_size :
    > 1  le MLP domine, genre et coût confondus (cas Llama-2 : 2,6875)
    ~ 1  ils s'égalisent, le confondant se casse
    < 1  l'attention domine, l'ordre par coût s'INVERSE contre l'ordre par
         genre — c'est là que les deux clés se contredisent, donc là que la
         mesure tranche.

Ne mesure rien : lit les config.json. Aucune carte, aucun poids.
"""
import glob, json, os, sys
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

RACINES = [MODELES, str(_p.Path(MODELES).parent)]

def octets(n_poids, bits_base=4.5, bits_cible=8.1875):
    return n_poids * (bits_cible - bits_base) / 8 / 2 ** 20

def tailles(c):
    h = c.get("hidden_size") or c.get("n_embd")
    if not h:
        return None
    nh = c.get("num_attention_heads") or c.get("n_head") or 1
    nkv = c.get("num_key_value_heads", nh) or nh
    hd = c.get("head_dim") or (h // nh if nh else h)
    # le MLP d'un MoE : moe_intermediate_size par expert, pas intermediate_size
    inter = (c.get("moe_intermediate_size") or c.get("intermediate_size")
             or c.get("ffn_hidden_size"))
    if not inter:
        return None
    return {
        "q_proj": h * nh * hd, "o_proj": nh * hd * h,
        "k_proj": h * nkv * hd, "v_proj": h * nkv * hd,
        "gate_proj": h * inter, "up_proj": h * inter, "down_proj": inter * h,
    }, h, inter

lignes = []
vus = set()
for racine in RACINES:
    for cfg in sorted(glob.glob(os.path.join(racine, "*", "config.json"))):
        nom = os.path.basename(os.path.dirname(cfg))
        if nom in vus:
            continue
        vus.add(nom)
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        # une config imbriquée (VLM, MoE) porte les tailles plus bas
        for cle in ("text_config", "llm_config", "language_config"):
            if cle in c and isinstance(c[cle], dict):
                c = {**c[cle], **{k: v for k, v in c.items() if k not in c[cle]}}
                break
        r = tailles(c)
        if not r:
            continue
        t, h, inter = r
        att_max = max(t["q_proj"], t["o_proj"], t["k_proj"], t["v_proj"])
        mlp_min = min(t["gate_proj"], t["up_proj"], t["down_proj"])
        lignes.append({
            "modele": nom, "hidden": h, "inter": inter,
            "rapport": inter / h,
            "att_max_mio": octets(att_max), "mlp_min_mio": octets(mlp_min),
            "recouvrement": att_max >= mlp_min,
            # UN DETECTEUR QUI RATE FABRIQUE UN CANDIDAT. Premiere version :
            # num_experts / num_local_experts / n_routed_experts seulement.
            # ernie-21b-a3b porte `moe_num_experts` et passait donc pour DENSE
            # — c'etait le seul candidat dense du parc, et il n'existait pas.
            # Toute cle contenant `expert` compte, plus l'architecture.
            "moe": bool(
                any(("expert" in k.lower() or k.lower().startswith("moe_"))
                    and c.get(k) for k in c)
                or any("moe" in a.lower() or "mixtral" in a.lower()
                       for a in (c.get("architectures") or []))),
        })

if not lignes:
    sys.exit("aucune config lue — les disques de modeles sont-ils montes ?")

# LES DEUX ENSEMBLES SONT LE MEME, verifie et non suppose : les 35 modeles qui
# cassent le confondant sont EXACTEMENT les 35 MoE du parc. Le controle est
# ecrit ici parce que « 35 et 35 » ne dit pas « les memes 35 ».
_casse = {d["modele"] for d in lignes if d["recouvrement"]}
_moe = {d["modele"] for d in lignes if d["moe"]}
print(f"casser le confondant et etre MoE : {len(_casse)} et {len(_moe)}, "
      f"ensembles identiques = {_casse == _moe}")
if _casse != _moe:
    print(f"  casse mais dense  : {sorted(_casse - _moe)[:4]}")
    print(f"  MoE mais confondu : {sorted(_moe - _casse)[:4]}")
print()

lignes.sort(key=lambda d: d["rapport"])
denses = [d for d in lignes if d["recouvrement"] and not d["moe"]]
print(f"{len(lignes)} modeles lus, dont {sum(d['moe'] for d in lignes)} MoE")
print(f"DENSES dont le confondant est casse : {len(denses)}")
for d in sorted(denses, key=lambda d: d["rapport"]):
    print(f"   {d['modele'][:50]:50s} rapport {d['rapport']:.4f}  "
          f"att {d['att_max_mio']:.2f}  mlp {d['mlp_min_mio']:.2f} Mio")
if not denses:
    print("   AUCUN. Le confondant ne se casse qu'en QUITTANT le regime dense,")
    print("   et un gain ne se transporte pas hors de son regime.")
print()
print(f"{'modele':46s}{'inter/hidden':>13s}{'att max':>9s}{'mlp min':>9s}{'MoE':>5s}")
print("-" * 82)
casse = [d for d in lignes if d["recouvrement"]]
for d in lignes[:14]:
    print(f"{d['modele'][:46]:46s}{d['rapport']:13.4f}{d['att_max_mio']:8.2f} "
          f"{d['mlp_min_mio']:8.2f} {'oui' if d['moe'] else '':>5s}")
print("...")
for d in lignes[-3:]:
    print(f"{d['modele'][:46]:46s}{d['rapport']:13.4f}{d['att_max_mio']:8.2f} "
          f"{d['mlp_min_mio']:8.2f} {'oui' if d['moe'] else '':>5s}")

print()
# LA MARGE, et non seulement le compte. « Aucun » sans distance au seuil laisse
# croire a une impossibilite ; le meilleur dense est a un facteur 1,5, ce qui
# est une architecture non attestee, pas une architecture interdite.
_den = [d for d in lignes if not d["moe"]]
if _den:
    _best = max(_den, key=lambda d: d["mlp_min_mio"] and
                d["att_max_mio"] / d["mlp_min_mio"])
    _r = _best["att_max_mio"] / _best["mlp_min_mio"]
    print(f"MARGE DU MEILLEUR DENSE : {_best['modele'][:40]} atteint "
          f"att/mlp = {_r:.4f}, soit un facteur {1 / _r:.2f} du seuil de 1,0. "
          f"Llama-2 est a {7.38 / 19.82:.4f}, facteur {19.82 / 7.38:.2f}.")
    print("  Une architecture dense qui casserait le confondant n'est donc pas")
    print("  INTERDITE, elle est NON ATTESTEE — la nuance change ce qu'on peut")
    print("  ecrire : « aucun dans ce parc de 92 », pas « aucune dense au monde ».")
print()
print(f"CONFONDANT CASSE (une projection d'attention >= la plus petite du MLP) : "
      f"{len(casse)} modeles sur {len(lignes)}")
print("Llama-2-7B pour comparaison : rapport 2,6875, att max 7,38, mlp min 19,82")
print()
print("Les candidats, du plus discriminant au moins :")
for d in casse[:8]:
    print(f"  {d['modele'][:44]:44s} rapport {d['rapport']:.4f}  "
          f"att {d['att_max_mio']:.2f} contre mlp {d['mlp_min_mio']:.2f} Mio")
if not casse:
    print("  AUCUN — le parc ne peut pas trancher le confondant, et c'est un")
    print("  resultat : il faudrait convertir un modele qu'on n'a pas.")
