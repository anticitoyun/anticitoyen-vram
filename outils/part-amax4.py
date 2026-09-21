#!/usr/bin/env python
"""Q1 « Four Over Six » — contrôle indépendant de la part de blocs ayant choisi amax/4, relue dans les CODES stockés
d un alias nvfp4 (pas dans le manifeste) : dans un bloc, le plus grand code de magnitude vaut 7 (niveau 6) sous amax/6
et 6 (niveau 4) sous amax/4 — l arrondi E4M3 de l échelle (≤ 6,25 %) ne déplace jamais ce maximum d un niveau.
Bloc clampé = échelle de bloc E4M3 == 448 (0x7E) : les deux candidats y sont confondus (amax/4 impossible).

    python outils/part-amax4.py <dossier alias> [--par-tenseur]

Imprime `RESULTAT {...}` : total (blocs, part_amax4, part_clampes, part_nuls) et, si le manifeste porte `echelle_nvfp4`,
l écart avec lui (`ecart_manifeste`, attendu 0). Rend 0 ; 2 si aucun tenseur nvfp4."""
import os, sys, json
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acvram.quant.nvfp4 import unpack_e2m1, BLOCK   # noqa: E402

E4M3_MAX_OCTET = 0x7E


def compter(qweight: torch.Tensor, block_scale_u8: torch.Tensor) -> dict:
    codes = unpack_e2m1(qweight) & 0x07                                   # [out, k]
    out, k = codes.shape
    blocs = codes.view(out, k // BLOCK, BLOCK)
    maxi = blocs.amax(dim=-1)
    clampe = block_scale_u8.view(out, k // BLOCK) == E4M3_MAX_OCTET       # échelle saturée : ni amax/6 ni amax/4 réels
    return {"blocs": int(maxi.numel()), "six": int(((maxi == 7) & ~clampe).sum()), "quatre": int(((maxi == 6) & ~clampe).sum()),
            "nuls": int((maxi == 0).sum()), "autres": int(((maxi != 7) & (maxi != 6) & (maxi != 0) & ~clampe).sum()),
            "clampes": int(clampe.sum())}


def main(dossier: str, par_tenseur: bool = False) -> int:
    man = json.load(open(os.path.join(dossier, "acvram_manifest.json")))
    wm = man.get("weight_map", {})
    total = {"blocs": 0, "six": 0, "quatre": 0, "nuls": 0, "autres": 0, "clampes": 0}
    detail = {}
    fichiers = {}
    for nom, ent in man.get("tensors", {}).items():
        if ent.get("format") != "nvfp4":
            continue
        kq, kb = f"{nom}.qweight", f"{nom}.block_scale"
        if kq not in wm:
            continue
        for k in (kq, kb):
            f = wm[k]
            if f not in fichiers:
                fichiers[f] = safe_open(os.path.join(dossier, f), "pt")
        c = compter(fichiers[wm[kq]].get_tensor(kq), fichiers[wm[kb]].get_tensor(kb).view(torch.uint8))
        for cle in total:
            total[cle] += c[cle]
        if par_tenseur:
            detail[nom] = {"blocs": c["blocs"], "part_amax4": round(c["quatre"] / max(1, c["blocs"]), 4),
                           "part_clampes": round(c["clampes"] / max(1, c["blocs"]), 4),
                           "manifeste": (ent.get("echelle") or {}).get("part_amax4")}
    if not total["blocs"]:
        print("RESULTAT aucun tenseur nvfp4 dans " + dossier); return 2
    n = total["blocs"]
    r = {"dossier": os.path.basename(os.path.abspath(dossier)), "regle_manifeste": (man.get("echelle_nvfp4") or {}).get("regle", man.get("options", {}).get("echelle_nvfp4")),
         "blocs": n, "part_amax4": round(total["quatre"] / n, 4), "part_max6": round(total["six"] / n, 4),   # hors clampés
         "part_clampes": round(total["clampes"] / n, 4), "part_nuls": round(total["nuls"] / n, 4), "part_autres": round(total["autres"] / n, 4)}
    em = man.get("echelle_nvfp4")
    if em:
        r["ecart_manifeste"] = {"blocs_amax4": total["quatre"] - em["amax4"], "blocs_clampes": total["clampes"] - em["clampes"],
                                "blocs": n - em["blocs"]}                # comptes exacts, pas des parts arrondies
    if par_tenseur:
        r["tenseurs"] = detail
    print("RESULTAT " + json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], "--par-tenseur" in sys.argv[2:]))
