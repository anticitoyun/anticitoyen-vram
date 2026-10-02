"""Plan du chargeur à sec (zzs, amendement 3, poste5 02/10) : rig lu par nvidia-smi (sous-processus, CUDA_VISIBLE_DEVICES=0, torch.cuda
masqué), puis _plan_from_manifest avec torch.cuda.mem_get_info remplacé ; aucun contexte CUDA. Usage : rig | plan F_MIO CTX…"""
import contextlib, io, json, os, pickle, sys
MODELE = "/mnt/AI_GENERATOR/models_acvram/Kimi-Linear-35B-kda-nvfp4"
if sys.argv[1] == "rig":
    import torch
    torch.cuda.is_available = lambda: False
    from acvram.hardware.detect import detect_rig
    r = detect_rig()
    open(sys.argv[2], "wb").write(pickle.dumps(r)); sys.exit(0)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "aucun contexte CUDA : CUDA_VISIBLE_DEVICES vide"
import torch
rig = pickle.loads(open(sys.argv[2], "rb").read())
F = int(float(sys.argv[3]) * 2**20)
tot = rig.gpus[0].total_mem
torch.cuda.mem_get_info = lambda *a, **k: (F, tot)
import acvram.hardware.detect as det
det.detect_rig = lambda profile=None: rig
from acvram.engine import loader
from acvram.engine.config import load_model_spec
man = json.load(open(os.path.join(MODELE, "acvram_manifest.json")))
for ctx in map(int, sys.argv[4:]):
    spec = load_model_spec(MODELE)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p = loader._plan_from_manifest(man, spec, max_model_len=ctx, max_concurrent_seqs=1)
    exil = loader._mlp_exiles(p)
    part = sum(1 for l in p.layers if l.experts_residents is not None)
    lignes = [l for l in buf.getvalue().splitlines() if "plan réajusté" in l or "exil" in l.lower()]
    print(f"ctx {ctx} : MLP exilés {exil}, couches à experts partiels {part} | " + " / ".join(x[:150] for x in lignes))
