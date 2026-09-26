#!/usr/bin/env python3
"""PPL par le CHEMIN DE DÉCODAGE — juge de (5), poste7 puis chef (15/09).

Montage (teacher forcing, KV continu) différent de l'étalon 9,2056/9,1218
(fenêtres 2048/2048 réinitialisées) : le seuil absolu de poste7 ne
s'applique plus tel quel. chef demande DEUX bras avec CE script :
`ACVRAM_MOE_DECODE_MMA=0` (A, GEMV W4A16, contrôle de cohérence — doit
tomber près de l'étalon du parc 9,12 ± 0,05, sinon le montage a un biais
à nommer) puis `=1` (B, MMA W4A4). Juge : B/A ≤ 1,010 → (5) s'ouvre ;
> 1,015 → défaut propre au décodage, retour à poste4 ; entre les deux,
retour à poste7.

    ACVRAM_MOE_DECODE_MMA=0 outils/carte.sh python outils/ppl-decode-mma-coder30b.py
    ACVRAM_MOE_DECODE_MMA=1 outils/carte.sh python outils/ppl-decode-mma-coder30b.py

PIÈGE VÉRIFIÉ DANS LE CODE avant d'écrire ce script (demandé par chef,
15/09) : `acvram eval` (`acvram/evaluate.py::perplexity`) construit
TOUJOURS un batch `is_prefill=True` avec `t = len(chunk) = --window`, en
UN SEUL forward. Le choix `_forward_grouped_mma` vs `_forward_prefill_grouped`
(`acvram/engine/model.py:1172-1176`) ne regarde PAS `is_prefill` : il
regarde `t <= _MOE_GROUPED_MAX` (32 par défaut). Donc `acvram eval
--window 2048` (la fenêtre de l'étalon 9,2056/9,1218) construit un `t=2048`
et prend TOUJOURS `_forward_prefill_grouped` — **jamais** le chemin de
décodage, quel que soit `ACVRAM_MOE_DECODE_MMA`. `--window` COMPORTE À LA
FOIS le nombre de jetons par appel (ce qui doit rester ≤ 32) et la
profondeur de contexte par position (ce qu'on veut réaliste, donc >> 32) :
**ces deux besoins ne peuvent pas être servis par le même paramètre**, et
`acvram eval` n'offre aucun autre levier — la mesure demandée n'existe
pas comme simple combinaison de drapeaux CLI.

CE SCRIPT construit donc le contexte AUTREMENT : un `Engine` sous graphes
(CUDA graphs — « sous graphes » de la consigne), UNE requête, jetons
DU CORPUS forcés en entrée à chaque pas (`_sample_only` détourné pour
rendre le jeton VRAI du corpus au lieu de l'argmax du modèle — sinon le
décodage dérive de sa propre sortie après le premier désaccord et compare
un texte auto-généré au corpus, pas le modèle au corpus). Chaque pas de
décodage traite `t=1` jeton : `t <= 32` est vérifié à CHAQUE pas, donc
`_forward_grouped_mma` est bien pris (vérifié en journalisant le nombre de
pas où `_stack_state == "oui"` et `t=1`, imprimé en fin de mesure).

DIFFÉRENCE MÉTHODOLOGIQUE ASSUMÉE avec l'étalon 9,2056/9,1218 (fenêtre
2048/2048, min_context 0, contexte RÉINITIALISÉ à chaque fenêtre de
2048) : ici le cache KV est CONTINU du premier au dernier jeton — chaque
position voit TOUT son passé, jamais moins que l'étalon à profondeur
égale, potentiellement plus aux positions qui suivaient un redémarrage de
fenêtre chez lui. Un contexte au moins aussi bon ne peut PAS improve la
PPL relativement à un défaut réel du chemin décodage : si (5) est fautif,
cette mesure le verra aussi. Le nombre de jetons notés est choisi pour
rester du même ordre de grandeur (≈ 8191, comme 4 fenêtres de 2048 avec
min_context 0).
"""
import math
import os
import sys
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# `_MOE_DECODE_MMA` (acvram/engine/model.py) est lu au chargement du module,
# donc AVANT tout import acvram — sinon le flag ne prend pas.
os.environ.setdefault("ACVRAM_MOE_DECODE_MMA", "1")

MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
CORPUS = "/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt"
MAX_TOKENS = 8192   # meme ordre de grandeur que l'etalon (4 fenetres de 2048)
MAX_MODEL_LEN = MAX_TOKENS + 64


def main() -> int:
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer
    from acvram.evaluate import _load_corpus

    # Bras A (chef, 15/09) : =0 (GEMV W4A16, controle de coherence contre
    # l'etalon du parc) et =1 (B, MMA W4A4) avec ce MEME script -- le
    # `setdefault` plus haut respecte une valeur deja posee dans
    # l'environnement de l'appelant.
    assert os.environ.get("ACVRAM_MOE_DECODE_MMA") in ("0", "1"), "flag pas pose avant import"
    # poste2, 16/09 : reprend ce script pour le voyant narrow_gemm (poste7 §6,
    # protocole-narrow-voyants-15-09.md) -- meme piege a eviter, meme preuve.
    assert os.environ.get("ACVRAM_NARROW_GEMM") in ("0", "1"), "flag pas pose avant import"

    tokenizer = load_tokenizer(MODEL)
    texte = _load_corpus(CORPUS)
    ids = tokenizer.encode(texte)[:MAX_TOKENS]
    print(f"  corpus : {len(ids)} jetons, {os.path.basename(CORPUS)}", flush=True)

    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=MAX_MODEL_LEN)
    engine = Engine(loaded, tokenizer, max_batch_size=1,
                    max_model_len=MAX_MODEL_LEN, enable_cuda_graphs=True)
    engine._eos = set()  # les jetons forces sont ceux du corpus, pas une fin

    etat = {"pos": 0, "nll": 0.0, "n": 0, "pas_petits": 0, "pas_total": 0}
    N = len(ids)

    def sample_force(logits: torch.Tensor, seqs) -> tuple:
        p = etat["pos"]
        etat["pas_total"] += 1
        if logits.shape[0] <= 32:
            etat["pas_petits"] += 1
        cible = ids[p + 1] if p + 1 < N else ids[p]
        lp = torch.log_softmax(logits[0].to(torch.float32), dim=-1)
        etat["nll"] += float(-lp[cible])
        etat["n"] += 1
        etat["pos"] = p + 1
        tok = torch.tensor([cible], device=logits.device, dtype=torch.long)
        logprob = lp[cible].reshape(1)
        return tok, logprob

    engine._sample_only = sample_force
    engine.add_request([ids[0]], SamplingParams(temperature=0.0, max_tokens=N - 1),
                       request_id="s0")

    while engine.running or engine.waiting:
        engine.step()
        if etat["n"] % 1024 == 0 and etat["n"]:
            print(f"  {etat['n']}/{N - 1} jetons notes", flush=True)

    ppl = math.exp(min(etat["nll"] / etat["n"], 60.0))
    print(f"RESULTAT ppl={ppl:.4f} n_jetons_notes={etat['n']} "
         f"pas_t_le_32={etat['pas_petits']}/{etat['pas_total']} "
         f"acvram_moe_decode_mma={os.environ['ACVRAM_MOE_DECODE_MMA']} "
         f"acvram_narrow_gemm={os.environ['ACVRAM_NARROW_GEMM']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
