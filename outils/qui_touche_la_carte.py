"""Quels tests allouent REELLEMENT sur la carte ? Mesure, au lieu de grep.

    pytest tests/ -p outils.qui_touche_la_carte

Trois inventaires du meme parc ont donne trois nombres ce soir — 26, 21, 19 —
parce que « toucher la carte » n'avait pas la meme definition. Un `grep` ne
peut pas trancher : il manque le fichier qui alloue via un module importe, et
il compte celui qui ecrit « cuda » dans un commentaire.

On mesure donc la VRAM allouee avant et apres chaque test. Ce que le greffon
rend n'est pas une opinion sur le code, c'est ce qui s'est passe.

Et il rend aussi le RESIDU : ce qu'un test laisse derriere lui. Un test qui
alloue et ne rend pas est le suspect naturel quand un voisin tombe apres lui.
"""
import pytest, torch

_alloc = {}
_residu = []

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not torch.cuda.is_available():
        yield; return
    torch.cuda.synchronize()
    avant = torch.cuda.memory_allocated()
    pic0 = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    yield
    torch.cuda.synchronize()
    apres = torch.cuda.memory_allocated()
    pic = torch.cuda.max_memory_allocated()
    _alloc[item.nodeid] = (pic, apres - avant)
    if apres > avant:
        _residu.append((item.nodeid, apres - avant))

_collectes = []

def pytest_collection_modifyitems(session, config, items):
    _collectes.extend(i.nodeid for i in items)

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _alloc:
        return
    t = terminalreporter
    touche = {k: v for k, v in _alloc.items() if v[0] > 0}
    t.write_sep("=", "qui touche la carte (mesure, pas suppose)")
    # COMBIEN AI-JE VU ? Un test qui saute, echoue a l import ou n est pas
    # collecte ne traverse pas le hookwrapper — et son absence du tableau se
    # lit comme « ne touche pas la carte ». Un instrument doit dire ce qu il
    # n a pas vu, sinon son silence passe pour une observation.
    manques = [k for k in _collectes if k not in _alloc]
    t.write_line(f"{len(_alloc)} tests traverses sur {len(_collectes)} collectes")
    if manques:
        t.write_line(f"NON VUS par ce greffon ({len(manques)}) — leur absence "
                     f"du tableau ne veut PAS dire qu'ils ne touchent pas la carte :")
        for k in manques[:10]:
            t.write_line(f"    {k}")
    t.write_line(f"{len(touche)} tests sur {len(_alloc)} ont alloue sur la carte")
    for k, (pic, _) in sorted(touche.items(), key=lambda x: -x[1][0])[:15]:
        t.write_line(f"  {pic/(1<<20):9.1f} Mio de pic   {k}")
    if _residu:
        t.write_sep("-", "RESIDU : ce qui n'a pas ete rendu")
        t.write_line("un test qui alloue sans rendre est le suspect naturel "
                     "quand un voisin tombe apres lui")
        for k, d in sorted(_residu, key=lambda x: -x[1])[:15]:
            t.write_line(f"  {d/(1<<20):9.1f} Mio laisses   {k}")
    else:
        t.write_line("aucun residu : chaque test rend ce qu'il a pris")
