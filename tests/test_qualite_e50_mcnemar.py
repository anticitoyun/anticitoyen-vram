"""e50.3 § 4 (poste4, 01/10) : McNemar item par item (`outils/qualite-e50-mcnemar.py`) -- pure,
sans carte, sans lm-eval. Le test qui compte ici (`test_detecte_une_derive_masquee_par_un_S_identique`)
CASSE si on revient à la garde de cohérence simple (écart de S) de l'ancienne
`campagne-qualite-e50.py` : construit un cas où les deux passes ont EXACTEMENT le même S (donc
la garde simple dirait « témoin TENU », écart = 0) alors que la moitié des items ont changé de
verdict dans chaque sens -- McNemar doit le voir, un delta de S ne le peut jamais par
construction (les deux sens se compensent dans la moyenne)."""
import importlib.util
import os

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("qe50_mcnemar", os.path.join(ICI, "outils", "qualite-e50-mcnemar.py"))
mcn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcn)


def test_deux_passes_identiques_p_vaut_1():
    a = {i: (i % 2 == 0) for i in range(30)}
    b = dict(a)
    disc_b, disc_c, p = mcn.mcnemar(a, b)
    assert (disc_b, disc_c) == (0, 0)
    assert p == 1.0


def test_detecte_une_derive_masquee_par_un_S_identique():
    """30 items, passe A : 15 justes (0-14), 15 faux (15-29) -> S_a = 0,5.
    Passe B : les 15 premiers basculent faux, les 15 derniers basculent justes -> S_b = 0,5
    aussi (même moyenne). Une garde « écart de S > seuil » dirait TENU (écart = 0) : FAUSSE
    sécurité -- l'instrument a pourtant divergé sur LES 30 items. McNemar le voit : b=15
    (juste->faux), c=15 (faux->juste), n=30, p extrêmement petit (quasi tout l'effectif a
    changé de sens de façon parfaitement symétrique -- improbable sous H0 autant que ce soit)."""
    a = {i: (i < 15) for i in range(30)}       # 15 justes, 15 faux
    b = {i: not (i < 15) for i in range(30)}   # exactement l'inverse : même S, tout a changé
    s_a = sum(a.values()) / len(a)
    s_b = sum(b.values()) / len(b)
    assert s_a == s_b == 0.5   # la garde simple (écart de S) ne verrait RIEN ici

    disc_b, disc_c, p = mcn.mcnemar(a, b)
    assert disc_b == 15 and disc_c == 15
    # p exact pour b=c=15, n=30 (binomial symétrique) : proche de 1, PAS proche de 0 --
    # contre-intuitif mais correct : McNemar teste une dérive DIRECTIONNELLE (b != c), pas le
    # volume de discordances. Le vrai signal de dérive ici est le VOLUME n=b+c=30 sur 30 items
    # (100 % de discordance), que mcnemar_global rapporte explicitement à part (voir plus bas) --
    # c'est ce nombre, pas p, que la garde doit surveiller en plus de p.
    assert p > 0.5
    assert disc_b + disc_c == len(a)   # 100 % des items ont changé : jamais vu avec la garde simple


def test_derive_directionnelle_p_petit():
    """Cas plus réaliste d'une vraie dérive (pas symétrique) : 12 items basculent juste->faux,
    1 seul faux->juste -- un déplacement net, p doit être petit (< 0,05)."""
    a = {i: True for i in range(20)}
    b = dict(a)
    for i in range(12):
        b[i] = False
    b[15] = False
    a[15] = False
    b[15] = True
    disc_b, disc_c, p = mcn.mcnemar(a, b)
    assert disc_b == 12 and disc_c == 1
    assert p < 0.05


def test_mcnemar_global_fusionne_les_taches_sans_collision_d_id():
    a = {"mmlu_e50_hsm": {0: True, 1: False}, "gsm8k_e50": {0: True}}
    b = {"mmlu_e50_hsm": {0: True, 1: False}, "gsm8k_e50": {0: False}}
    disc_b, disc_c, p = mcn.mcnemar_global(a, b)
    assert (disc_b, disc_c) == (1, 0)   # seul gsm8k_e50:0 diffère ; les deux mmlu_e50_hsm:0 ne collisionnent pas


def test_ne_compare_que_les_doc_id_communs():
    a = {0: True, 1: False, 99: True}   # 99 absent de b (ex. lm-eval a sauté un item)
    b = {0: True, 1: True}
    disc_b, disc_c, p = mcn.mcnemar(a, b)
    assert (disc_b, disc_c) == (0, 1)   # seul l'item 1 (commun) diffère ; 99 ignoré


def test_temoin_tenu_rejette_la_derive_symetrique_que_la_garde_simple_laisserait_passer():
    """LE test qui casse si campagne-qualite-e50.py revient à l'ancienne garde (écart de S >
    0,05) : construit un témoin dont les deux passes ont S IDENTIQUE (garde simple -> TENU,
    écart = 0) mais 100 % de discordance item par item (garde McNemar -> FAUX, via le taux de
    discordance, méthode §4/E1 « réponses identiques ≥ 90 % »). `temoin_tenu` DOIT rendre
    tenu=False ici ; s'il rendait True, la pièce serait revenue, en pratique, au même aveuglement
    que la garde simple qu'elle remplace."""
    a_mmlu = {i: (i < 15) for i in range(30)}
    b_mmlu = {i: not (i < 15) for i in range(30)}
    assert sum(a_mmlu.values()) == sum(b_mmlu.values())   # même S sur cette tâche

    tenu, disc_b, disc_c, p, taux = mcn.temoin_tenu({"mmlu_e50_hsm": a_mmlu}, {"mmlu_e50_hsm": b_mmlu})
    assert tenu is False, (
        f"régression : une dérive symétrique à 100 % de discordance (b={disc_b}, c={disc_c}, "
        f"p={p:.3f}, taux={taux:.0%}) est passée TENUE -- la garde McNemar doit la rejeter via "
        "le taux de discordance (E1 : réponses identiques >= 90 %), pas seulement via p"
    )
    assert taux == 1.0   # 100 % des items ont changé de verdict, bien au-dessus du seuil de 10 %
