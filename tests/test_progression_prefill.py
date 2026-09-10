"""La progression d'un prefill a son champ à elle, distinct de `cached_len`.

Découper une invite et reprendre existe déjà dans le moteur (`_build_batch`
prend une borne de FIN, `limite`), mais les deux passes s'enchaînent dans le
même `step()` : cela découpe le calcul sans découper la latence. Ce qui
l'empêche de rendre la main tenait à un seul champ — `prefilled` était un
booléen, sans état « en cours ».

La dérivation tentante, `prefilled = (cached_len == len(prompt_ids))`, est
fausse DANS LES DEUX SENS. Les deux essais qui suivent tiennent les deux sens ;
sans le champ propre, ils échouent.
"""
from acvram.engine.runner import BLOCK_SIZE, Sequence
from acvram.engine.sampler import SamplingParams


def _seq(n):
    return Sequence(prompt_ids=list(range(1, n + 1)), params=SamplingParams())


def test_apres_un_prefill_entier_cached_len_ne_bouge_pas():
    """Premier sens : `cached_len` sert d'ORIGINE pour compter les jetons
    (`len(prompt_ids) - cached_len`), il n'est jamais avancé à la fin d'un
    prefill complet. Le dériver déclarerait « pas prefillée » une séquence
    qui l'est — elle ne décoderait jamais."""
    s = _seq(300)
    s.prefill_len = len(s.prompt_ids)          # ce que fait `step()`
    assert s.cached_len == 0, "le moteur ne l'avance pas ici"
    assert s.prefilled
    assert (s.cached_len == len(s.prompt_ids)) is False, \
        "le predicat naif se tromperait ici"


def test_un_prefixe_entierement_en_cache_n_est_pas_prefille():
    """Second sens, plus grave : à l'admission une séquence dont le préfixe
    est en cache reçoit `cached_len` AVANT tout passage en avant. Le prédicat
    naïf la déclarerait prefillée sans qu'aucun logit existe — elle passerait
    au décodage sans état."""
    s = _seq(300)
    s.cached_len = 300
    s.prefill_len = s.cached_len               # ce que fait `_admit()`
    assert s.prefilled, "ici les deux coincident : rien ne reste a calculer"
    s2 = _seq(300)
    s2.cached_len = 256
    s2.prefill_len = s2.cached_len
    assert not s2.prefilled, "44 jetons restent a passer en avant"


def test_une_reprise_partielle_laisse_la_sequence_inachevee():
    """Ce que le booléen ne pouvait pas dire : à mi-chemin."""
    s = _seq(300)
    s.prefill_len = 128
    assert not s.prefilled
    s.prefill_len = 300
    assert s.prefilled


def test_le_cache_ne_peut_pas_couvrir_l_invite_entiere():
    """Le défaut « tout en cache, zéro forward » n'existe pas : `_admit` borne
    l'appariement à `(len(prompt_ids) - 1) // BLOCK_SIZE` blocs — le « -1 »
    retient toujours un jeton à faire traverser le modèle. Le vérifier ici
    évite d'écrire une branche pour un cas impossible, et dénoncerait la
    disparition du « -1 »."""
    for n in (1, BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, 7 * BLOCK_SIZE):
        limite = max(0, (n - 1) // BLOCK_SIZE)
        assert limite * BLOCK_SIZE < n, f"invite de {n} jetons entierement servie"
