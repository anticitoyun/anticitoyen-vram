"""Pièce e50.2 (poste2, ordre chef, 02/10, CPU, sans carte) : diagnostic du repli ChatML
silencieux sur les 8 alias Mistral 24B (ariel-alloy-v1-24b, cydonia-24b, cydonia-magnum-
diamond-24b, dolphin-mistral-glm47-24b-venice), même cause que le gpt-oss d'poste1
(558999068, pas encore fusionné sur main) : `acvram/server/chat.py:_render_jinja` construit
l'`Environment` Jinja PARTAGÉE par tous les gabarits sans la globale `strftime_now` que
transformers fournit (`chat_template_utils`) — le gabarit Mistral l'appelle pour dater le
message système, lève `jinja2.exceptions.UndefinedError`, attrapée par le `except Exception`
générique d'`apply_chat_template` (chat.py:128) → repli ChatML SILENCIEUX (un seul
avertissement stderr, `Tokenizer._repli_averti`, jamais par alias). Gabarit réduit ci-dessous :
forme réelle des gabarits Mistral-v7/v11 (bloc système daté par `strftime_now`, cf.
tokenizer_config.json des dépôts HF `mistralai/*-Instruct-2506`), pas une copie du fichier
déployé (hors dépôt, carte non prise pour aller le lire)."""
import pathlib
import sys

ICI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ICI))

from acvram.server.chat import Tokenizer, messages_pour_gabarit  # noqa: E402
from acvram.server.protocol import ChatMessage  # noqa: E402
from jinja2 import Environment  # noqa: E402
from jinja2.exceptions import TemplateError  # noqa: E402

GABARIT_MISTRAL_24B = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content'] %}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}"
    "{%- set loop_messages = messages %}"
    "{%- endif %}"
    "{{- '[SYSTEM_PROMPT]' }}"
    "{%- if system_message is defined %}{{- system_message }}{%- endif %}"
    "{{- '\\nToday Date: ' + strftime_now('%Y-%m-%d') }}"
    "{{- '[/SYSTEM_PROMPT]' }}"
    "{%- for message in loop_messages %}"
    "{%- if message['role'] == 'user' %}{{- '[INST]' + message['content'] + '[/INST]' }}"
    "{%- elif message['role'] == 'assistant' %}{{- message['content'] }}"
    "{%- endif %}"
    "{%- endfor %}"
)

TOUR = [ChatMessage(role="system", content="Tu es utile."),
        ChatMessage(role="user", content="bonjour")]


def _raise_exception(msg):
    raise TemplateError(msg)


def test_sans_strftime_now_le_gabarit_mistral_24b_bascule_en_chatml_repli_silencieux():
    """Mécanisme exact de la faute (AVANT 558999068 / avant le correctif ci-contre) : une
    Environment sans `strftime_now`, posée directement (comme le ferait la construction
    paresseuse de `_render_jinja` sans la ligne du correctif) — `apply_chat_template` avale
    l'`UndefinedError` et bascule en ChatML sans le dire à l'appelant."""
    tok = Tokenizer(None, {}, GABARIT_MISTRAL_24B, "test-sans-correctif")
    env_casse = Environment(trim_blocks=True, lstrip_blocks=True)
    env_casse.globals["raise_exception"] = _raise_exception
    # PAS de strftime_now : reproduit l'état du code avant correctif
    tok._env = env_casse
    tok._templates = {}

    texte = tok.apply_chat_template(messages_pour_gabarit(TOUR), True)
    assert tok.gabarit_effectif == "chatml-repli"
    assert "Today Date" not in texte  # invite ChatML générique, pas le gabarit du modèle


def test_avec_le_correctif_le_gabarit_mistral_24b_se_rend_en_jinja():
    """Chemin réel du serveur (construction paresseuse de `_render_jinja`, correctif en
    place) : casse si la ligne `env.globals["strftime_now"] = ...` disparaît de
    `acvram/server/chat.py` — `apply_chat_template` retomberait alors en `chatml-repli` et
    cette assertion échouerait."""
    tok = Tokenizer(None, {}, GABARIT_MISTRAL_24B, "test-avec-correctif")
    texte = tok.apply_chat_template(messages_pour_gabarit(TOUR), True)
    assert tok.gabarit_effectif == "jinja", texte
    assert "Today Date: " in texte
    assert "[INST]bonjour[/INST]" in texte
