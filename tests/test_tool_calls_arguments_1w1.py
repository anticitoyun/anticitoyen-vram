"""1w1 (poste6, 30/09, hypothèse poste1 sf2 28/09) : /v1/chat/completions reçoit ``tool_calls[].function.arguments`` en
CHAÎNE JSON (contrat OpenAI) et `messages_pour_gabarit` la passait telle quelle au gabarit ; le VRAI gabarit de
Qwen3-Coder-30B-A3B-Instruct (tests/gabarits/, copié du dossier HF) fait ``tool_call.arguments|items`` → TypeError →
repli ChatML SANS outils : kimi et tout client OpenAI qui rejoue un appel d'outil perdaient le second tour.
Correctif : `arguments_en_objet` décode avant le rendu (comme /v1/messages garde l'objet), repli nommé
``{"arguments": <chaîne>}`` sur un JSON invalide, requête non modifiée, objet déjà décodé inchangé.
Cassure : rendre `arguments_en_objet` = identité → test_gabarit_qwen3_coder_rend_les_parametres rouge (chatml-repli)."""
import json
import pathlib

import pytest

from acvram.server.chat import Tokenizer, arguments_en_objet, messages_pour_gabarit
from acvram.server.protocol import ChatMessage

GABARIT = (pathlib.Path(__file__).parent / "gabarits" / "qwen3-coder-30b-a3b-instruct.jinja").read_text(encoding="utf-8")


def _tok():
    return Tokenizer(backend=None, config={"eos_token": "<|im_end|>"}, template=GABARIT, template_source="test")


def _tour(arguments):
    return [ChatMessage(role="user", content="lis a.py"),
            ChatMessage(role="assistant", content="", tool_calls=[{"id": "call_1", "type": "function",
                        "function": {"name": "lire", "arguments": arguments}}]),
            ChatMessage(role="tool", content="print(1)", tool_call_id="call_1")]


def test_gabarit_qwen3_coder_rend_les_parametres():
    tok = _tok()
    texte = tok.apply_chat_template(messages_pour_gabarit(_tour(json.dumps({"chemin": "a.py", "n": 3, "opts": {"a": 1}}))), True)
    assert tok.gabarit_effectif == "jinja", texte[:300]
    assert "<function=lire>" in texte and "<parameter=chemin>\na.py\n</parameter>" in texte
    assert "<parameter=n>\n3\n</parameter>" in texte and '<parameter=opts>\n{"a": 1}\n</parameter>' in texte
    assert "<tool_response>\nprint(1)\n</tool_response>" in texte


def test_sans_correctif_le_gabarit_tombait_en_chatml(capsys):
    """Le témoin de la cause : la chaîne brute fait échouer `arguments|items` → repli ChatML, sans outils."""
    tok = _tok()
    bruts = [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                                                                  "function": {"name": "lire", "arguments": '{"chemin": "a.py"}'}}]}]
    texte = tok.apply_chat_template(bruts, True)
    assert tok.gabarit_effectif == "chatml-repli" and "<parameter=" not in texte


def test_json_invalide_repli_nomme_sans_exception():
    tok = _tok()
    texte = tok.apply_chat_template(messages_pour_gabarit(_tour("{pas du json")), True)
    assert tok.gabarit_effectif == "jinja" and "<parameter=arguments>\n{pas du json\n</parameter>" in texte


def test_objet_deja_decode_et_requete_intacte():
    msgs = _tour('{"chemin": "a.py"}')
    avant = json.dumps(msgs[1].tool_calls)
    rendu = messages_pour_gabarit(msgs)
    assert rendu[1]["tool_calls"][0]["function"]["arguments"] == {"chemin": "a.py"}
    assert json.dumps(msgs[1].tool_calls) == avant, "la requête a été modifiée en place"
    deja = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": {"k": 1}}}]
    assert arguments_en_objet(deja) == deja
    plat = [{"name": "f", "arguments": '{"k": 1}'}]
    assert arguments_en_objet(plat) == [{"name": "f", "arguments": {"k": 1}}]
    assert arguments_en_objet([{"function": {"name": "f", "arguments": "[1, 2]"}}])[0]["function"]["arguments"] == {"arguments": [1, 2]}
