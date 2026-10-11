"""``raw_system_prompt`` sends the session instructions verbatim.

No channel rules, tool section, or language instruction are added, history
compaction is disabled, and unset sampling params are left to the provider.
The OpenAI client is faked, so this runs with no network and no GPU.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline.messages import GenerateResponseRequest

PROMPT = "You are being interviewed.\nAnswer however you like.  "


class _FakeCompletions:
    def __init__(self):
        self.next_result = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_result


class _FakeClient:
    def __init__(self, *a, **k):
        self.chat = SimpleNamespace(completions=_FakeCompletions())

    def with_options(self, **kwargs):
        return self


def _make_handler(**overrides):
    orig_openai = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
    try:
        return ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=dict(
                model_name="test-model",
                base_url="http://fake/v1",
                api_key="k",
                stream=False,
                disable_thinking=True,
                compact_history=True,
                enable_lang_prompt=True,
                **overrides,
            ),
        )
    finally:
        base_mod.OpenAI = orig_openai


def _last_call(handler):
    chat = Chat(10)
    chat.add_item(make_user_message("Hej, hur mar du?"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions=PROMPT)
    rc = RuntimeConfig(chat=chat, session=session)
    req = GenerateResponseRequest(runtime_config=rc, language_code="sv-auto", turn_id="t", turn_revision=0)

    list(handler.process(req))

    calls = handler.client.chat.completions.calls
    assert calls, "the backend never issued a request"
    return calls[-1]


def _system_contents(call):
    return [m.get("content") for m in call["messages"] if m.get("role") == "system"]


def test_raw_mode_sends_instructions_verbatim():
    handler = _make_handler(raw_system_prompt=True)

    assert _system_contents(_last_call(handler)) == [PROMPT]


def test_default_mode_still_wraps_instructions():
    handler = _make_handler()

    (content,) = _system_contents(_last_call(handler))
    assert content != PROMPT and PROMPT.strip() in content


def test_raw_mode_disables_compaction():
    assert _make_handler(raw_system_prompt=True).compactor is None
    assert _make_handler().compactor is not None


def test_sampling_params_unset_by_default():
    call = _last_call(_make_handler(raw_system_prompt=True))

    assert "temperature" not in call and "max_tokens" not in call


def test_sampling_params_forwarded_when_set():
    call = _last_call(_make_handler(raw_system_prompt=True, temperature=0.7, max_tokens=2048))

    assert call["temperature"] == 0.7 and call["max_tokens"] == 2048


def test_request_log_records_exact_system_message_without_key(tmp_path):
    log = tmp_path / "requests.jsonl"
    handler = _make_handler(raw_system_prompt=True, request_log=str(log))
    _last_call(handler)

    import json

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["purpose"] for r in records] == ["warmup", "response", "response_meta"]
    response = records[1]
    system = [m["content"] for m in response["input"] if m["role"] == "system"]
    assert system == [PROMPT]
    assert response["model"] == "test-model"
    assert "api_key" not in log.read_text()


def test_extra_body_is_merged_into_every_request():
    pin = '{"provider": {"order": ["anthropic"], "allow_fallbacks": false}}'
    handler = _make_handler(raw_system_prompt=True, extra_body=pin)
    call = _last_call(handler)

    assert call["extra_body"]["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
    # The reasoning-related fields built from other flags are kept alongside.
    assert "chat_template_kwargs" in call["extra_body"]


def test_request_log_records_who_answered(tmp_path):
    import json

    log = tmp_path / "requests.jsonl"
    handler = _make_handler(raw_system_prompt=True, request_log=str(log))
    handler.client.chat.completions.next_result.id = "gen-123"
    handler.client.chat.completions.next_result.model = "anthropic/claude-4.5-sonnet-20250929"
    handler.client.chat.completions.next_result.provider = "Anthropic"
    _last_call(handler)

    meta = [json.loads(line) for line in log.read_text().splitlines() if '"response_meta"' in line]
    assert meta[-1]["id"] == "gen-123"
    assert meta[-1]["model"] == "anthropic/claude-4.5-sonnet-20250929"
    assert meta[-1]["provider"] == "Anthropic"
