import os
from types import SimpleNamespace

import pytest
from openai import APIConnectionError

from radar.annotation import llm_client
from radar.annotation.llm_client import (PROVIDERS, annotate_batch, get_client, list_models,
                                         parse_json, render_prompt)

# Live tests call the real APIs; run them with:  $env:RADAR_LIVE_TESTS = "1"; uv run pytest tests/test_llm_client.py
LIVE = os.getenv('RADAR_LIVE_TESTS') == '1'
live = pytest.mark.skipif(not LIVE, reason='set RADAR_LIVE_TESTS=1 to call the real APIs')


class FakeClient:
    """Stands in for the OpenAI client: returns a fixed answer or raises a fixed error."""

    def __init__(self, content=None, error=None, model='fake-model'):
        self.content, self.error, self.model = content, error, model
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        response = SimpleNamespace(model=self.model, choices=[SimpleNamespace(message=message)])
        response.model_dump = lambda mode: {'model': self.model, 'content': self.content}
        return response


@pytest.fixture
def fake(monkeypatch):
    """Install a fake client for every provider; returns a function to configure it."""
    def install(**kwargs):
        client = FakeClient(**kwargs)
        monkeypatch.setattr(llm_client, '_clients', {name: client for name in PROVIDERS})
        return client
    return install


# --- configuration ----------------------------------------------------------------------

@pytest.mark.parametrize('provider', PROVIDERS)
def test_provider_config_is_complete(provider):
    config = PROVIDERS[provider]
    assert config['base_url'].startswith('https://')
    assert config['key_env'] and config['default_model']


@pytest.mark.parametrize('provider', PROVIDERS)
def test_provider_key_is_set(provider):
    assert os.getenv(PROVIDERS[provider]['key_env']), f'{PROVIDERS[provider]["key_env"]} missing in .env'


@pytest.mark.parametrize('provider', PROVIDERS)
def test_get_client_uses_provider_url(provider, monkeypatch):
    monkeypatch.setattr(llm_client, '_clients', {})
    monkeypatch.setenv(PROVIDERS[provider]['key_env'], 'test-key')
    client = get_client(provider)
    assert str(client.base_url).rstrip('/') == PROVIDERS[provider]['base_url']
    assert get_client(provider) is client  # cached


def test_get_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match='unknown provider'):
        get_client('nope')


# --- prompt and parsing -----------------------------------------------------------------

def test_render_prompt_keeps_json_braces():
    assert render_prompt('Thema „{THEMA}“ {"id": 1}', 'E-Auto') == 'Thema „E-Auto“ {"id": 1}'


@pytest.mark.parametrize('content', [
    '{"ergebnisse": []}',
    '```json\n{"ergebnisse": []}\n```',
    '  ```\n{"ergebnisse": []}\n```  ',
])
def test_parse_json_accepts_plain_and_fenced(content):
    assert parse_json(content) == {'ergebnisse': []}


@pytest.mark.parametrize('content', ['', 'Hier ist das Ergebnis: {', '[1, 2]'])
def test_parse_json_rejects_invalid(content):
    with pytest.raises(ValueError):
        parse_json(content)


# --- annotate_batch with a fake client --------------------------------------------------

@pytest.mark.parametrize('provider', PROVIDERS)
def test_annotate_batch_sends_to_provider_default_model(provider, fake):
    client = fake(content='{"ergebnisse": []}')
    result = annotate_batch('system', [{'id': 1, 'text': 'Reichweite ätzend'}], provider=provider)
    call = client.calls[0]
    assert call['model'] == PROVIDERS[provider]['default_model']
    assert call['messages'][0] == {'role': 'system', 'content': 'system'}
    assert 'ätzend' in call['messages'][1]['content']  # umlauts not escaped
    assert result['parsed'] == {'ergebnisse': []} and result['error'] is None


def test_annotate_batch_explicit_model(fake):
    client = fake(content='{"ergebnisse": []}')
    annotate_batch('system', [], provider='idt', model='other-model')
    assert client.calls[0]['model'] == 'other-model'


def test_annotate_batch_connection_error_does_not_raise(fake):
    fake(error=APIConnectionError(request=None))
    result = annotate_batch('system', [{'id': 1, 'text': 'x'}])
    assert result['raw'] is None and result['parsed'] is None
    assert result['error'].startswith('APIConnectionError')


def test_annotate_batch_invalid_json_keeps_raw(fake):
    fake(content='kein JSON')
    result = annotate_batch('system', [{'id': 1, 'text': 'x'}])
    assert result['raw'] is not None and result['parsed'] is None
    assert result['error'].startswith('invalid JSON')


# --- live: the real APIs ----------------------------------------------------------------

@live
@pytest.mark.parametrize('provider', PROVIDERS)
def test_live_default_model_is_offered(provider):
    assert PROVIDERS[provider]['default_model'] in list_models(provider)


@live
@pytest.mark.parametrize('provider', PROVIDERS)
def test_live_annotate_batch(provider):
    result = annotate_batch('Antworte ausschließlich mit {"ergebnisse": []}.',
                            [{'id': 1, 'text': 'Test'}], provider=provider)
    assert result['error'] is None, result['error']
    assert result['model']  # reported name may differ from the requested alias (KI:connect)
    assert isinstance(result['parsed'], dict)
