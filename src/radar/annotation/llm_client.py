# llm_client.py — the only module that talks to an LLM (OpenAI-compatible APIs, see config.example.toml).
# Knows no database: takes a prompt and plain dicts, returns plain dicts.

import json
import logging
import os
import re

from openai import OpenAI, OpenAIError

from radar.config import providers

logger = logging.getLogger(__name__)

# Providers come from the [llm] section of config.toml / config.example.toml.
# Each entry: base_url, key_env, default_model.
PROVIDERS, DEFAULT_PROVIDER = providers('llm')
DEFAULT_TEMPERATURE = 0.0
TIMEOUT = 600
MAX_RETRIES = 3

# just in the case of the model wrapping the JSON
CODE_FENCE = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.DOTALL)

_clients = {}


def get_client(provider):
    """Create the API client of a provider once and reuse it (keeps the HTTP connection open)."""
    if provider not in PROVIDERS:
        raise ValueError(f'unknown provider {provider!r}, known: {", ".join(PROVIDERS)}')
    if provider not in _clients:
        config = PROVIDERS[provider]
        _clients[provider] = OpenAI(base_url=config['base_url'], api_key=os.environ[config['key_env']],
                                    timeout=TIMEOUT, max_retries=MAX_RETRIES)
    return _clients[provider]


def list_models(provider=DEFAULT_PROVIDER):
    """Return the model ids a provider offers (to look up the exact name for a run)."""
    return sorted(model.id for model in get_client(provider).models.list())


def render_prompt(template, topic):
    """Insert the topic into a prompt template.

    str.replace instead of str.format: the template contains JSON braces,
    which format() would try to interpret as placeholders.
    """
    return template.replace('{THEMA}', topic)


def annotate_batch(system_prompt, items, *, provider=DEFAULT_PROVIDER, model=None,
                   temperature=DEFAULT_TEMPERATURE, thinking=None):
    """Send one batch of comments and return the answer.

    :param system_prompt: rendered prompt (see render_prompt)
    :param items: list of dicts as the prompt describes them, e.g. {'id': 17, 'text': '...'}
    :param provider: key of PROVIDERS
    :param model: model id at that provider; None means the provider's default model
    :param thinking: True/False switches the reasoning phase of Qwen models on/off
        (chat_template_kwargs.enable_thinking, tested with IDT 18.11.); None sends nothing,
        the provider's default applies
    :return: dict with
        'model':   model name as reported by the API (can differ from the requested one)
        'raw':     the answer as a dict shaped like a non-streamed response, including the
                   reasoning text and token usage (for batch.raw_response), None on error
        'content': the answer text, None on error
        'parsed':  the answer as JSON (dict), None if it is not valid JSON
        'error':   None, or a short description of what went wrong
    Never raises for API or parsing problems: a failed batch is recorded and resent later.
    """
    model = model or PROVIDERS[provider]['default_model']
    user_message = json.dumps(items, ensure_ascii=False)
    extra_body = {} if thinking is None else {'chat_template_kwargs': {'enable_thinking': thinking}}
    try:
        stream = get_client(provider).chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            stream=True,
            stream_options={'include_usage': True},
            extra_body=extra_body,
        )
        raw = collect_stream(stream)
    except OpenAIError as error:
        logger.warning('request failed: %s', error)
        return {'model': None, 'raw': None, 'content': None, 'parsed': None,
                'error': f'{type(error).__name__}: {error}'}

    content = raw['choices'][0]['message']['content']
    result = {'model': raw['model'], 'raw': raw, 'content': content, 'parsed': None, 'error': None}
    try:
        result['parsed'] = parse_json(content)
    except ValueError as error:
        logger.warning('answer is not valid JSON: %s', error)
        result['error'] = f'invalid JSON: {error}'
    return result


def collect_stream(stream):
    """Join a streamed answer into one dict shaped like a non-streamed response.

    Streaming keeps the connection busy while the model is still reasoning; the IDT proxy
    closes connections that stay silent for about 50 s (18.11.).
    """
    model = None
    content, reasoning = [], []
    finish_reason = None
    usage = None
    for chunk in stream:
        model = model or chunk.model
        if chunk.usage is not None:  # last chunk, only with stream_options include_usage
            usage = chunk.usage.model_dump(mode='json')
        for choice in chunk.choices:
            delta = choice.delta
            if delta.content:
                content.append(delta.content)
            thought = getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None)
            if thought:
                reasoning.append(thought)
            finish_reason = choice.finish_reason or finish_reason
    message = {'role': 'assistant', 'content': ''.join(content), 'reasoning': ''.join(reasoning) or None}
    return {'model': model, 'stream': True, 'usage': usage,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}]}


def parse_json(content):
    """Parse the model's answer as a JSON object, tolerating a Markdown code fence around it."""
    text = content.strip()
    match = CODE_FENCE.match(text)
    if match:
        text = match.group(1)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f'expected a JSON object, got {type(data).__name__}')
    return data
