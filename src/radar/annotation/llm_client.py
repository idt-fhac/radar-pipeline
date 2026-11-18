# llm_client.py — the only module that talks to an LLM (OpenAI-compatible APIs: KI:connect, IDT).
# Knows no database: takes a prompt and plain dicts, returns plain dicts.

import json
import logging
import os
import re

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

PROVIDERS = {
    'kiconnect': {'base_url': 'https://chat.kiconnect.nrw/api/v1', 'key_env': 'KI_API_KEY',
                  'default_model': 'qwen-qwen3-8-27b'},
    'idt':       {'base_url': 'https://chat.idt.fh-aachen.de/api/v1', 'key_env': 'IDT_API_KEY',
                  'default_model': 'qwen3-8-flash-next-int4'},
}
DEFAULT_PROVIDER = 'kiconnect'
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
        'raw':     the complete API response as a dict (for batch.raw_response), None on error
        'content': the answer text, None on error
        'parsed':  the answer as JSON (dict), None if it is not valid JSON
        'error':   None, or a short description of what went wrong
    Never raises for API or parsing problems: a failed batch is recorded and resent later.
    """
    model = model or PROVIDERS[provider]['default_model']
    user_message = json.dumps(items, ensure_ascii=False)
    extra_body = {} if thinking is None else {'chat_template_kwargs': {'enable_thinking': thinking}}
    try:
        response = get_client(provider).chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            extra_body=extra_body,
        )
    except OpenAIError as error:
        logger.warning('request failed: %s', error)
        return {'model': None, 'raw': None, 'content': None, 'parsed': None,
                'error': f'{type(error).__name__}: {error}'}

    content = response.choices[0].message.content or ''
    result = {'model': response.model, 'raw': response.model_dump(mode='json'),
              'content': content, 'parsed': None, 'error': None}
    try:
        result['parsed'] = parse_json(content)
    except ValueError as error:
        logger.warning('answer is not valid JSON: %s', error)
        result['error'] = f'invalid JSON: {error}'
    return result


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
