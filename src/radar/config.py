# config.py — settings of the pipeline: config.example.toml (in the repository) holds every
# setting with its default; config.toml (local, not in the repository) overrides single values.

import tomllib

from radar import ROOT

EXAMPLE_FILE = ROOT / 'config.example.toml'
LOCAL_FILE = ROOT / 'config.toml'


def load(path):
    with open(path, 'rb') as f:  # tomllib only reads binary files
        return tomllib.load(f)


def merge(base, override):
    """Tables are merged key by key, everything else is replaced by the override."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


CONFIG = load(EXAMPLE_FILE)
if LOCAL_FILE.exists():
    CONFIG = merge(CONFIG, load(LOCAL_FILE))
SOURCE = 'config.toml / config.example.toml' if LOCAL_FILE.exists() else 'config.example.toml'


def setting(section, key):
    """One value, e.g. setting('annotation', 'batch_size'); a missing key is an error, not a silent default."""
    try:
        return CONFIG[section][key]
    except KeyError:
        raise KeyError(f'setting [{section}] {key} is missing in {SOURCE}') from None


def providers(section):
    """Return (providers, default) of a section such as 'llm' or 'embedding'; checks the default exists."""
    entries = CONFIG.get(section, {}).get('providers', {})
    default = CONFIG.get(section, {}).get('default')
    if default not in entries:
        raise ValueError(f'default {section} provider {default!r} is not defined in {SOURCE}')
    return entries, default
