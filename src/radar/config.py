# config.py — reads config.toml (local, not in the repository) or, if missing, config.example.toml

import tomllib

from radar import ROOT

CONFIG_FILE = ROOT / 'config.toml'
if not CONFIG_FILE.exists():
    CONFIG_FILE = ROOT / 'config.example.toml'

with open(CONFIG_FILE, 'rb') as f:  # tomllib only reads binary files
    CONFIG = tomllib.load(f)


def providers(section):
    """Return (providers, default) of a section such as 'llm' or 'embedding'; checks the default exists."""
    if section not in CONFIG:
        raise ValueError(f'section [{section}] is missing in {CONFIG_FILE.name}')
    entries = CONFIG[section].get('providers', {})
    default = CONFIG[section].get('default')
    if default not in entries:
        raise ValueError(f'default {section} provider {default!r} is not defined in {CONFIG_FILE.name}')
    return entries, default
