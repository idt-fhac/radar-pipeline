import hashlib
from sqlalchemy import select
from radar.db.models.annotation import PromptVersion
from radar import ROOT

def file_hash(path):
    text = path.read_text(encoding='utf-8')
    text = text.encode('utf-8')
    return hashlib.sha256(text).hexdigest()

def text_hash(text):
    text = text.encode('utf-8')
    return hashlib.sha256(text).hexdigest()

def get_or_create_prompt_version(session, name):
    """Return the stored prompt version for prompts/<name>.txt, creating it on first use.

    Prompts are never edited in place: once a name is stored, its text is fixed.
    Raises if the file changed since (same name, different hash) or if the same
    text is already stored under another name. The caller commits.
    """
    file = ROOT / 'prompts' / f'{name}.txt'
    text = file.read_text(encoding='utf-8')  # FileNotFoundError if the prompt does not exist
    fingerprint = text_hash(text)

    prompt_version = session.scalar(select(PromptVersion).where(PromptVersion.name == name))
    if prompt_version is not None:
        if prompt_version.hash != fingerprint:
            raise ValueError(
                f'prompt {name!r} was edited after it was first used; '
                f'restore the file or save the change under a new name (e.g. a new version)')
        return prompt_version

    same_text = session.scalar(select(PromptVersion).where(PromptVersion.hash == fingerprint))
    if same_text is not None:
        raise ValueError(
            f'prompt {name!r} has the same text as the stored prompt {same_text.name!r}; '
            f'use that name or change the text')

    prompt_version = PromptVersion(name=name, text=text, hash=fingerprint, annotation_runs=[])
    session.add(prompt_version)
    return prompt_version
