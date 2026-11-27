import pytest
from sqlalchemy import func, select

from radar.annotation import prompts
from radar.annotation.prompts import file_hash, get_or_create_prompt_version, text_hash
from radar.db.models.annotation import PromptVersion


@pytest.fixture
def prompt_folder(tmp_path, monkeypatch):
    """prompts/ in a temporary folder; returns write(name, text)."""
    folder = tmp_path / 'prompts'
    folder.mkdir()
    monkeypatch.setattr(prompts, 'ROOT', tmp_path)

    def write(name, text):
        (folder / f'{name}.txt').write_text(text, encoding='utf-8')
    return write


def count(session):
    return session.scalar(select(func.count()).select_from(PromptVersion))


def test_first_use_stores_name_text_and_hash(database, prompt_folder):
    prompt_folder('extraction_v1', 'Thema „{THEMA}“: Finde die Kriterien.')
    with database() as session:
        get_or_create_prompt_version(session, 'extraction_v1')
        session.commit()
    with database() as session:
        stored = session.scalar(select(PromptVersion))
        assert stored.name == 'extraction_v1'
        assert stored.text == 'Thema „{THEMA}“: Finde die Kriterien.'
        assert stored.hash == text_hash(stored.text)


def test_second_use_returns_the_stored_version(database, prompt_folder):
    prompt_folder('extraction_v1', 'Text')
    with database() as session:
        first = get_or_create_prompt_version(session, 'extraction_v1')
        session.commit()
        again = get_or_create_prompt_version(session, 'extraction_v1')
        assert again.id == first.id and count(session) == 1


def test_an_edited_prompt_file_is_refused(database, prompt_folder):
    prompt_folder('extraction_v1', 'Text')
    with database() as session:
        get_or_create_prompt_version(session, 'extraction_v1')
        session.commit()
        prompt_folder('extraction_v1', 'Text, nachträglich geändert')
        with pytest.raises(ValueError, match='was edited after it was first used'):
            get_or_create_prompt_version(session, 'extraction_v1')
        assert session.scalar(select(PromptVersion.text)) == 'Text'


def test_the_same_text_under_another_name_is_refused(database, prompt_folder):
    prompt_folder('extraction_v1', 'Text')
    prompt_folder('kopie', 'Text')
    with database() as session:
        get_or_create_prompt_version(session, 'extraction_v1')
        session.commit()
        with pytest.raises(ValueError, match="same text as the stored prompt 'extraction_v1'"):
            get_or_create_prompt_version(session, 'kopie')
        assert count(session) == 1


def test_a_missing_prompt_file_is_an_error(database, prompt_folder):
    with database() as session:
        with pytest.raises(FileNotFoundError):
            get_or_create_prompt_version(session, 'gibt_es_nicht')


def test_hash_does_not_depend_on_the_line_endings(tmp_path):
    # read as text: Windows line endings become '\n', so the same prompt has the same hash everywhere
    (tmp_path / 'unix.txt').write_bytes('Zeile 1\nZeile 2 ä'.encode('utf-8'))
    (tmp_path / 'windows.txt').write_bytes('Zeile 1\r\nZeile 2 ä'.encode('utf-8'))
    assert file_hash(tmp_path / 'unix.txt') == file_hash(tmp_path / 'windows.txt') == text_hash('Zeile 1\nZeile 2 ä')


def test_hash_tells_texts_apart():
    assert text_hash('Text') != text_hash('Text ')
    assert len(text_hash('Text')) == 64  # SHA-256 as hex
