import shutil
import subprocess

import pytest

from radar import provenance
from radar.provenance import CODE_PATHS, get_code_version


# --- with a faked git ------------------------------------------------------------------------

def fake_git(monkeypatch, changes=''):
    """Replace provenance.git: HEAD is 'abc123', `git status` reports the given changes."""
    calls = []

    def git(*args):
        calls.append(args)
        return 'abc123' if args[0] == 'rev-parse' else changes
    monkeypatch.setattr(provenance, 'git', git)
    return calls


def test_clean_tree_gives_the_commit(monkeypatch):
    fake_git(monkeypatch)
    assert get_code_version() == 'abc123'
    assert get_code_version(allow_dirty=True) == 'abc123'


def test_dirty_tree_is_refused_and_names_the_changes(monkeypatch):
    fake_git(monkeypatch, changes=' M src/radar/cli.py')
    with pytest.raises(RuntimeError, match='uncommitted changes') as error:
        get_code_version()
    assert 'src/radar/cli.py' in str(error.value)


def test_dirty_tree_is_marked_if_allowed(monkeypatch):
    fake_git(monkeypatch, changes=' M src/radar/cli.py')
    assert get_code_version(allow_dirty=True) == 'abc123-dirty'


def test_only_the_code_paths_are_checked(monkeypatch):
    calls = fake_git(monkeypatch)
    get_code_version()
    assert calls[1] == ('status', '--porcelain', '--', *CODE_PATHS)


# --- with a real repository ------------------------------------------------------------------

@pytest.fixture
def repository(tmp_path, monkeypatch):
    """A Git repository with one commit in a temporary folder, used as the project root."""
    if shutil.which('git') is None:
        pytest.skip('git is not installed')

    def git(*args):
        subprocess.run(['git', '-c', 'user.name=test', '-c', 'user.email=test@example.org',
                        '-c', 'commit.gpgsign=false', *args],
                       cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / 'src').mkdir()
    (tmp_path / 'migrations').mkdir()
    (tmp_path / 'src' / 'code.py').write_text('x = 1\n', encoding='utf-8')
    (tmp_path / 'migrations' / 'first.py').write_text('y = 1\n', encoding='utf-8')
    (tmp_path / 'uv.lock').write_text('version = 1\n', encoding='utf-8')
    (tmp_path / 'README.md').write_text('readme\n', encoding='utf-8')
    git('init')
    git('add', '.')
    git('commit', '-m', 'initial')
    monkeypatch.setattr(provenance, 'ROOT', tmp_path)
    return tmp_path


def test_code_version_is_the_full_commit_hash(repository):
    version = get_code_version()
    assert len(version) == 40 and set(version) <= set('0123456789abcdef')


def test_changes_outside_the_code_paths_do_not_count(repository):
    (repository / 'README.md').write_text('changed\n', encoding='utf-8')
    (repository / 'notes.txt').write_text('new\n', encoding='utf-8')
    assert not get_code_version().endswith('-dirty')


def test_code_paths_cover_code_migrations_and_library_versions():
    assert CODE_PATHS == ['src', 'migrations', 'uv.lock']


@pytest.mark.parametrize('path', ['src/code.py', 'migrations/first.py', 'uv.lock'])
def test_changed_code_counts(repository, path):
    (repository / path).write_text('changed = True\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='uncommitted changes'):
        get_code_version()
    assert get_code_version(allow_dirty=True).endswith('-dirty')


def test_new_untracked_code_counts(repository):
    (repository / 'src' / 'new_module.py').write_text('z = 1\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='new_module.py'):
        get_code_version()
