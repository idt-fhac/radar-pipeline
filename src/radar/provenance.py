# provenance.py — which code produced a result

import subprocess

from radar import ROOT

# Only changes here alter what a pipeline run executes; tests or docs do not.
CODE_PATHS = ['src', 'migrations']


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def code_version(allow_dirty=False):
    """Return the Git commit of the running code, to be stored as code_version.

    With uncommitted changes in the code paths the commit hash would not describe the
    code that actually ran, so this raises unless allow_dirty is set (for development);
    then the hash gets the suffix '-dirty' so such results stay recognisable.
    """
    commit = git('rev-parse', 'HEAD')
    changes = git('status', '--porcelain', '--', *CODE_PATHS)
    if changes:
        if not allow_dirty:
            raise RuntimeError('uncommitted changes in the code, commit them first '
                               '(or pass allow_dirty=True while developing):\n' + changes)
        return commit + '-dirty'
    return commit
