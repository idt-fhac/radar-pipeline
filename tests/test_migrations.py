# Data migrations: their SQL is run against a temporary database built from the models.

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, text

from radar.db.models.collection import Container, Post

VERSIONS = Path(__file__).resolve().parents[1] / 'migrations' / 'versions'
NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def load_migration(revision):
    """Import a migration script by its revision id (the folder is no package)."""
    path, = VERSIONS.glob(f'{revision}_*.py')
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migrations_form_one_chain():
    revisions = {}
    for path in VERSIONS.glob('*.py'):
        migration = load_migration(path.name.split('_')[0])
        revisions[migration.revision] = migration.down_revision
    heads = set(revisions) - set(revisions.values())
    roots = [revision for revision, parent in revisions.items() if parent is None]
    assert len(heads) == 1 and len(roots) == 1  # no branches: "alembic upgrade head" has one way


def test_repair_container_titles_takes_the_title_stored_with_the_root_post(database):
    migration = load_migration('271812c08c36')

    def container(external_id, title, raw=None, comment_raw=None):
        new = Container(source='youtube', external_id=external_id, title=title)
        if raw is not None:
            root = Post(container=new, external_id=external_id, text='Beschreibung', published_at=NOW, raw=raw)
            session.add(root)
            if comment_raw is not None:
                session.add(Post(container=new, parent=root, external_id='c1', text='Kommentar',
                                 published_at=NOW, raw=comment_raw))
        session.add(new)

    with database() as session:
        container('escaped', 'Reichweite &amp; Verbrauch', raw={'snippet': {'title': 'Reichweite & Verbrauch'}})
        container('mention', 'Test | @kanalname', raw={'snippet': {'title': 'Test | @3fa9c0e1b2d4'}},
                  comment_raw={'snippet': {'title': 'nicht der Titel des Videos'}})
        container('correct', 'Schon richtig', raw={'snippet': {'title': 'Schon richtig'}})
        container('no-title-in-raw', 'bleibt', raw={})
        container('no-root-post', 'bleibt auch')
        session.commit()

        session.execute(text(migration.REPAIR_TITLES))
        session.commit()

        assert dict(session.execute(select(Container.external_id, Container.title)).all()) == {
            'escaped': 'Reichweite & Verbrauch',
            'mention': 'Test | @3fa9c0e1b2d4',
            'correct': 'Schon richtig',
            'no-title-in-raw': 'bleibt',
            'no-root-post': 'bleibt auch',
        }
