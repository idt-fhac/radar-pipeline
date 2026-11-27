# conftest.py — shared by all tests: a safe environment, an empty temporary database and a
# builder for collection data.

import os

# Set before anything imports radar. privacy.py reads SALT_HEX at import time: the tests must
# neither need nor use the real key from .env (load_dotenv keeps values that are already set).
os.environ['SALT_HEX'] = 'ab' * 32
# A test that used the global Session by mistake must never reach the production database.
os.environ['RADAR_DB'] = 'test'

from datetime import datetime, timedelta, timezone  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from radar.db import models  # noqa: E402, F401 (registers every table on Base.metadata)
from radar.db.base import Base  # noqa: E402
from radar.db.engine import enable_foreign_keys, engine as real_engine  # noqa: E402
from radar.db.models.collection import (  # noqa: E402
    Author, CollectionRun, Container, Finding, Post, SearchQuery, Topic,
)

START = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def database(tmp_path):
    """Session factory for an empty SQLite file with the full schema.

    Set up like radar.db.engine: the same JSON serializer (sorted keys) and the same
    foreign key listener, but in a temporary folder.
    """
    engine = create_engine(f'sqlite:///{(tmp_path / "test.db").as_posix()}',
                           json_serializer=real_engine.dialect._json_serializer)
    event.listen(engine, 'connect', enable_foreign_keys)
    Base.metadata.create_all(engine)
    yield sessionmaker(engine)
    engine.dispose()  # Windows cannot delete a file that is still open


class CorpusBuilder:
    """Builds collection data in a session: topic -> search query -> collection run -> videos -> comments.

    Nothing is committed here. Build everything first and commit then: a commit expires the
    objects, and their relationships (lazy='raise') cannot be read any more.
    """

    def __init__(self, session, topic='E-Auto', search_term='Verbrenner-Aus'):
        self.session = session
        self.topic = Topic(name=topic, search_queries=[], samples=[])
        session.add(self.topic)
        self.collection_run = None
        self.posts = 0
        self.authors = {}
        self.new_run(search_term)

    def new_run(self, search_term):
        """A collection run for the search term (a new search query of the topic if needed); later videos belong to it."""
        search_query = next((query for query in self.topic.search_queries if query.search_term == search_term), None)
        if search_query is None:
            search_query = SearchQuery(source='youtube', search_term=search_term, search_params={})
            self.topic.search_queries.append(search_query)
        self.collection_run = CollectionRun(search_query=search_query, started_at=START, finished_at=START,
                                            fetch_params={}, code_version='test-version')
        self.session.add(self.collection_run)
        self.rank = 0
        return self.collection_run

    def video(self, external_id, title='Verbrenner-Aus: Was nun?', source='youtube'):
        """A container found by the current run, with its root post (returned)."""
        container = Container(source=source, external_id=external_id, title=title)
        root = Post(container=container, external_id=external_id, text='Beschreibung des Videos',
                    published_at=START, raw={})
        self.session.add(root)
        self.find(root)
        return root

    def find(self, root):
        """The current run finds the container of this root post (again)."""
        self.rank += 1
        self.session.add(Finding(collection_run=self.collection_run, container=root.container, rank=self.rank))

    def comment(self, parent, text, published_at=None, author=None, external_id=None):
        """A comment below parent: the root post (top-level comment) or a comment (reply)."""
        self.posts += 1
        if published_at is None:
            published_at = START + timedelta(minutes=self.posts)  # later comments are newer
        if author is not None and author not in self.authors:
            self.authors[author] = Author(pseudonym=author)
        post = Post(container=parent.container, parent=parent, external_id=external_id or f'c{self.posts}',
                    text=text, published_at=published_at, raw={}, author=self.authors.get(author))
        self.session.add(post)
        return post


@pytest.fixture
def Corpus():
    """The CorpusBuilder class; call it with a session."""
    return CorpusBuilder
