import random
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from radar.db.engine import Session
from radar.db.models.collection import Post, Container, Finding, CollectionRun, SearchQuery, Topic
from radar.db.models.annotation import Sample
from radar.config import setting
from radar.provenance import get_code_version
from radar.sampling.filters import remove_duplicates, remove_empty, remove_short

MIN_WORDS = setting('sampling', 'min_words')
SEED = setting('sampling', 'seed')

def _load_posts_stmt(session, topic_name, *, source=None, video_ids=None,
               published_after=None, published_before=None,
               include_root_posts=False):
    published_after, published_before = _utc(published_after), _utc(published_before)
    stmt = select(Post).join(Post.container).join(Container.findings).join(Finding.collection_run).join(
        CollectionRun.search_query).join(SearchQuery.topics).where(Topic.name == topic_name).distinct()
    if source is not None:
        stmt = stmt.where(Container.source == source)
    if video_ids is not None:
        stmt = stmt.where(Container.external_id.in_(video_ids))
    if published_after is not None:
        stmt = stmt.where(Post.published_at >= published_after)
    if published_before is not None:
        stmt = stmt.where(Post.published_at < published_before)
    if not include_root_posts:
        stmt = stmt.where(Post.parent_id.is_not(None))
    return stmt

def load_posts(session, topic_name, *, source=None, video_ids=None,
               published_after=None, published_before=None,
               include_root_posts=False):
    """Return all posts belonging to a topic, optionally narrowed down.

    :param session: open database session
    :param topic_name: name of the topic (``topic.name``)
    :param source: only posts from this source, e.g. ``'youtube'``
    :param video_ids: only posts from these videos; a list of **external** ids
        (YouTube video ids as in ``container.external_id``), not internal ``container.id``
    :param published_after: only posts published at or after this time (inclusive); a datetime
        with another time zone is converted to UTC, one without time zone counts as UTC
    :param published_before: only posts published before this time (exclusive), as above
    :param include_root_posts: also return root posts (the video itself), default False
    :return: list of matching posts, each at most once
    """
    stmt = _load_posts_stmt(session, topic_name=topic_name, source = source, video_ids=video_ids, published_after = published_after, published_before=published_before, include_root_posts=include_root_posts)
    return session.scalars(stmt).all()


def create_sample(session, name, topic_name, *, source=None, video_ids=None,
                  published_after=None, published_before=None, include_root_posts=False,
                  drop_empty=True, min_words=MIN_WORDS, drop_duplicates=True, allow_dirty=False):
    """Load the posts of a topic, filter them and freeze the result as a new sample.

    The definition is built from the same arguments that select and filter the posts,
    so it always describes what actually happened. The caller commits.

    :param name: unique, human-chosen name, e.g. ``'e-auto-main'``
    :param drop_empty: drop posts without any letters (only links, mentions, emojis)
    :param min_words: drop posts with fewer words; ``None`` keeps all lengths
    :param drop_duplicates: keep only the earliest post per normalised text
    :param allow_dirty: allow uncommitted code changes (development only)
    Other parameters as in :func:`load_posts`.
    """
    topic = session.scalar(select(Topic).where(Topic.name == topic_name))
    if topic is None:
        raise ValueError(f'unknown topic {topic_name!r}')
    # here already, so the definition records the same UTC times the query uses
    published_after, published_before = _utc(published_after), _utc(published_before)

    posts = load_posts(session, topic_name, source=source, video_ids=video_ids,
                       published_after=published_after, published_before=published_before,
                       include_root_posts=include_root_posts)

    # A list, because the order matters: each entry records how many posts remained after it.
    # (A dict would not keep the order: the JSON serializer sorts keys.)
    filters = []
    loaded = len(posts)
    if drop_empty:
        posts = remove_empty(posts)
        filters.append({'name': 'remove_empty', 'remaining': len(posts)})
    if min_words is not None:
        posts = remove_short(posts, min_words)
        filters.append({'name': 'remove_short', 'min_words': min_words, 'remaining': len(posts)})
    if drop_duplicates:
        posts = remove_duplicates(posts)
        filters.append({'name': 'remove_duplicates', 'remaining': len(posts)})

    definition = {
        'topic': topic_name,
        'source': source,
        'video_ids': video_ids,
        'published_after': _iso(published_after),
        'published_before': _iso(published_before),
        'include_root_posts': include_root_posts,
        'loaded': loaded,
        'filters': filters,
    }
    sample = Sample(name=name, topic=topic, definition=definition,
                    code_version=get_code_version(allow_dirty=allow_dirty),
                    created_at=datetime.now(timezone.utc), posts=posts, subsets=[])
    session.add(sample)
    return sample


def create_subset(session, name, base_name, n, *, seed=SEED, allow_dirty=False):
    """Draw n posts at random from an existing sample and freeze them as a subset.

    Reproducible: the same base sample, n and seed always give the same posts.
    Topic and filters are inherited from the base sample. The caller commits.

    :param name: unique, human-chosen name, e.g. ``'e-auto-300'``
    :param base_name: name of the sample to draw from
    :param n: number of posts to draw
    :param seed: seed of the random generator
    """
    base = session.scalar(select(Sample).where(Sample.name == base_name)
                          .options(selectinload(Sample.posts), selectinload(Sample.topic)))
    if base is None:
        raise ValueError(f'unknown sample {base_name!r}')
    if n > len(base.posts):
        raise ValueError(f'sample {base_name!r} has only {len(base.posts)} posts, cannot draw {n}')

    # Sort first: the database returns posts in no guaranteed order,
    # and the same seed only gives the same draw on the same input order.
    candidates = sorted(base.posts, key=lambda p: p.id)
    posts = random.Random(seed).sample(candidates, n)

    definition = {
        'base_sample': base_name,
        'n': n,
        'seed': seed,
        'base_size': len(candidates),
    }
    subset = Sample(name=name, topic=base.topic, base=base, definition=definition,
                    code_version=get_code_version(allow_dirty=allow_dirty),
                    created_at=datetime.now(timezone.utc), posts=posts, subsets=[])
    session.add(subset)
    return subset


def _utc(value):
    """A time filter as UTC: an aware datetime is converted, a naive one counts as UTC.

    SQLite stores no time zone and SQLAlchemy drops it when binding a value, so
    '00:00+01:00' would be compared with the stored UTC times as if it were 00:00 UTC.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value):
    """datetime -> ISO string for the JSON definition (JSON knows no datetime)."""
    return value.isoformat() if value is not None else None


def main():
    with Session() as session:
        print(len(load_posts(session, 'E-Auto')))
        print(len(load_posts(session, 'E-Auto', include_root_posts = True)))

if __name__ == "__main__":
    main()