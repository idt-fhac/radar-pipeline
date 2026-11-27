from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from radar.db.base import MAX_REPR_LENGTH, Base
from radar.db.engine import engine as real_engine
from radar.db.models.annotation import Annotation, CriterionMention
from radar.db.models.collection import Container, Post, SearchQuery, Topic

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def new_post(container, external_id='c1', **values):
    return Post(container=container, external_id=external_id, text='Text', published_at=NOW, raw={}, **values)


# --- engine ----------------------------------------------------------------------------------

def test_json_is_serialised_with_sorted_keys_and_readable_umlauts():
    serialise = real_engine.dialect._json_serializer
    assert serialise({'regionCode': 'DE', 'b': {'z': 1, 'a': 'ä'}, 'a': 2}) == \
        '{"a": 2, "b": {"a": "ä", "z": 1}, "regionCode": "DE"}'


def test_foreign_keys_are_enforced(database):
    with database() as session:
        assert session.execute(text('PRAGMA foreign_keys')).scalar() == 1
        session.add(Post(container_id=999, external_id='c1', text='Text', published_at=NOW, raw={}))
        with pytest.raises(IntegrityError):
            session.commit()


def test_all_tables_of_the_data_model_exist(database):
    with database() as session:
        tables = set(session.scalars(text("SELECT name FROM sqlite_master WHERE type = 'table'")))
    assert tables == set(Base.metadata.tables)
    assert {'topic', 'post', 'snapshot', 'sample', 'llm_run', 'human_run', 'criterion_mention',
            'embedding', 'mapping_input', 'assignment'} <= tables


# --- constraints -----------------------------------------------------------------------------

def test_search_query_is_unique_whatever_the_order_of_its_params(database):
    # the reason for the sorted keys: the same params must give the same JSON text
    with database() as session:
        session.add(SearchQuery(source='youtube', search_term='Verbrenner-Aus',
                                search_params={'regionCode': 'DE', 'relevanceLanguage': 'de'}))
        session.commit()
        found = session.scalar(select(SearchQuery).where(
            SearchQuery.search_params == {'relevanceLanguage': 'de', 'regionCode': 'DE'}))
        assert found is not None

        session.add(SearchQuery(source='youtube', search_term='Verbrenner-Aus',
                                search_params={'relevanceLanguage': 'de', 'regionCode': 'DE'}))
        with pytest.raises(IntegrityError):
            session.commit()


def test_same_search_term_with_other_params_is_another_query(database):
    with database() as session:
        session.add_all([
            SearchQuery(source='youtube', search_term='Verbrenner-Aus', search_params={'regionCode': 'DE'}),
            SearchQuery(source='youtube', search_term='Verbrenner-Aus', search_params={'regionCode': 'AT'}),
            SearchQuery(source='youtube', search_term='Verbrenner-Aus', search_params={}),
            SearchQuery(source='reddit', search_term='Verbrenner-Aus', search_params={}),
        ])
        session.commit()


def test_topic_names_are_unique(database):
    with database() as session:
        session.add_all([Topic(name='E-Auto'), Topic(name='E-Auto')])
        with pytest.raises(IntegrityError):
            session.commit()


def test_external_ids_are_unique_per_container_only(database):
    with database() as session:
        first = Container(source='youtube', external_id='v1', title='Video 1')
        second = Container(source='youtube', external_id='v2', title='Video 2')
        session.add_all([new_post(first, 'c1'), new_post(second, 'c1')])
        session.commit()
        session.add(Post(container_id=first.id, external_id='c1', text='Text', published_at=NOW, raw={}))
        with pytest.raises(IntegrityError):
            session.commit()


def test_same_video_id_can_exist_in_two_sources(database):
    with database() as session:
        session.add_all([Container(source='youtube', external_id='abc', title='Video'),
                         Container(source='reddit', external_id='abc', title='Thread')])
        session.commit()
        session.add(Container(source='youtube', external_id='abc', title='Video again'))
        with pytest.raises(IntegrityError):
            session.commit()


def test_post_without_author_and_parent_is_allowed(database):
    # deleted users must not collapse into one pseudonym; a root post has no parent
    with database() as session:
        session.add(new_post(Container(source='youtube', external_id='v1', title='Video')))
        session.commit()
        post = session.scalar(select(Post))
        assert post.author_pseudonym is None and post.parent_id is None


def test_criterion_mention_needs_its_annotation(database):
    with database() as session:
        post = new_post(Container(source='youtube', external_id='v1', title='Video'))
        session.add(post)
        session.commit()
        # no annotation (run 1, this post) exists: the composite foreign key refuses the mention
        session.add(CriterionMention(run_id=1, post_id=post.id, label='Preis'))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        assert session.get(Annotation, (1, post.id)) is None


# --- Base.__repr__ ---------------------------------------------------------------------------

def test_repr_shows_the_column_values_and_cuts_long_ones():
    post = Post(id=3, container_id=1, external_id='Ug1', text='Die Reichweite ist ' * 10)
    shown = repr(post)
    assert shown.startswith("Post(id=3, container_id=1, external_id='Ug1', ")
    assert "text='Die Reichweite ist Die Reichweite ist D…" in shown
    assert len("'Die Reichweite ist Die Reichweite ist D") == MAX_REPR_LENGTH
    assert 'published_at=…' in shown  # never set


def test_repr_of_an_expired_object_triggers_no_query(database):
    with database() as session:
        topic = Topic(name='E-Auto')
        session.add(topic)
        session.commit()  # expires the object
        assert repr(topic) == 'Topic(id=…, name=…)'
