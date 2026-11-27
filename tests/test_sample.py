from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from radar.db.models.annotation import Sample
from radar.db.models.collection import Post
from radar.sampling import sample as sampling
from radar.sampling.sample import create_sample, create_subset, load_posts

DAY = datetime(2026, 11, 1, tzinfo=timezone.utc)
LONG = 'Die Reichweite im Winter ist ein Witz'  # passes every filter


@pytest.fixture(autouse=True)
def fixed_code_version(monkeypatch):
    """The tests must not depend on the state of the Git tree."""
    monkeypatch.setattr(sampling, 'get_code_version',
                        lambda allow_dirty=False: 'test-version-dirty' if allow_dirty else 'test-version')


def texts(posts):
    return sorted(post.text for post in posts)


def sample_texts(session, name):
    """Texts of the posts stored for a sample, read from the database."""
    return sorted(session.scalars(select(Post.text).join(Post.samples).where(Sample.name == name)))


def sample_ids(session, name):
    return set(session.scalars(select(Post.id).join(Post.samples).where(Sample.name == name)))


# --- load_posts ------------------------------------------------------------------------------

def test_load_posts_returns_the_comments_of_the_topic_only(database, Corpus):
    with database() as session:
        e_auto = Corpus(session, 'E-Auto')
        video = e_auto.video('v1')
        top = e_auto.comment(video, 'Kommentar')
        e_auto.comment(top, 'Antwort')
        other = Corpus(session, 'Tempolimit', search_term='Tempolimit')
        other.comment(other.video('v2'), 'anderes Thema')
        session.commit()

        assert texts(load_posts(session, 'E-Auto')) == ['Antwort', 'Kommentar']
        assert texts(load_posts(session, 'Tempolimit')) == ['anderes Thema']
        assert load_posts(session, 'Unbekannt') == []


def test_root_posts_are_left_out_unless_asked_for(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        corpus.comment(corpus.video('v1'), 'Kommentar')
        session.commit()

        assert texts(load_posts(session, 'E-Auto')) == ['Kommentar']
        assert texts(load_posts(session, 'E-Auto', include_root_posts=True)) == [
            'Beschreibung des Videos', 'Kommentar']


def test_a_video_found_several_times_gives_each_post_once(database, Corpus):
    with database() as session:
        corpus = Corpus(session, search_term='Verbrenner-Aus')
        video = corpus.video('v1')
        corpus.comment(video, 'Kommentar')
        corpus.new_run('Verbrenner-Aus')        # the same search again
        corpus.find(video)
        corpus.new_run('Elektroautos Test')     # another search term of the topic
        corpus.find(video)
        session.commit()

        assert texts(load_posts(session, 'E-Auto')) == ['Kommentar']


def test_load_posts_filters_by_source_and_video(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        corpus.comment(corpus.video('v1'), 'unter Video 1')
        corpus.comment(corpus.video('v2'), 'unter Video 2')
        corpus.comment(corpus.video('t1', source='reddit'), 'aus Reddit')
        session.commit()

        assert texts(load_posts(session, 'E-Auto', source='youtube')) == ['unter Video 1', 'unter Video 2']
        assert texts(load_posts(session, 'E-Auto', source='reddit')) == ['aus Reddit']
        assert texts(load_posts(session, 'E-Auto', video_ids=['v2'])) == ['unter Video 2']
        assert texts(load_posts(session, 'E-Auto', video_ids=['v1', 't1'])) == ['aus Reddit', 'unter Video 1']
        assert load_posts(session, 'E-Auto', video_ids=[]) == []


def test_time_window_includes_the_start_and_excludes_the_end(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1')
        for day in range(4):
            corpus.comment(video, f'Tag {day}', published_at=DAY + timedelta(days=day))
        session.commit()

        after, before = DAY + timedelta(days=1), DAY + timedelta(days=3)
        assert texts(load_posts(session, 'E-Auto', published_after=after)) == ['Tag 1', 'Tag 2', 'Tag 3']
        assert texts(load_posts(session, 'E-Auto', published_before=before)) == ['Tag 0', 'Tag 1', 'Tag 2']
        assert texts(load_posts(session, 'E-Auto', published_after=after, published_before=before)) == [
            'Tag 1', 'Tag 2']


# --- create_sample ---------------------------------------------------------------------------

def test_create_sample_freezes_the_filtered_posts_and_records_the_attrition(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1')
        corpus.comment(video, LONG)
        corpus.comment(video, LONG.upper())                           # duplicate
        corpus.comment(video, 'Der Strom kommt doch eh aus Kohle')
        corpus.comment(video, 'zu kurz')                              # short
        corpus.comment(video, '👍👍👍')                                 # empty
        create_sample(session, 'e-auto-main', 'E-Auto')
        session.commit()

    with database() as session:
        sample = session.scalar(select(Sample).where(Sample.name == 'e-auto-main'))
        assert sample_texts(session, 'e-auto-main') == ['Der Strom kommt doch eh aus Kohle', LONG]
        assert sample.code_version == 'test-version'
        assert sample.base_sample_id is None
        assert sample.definition == {
            'topic': 'E-Auto', 'source': None, 'video_ids': None,
            'published_after': None, 'published_before': None, 'include_root_posts': False,
            'loaded': 5,
            'filters': [{'name': 'remove_empty', 'remaining': 4},
                        {'name': 'remove_short', 'min_words': 5, 'remaining': 3},
                        {'name': 'remove_duplicates', 'remaining': 2}],
        }


def test_create_sample_keeps_the_order_of_the_filter_steps(database, Corpus):
    # the JSON serializer sorts dict keys, so the steps are a list
    with database() as session:
        corpus = Corpus(session)
        corpus.comment(corpus.video('v1'), LONG)
        create_sample(session, 's', 'E-Auto')
        session.commit()
    with database() as session:
        definition = session.scalar(select(Sample.definition))
        assert [step['name'] for step in definition['filters']] == [
            'remove_empty', 'remove_short', 'remove_duplicates']


def test_filters_can_be_switched_off_and_are_then_not_recorded(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1')
        for text in [LONG, LONG, 'zu kurz', '👍']:
            corpus.comment(video, text)
        sample = create_sample(session, 'alles', 'E-Auto', drop_empty=False, min_words=None,
                               drop_duplicates=False)
        assert sample.definition['loaded'] == 4 and sample.definition['filters'] == []
        only_length = create_sample(session, 'laenge', 'E-Auto', drop_empty=False, min_words=2,
                                    drop_duplicates=False)
        assert only_length.definition['filters'] == [{'name': 'remove_short', 'min_words': 2, 'remaining': 3}]
        session.commit()

        assert len(sample_ids(session, 'alles')) == 4
        assert sample_texts(session, 'laenge') == [LONG, LONG, 'zu kurz']


def test_create_sample_stores_its_arguments_in_the_definition(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        corpus.comment(corpus.video('v1'), LONG, published_at=DAY)
        sample = create_sample(session, 's', 'E-Auto', source='youtube', video_ids=['v1'],
                               published_after=DAY, published_before=DAY + timedelta(days=1),
                               include_root_posts=True, allow_dirty=True)
        assert sample.code_version == 'test-version-dirty'
        assert sample.definition['source'] == 'youtube'
        assert sample.definition['video_ids'] == ['v1']
        assert sample.definition['published_after'] == '2026-11-01T00:00:00+00:00'
        assert sample.definition['published_before'] == '2026-11-02T00:00:00+00:00'
        assert sample.definition['include_root_posts'] is True
        assert sample.definition['loaded'] == 2  # the comment and the video itself


def test_create_sample_refuses_an_unknown_topic(database, Corpus):
    with database() as session:
        Corpus(session)
        with pytest.raises(ValueError, match="unknown topic 'Tempolimit'"):
            create_sample(session, 's', 'Tempolimit')


def test_sample_names_are_unique(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        corpus.comment(corpus.video('v1'), LONG)
        create_sample(session, 's', 'E-Auto')
        session.commit()
        create_sample(session, 's', 'E-Auto')
        with pytest.raises(IntegrityError):
            session.commit()


def test_a_sample_does_not_change_when_new_posts_arrive(database, Corpus):
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1')
        corpus.comment(video, LONG)
        create_sample(session, 's', 'E-Auto')
        corpus.comment(video, 'Ein neuer Kommentar nach der Stichprobe')
        session.commit()

        assert sample_texts(session, 's') == [LONG]
        assert len(load_posts(session, 'E-Auto')) == 2


# --- create_subset ---------------------------------------------------------------------------

@pytest.fixture
def with_main_sample(database, Corpus):
    """A database with the sample 'main' of 30 posts."""
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1')
        for number in range(30):
            corpus.comment(video, f'{LONG} Nummer {number}')
        create_sample(session, 'main', 'E-Auto')
        session.commit()
    return database


def test_subset_draws_n_posts_of_the_base_sample(with_main_sample):
    with with_main_sample() as session:
        subset = create_subset(session, 'sub', 'main', 10, seed=42)
        assert subset.definition == {'base_sample': 'main', 'n': 10, 'seed': 42, 'base_size': 30}
        session.commit()

        drawn = sample_ids(session, 'sub')
        assert len(drawn) == 10 and drawn < sample_ids(session, 'main')
        subset = session.scalar(select(Sample).where(Sample.name == 'sub'))
        main = session.scalar(select(Sample).where(Sample.name == 'main'))
        assert subset.base_sample_id == main.id
        assert subset.topic_id == main.topic_id  # the topic is inherited
        assert subset.code_version == 'test-version'


def test_same_seed_gives_the_same_posts_and_another_seed_others(with_main_sample):
    with with_main_sample() as session:
        create_subset(session, 'a', 'main', 10, seed=42)
        create_subset(session, 'b', 'main', 10, seed=42)
        create_subset(session, 'c', 'main', 10, seed=7)
        session.commit()

        assert sample_ids(session, 'a') == sample_ids(session, 'b')
        assert sample_ids(session, 'a') != sample_ids(session, 'c')


def test_subset_uses_the_seed_of_the_config_by_default(with_main_sample):
    with with_main_sample() as session:
        subset = create_subset(session, 'sub', 'main', 5)
        assert subset.definition['seed'] == sampling.SEED


def test_subsets_can_be_nested(with_main_sample):
    with with_main_sample() as session:
        create_subset(session, 'llm', 'main', 20, seed=1)
        session.commit()
        create_subset(session, 'human', 'llm', 5, seed=1)
        session.commit()

        assert sample_ids(session, 'human') < sample_ids(session, 'llm') < sample_ids(session, 'main')
        human = session.scalar(select(Sample).where(Sample.name == 'human'))
        llm = session.scalar(select(Sample).where(Sample.name == 'llm'))
        assert human.base_sample_id == llm.id
        assert human.definition['base_size'] == 20


def test_subset_of_the_full_size_takes_every_post(with_main_sample):
    with with_main_sample() as session:
        create_subset(session, 'alle', 'main', 30)
        session.commit()
        assert sample_ids(session, 'alle') == sample_ids(session, 'main')


def test_subset_refuses_an_unknown_base_and_too_many_posts(with_main_sample):
    with with_main_sample() as session:
        with pytest.raises(ValueError, match="unknown sample 'nope'"):
            create_subset(session, 'sub', 'nope', 5)
        with pytest.raises(ValueError, match='has only 30 posts, cannot draw 31'):
            create_subset(session, 'sub', 'main', 31)
