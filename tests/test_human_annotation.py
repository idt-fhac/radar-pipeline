import copy
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

from radar.annotation import prompts
from radar.annotation.human import store, transfer
from radar.annotation.human.app import STATIC_DIR, create_app
from radar.annotation.run_llm import open_posts
from radar.db import models  # noqa: F401 (registers every table on Base.metadata)
from radar.db.base import Base
from radar.db.models.annotation import (Annotation, AnnotationRun, CriterionMention, HumanRun, LlmRun,
                                        PromptVersion, Sample)
from radar.db.models.collection import Container, Post, Topic
from radar.db.models.mapping import MappingRun

PROMPT = 'guide_v1'
PROMPT_TEXT = 'Leitfaden zum Thema „{THEMA}“.'
TEXTS = {
    'c1': 'Die Reichweite im Winter ist ein Witz, aber das Laden zu Hause ist billig.',
    'c2': 'So ein Schwachsinn, typisch Deutschland',
    'c3': '@3fa9c0e1b2d4 der Strom kommt doch eh aus Kohle.\nUnd wer zahlt die Ladesäulen?',
    'c4': 'Nicht in der Stichprobe.',
}
CRITERIA = [
    {'label': 'Reichweite', 'sentiment': 'negative', 'quote': 'Die Reichweite im Winter ist ein Witz'},
    {'label': 'Ladekosten', 'sentiment': 'positive', 'quote': None},
]


@pytest.fixture
def create_database(tmp_path, monkeypatch):
    """Returns create(name, external_ids) -> session factory for a new SQLite file in a temporary folder.

    Every database holds one video with the comments c1 to c4, stored in the given order
    (so the same comment can have different post ids in two databases), and the sample
    'e-auto-3' with c1 to c3. Prompts are read from the temporary folder and the code version
    is fixed, so the tests depend neither on prompts/ nor on the state of the Git tree.
    """
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / f'{PROMPT}.txt').write_text(PROMPT_TEXT, encoding='utf-8')
    monkeypatch.setattr(prompts, 'ROOT', tmp_path)
    monkeypatch.setattr(store, 'get_code_version', lambda allow_dirty=False: 'test-version')
    engines = []

    def create(name, external_ids=('c1', 'c2', 'c3', 'c4')):
        engine = create_engine(f'sqlite:///{(tmp_path / f"{name}.db").as_posix()}')
        event.listen(engine, 'connect', lambda connection, record: connection.execute('PRAGMA foreign_keys=ON'))
        engines.append(engine)
        Base.metadata.create_all(engine)
        Session = sessionmaker(engine)
        with Session() as session:
            container = Container(source='youtube', external_id='video1', title='Verbrenner-Aus: Was nun?')
            posts = [Post(container=container, external_id=external_id, text=TEXTS[external_id], raw={},
                          published_at=datetime(2026, 11, 1, tzinfo=timezone.utc))
                     for external_id in external_ids]
            session.add_all(posts)
            session.flush()  # post ids follow the given order
            session.add(Sample(name='e-auto-3', topic=Topic(name='E-Auto'), definition={},
                               code_version='test-version', created_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
                               posts=[post for post in posts if post.external_id != 'c4']))
            session.commit()
        return Session

    yield create
    for engine in engines:
        engine.dispose()  # Windows cannot delete a file that is still open


@pytest.fixture
def Session(create_database):
    return create_database('main')


def post_id(session, external_id):
    return session.scalar(select(Post.id).where(Post.external_id == external_id))


def start(session, annotator='Anna'):
    annotation_run, created = store.start_human_run(session, 'e-auto-3', PROMPT, annotator)
    return annotation_run


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


# --- creating and continuing a run -----------------------------------------------------------

def test_start_creates_a_human_run_that_is_no_llm_run(Session):
    with Session() as session:
        annotation_run, created = store.start_human_run(session, 'e-auto-3', PROMPT, ' Anna ')
        assert created
        assert annotation_run.human_run.annotator == 'Anna'
        assert annotation_run.code_version == 'test-version'
        assert annotation_run.prompt_version.name == PROMPT
        assert session.get(LlmRun, annotation_run.id) is None  # a run is never both


def test_start_again_continues_the_run_of_the_same_annotator(Session):
    with Session() as session:
        first = start(session).id
    with Session() as session:
        annotation_run, created = store.start_human_run(session, 'e-auto-3', PROMPT, 'Anna')
        assert (annotation_run.id, created) == (first, False)
        other, created = store.start_human_run(session, 'e-auto-3', PROMPT, 'Ben')
        assert other.id != first and created
        assert count(session, HumanRun) == 2


def test_start_rejects_unknown_sample_and_missing_name(Session):
    with Session() as session:
        with pytest.raises(ValueError, match='sample'):
            store.start_human_run(session, 'nope', PROMPT, 'Anna')
        with pytest.raises(ValueError, match='name'):
            store.start_human_run(session, 'e-auto-3', PROMPT, '  ')
        assert count(session, AnnotationRun) == 0


def test_llm_run_is_not_a_human_run(Session):
    with Session() as session:
        sample = session.scalar(select(Sample))
        llm = AnnotationRun(sample_id=sample.id, code_version='x', started_at=datetime.now(timezone.utc),
                            prompt_version=PromptVersion(name='p', text='t', hash='h'))
        llm.llm_run = LlmRun(model='m', temperature=0.0, batch_size=20, context_strategy='text', params={})
        session.add(llm)
        session.commit()
        with pytest.raises(ValueError, match='not a human run'):
            store.load_human_run(session, llm.id)


# --- saving ----------------------------------------------------------------------------------

def test_save_with_criteria(Session):
    with Session() as session:
        annotation_run = start(session)
        run_id, c1 = annotation_run.id, post_id(session, 'c1')
        store.save_annotation(session, annotation_run, c1, 'mixed', CRITERIA)
    with Session() as session:  # a new session: read what is really in the file
        post = store.get_post(session, store.load_human_run(session, run_id), c1)
        assert post['text'] == TEXTS['c1'] and post['video_title'] == 'Verbrenner-Aus: Was nun?'
        assert post['annotation'] == {'stance': 'mixed', 'criteria': CRITERIA}


def test_run_state_tells_what_is_done_and_shows_the_guideline(Session):
    with Session() as session:
        annotation_run = start(session)
        ids = [post_id(session, external_id) for external_id in ['c1', 'c2', 'c3']]
        store.save_annotation(session, annotation_run, ids[0], 'pro', [])
        state = store.run_state(session, store.load_human_run(session, annotation_run.id))
        assert state['post_ids'] == ids  # the post outside the sample is not offered
        assert state['annotated_ids'] == ids[:1]
        assert state['guideline'] == 'Leitfaden zum Thema „E-Auto“.'
        assert state['locked_by'] == []


def test_no_criteria_is_an_annotation_of_its_own(Session):
    with Session() as session:
        annotation_run = start(session)
        c2 = post_id(session, 'c2')
        store.save_annotation(session, annotation_run, c2, 'contra', [])
        assert session.get(Annotation, (annotation_run.id, c2)).stance == 'contra'
        assert count(session, CriterionMention) == 0
        assert c2 not in [post.id for post in open_posts(session, annotation_run)]  # done, not open
        post = store.get_post(session, store.load_human_run(session, annotation_run.id), c2)
        assert post['annotation'] == {'stance': 'contra', 'criteria': []}


def test_saving_again_replaces_stance_and_criteria(Session):
    corrected = [{'label': 'Winterreichweite', 'sentiment': 'negative', 'quote': 'Reichweite im Winter'}]
    with Session() as session:
        annotation_run = start(session)
        run_id, c1 = annotation_run.id, post_id(session, 'c1')
        store.save_annotation(session, annotation_run, c1, 'mixed', CRITERIA)
    with Session() as session:
        store.save_annotation(session, store.load_human_run(session, run_id), c1, 'contra', corrected)
    with Session() as session:
        post = store.get_post(session, store.load_human_run(session, run_id), c1)
        assert post['annotation'] == {'stance': 'contra', 'criteria': corrected}
        assert count(session, Annotation) == 1 and count(session, CriterionMention) == 1


def test_quote_and_label_are_stripped_and_an_empty_quote_is_none(Session):
    with Session() as session:
        annotation_run = start(session)
        c3 = post_id(session, 'c3')
        store.save_annotation(session, annotation_run, c3, 'contra', [
            {'label': ' Strommix ', 'sentiment': 'negative', 'quote': ' kommt doch eh aus Kohle.\nUnd '},
            {'label': 'Kosten', 'sentiment': 'neutral', 'quote': '  '}])
        annotation_run = store.load_human_run(session, annotation_run.id)
        assert store.get_post(session, annotation_run, c3)['annotation']['criteria'] == [
            {'label': 'Strommix', 'sentiment': 'negative', 'quote': 'kommt doch eh aus Kohle.\nUnd'},
            {'label': 'Kosten', 'sentiment': 'neutral', 'quote': None}]


@pytest.mark.parametrize('stance, criteria, message', [
    ('dafuer', [], 'invalid stance'),
    (None, [], 'invalid stance'),
    ('pro', [{'label': 'Preis', 'sentiment': 'positiv', 'quote': None}], 'invalid sentiment'),
    ('pro', [{'label': 'Preis', 'sentiment': None, 'quote': None}], 'invalid sentiment'),
    ('pro', [{'label': '  ', 'sentiment': 'neutral', 'quote': None}], 'without label'),
    ('pro', [{'label': 'Preis', 'sentiment': 'neutral', 'quote': None},
             {'label': 'preis', 'sentiment': 'negative', 'quote': None}], 'twice'),
    ('pro', [{'label': 'Preis', 'sentiment': 'neutral', 'quote': 'die reichweite'}], 'does not occur'),
    ('pro', None, 'must be a list'),
])
def test_invalid_input_is_rejected_and_nothing_is_stored(Session, stance, criteria, message):
    with Session() as session:
        annotation_run = start(session)
        with pytest.raises(ValueError, match=message):
            store.save_annotation(session, annotation_run, post_id(session, 'c1'), stance, criteria)
        assert count(session, Annotation) == 0 and count(session, CriterionMention) == 0


def test_post_outside_the_sample_is_rejected(Session):
    with Session() as session:
        annotation_run = start(session)
        c4 = post_id(session, 'c4')
        with pytest.raises(ValueError, match='not part of sample'):
            store.save_annotation(session, annotation_run, c4, 'pro', [])
        with pytest.raises(ValueError, match='not part of sample'):
            store.get_post(session, annotation_run, c4)


def test_annotations_of_other_runs_stay_hidden(Session):
    with Session() as session:
        c1 = post_id(session, 'c1')
        store.save_annotation(session, start(session, 'Ben'), c1, 'contra', CRITERIA)
        mine = start(session, 'Anna')
        assert store.get_post(session, mine, c1)['annotation'] is None
        assert store.run_state(session, mine)['annotated_ids'] == []


def test_run_in_a_mapping_run_is_locked(Session):
    with Session() as session:
        annotation_run = start(session)
        run_id, c1, c2 = annotation_run.id, post_id(session, 'c1'), post_id(session, 'c2')
        store.save_annotation(session, annotation_run, c1, 'mixed', CRITERIA)
        mapping_run = MappingRun(method='hdbscan', params={}, code_version='x', created_at=datetime.now(timezone.utc),
                                 annotation_runs=[store.load_human_run(session, run_id)])
        session.add(mapping_run)
        session.commit()

        annotation_run = store.load_human_run(session, run_id)
        assert store.run_state(session, annotation_run)['locked_by'] == [mapping_run.id]
        with pytest.raises(store.RunLocked, match='mapping run'):
            store.save_annotation(session, annotation_run, c1, 'pro', [])  # editing
        with pytest.raises(store.RunLocked):
            store.save_annotation(session, annotation_run, c2, 'pro', [])  # a new annotation too
        assert store.get_post(session, annotation_run, c1)['annotation'] == {'stance': 'mixed', 'criteria': CRITERIA}
        assert count(session, Annotation) == 1


# --- help with labels ------------------------------------------------------------------------

def criterion(label):
    return {'label': label, 'sentiment': 'neutral', 'quote': None}


def test_run_labels_counts_only_the_labels_of_this_run(Session):
    with Session() as session:
        ids = [post_id(session, external_id) for external_id in ['c1', 'c2', 'c3']]
        mine = start(session, 'Anna')
        assert store.run_labels(session, mine) == []
        store.save_annotation(session, mine, ids[0], 'pro', [criterion('Reichweite'), criterion('Preis')])
        store.save_annotation(session, mine, ids[1], 'pro', [criterion('Reichweite'), criterion('Ladezeit')])
        store.save_annotation(session, mine, ids[2], 'pro', [])
        other = start(session, 'Ben')
        store.save_annotation(session, other, ids[0], 'pro', [criterion('Umwelt'), criterion('Reichweite')])

        assert store.run_labels(session, mine) == [  # most frequent first, then alphabetical
            {'label': 'Reichweite', 'count': 2}, {'label': 'Ladezeit', 'count': 1}, {'label': 'Preis', 'count': 1}]
        assert store.run_labels(session, other) == [
            {'label': 'Reichweite', 'count': 1}, {'label': 'Umwelt', 'count': 1}]


@pytest.mark.parametrize('a, b, distance', [
    ('Ladeinfrastrukur', 'Ladeinfrastruktur', 1),  # one letter missing
    ('Reichweite', 'Reichwiete', 2),               # two letters swapped = two replacements
    ('Preis', 'Preise', 1),
    ('Ladezeit', 'Ladeleistung', 5),               # z -> l, then s and ung inserted
    ('', 'abc', 3),
    ('Akku', 'Akku', 0),
])
def test_edit_distance(a, b, distance):
    assert store.edit_distance(a, b) == distance
    assert store.edit_distance(b, a) == distance


LABELS = [{'label': 'Ladeinfrastruktur', 'count': 7}, {'label': 'Ladezeit', 'count': 3},
          {'label': 'Preis', 'count': 2}, {'label': 'Preise', 'count': 1}, {'label': 'CO2', 'count': 1}]


@pytest.mark.parametrize('typed, expected', [
    ('Ladeinfrastrukur', 'Ladeinfrastruktur'),   # the typo this hint was built for
    (' ladeinfrastruktur ', 'Ladeinfrastruktur'),  # only the case differs
    ('Ladeinfrastruktur', None),                 # is a label of the run already
    ('Ladeleistung', None),                      # similar topic, but another criterion
    ('Ladezeiten', 'Ladezeit'),
    ('Preiss', 'Preis'),                         # as close to 'Preise': the more frequent label wins
    ('Preise', None),
    ('Reis', 'Preis'),                           # short labels: one edit
    ('Raus', None),                              # two edits are too many for a short label
    ('CO3', None),                               # very short labels: no edit at all
    ('co2', 'CO2'),
    ('', None),
])
def test_similar_label(typed, expected):
    similar = store.similar_label(typed, LABELS)
    assert (similar and similar['label']) == expected
    if similar:
        assert similar in LABELS  # the entry with its count


def test_similar_label_prefers_the_closer_label():
    labels = [{'label': 'Ladezeiten', 'count': 9}, {'label': 'Ladezeit', 'count': 1}]
    assert store.similar_label('Ladezeid', labels)['label'] == 'Ladezeit'


# --- export and import -----------------------------------------------------------------------

@pytest.fixture
def OtherSession(create_database):
    """The database of a second annotator: the same comments, but stored in another order, so the post ids differ."""
    return create_database('other', external_ids=('c4', 'c3', 'c2', 'c1'))


@pytest.fixture
def export(OtherSession):
    """A second annotator annotates c1 and c2 in the other database; this is the export of that run."""
    with OtherSession() as session:
        annotation_run = start(session, 'Ben')
        run_id = annotation_run.id
        store.save_annotation(session, annotation_run, post_id(session, 'c1'), 'mixed', CRITERIA)
        store.save_annotation(session, annotation_run, post_id(session, 'c2'), 'contra', [])
        return transfer.export_run(session, run_id)


def test_export_identifies_posts_by_external_ids(export):
    assert export['format_version'] == 1
    assert export['sample'] == 'e-auto-3' and export['annotator'] == 'Ben'
    assert export['prompt'] == {'name': PROMPT, 'hash': prompts.text_hash(PROMPT_TEXT)}
    assert export['code_version'] == 'test-version'
    assert datetime.fromisoformat(export['started_at']).utcoffset().total_seconds() == 0  # UTC is written out
    assert export['annotations'] == [
        {'container_external_id': 'video1', 'post_external_id': 'c1', 'stance': 'mixed', 'criteria': CRITERIA},
        {'container_external_id': 'video1', 'post_external_id': 'c2', 'stance': 'contra', 'criteria': []},
    ]


def test_export_then_import_gives_the_same_run(export, Session, OtherSession, tmp_path):
    file = tmp_path / 'exports' / 'run.json'
    transfer.save_export(export, file)
    with Session() as session, OtherSession() as other_session:
        assert post_id(session, 'c1') != post_id(other_session, 'c1')
        run_id = transfer.import_run(session, transfer.load_export(file))
    with Session() as session:
        assert transfer.export_run(session, run_id) == export
        annotation_run = store.load_human_run(session, run_id)
        assert annotation_run.human_run.annotator == 'Ben'
        assert session.get(LlmRun, run_id) is None
        assert [post.external_id for post in open_posts(session, annotation_run)] == ['c3']  # can be continued


def test_import_refuses_a_run_that_is_already_there(export, Session, OtherSession):
    with Session() as session:
        transfer.import_run(session, export)
        with pytest.raises(ValueError, match='already in the database'):
            transfer.import_run(session, export)
        assert count(session, AnnotationRun) == 1 and count(session, Annotation) == 2
    with OtherSession() as session:  # the database it was exported from has it as well
        with pytest.raises(ValueError, match='already in the database'):
            transfer.import_run(session, export)


def changed(export, edit_first_annotation=None, **top_level):
    """A copy of the export with top-level keys replaced or its first annotation edited by a function."""
    export = copy.deepcopy(export)
    export.update(top_level)
    if edit_first_annotation:
        edit_first_annotation(export['annotations'][0])
    return export


@pytest.mark.parametrize('changes, message', [
    ({'format_version': 2}, 'format_version'),
    ({'sample': 'e-auto-300'}, 'Unknown sample'),
    ({'prompt': {'name': PROMPT, 'hash': 'abc'}}, 'another text'),
    ({'prompt': {'name': 'missing_v9', 'hash': 'abc'}}, 'neither stored'),
    ({'annotator': ' '}, 'no annotator'),
    ({'started_at': 'yesterday'}, 'invalid started_at'),
    ({'edit_first_annotation': lambda annotation: annotation.update(post_external_id='c4')}, 'not part of sample'),
    ({'edit_first_annotation': lambda annotation: annotation.update(post_external_id='c2')}, 'appears twice'),
    ({'edit_first_annotation': lambda annotation: annotation.update(stance='dafuer')}, 'invalid stance'),
    ({'edit_first_annotation': lambda annotation: annotation['criteria'][0].update(sentiment='positiv')},
     'invalid sentiment'),
    ({'edit_first_annotation': lambda annotation: annotation['criteria'][0].update(quote='steht nicht im Text')},
     'does not occur'),
])
def test_import_checks_fail_and_write_nothing(export, Session, changes, message):
    with Session() as session:
        with pytest.raises(ValueError, match=message):
            transfer.import_run(session, changed(export, **changes))
        session.commit()  # even a caller that commits afterwards stores nothing
    with Session() as session:
        for model in [AnnotationRun, HumanRun, Annotation, CriterionMention, PromptVersion]:
            assert count(session, model) == 0


# --- the web server --------------------------------------------------------------------------

def test_page_loads_nothing_from_the_internet():
    page = (STATIC_DIR / 'index.html').read_text(encoding='utf-8')
    assert 'http://' not in page and 'https://' not in page and '="//' not in page


def test_routes_show_save_and_report_errors(Session):
    with Session() as session:
        run_id, c1 = start(session).id, post_id(session, 'c1')
    client = create_app(Session, run_id).test_client()

    assert 'RADAR Annotation' in client.get('/').get_data(as_text=True)
    state = client.get('/api/run').get_json()
    assert state['annotator'] == 'Anna' and state['annotated_ids'] == []
    assert client.get(f'/api/posts/{c1}').get_json()['annotation'] is None

    saved = client.put(f'/api/posts/{c1}', json={'stance': 'mixed', 'criteria': CRITERIA})
    assert saved.status_code == 200
    assert client.get(f'/api/posts/{c1}').get_json()['annotation'] == {'stance': 'mixed', 'criteria': CRITERIA}
    assert client.get('/api/run').get_json()['annotated_ids'] == [c1]

    assert client.get('/api/labels').get_json() == [{'label': 'Ladekosten', 'count': 1},
                                                    {'label': 'Reichweite', 'count': 1}]
    similar = client.get('/api/similar-label', query_string={'label': 'Reichwiete'}).get_json()
    assert similar == {'similar': {'label': 'Reichweite', 'count': 1}}
    assert client.get('/api/similar-label', query_string={'label': 'Reichweite'}).get_json() == {'similar': None}
    assert client.get('/api/similar-label').get_json() == {'similar': None}

    invalid = client.put(f'/api/posts/{c1}', json={'stance': 'dafuer', 'criteria': []})
    assert invalid.status_code == 400 and 'invalid stance' in invalid.get_json()['error']
    assert client.get('/api/posts/999').status_code == 400

    with Session() as session:
        session.add(MappingRun(method='hdbscan', params={}, code_version='x', created_at=datetime.now(timezone.utc),
                               annotation_runs=[store.load_human_run(session, run_id)]))
        session.commit()
    locked = client.put(f'/api/posts/{c1}', json={'stance': 'pro', 'criteria': []})
    assert locked.status_code == 409 and 'mapping run' in locked.get_json()['error']
