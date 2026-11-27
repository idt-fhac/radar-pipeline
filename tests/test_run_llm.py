import json
import logging
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from radar.annotation import prompts, run_llm
from radar.annotation.llm_client import DEFAULT_PROVIDER, PROVIDERS
from radar.annotation.run_llm import (build_items, check_reported_model, load_run, make_batches,
                                      next_batch_number, open_posts, process_run, resume_run,
                                      start_llm_run)
from radar.db.models.annotation import (Annotation, AnnotationRun, Batch, BatchPost, CriterionMention,
                                        HumanRun, LlmRun, PromptVersion, Sample)
from radar.db.models.collection import Post

PROMPT = 'extraction_test'
POSTS = 5


@pytest.fixture
def Session(database, Corpus, tmp_path, monkeypatch):
    """A database with one video, POSTS comments and the sample 's5' of all of them.

    The prompt comes from a temporary folder and the code version is fixed, so the tests
    depend neither on prompts/ nor on the state of the Git tree.
    """
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / f'{PROMPT}.txt').write_text('Kommentare zum Thema „{THEMA}“.', encoding='utf-8')
    monkeypatch.setattr(prompts, 'ROOT', tmp_path)
    monkeypatch.setattr(run_llm, 'get_code_version', lambda allow_dirty=False: 'test-version')
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1', title='Verbrenner-Aus: Was nun?')
        comments = [corpus.comment(video, f'Kommentar {number}') for number in range(1, POSTS + 1)]
        session.add(Sample(name='s5', topic=corpus.topic, definition={}, code_version='test-version',
                           created_at=datetime(2026, 11, 2, tzinfo=timezone.utc), posts=comments, subsets=[]))
        session.commit()
    return database


class FakeLLM:
    """Stands in for llm_client.annotate_batch; `answer(items, call_number)` decides what comes back.

    answer returns the list for "ergebnisse", a string (sent back as it is, e.g. broken JSON)
    or None (the request fails). Calls are recorded with their settings.
    """

    def __init__(self, answer=None, model='fake-model'):
        self.answer = answer or (lambda items, call_number: [good(item['id']) for item in items])
        self.model = model
        self.calls = []

    def __call__(self, system_prompt, items, **settings):
        self.calls.append({'system_prompt': system_prompt, 'items': items, **settings})
        answer = self.answer(items, len(self.calls))
        if answer is None:
            return {'model': None, 'raw': None, 'content': None, 'parsed': None,
                    'error': 'APIConnectionError: Connection error.'}
        content = answer if isinstance(answer, str) else json.dumps({'ergebnisse': answer})
        raw = {'model': self.model, 'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
        try:
            return {'model': self.model, 'raw': raw, 'content': content, 'parsed': json.loads(content), 'error': None}
        except ValueError:
            return {'model': self.model, 'raw': raw, 'content': content, 'parsed': None, 'error': 'invalid JSON'}

    def sent_ids(self):
        """Post ids per call, in the order of the calls."""
        return [[item['id'] for item in call['items']] for call in self.calls]


def good(post_id, criteria=(('Preis', 'positiv', 'billig'),)):
    return {'id': post_id, 'position': 'dafuer', 'kriterien': [
        {'kriterium': label, 'sentiment': sentiment, 'zitat': quote} for label, sentiment, quote in criteria]}


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeLLM in run_llm; returns install(answer=None, model='fake-model')."""
    def install(answer=None, **kwargs):
        llm = FakeLLM(answer, **kwargs)
        monkeypatch.setattr(run_llm, 'annotate_batch', llm)
        return llm
    return install


def run(session, **options):
    """A complete run on 's5' with batches of two, one request at a time (deterministic order)."""
    options = {'batch_size': 2, 'workers': 1, **options}
    return run_llm.run_llm(session, 's5', PROMPT, **options)


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def statuses(session, run_id):
    return list(session.scalars(select(Batch.status).where(Batch.run_id == run_id).order_by(Batch.number)))


def post_ids(session):
    return list(session.scalars(select(Post.id).where(Post.parent_id.is_not(None)).order_by(Post.id)))


# --- make_batches and build_items ------------------------------------------------------------

@pytest.mark.parametrize('items, size, expected', [
    ([], 2, []),
    ([1], 20, [[1]]),
    ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
    ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
    ([1, 2, 3], 1, [[1], [2], [3]]),
])
def test_make_batches(items, size, expected):
    assert make_batches(items, size) == expected


def test_build_items_follows_the_context_strategy(Session):
    with Session() as session:
        annotation_run = start_llm_run(session, 's5', PROMPT)
        posts = open_posts(session, annotation_run)[:2]
        ids = [post.id for post in posts]

        assert build_items(posts, 'text') == [
            {'id': ids[0], 'text': 'Kommentar 1'}, {'id': ids[1], 'text': 'Kommentar 2'}]
        assert build_items(posts, 'text+video_title') == [
            {'id': ids[0], 'video': 'Verbrenner-Aus: Was nun?', 'text': 'Kommentar 1'},
            {'id': ids[1], 'video': 'Verbrenner-Aus: Was nun?', 'text': 'Kommentar 2'}]
        with pytest.raises(ValueError, match='Unknown context_strategy'):
            build_items(posts, 'text+everything')


# --- start_llm_run ---------------------------------------------------------------------------

def test_start_stores_the_run_with_all_its_settings(Session):
    with Session() as session:
        run_id = start_llm_run(session, 's5', PROMPT, temperature=0.3, batch_size=7,
                               context_strategy='text', thinking=False, json_mode=True).id
    with Session() as session:
        annotation_run = load_run(session, run_id)
        llm_run = annotation_run.llm_run
        assert annotation_run.sample.name == 's5'
        assert annotation_run.prompt_version.name == PROMPT
        assert annotation_run.code_version == 'test-version'
        assert llm_run.model == PROVIDERS[DEFAULT_PROVIDER]['default_model']
        assert (llm_run.temperature, llm_run.batch_size, llm_run.context_strategy) == (0.3, 7, 'text')
        assert llm_run.params == {'provider': DEFAULT_PROVIDER, 'base_url': PROVIDERS[DEFAULT_PROVIDER]['base_url'],
                                  'thinking': False, 'json_mode': True}
        assert session.get(HumanRun, run_id) is None  # a run is never both
        assert count(session, Batch) == 0  # nothing is sent yet


def test_start_takes_an_explicit_model(Session):
    with Session() as session:
        annotation_run = start_llm_run(session, 's5', PROMPT, model='other-model')
        assert load_run(session, annotation_run.id).llm_run.model == 'other-model'


def test_two_runs_share_the_prompt_version(Session):
    with Session() as session:
        start_llm_run(session, 's5', PROMPT)
        start_llm_run(session, 's5', PROMPT)
        assert count(session, AnnotationRun) == 2 and count(session, PromptVersion) == 1


def test_start_refuses_unknown_sample_and_context_strategy(Session):
    with Session() as session:
        with pytest.raises(ValueError, match='Unknown sample_name: nope'):
            start_llm_run(session, 'nope', PROMPT)
        with pytest.raises(ValueError, match='Unknown context_strategy'):
            start_llm_run(session, 's5', PROMPT, context_strategy='text+everything')
        assert count(session, AnnotationRun) == 0 and count(session, LlmRun) == 0


# --- a complete run --------------------------------------------------------------------------

def test_run_annotates_every_post_of_the_sample(Session, fake):
    llm = fake()
    with Session() as session:
        run_id = run(session).id
    with Session() as session:
        ids = post_ids(session)
        assert llm.sent_ids() == [ids[0:2], ids[2:4], ids[4:5]]
        assert statuses(session, run_id) == ['done', 'done', 'done']
        assert count(session, Annotation) == POSTS and count(session, CriterionMention) == POSTS
        mention = session.scalar(select(CriterionMention).where(CriterionMention.post_id == ids[0]))
        assert (mention.run_id, mention.label, mention.sentiment, mention.quote) == (
            run_id, 'Preis', 'positive', 'billig')
        assert session.get(Annotation, (run_id, ids[0])).stance == 'pro'
        assert open_posts(session, load_run(session, run_id)) == []


def test_run_sends_the_rendered_prompt_and_the_stored_settings(Session, fake):
    llm = fake()
    with Session() as session:
        run(session, model='some-model', temperature=0.2, thinking=False, json_mode=True,
            context_strategy='text+video_title')
    call = llm.calls[0]
    assert call['system_prompt'] == 'Kommentare zum Thema „E-Auto“.'
    assert call['items'][0]['video'] == 'Verbrenner-Aus: Was nun?'
    assert {key: call[key] for key in ['provider', 'model', 'temperature', 'thinking', 'json_mode']} == {
        'provider': DEFAULT_PROVIDER, 'model': 'some-model', 'temperature': 0.2, 'thinking': False, 'json_mode': True}


def test_batches_record_what_was_sent_and_what_came_back(Session, fake):
    fake()
    with Session() as session:
        run_id = run(session).id
    with Session() as session:
        ids = post_ids(session)
        batches = session.scalars(select(Batch).where(Batch.run_id == run_id).order_by(Batch.number)
                                  .options(selectinload(Batch.batch_posts))).all()
        assert [batch.number for batch in batches] == [1, 2, 3]
        sent = [[(batch_post.position, batch_post.post_id)
                 for batch_post in sorted(batch.batch_posts, key=lambda batch_post: batch_post.position)]
                for batch in batches]
        assert sent == [[(1, ids[0]), (2, ids[1])], [(1, ids[2]), (2, ids[3])], [(1, ids[4])]]
        assert batches[0].raw_response['model'] == 'fake-model'
        assert 'ergebnisse' in batches[0].raw_response['choices'][0]['message']['content']


def test_post_without_criteria_gets_an_empty_annotation(Session, fake):
    fake(lambda items, call_number: [good(item['id'], criteria=()) for item in items])
    with Session() as session:
        run_id = run(session).id
        assert count(session, Annotation) == POSTS and count(session, CriterionMention) == 0
        assert open_posts(session, load_run(session, run_id)) == []  # annotated, not open


def test_several_criteria_per_post_are_stored_in_order(Session, fake):
    fake(lambda items, call_number: [good(item['id'], [('Preis', 'positiv', 'billig'), ('Reichweite', 'negativ', None)])
                                     for item in items])
    with Session() as session:
        run(session)
        first = post_ids(session)[0]
        mentions = session.scalars(select(CriterionMention).where(CriterionMention.post_id == first)
                                   .order_by(CriterionMention.id)).all()
        assert [(mention.label, mention.sentiment, mention.quote) for mention in mentions] == [
            ('Preis', 'positive', 'billig'), ('Reichweite', 'negative', None)]


def test_parallel_requests_store_every_batch_once(Session, fake):
    fake()
    with Session() as session:
        run_id = run(session, batch_size=1, workers=3).id
        assert sorted(statuses(session, run_id)) == ['done'] * POSTS
        assert list(session.scalars(select(Batch.number).order_by(Batch.number))) == list(range(1, POSTS + 1))
        assert count(session, Annotation) == POSTS


# --- answers that are incomplete, broken or missing ------------------------------------------

def test_skipped_post_makes_the_batch_partial_and_is_sent_again(Session, fake):
    def skip_first_post_once(items, call_number):
        return [good(item['id']) for item in items][1 if call_number == 1 else 0:]
    llm = fake(skip_first_post_once)
    with Session() as session:
        run_id = run(session).id
        ids = post_ids(session)
        assert llm.sent_ids() == [ids[0:2], ids[2:4], ids[4:5], [ids[0]]]
        assert statuses(session, run_id) == ['partial', 'done', 'done', 'done']  # numbers go on in round 2
        assert count(session, Annotation) == POSTS


def test_failed_request_is_an_error_without_response_and_is_sent_again(Session, fake):
    llm = fake(lambda items, call_number: None if call_number == 1 else [good(item['id']) for item in items])
    with Session() as session:
        run_id = run(session).id
        ids = post_ids(session)
        assert statuses(session, run_id) == ['error', 'done', 'done', 'done']
        assert llm.sent_ids()[3] == ids[0:2]
        # SQL NULL, not the JSON text 'null': "raw_response IS NULL" must find failed batches
        failed = session.scalars(select(Batch.number).where(Batch.raw_response.is_(None))).all()
        assert failed == [1]
        assert count(session, Annotation) == POSTS


def test_answer_that_is_no_json_is_invalid_but_kept(Session, fake):
    fake(lambda items, call_number: 'Hier sind die Ergebnisse: {' if call_number == 1
         else [good(item['id']) for item in items])
    with Session() as session:
        run_id = run(session).id
        assert statuses(session, run_id) == ['invalid', 'done', 'done', 'done']
        first = session.scalar(select(Batch).where(Batch.number == 1))
        assert first.raw_response['choices'][0]['message']['content'] == 'Hier sind die Ergebnisse: {'
        assert count(session, Annotation) == POSTS


def test_valid_json_without_usable_entries_is_invalid(Session, fake):
    fake(lambda items, call_number: [{'id': item['id'], 'position': 'vielleicht', 'kriterien': []} for item in items]
         if call_number == 1 else [good(item['id']) for item in items])
    with Session() as session:
        run_id = run(session).id
        assert statuses(session, run_id)[0] == 'invalid'
        assert count(session, Annotation) == POSTS


def test_run_stops_after_max_rounds_and_can_be_resumed(Session, fake, caplog):
    llm = fake(lambda items, call_number: None)
    with Session() as session:
        with caplog.at_level(logging.WARNING, logger='radar.annotation.run_llm'):
            run_id = run(session, max_rounds=2).id
        assert statuses(session, run_id) == ['error'] * 6  # 3 batches in each of 2 rounds
        assert count(session, Annotation) == 0
        assert f'run {run_id}: {POSTS} posts still without annotation after 2 rounds' in caplog.text

    working = fake()
    with Session() as session:
        resume_run(session, run_id, workers=1)
        assert len(llm.calls) == 6 and len(working.calls) == 3
        assert statuses(session, run_id) == ['error'] * 6 + ['done'] * 3  # batch numbers continue
        assert count(session, Annotation) == POSTS


def test_resume_sends_only_the_open_posts(Session, fake):
    fake(lambda items, call_number: [good(item['id']) for item in items] if call_number == 1 else None)
    with Session() as session:
        run_id = run(session, max_rounds=1).id
        ids = post_ids(session)
        assert count(session, Annotation) == 2

    llm = fake()
    with Session() as session:
        resume_run(session, run_id, workers=1)
        assert llm.sent_ids() == [ids[2:4], ids[4:5]]
        assert count(session, Annotation) == POSTS


def test_resume_of_a_complete_run_sends_nothing(Session, fake):
    fake()
    with Session() as session:
        run_id = run(session).id
    llm = fake()
    with Session() as session:
        resume_run(session, run_id)
        assert llm.calls == []
        assert count(session, Batch) == 3


def test_resume_uses_the_settings_stored_with_the_run(Session, fake):
    fake(lambda items, call_number: None)
    with Session() as session:
        run_id = run(session, max_rounds=1, model='stored-model', temperature=0.4, batch_size=3,
                     thinking=False, json_mode=True, context_strategy='text').id
    llm = fake()
    with Session() as session:
        resume_run(session, run_id, workers=1)
    call = llm.calls[0]
    assert (call['model'], call['temperature'], call['thinking'], call['json_mode']) == (
        'stored-model', 0.4, False, True)
    assert len(call['items']) == 3 and 'video' not in call['items'][0]


def test_run_without_json_mode_in_its_params_resumes_without_it(Session, fake):
    with Session() as session:
        annotation_run = start_llm_run(session, 's5', PROMPT)
        run_id = annotation_run.id
        load_run(session, run_id).llm_run.params = {'provider': DEFAULT_PROVIDER}  # as stored by older code
        session.commit()
    llm = fake()
    with Session() as session:
        resume_run(session, run_id, workers=1)
    assert llm.calls[0]['json_mode'] is False
    assert llm.calls[0]['thinking'] is None  # nothing is sent, the provider's default applies


# --- loading a run ---------------------------------------------------------------------------

def test_load_run_refuses_unknown_and_human_runs(Session):
    with Session() as session:
        sample = session.scalar(select(Sample))
        human = AnnotationRun(sample_id=sample.id, code_version='x', started_at=datetime.now(timezone.utc),
                              prompt_version=PromptVersion(name='p', text='t', hash='h'))
        human.human_run = HumanRun(annotator='Anna')
        session.add(human)
        session.commit()

        with pytest.raises(ValueError, match='unknown annotation run 999'):
            load_run(session, 999)
        with pytest.raises(ValueError, match='is not an LLM run'):
            load_run(session, human.id)
        with pytest.raises(ValueError, match='is not an LLM run'):
            resume_run(session, human.id)


def test_open_posts_looks_at_one_run_only(Session, fake):
    fake()
    with Session() as session:
        run(session)  # every post is annotated in this run ...
        other = start_llm_run(session, 's5', PROMPT)
        assert len(open_posts(session, other)) == POSTS  # ... but not in the other one
        assert [post.id for post in open_posts(session, other)] == post_ids(session)  # ordered by id


def test_process_run_can_be_called_on_a_started_run(Session, fake):
    fake()
    with Session() as session:
        annotation_run = start_llm_run(session, 's5', PROMPT, batch_size=5)
        assert next_batch_number(session, load_run(session, annotation_run.id).llm_run) == 1
        process_run(session, annotation_run, workers=1)
        assert count(session, Annotation) == POSTS and count(session, BatchPost) == POSTS
        assert next_batch_number(session, load_run(session, annotation_run.id).llm_run) == 2


# --- the model the API reports ---------------------------------------------------------------

class Stub:
    """Just the attributes check_reported_model reads."""
    def __init__(self, **values):
        self.__dict__.update(values)


def test_reported_model_warns_once_if_it_differs_and_again_if_it_changes(caplog):
    llm_run, batch = Stub(model='qwen-qwen3-8-27b'), Stub(number=4)
    with caplog.at_level(logging.WARNING, logger='radar.annotation.run_llm'):
        assert check_reported_model(llm_run, batch, 'Qwen/Qwen3.8-27B', None) == 'Qwen/Qwen3.8-27B'
        assert "batch 4: answered by 'Qwen/Qwen3.8-27B', requested 'qwen-qwen3-8-27b'" in caplog.text
        caplog.clear()

        assert check_reported_model(llm_run, batch, 'Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-27B') == 'Qwen/Qwen3.8-27B'
        assert caplog.text == ''  # the alias is reported only for the first answer

        assert check_reported_model(llm_run, batch, 'other-model', 'Qwen/Qwen3.8-27B') == 'other-model'
        assert "model changed within the run from 'Qwen/Qwen3.8-27B' to 'other-model'" in caplog.text


def test_reported_model_is_silent_if_it_matches_or_is_missing(caplog):
    llm_run, batch = Stub(model='fake-model'), Stub(number=1)
    with caplog.at_level(logging.WARNING, logger='radar.annotation.run_llm'):
        assert check_reported_model(llm_run, batch, 'fake-model', None) == 'fake-model'
        assert check_reported_model(llm_run, batch, None, 'fake-model') == 'fake-model'  # a failed request
        assert caplog.text == ''
