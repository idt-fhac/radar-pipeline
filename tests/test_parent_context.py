# The comment a reply stands below as context: for the LLM (context strategy
# 'text+video_title+parent') and for humans (the annotation tool shows it).

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from radar.annotation import prompts, run_llm
from radar.annotation.human import store
from radar.annotation.human.app import STATIC_DIR, create_app
from radar.annotation.run_llm import CONTEXT_STRATEGIES, build_items, load_run, open_posts, parent_comment
from radar.db.models.annotation import Sample
from radar.db.models.collection import Post

PROMPT = 'extraction_test'
TOP = 'Die Reichweite reicht im Alltag völlig.'
REPLY = '@3fa9c0e1b2d4 im Winter aber nicht.'
OTHER = 'Der Strom kommt doch eh aus Kohle.'


@pytest.fixture
def Session(database, Corpus, tmp_path, monkeypatch):
    """A video with a top-level comment, a reply to it and another top-level comment.

    The sample 'thread' holds the three comments and, to cover that case too, the video itself.
    """
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / f'{PROMPT}.txt').write_text('Kommentare zum Thema „{THEMA}“.', encoding='utf-8')
    monkeypatch.setattr(prompts, 'ROOT', tmp_path)
    monkeypatch.setattr(run_llm, 'get_code_version', lambda allow_dirty=False: 'test-version')
    monkeypatch.setattr(store, 'get_code_version', lambda allow_dirty=False: 'test-version')
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1', title='Verbrenner-Aus: Was nun?')
        top = corpus.comment(video, TOP, external_id='top')
        reply = corpus.comment(top, REPLY, external_id='reply')
        other = corpus.comment(video, OTHER, external_id='other')
        session.add(Sample(name='thread', topic=corpus.topic, definition={}, code_version='test-version',
                           created_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
                           posts=[video, top, reply, other], subsets=[]))
        session.commit()
    return database


def by_external_id(posts):
    return {post.external_id: post for post in posts}


def post_id(session, external_id):
    return session.scalar(select(Post.id).where(Post.external_id == external_id))


# --- which post has a parent comment ---------------------------------------------------------

def test_only_a_reply_has_a_parent_comment(Session):
    with Session() as session:
        annotation_run = run_llm.start_llm_run(session, 'thread', PROMPT)
        posts = by_external_id(open_posts(session, annotation_run))  # loads the parents
        assert parent_comment(posts['reply']) is posts['top']
        assert parent_comment(posts['top']) is None    # its parent is the video, not a comment
        assert parent_comment(posts['other']) is None
        assert parent_comment(posts['v1']) is None     # the video itself has no parent at all


# --- LLM runs --------------------------------------------------------------------------------

def test_parent_strategy_adds_the_parent_comment_to_replies_only(Session):
    with Session() as session:
        annotation_run = run_llm.start_llm_run(session, 'thread', PROMPT)
        posts = open_posts(session, annotation_run)
        items = {post.external_id: item for post, item in zip(posts, build_items(posts, 'text+video_title+parent'))}

        assert items['reply'] == {'id': items['reply']['id'], 'video': 'Verbrenner-Aus: Was nun?',
                                  'antwort_auf': TOP, 'text': REPLY}
        assert list(items['reply']) == ['id', 'video', 'antwort_auf', 'text']  # the comment itself comes last
        for external_id in ['top', 'other', 'v1']:
            assert list(items[external_id]) == ['id', 'video', 'text']


@pytest.mark.parametrize('context_strategy', ['text', 'text+video_title'])
def test_other_strategies_never_send_the_parent_comment(Session, context_strategy):
    with Session() as session:
        annotation_run = run_llm.start_llm_run(session, 'thread', PROMPT)
        items = build_items(open_posts(session, annotation_run), context_strategy)
        assert all('antwort_auf' not in item for item in items)


def test_run_with_parent_strategy_sends_and_stores_it(Session, monkeypatch):
    sent = []

    def annotate_batch(system_prompt, items, **settings):
        sent.extend(items)
        answer = {'ergebnisse': [{'id': item['id'], 'position': 'unklar', 'kriterien': []} for item in items]}
        return {'model': 'fake-model', 'raw': {'model': 'fake-model'}, 'content': json.dumps(answer),
                'parsed': answer, 'error': None}
    monkeypatch.setattr(run_llm, 'annotate_batch', annotate_batch)

    with Session() as session:
        run_id = run_llm.run_llm(session, 'thread', PROMPT, context_strategy='text+video_title+parent',
                                 workers=1).id
        assert load_run(session, run_id).llm_run.context_strategy == 'text+video_title+parent'
        reply_id = post_id(session, 'reply')
    assert [item.get('antwort_auf') for item in sent if item['id'] == reply_id] == [TOP]
    assert sum('antwort_auf' in item for item in sent) == 1


def test_parent_strategy_is_a_known_context_strategy():
    assert CONTEXT_STRATEGIES == ['text', 'text+video_title', 'text+video_title+parent']


# --- the annotation tool ---------------------------------------------------------------------

def test_tool_gets_the_parent_comment_of_a_reply(Session):
    with Session() as session:
        annotation_run, created = store.start_human_run(session, 'thread', PROMPT, 'Anna')
        reply = store.get_post(session, annotation_run, post_id(session, 'reply'))
        assert (reply['text'], reply['parent_text']) == (REPLY, TOP)
        assert reply['video_title'] == 'Verbrenner-Aus: Was nun?'
        for external_id in ['top', 'other', 'v1']:
            assert store.get_post(session, annotation_run, post_id(session, external_id))['parent_text'] is None


def test_quote_of_a_reply_must_come_from_the_reply_not_from_its_parent(Session):
    with Session() as session:
        annotation_run, created = store.start_human_run(session, 'thread', PROMPT, 'Anna')
        reply_id = post_id(session, 'reply')
        from_parent = [{'label': 'Reichweite', 'sentiment': 'positive', 'quote': 'reicht im Alltag'}]
        with pytest.raises(ValueError, match='does not occur in the text'):
            store.save_annotation(session, annotation_run, reply_id, 'contra', from_parent)
        from_reply = [{'label': 'Reichweite', 'sentiment': 'negative', 'quote': 'im Winter aber nicht'}]
        store.save_annotation(session, annotation_run, reply_id, 'contra', from_reply)


def test_route_and_page_show_the_parent_comment(Session):
    with Session() as session:
        annotation_run, created = store.start_human_run(session, 'thread', PROMPT, 'Anna')
        run_id, reply_id, top_id = annotation_run.id, post_id(session, 'reply'), post_id(session, 'top')
    client = create_app(Session, run_id).test_client()
    assert client.get(f'/api/posts/{reply_id}').get_json()['parent_text'] == TOP
    assert client.get(f'/api/posts/{top_id}').get_json()['parent_text'] is None

    page = (STATIC_DIR / 'index.html').read_text(encoding='utf-8')
    assert 'id="parent-text"' in page and 'post.parent_text' in page
