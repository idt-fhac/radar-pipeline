from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from radar.analysis.quotes import non_verbatim_quotes, quote_fidelity
from radar.db.base import Base
from radar.db.models.annotation import (Annotation, AnnotationRun, CriterionMention, LlmRun, PromptVersion,
                                        Sample)
from radar.db.models.collection import Post

FIRST = 'Die Reichweite im Winter ist ein Witz, aber das Laden zu Hause ist billig.'
SECOND = 'Er sagt „die Infrastruktur" fehlt.\nUnd wer zahlt die Ladesäulen?'


@pytest.fixture
def Session(database, Corpus):
    """Two comments and a function add_run(mentions) -> run id, mentions = [(post, label, quote), ...]."""
    with database() as session:
        corpus = Corpus(session)
        video = corpus.video('v1')
        corpus.comment(video, FIRST, external_id='first')
        corpus.comment(video, SECOND, external_id='second')
        session.add(Sample(name='s', topic=corpus.topic, definition={}, code_version='x',
                           created_at=datetime(2026, 11, 2, tzinfo=timezone.utc), posts=[], subsets=[]))
        session.add(PromptVersion(name='p', text='t', hash='h'))
        session.commit()
    return database


def add_run(session, mentions):
    """An annotation run with the given criterion mentions: [(external id of the post, label, quote), ...]."""
    annotation_run = AnnotationRun(sample_id=session.scalar(select(Sample.id)), code_version='x',
                                   prompt_version_id=session.scalar(select(PromptVersion.id)),
                                   started_at=datetime.now(timezone.utc))
    annotation_run.llm_run = LlmRun(model='m', temperature=0.0, batch_size=20, context_strategy='text', params={})
    session.add(annotation_run)
    session.flush()
    post_ids = dict(session.execute(select(Post.external_id, Post.id)).all())
    for external_id in {external_id for external_id, label, quote in mentions}:
        session.add(Annotation(run_id=annotation_run.id, post_id=post_ids[external_id], stance='pro',
                               criterion_mentions=[
                                   CriterionMention(run_id=annotation_run.id, post_id=post_ids[external_id],
                                                    label=label, quote=quote)
                                   for mention_post, label, quote in mentions if mention_post == external_id]))
    session.commit()
    return annotation_run.id


def test_counts_mentions_quotes_and_verbatim_quotes(Session):
    with Session() as session:
        run_id = add_run(session, [
            ('first', 'Reichweite', 'Die Reichweite im Winter ist ein Witz'),
            ('first', 'Ladekosten', 'das Laden zu Hause ist günstig'),  # paraphrased
            ('first', 'Komfort', None),                                 # no quote at all
            ('second', 'Kosten', 'wer zahlt die Ladesäulen?'),
        ])
        assert quote_fidelity(session) == [
            {'run_id': run_id, 'mentions': 4, 'quotes': 3, 'verbatim': 2, 'share': pytest.approx(2 / 3)}]


@pytest.mark.parametrize('quote, verbatim', [
    (FIRST, True),                                   # the whole comment
    ('ein Witz, aber das Laden', True),
    ('billig.', True),
    ('die reichweite im winter', False),             # another case
    ('Die  Reichweite im Winter', False),            # another blank
    ('Die Reichweite im Winter ist ein Witz.', False),  # punctuation added
    ('Reichweite ... ist ein Witz', False),          # shortened in the middle
])
def test_verbatim_means_the_exact_string_as_for_human_annotators(Session, quote, verbatim):
    with Session() as session:
        add_run(session, [('first', 'Reichweite', quote)])
        assert quote_fidelity(session)[0]['verbatim'] == (1 if verbatim else 0)
        assert (quote in FIRST) is verbatim  # the rule of the annotation tool


def test_special_characters_and_line_breaks_are_compared_as_they_are(Session):
    with Session() as session:
        add_run(session, [
            ('second', 'Infrastruktur', '„die Infrastruktur" fehlt'),
            ('second', 'Kosten', 'fehlt.\nUnd wer zahlt'),
            ('second', 'Infrastruktur 2', '"die Infrastruktur" fehlt'),  # another opening quotation mark
            ('second', 'Kosten 2', 'fehlt. Und wer zahlt'),               # the line break became a blank
        ])
        assert quote_fidelity(session)[0]['verbatim'] == 2


def test_a_quote_must_come_from_its_own_post(Session):
    with Session() as session:
        add_run(session, [('second', 'Reichweite', 'Die Reichweite im Winter ist ein Witz')])  # text of 'first'
        assert quote_fidelity(session)[0]['verbatim'] == 0


def test_each_run_is_counted_on_its_own_and_can_be_selected(Session):
    with Session() as session:
        good = add_run(session, [('first', 'Reichweite', 'Reichweite im Winter'),
                                 ('second', 'Kosten', 'wer zahlt')])
        bad = add_run(session, [('first', 'Reichweite', 'geringe Reichweite'),
                                ('second', 'Kosten', 'wer zahlt')])
        assert [(row['run_id'], row['verbatim'], row['share']) for row in quote_fidelity(session)] == [
            (good, 2, 1.0), (bad, 1, 0.5)]
        assert [row['run_id'] for row in quote_fidelity(session, [bad])] == [bad]
        assert quote_fidelity(session, []) == []
        assert quote_fidelity(session, [999]) == []


def test_run_without_quotes_has_no_share_and_run_without_mentions_no_row(Session):
    with Session() as session:
        no_quotes = add_run(session, [('first', 'Reichweite', None)])
        add_run(session, [])
        assert quote_fidelity(session) == [
            {'run_id': no_quotes, 'mentions': 1, 'quotes': 0, 'verbatim': 0, 'share': None}]


def test_non_verbatim_quotes_lists_what_to_look_at(Session):
    with Session() as session:
        run_id = add_run(session, [
            ('first', 'Reichweite', 'Die Reichweite im Winter ist ein Witz'),
            ('first', 'Ladekosten', 'das Laden zu Hause ist günstig'),
            ('first', 'Komfort', None),
            ('second', 'Kosten', 'wer bezahlt die Ladesäulen'),
        ])
        other = add_run(session, [('first', 'Preis', 'viel zu teuer')])
        first, second = (session.scalar(select(Post.id).where(Post.external_id == external_id))
                         for external_id in ['first', 'second'])

        assert non_verbatim_quotes(session, run_id) == [
            {'post_id': first, 'label': 'Ladekosten', 'quote': 'das Laden zu Hause ist günstig', 'text': FIRST},
            {'post_id': second, 'label': 'Kosten', 'quote': 'wer bezahlt die Ladesäulen', 'text': SECOND},
        ]
        assert [row['label'] for row in non_verbatim_quotes(session, other)] == ['Preis']
        assert non_verbatim_quotes(session, 999) == []


def test_the_measure_needs_no_table_or_column_of_its_own():
    # computed from criterion_mention.quote and post.text; the data model stays as it is
    assert not any('quote' in table or 'fidelity' in table for table in Base.metadata.tables)
    assert 'verbatim' not in CriterionMention.__table__.columns
