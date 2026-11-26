# store.py — reads and writes human annotations. Knows no web server: a session and plain
# values in, ORM objects or plain dicts out, so everything here can be tested without Flask.

from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from radar.annotation import parse
from radar.annotation.llm_client import render_prompt
from radar.annotation.prompts import get_or_create_prompt_version
from radar.annotation.run_llm import open_posts
from radar.db.models.annotation import Annotation, AnnotationRun, CriterionMention, HumanRun, Sample
from radar.db.models.collection import Post
from radar.db.models.mapping import mapping_input
from radar.provenance import get_code_version

# The same values the LLM runs store (parse.py translates the German words of the prompt).
STANCES = list(parse.STANCES.values())
SENTIMENTS = list(parse.SENTIMENTS.values())


class RunLocked(ValueError):
    """The run is input of a mapping run and must not change any more."""


def start_human_run(session, sample_name, prompt_name, annotator, allow_dirty=False):
    """Return the human run of this annotator for a sample and a prompt, creating it on first use.

    (sample, prompt, annotator) identifies the run, so starting the tool again continues it.
    A new run is committed. Human runs are created only here and by the import
    (transfer.py): that keeps a run from being both llm_run and human_run.
    :return: (AnnotationRun as loaded by load_human_run, True if it was created)
    """
    annotator = annotator.strip()
    if not annotator:
        raise ValueError('the annotator needs a name')
    sample = session.scalar(select(Sample).where(Sample.name == sample_name))
    if sample is None:
        raise ValueError(f'Unknown sample_name: {sample_name}')
    prompt_version = get_or_create_prompt_version(session, prompt_name)
    session.flush()  # a prompt version created just now gets its id here

    existing = session.scalars(
        select(AnnotationRun.id)
        .join(AnnotationRun.human_run)
        .where(AnnotationRun.sample_id == sample.id,
               AnnotationRun.prompt_version_id == prompt_version.id,
               HumanRun.annotator == annotator)
        .order_by(AnnotationRun.id.desc())  # several are only possible after an import: take the newest
    ).first()
    if existing is not None:
        return load_human_run(session, existing), False

    annotation_run = AnnotationRun(sample_id=sample.id, prompt_version=prompt_version,
                                   code_version=get_code_version(allow_dirty),
                                   started_at=datetime.now(timezone.utc))
    annotation_run.human_run = HumanRun(annotator=annotator)
    session.add(annotation_run)
    session.commit()
    return load_human_run(session, annotation_run.id), True


def load_human_run(session, run_id):
    """Load an AnnotationRun with human_run, prompt_version and sample.topic; raises for an LLM run."""
    annotation_run = session.scalar(
        select(AnnotationRun)
        .where(AnnotationRun.id == run_id)
        .options(selectinload(AnnotationRun.human_run),
                 selectinload(AnnotationRun.prompt_version),
                 selectinload(AnnotationRun.sample).selectinload(Sample.topic)))
    if annotation_run is None:
        raise ValueError(f'unknown annotation run {run_id}')
    if annotation_run.human_run is None:
        raise ValueError(f'annotation run {run_id} is not a human run')
    return annotation_run


def mapping_run_ids(session, annotation_run):
    """Ids of the mapping runs this run is input of; a run with any of them is locked."""
    return list(session.scalars(
        select(mapping_input.c.mapping_run_id)
        .where(mapping_input.c.annotation_run_id == annotation_run.id)
        .order_by(mapping_input.c.mapping_run_id)))


def run_state(session, annotation_run):
    """Everything the tool shows about the run itself, as a plain dict.

    post_ids are all posts of the sample in the order they are annotated (by post.id);
    annotated_ids are those that already have an annotation in this run.
    """
    post_ids = list(session.scalars(
        select(Post.id).join(Post.samples)
        .where(Sample.id == annotation_run.sample_id).order_by(Post.id)))
    still_open = {post.id for post in open_posts(session, annotation_run)}
    topic = annotation_run.sample.topic.name
    return {
        'run_id': annotation_run.id,
        'annotator': annotation_run.human_run.annotator,
        'sample': annotation_run.sample.name,
        'prompt': annotation_run.prompt_version.name,
        'topic': topic,
        # humans read the very text the LLM gets as system prompt
        'guideline': render_prompt(annotation_run.prompt_version.text, topic),
        'post_ids': post_ids,
        'annotated_ids': [post_id for post_id in post_ids if post_id not in still_open],
        'locked_by': mapping_run_ids(session, annotation_run),
    }


def get_sample_post(session, annotation_run, post_id):
    """The post with its container, or ValueError if it is not part of the run's sample."""
    post = session.scalar(
        select(Post).join(Post.samples)
        .where(Post.id == post_id, Sample.id == annotation_run.sample_id)
        .options(selectinload(Post.container)))
    if post is None:
        raise ValueError(f'post {post_id} is not part of sample {annotation_run.sample.name!r}')
    return post


def get_post(session, annotation_run, post_id):
    """One post as the tool shows it: text, video title and its annotation in THIS run (or None).

    Annotations of other runs are never read here: the annotator must not see what an LLM
    or another person answered.
    """
    post = get_sample_post(session, annotation_run, post_id)
    annotation = session.scalar(
        select(Annotation)
        .where(Annotation.run_id == annotation_run.id, Annotation.post_id == post_id)
        .options(selectinload(Annotation.criterion_mentions)))
    result = {'id': post.id, 'text': post.text, 'video_title': post.container.title, 'annotation': None}
    if annotation is not None:
        mentions = sorted(annotation.criterion_mentions, key=lambda mention: mention.id)
        result['annotation'] = {
            'stance': annotation.stance,
            'criteria': [{'label': mention.label, 'sentiment': mention.sentiment, 'quote': mention.quote}
                         for mention in mentions],
        }
    return result


def clean_annotation(text, stance, criteria):
    """Check one annotation against the text of its post; return the criteria as clean dicts.

    The rules of the LLM runs (parse.py), but strict: a human can correct the input, so
    nothing is dropped silently. Each label once per post (case-insensitive); a quote is
    optional (None) but must occur word for word in the text.
    :param criteria: list of {'label', 'sentiment', 'quote'}; [] means "no criteria"
    :return: list of {'label', 'sentiment', 'quote'} with stripped values
    """
    if stance not in STANCES:
        raise ValueError(f'invalid stance {stance!r}, use one of: {", ".join(STANCES)}')
    if not isinstance(criteria, list):
        raise ValueError('criteria must be a list')
    mentions = []
    seen = set()
    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise ValueError(f'criterion is not an object: {criterion!r}')
        label = criterion.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('criterion without label')
        label = label.strip()
        if label.lower() in seen:
            raise ValueError(f'criterion {label!r} twice')
        seen.add(label.lower())

        sentiment = criterion.get('sentiment')
        if sentiment not in SENTIMENTS:
            raise ValueError(f'invalid sentiment {sentiment!r} for {label!r}, '
                             f'use one of: {", ".join(SENTIMENTS)}')

        quote = criterion.get('quote')
        if quote is not None and not isinstance(quote, str):
            raise ValueError(f'quote of {label!r} is not a text')
        quote = quote.strip() if quote and quote.strip() else None
        if quote is not None and quote not in text:
            raise ValueError(f'quote of {label!r} does not occur in the text: {quote!r}')

        mentions.append({'label': label, 'sentiment': sentiment, 'quote': quote})
    return mentions


def save_annotation(session, annotation_run, post_id, stance, criteria):
    """Store the annotation of one post, replacing an earlier one, and commit.

    Annotation and criteria are replaced together in one transaction. An empty criteria
    list is a result of its own ("no criteria"), not a missing annotation.
    Raises RunLocked once the run is input of a mapping run: `assignment` points to its
    criterion mentions, and the mapping would no longer describe the run.
    """
    locked_by = mapping_run_ids(session, annotation_run)
    if locked_by:
        raise RunLocked(f'annotation run {annotation_run.id} is input of mapping run(s) '
                        f'{", ".join(map(str, locked_by))} and cannot be changed any more')
    post = get_sample_post(session, annotation_run, post_id)
    mentions = clean_annotation(post.text, stance, criteria)

    # mentions first: they reference the annotation
    session.execute(delete(CriterionMention).where(CriterionMention.run_id == annotation_run.id,
                                                   CriterionMention.post_id == post_id))
    session.execute(delete(Annotation).where(Annotation.run_id == annotation_run.id,
                                             Annotation.post_id == post_id))
    session.add(Annotation(
        run_id=annotation_run.id,
        post_id=post_id,
        stance=stance,
        criterion_mentions=[CriterionMention(run_id=annotation_run.id, post_id=post_id, **mention)
                            for mention in mentions],
    ))
    session.commit()
