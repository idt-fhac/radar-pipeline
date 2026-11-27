
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from radar.annotation.llm_client import PROVIDERS, DEFAULT_PROVIDER, DEFAULT_TEMPERATURE, annotate_batch, render_prompt
from radar.annotation.parse import parse_response
from radar.annotation.prompts import get_or_create_prompt_version
from radar.db.models.annotation import AnnotationRun, Sample, PromptVersion, Annotation, LlmRun, Batch, BatchPost, CriterionMention
from sqlalchemy import select, exists, func
from sqlalchemy.orm import selectinload

from datetime import datetime, timezone

from radar.db.models.collection import Post, Container
from radar.config import SOURCE, setting
from radar.provenance import get_code_version

logger = logging.getLogger(__name__)

BATCH_SIZE = setting('annotation', 'batch_size')
DEFAULT_THINKING = setting('annotation', 'thinking')
DEFAULT_JSON_MODE = setting('annotation', 'json_mode')
MAX_ROUNDS = setting('annotation', 'max_rounds')
WORKERS = setting('annotation', 'workers')

CONTEXT_STRATEGIES = [
    'text',
    'text+video_title',
    'text+video_title+parent',  # replies also get the comment they stand below
]
DEFAULT_CONTEXT_STRATEGY = setting('annotation', 'context_strategy')
if DEFAULT_CONTEXT_STRATEGY not in CONTEXT_STRATEGIES:
    raise ValueError(f'[annotation] context_strategy {DEFAULT_CONTEXT_STRATEGY!r} in {SOURCE} is unknown, '
                     f'use one of: {", ".join(CONTEXT_STRATEGIES)}')

BATCH_STATUSES = ['sent', 'done', 'partial', 'invalid', 'error']

def build_items(posts, context_strategy=DEFAULT_CONTEXT_STRATEGY):
    if context_strategy not in CONTEXT_STRATEGIES:
        raise ValueError(f'Unknown context_strategy: {context_strategy}\nChoose from {CONTEXT_STRATEGIES = }')
    items = []
    for post in posts:
        item = {'id': post.id}
        # context strategies are implemented here
        match context_strategy:
            case 'text+video_title':
                item['video'] = post.container.title
            case 'text+video_title+parent':
                item['video'] = post.container.title
                parent = parent_comment(post)
                if parent is not None:
                    item['antwort_auf'] = parent.text
        item['text'] = post.text
        items.append(item)
    return items


def parent_comment(post):
    """The comment a reply stands below, or None for a post that is no reply.

    The parent of a top-level comment is the root post (on YouTube the video with its
    description as text). It is no context of its own: the title already stands for the video.
    post.parent must be loaded (open_posts does that).
    """
    parent = post.parent
    if parent is None or parent.parent_id is None:
        return None
    return parent


def open_posts(session, annotation_run):
    already_annotated = select(Annotation.post_id).where(Annotation.post_id == Post.id,Annotation.run_id == annotation_run.id)
    stmt = (
        select(Post)
        .join(Post.samples)
        .where(Sample.id == annotation_run.sample_id, ~exists(already_annotated))
        .options(selectinload(Post.container), selectinload(Post.parent))
        .order_by(Post.id)
    )
    posts = session.scalars(stmt).all()
    return posts


def start_llm_run(session, sample_name, prompt_name, provider =DEFAULT_PROVIDER, model = None, temperature = DEFAULT_TEMPERATURE, batch_size = BATCH_SIZE, context_strategy=DEFAULT_CONTEXT_STRATEGY, thinking=DEFAULT_THINKING, json_mode=DEFAULT_JSON_MODE, allow_dirty = False):
    """Create an AnnotationRun with its LlmRun for a sample and a prompt, commit and return it.

    Only creates the run; sending happens in process_run.
    """
    if model is None:
        model = PROVIDERS[provider]['default_model']
    sample = session.scalar(select(Sample).where(Sample.name == sample_name))
    if sample is None:
        raise ValueError(f'Unknown sample_name: {sample_name}')
    if context_strategy not in CONTEXT_STRATEGIES:
        raise ValueError(f'Unknown context_strategy: {context_strategy}\nChoose from {CONTEXT_STRATEGIES = }')
    prompt_version = get_or_create_prompt_version(session, prompt_name)
    annotation_run = AnnotationRun(sample_id = sample.id, prompt_version= prompt_version, code_version = get_code_version(allow_dirty), started_at = datetime.now(timezone.utc))
    # base_url too: the provider name alone does not say which server answered, the config can change
    params = {'provider': provider, 'base_url': PROVIDERS[provider]['base_url'], 'thinking': thinking,
              'json_mode': json_mode}
    llm_run = LlmRun(model=model, temperature = temperature, batch_size = batch_size, context_strategy = context_strategy, params=params)
    annotation_run.llm_run = llm_run
    session.add(annotation_run)
    session.add(llm_run)
    session.commit()
    return annotation_run


def make_batches(items, size=BATCH_SIZE):
    """Split a list into consecutive chunks of at most size elements; [] for an empty list."""
    if not items or len(items) == 0:
        return []
    batches = []
    i = 0
    while len(items)-i*size > size:
        batches.append(items[i*size:(i+1)*size])
        i += 1
    batches.append(items[i*size:])
    return batches

def run_llm(session, sample_name, prompt_name, provider=DEFAULT_PROVIDER, model=None,
            temperature=DEFAULT_TEMPERATURE, batch_size=BATCH_SIZE, context_strategy=DEFAULT_CONTEXT_STRATEGY,
            thinking=DEFAULT_THINKING, json_mode=DEFAULT_JSON_MODE, allow_dirty=False,
            max_rounds=MAX_ROUNDS, workers=WORKERS):
    """Start a new LLM run and process it until done. Convenience for start_llm_run + process_run.

    Parameters as in start_llm_run and process_run.
    :return: the AnnotationRun
    """
    annotation_run = start_llm_run(session, sample_name, prompt_name, provider=provider, model=model,
                                   temperature=temperature, batch_size=batch_size,
                                   context_strategy=context_strategy, thinking=thinking, json_mode=json_mode,
                                   allow_dirty=allow_dirty)
    process_run(session, annotation_run, max_rounds=max_rounds, workers=workers)
    return annotation_run


def resume_run(session, annotation_run_id, *, max_rounds=MAX_ROUNDS, workers=WORKERS):
    """Continue an aborted or incomplete run: sends only the posts that still lack an annotation.

    process_run computes the open posts afresh in every round, so resuming is just processing again.
    :return: the AnnotationRun
    """
    annotation_run = load_run(session, annotation_run_id)
    process_run(session, annotation_run, max_rounds=max_rounds, workers=workers)
    return annotation_run


# --- Processing ---------------------------------------------------------------------------

def load_run(session, run_id):
    """Load an AnnotationRun with everything process_run needs (llm_run, prompt_version, sample.topic).

    Needed because of lazy="raise": these relationships must be loaded explicitly.
    """
    stmt = (
        select(AnnotationRun)
        .where(AnnotationRun.id == run_id)
        .options(
            selectinload(AnnotationRun.llm_run),
            selectinload(AnnotationRun.prompt_version),
            selectinload(AnnotationRun.sample).selectinload(Sample.topic),
        )
    )
    annotation_run = session.scalar(stmt)
    if annotation_run is None:
        raise ValueError(f'unknown annotation run {run_id}')
    if annotation_run.llm_run is None:
        raise ValueError(f'annotation run {run_id} is not an LLM run')
    return annotation_run


def process_run(session, annotation_run, *, max_rounds=MAX_ROUNDS, workers=WORKERS):
    """Send all open posts of a run in batches until every post has an annotation or max_rounds is reached.

    Per round: open_posts -> build_items -> make_batches -> create_batches (commit)
    -> send_batch in parallel threads -> store_result in the main thread (commit per batch).
    Posts still open after max_rounds are logged; the run stays resumable.
    """
    annotation_run = load_run(session, annotation_run.id)
    llm_run = annotation_run.llm_run
    system_prompt = render_prompt(annotation_run.prompt_version.text, annotation_run.sample.topic.name)
    settings = {'provider': llm_run.params['provider'], 'model': llm_run.model,
                'temperature': llm_run.temperature, 'thinking': llm_run.params.get('thinking'),
                # older runs have no json_mode in params: they ran without it, so resume them without it
                'json_mode': llm_run.params.get('json_mode', False)}

    reported_model = None
    for round_number in range(1, max_rounds + 1):
        posts = open_posts(session, annotation_run)
        if not posts:
            logger.info('run %s complete', annotation_run.id)
            return
        logger.info('run %s, round %s: %s open posts', annotation_run.id, round_number, len(posts))

        items = build_items(posts, llm_run.context_strategy)
        chunks = make_batches(items, llm_run.batch_size)
        batches = create_batches(session, llm_run, chunks)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(send_batch, system_prompt, chunk, settings): (batch, chunk)
                       for batch, chunk in zip(batches, chunks)}
            for future in as_completed(futures):
                batch, chunk = futures[future]
                result, annotations, problems = future.result()
                reported_model = check_reported_model(llm_run, batch, result['model'], reported_model)
                store_result(session, llm_run, batch, chunk, result, annotations, problems)

    remaining = len(open_posts(session, annotation_run))
    if remaining:
        logger.warning('run %s: %s posts still without annotation after %s rounds; resume_run sends them again',
                       annotation_run.id, remaining, max_rounds)
    else:
        logger.info('run %s complete', annotation_run.id)


def create_batches(session, llm_run, chunks):
    """Store the batches of one round before sending: Batch(number, status='sent') + BatchPost(position).

    "Was sent" is recorded before the request, so a crash mid-request is still visible.
    :param chunks: list of item lists (from make_batches)
    :return: list of Batch, in the same order as chunks
    """
    first_number = next_batch_number(session, llm_run)
    batches = []
    for offset, chunk in enumerate(chunks):
        batch = Batch(
            run_id=llm_run.run_id,
            number=first_number + offset,
            status='sent',
            batch_posts=[BatchPost(post_id=item['id'], position=position)
                         for position, item in enumerate(chunk, start=1)],
        )
        session.add(batch)
        batches.append(batch)
    session.commit()
    return batches


def next_batch_number(session, llm_run):
    """Return the number for the next batch of this run (1 for the first; continues across rounds and resumes)."""
    highest = session.scalar(select(func.max(Batch.number)).where(Batch.run_id == llm_run.run_id))
    return (highest or 0) + 1


def send_batch(system_prompt, items, settings):
    """Send one batch and validate the answer. Runs in a worker thread: no session, no database.

    :param settings: plain dict (provider, model, temperature, thinking, json_mode); ORM objects stay in the main thread
    :return: (llm_client result, annotations, problems) from annotate_batch and parse_response
    """
    result = annotate_batch(system_prompt, items, provider=settings['provider'],
                            model=settings['model'], temperature=settings['temperature'],
                            thinking=settings['thinking'], json_mode=settings['json_mode'])
    if result['error'] is not None and result['raw'] is None:
        return result, {}, [result['error']]
    annotations, problems = parse_response(result['parsed'], [item['id'] for item in items])
    if result['error'] is not None:
        problems.insert(0, result['error'])
    return result, annotations, problems


def check_reported_model(llm_run, batch, reported, previous):
    """Warn once if the API reports another model than requested, and whenever it changes within a run.

    Some providers map the requested name to the deployment behind it (KI:connect:
    'qwen-qwen3-8-27b' -> 'Qwen/Qwen3.8-27B'), so a difference alone is no error and is
    reported only for the first answer; a change between batches may mean a model swap.
    The reported name is stored per batch in raw_response either way.
    :return: the model to compare the next batch with
    """
    if reported is None:
        return previous
    if previous is None and reported != llm_run.model:
        logger.warning('batch %s: answered by %r, requested %r', batch.number, reported, llm_run.model)
    elif previous is not None and reported != previous:
        logger.warning('batch %s: model changed within the run from %r to %r', batch.number, previous, reported)
    return reported


def store_result(session, llm_run, batch, items, result, annotations, problems):
    """Write the outcome of one batch in the main thread and commit.

    Sets batch.raw_response and batch.status ('done', 'partial', 'invalid' or 'error'),
    adds Annotation + CriterionMention rows.
    """
    for post_id, annotation in annotations.items():
        session.add(Annotation(
            run_id=llm_run.run_id,
            post_id=post_id,
            stance=annotation['stance'],
            criterion_mentions=[
                CriterionMention(run_id=llm_run.run_id, post_id=post_id, label=mention['label'],
                                 sentiment=mention['sentiment'], quote=mention['quote'])
                for mention in annotation['mentions']
            ],
        ))

    batch.raw_response = result['raw']
    if result['raw'] is None:
        batch.status = 'error'
    elif not annotations:
        batch.status = 'invalid'
    elif len(annotations) < len(items):
        batch.status = 'partial'
    else:
        batch.status = 'done'
    session.commit()

    if problems:
        logger.warning('batch %s (%s): %s problem(s), first: %s', batch.number, batch.status, len(problems), problems[0])
