# transfer.py — moves a human run between two copies of the database as a JSON file:
# a second annotator works on a copy, exports the run, and it is imported into the main database.
# Posts are identified by their external ids; post.id is only valid inside one database file.

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from radar.annotation.human.store import clean_annotation, load_human_run
from radar.annotation.prompts import get_or_create_prompt_version
from radar.db.models.annotation import Annotation, AnnotationRun, CriterionMention, HumanRun, PromptVersion, Sample
from radar.db.models.collection import Post

FORMAT_VERSION = 1
MAX_PROBLEMS_SHOWN = 10


def export_run(session, run_id):
    """Return a human run with all its annotations as a plain dict (see FORMAT_VERSION)."""
    annotation_run = load_human_run(session, run_id)
    annotations = session.scalars(
        select(Annotation)
        .where(Annotation.run_id == run_id)
        .options(selectinload(Annotation.post).selectinload(Post.container),
                 selectinload(Annotation.criterion_mentions))).all()
    # ordered by external ids, so the same run gives the same file from every database
    annotations = sorted(annotations, key=lambda annotation: (annotation.post.container.external_id,
                                                              annotation.post.external_id))
    return {
        'format_version': FORMAT_VERSION,
        'sample': annotation_run.sample.name,
        'prompt': {'name': annotation_run.prompt_version.name, 'hash': annotation_run.prompt_version.hash},
        'annotator': annotation_run.human_run.annotator,
        # SQLite returns the timestamp without time zone; it was stored in UTC
        'started_at': annotation_run.started_at.replace(tzinfo=timezone.utc).isoformat(),
        'code_version': annotation_run.code_version,
        'annotations': [
            {
                'container_external_id': annotation.post.container.external_id,
                'post_external_id': annotation.post.external_id,
                'stance': annotation.stance,
                'criteria': [
                    {'label': mention.label, 'sentiment': mention.sentiment, 'quote': mention.quote}
                    for mention in sorted(annotation.criterion_mentions, key=lambda mention: mention.id)
                ],
            }
            for annotation in annotations
        ],
    }


def save_export(data, path):
    """Write an export as JSON (UTF-8, readable), creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def load_export(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise ValueError(f'{path} does not exist') from None
    except json.JSONDecodeError as error:
        raise ValueError(f'{path} is not valid JSON: {error}') from None


def import_run(session, data):
    """Create a human run from an export and commit it. Returns the new AnnotationRun id.

    Everything is checked before the first row is written; any problem raises ValueError
    and leaves the database unchanged. Checked: format version, the sample exists under its
    name, the prompt hash matches the stored prompt version, the run was not imported
    before, every post belongs to the sample and appears once, the values are valid.
    """
    try:
        run_id = add_imported_run(session, data)
        session.commit()
        return run_id
    except Exception:
        session.rollback()  # also drops a prompt version created just for this import
        raise


def add_imported_run(session, data):
    """Check an export and add its run with all annotations to the session, without committing."""
    if not isinstance(data, dict) or data.get('format_version') != FORMAT_VERSION:
        found = data.get('format_version') if isinstance(data, dict) else None
        raise ValueError(f'unsupported format_version {found!r}, expected {FORMAT_VERSION}')
    for key in ['sample', 'prompt', 'annotator', 'started_at', 'code_version', 'annotations']:
        if key not in data:
            raise ValueError(f'the export has no {key!r}')

    sample = session.scalar(select(Sample).where(Sample.name == data['sample']))
    if sample is None:
        raise ValueError(f'Unknown sample_name: {data["sample"]}')
    prompt_version = find_prompt_version(session, data['prompt'])
    annotator = data['annotator'].strip() if isinstance(data['annotator'], str) else ''
    if not annotator:
        raise ValueError('the export names no annotator')
    started_at = parse_started_at(data['started_at'])
    if not isinstance(data['code_version'], str) or not data['code_version']:
        raise ValueError('the export has no code_version')
    session.flush()  # a prompt version created just now gets its id here

    same_run = session.scalar(
        select(AnnotationRun.id)
        .join(AnnotationRun.human_run)
        .where(AnnotationRun.sample_id == sample.id,
               AnnotationRun.prompt_version_id == prompt_version.id,
               AnnotationRun.started_at == started_at,
               HumanRun.annotator == annotator))
    if same_run is not None:
        raise ValueError(f'this run is already in the database as annotation run {same_run}')

    annotations = check_annotations(session, sample, data['annotations'])

    annotation_run = AnnotationRun(sample_id=sample.id, prompt_version=prompt_version,
                                   code_version=data['code_version'], started_at=started_at)
    annotation_run.human_run = HumanRun(annotator=annotator)
    session.add(annotation_run)
    session.flush()  # assigns annotation_run.id for the rows below
    for post_id, stance, mentions in annotations:
        session.add(Annotation(
            run_id=annotation_run.id,
            post_id=post_id,
            stance=stance,
            criterion_mentions=[CriterionMention(run_id=annotation_run.id, post_id=post_id, **mention)
                                for mention in mentions],
        ))
    return annotation_run.id


def find_prompt_version(session, prompt):
    """The stored prompt version of the export; its text must be the one the annotator read.

    If this database has never used the prompt, it is created from prompts/<name>.txt,
    which must then have the exported hash.
    """
    if not isinstance(prompt, dict) or not prompt.get('name') or not prompt.get('hash'):
        raise ValueError('the export needs a prompt with name and hash')
    prompt_version = session.scalar(select(PromptVersion).where(PromptVersion.name == prompt['name']))
    if prompt_version is None:
        try:
            prompt_version = get_or_create_prompt_version(session, prompt['name'])
        except FileNotFoundError:
            raise ValueError(f'prompt {prompt["name"]!r} is neither stored in the database '
                             f'nor a file in prompts/') from None
    if prompt_version.hash != prompt['hash']:
        raise ValueError(f'prompt {prompt["name"]!r} has another text here than the one the annotator read '
                         f'(hash {prompt_version.hash[:12]}… here, {prompt["hash"][:12]}… in the export)')
    return prompt_version


def parse_started_at(value):
    """ISO timestamp of the export -> datetime in UTC (a value without time zone counts as UTC)."""
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f'invalid started_at {value!r}') from None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def check_annotations(session, sample, entries):
    """Match the exported annotations to the posts of the sample and validate them.

    :return: list of (post_id, stance, mentions) ready to store
    Raises ValueError naming the problems (at most MAX_PROBLEMS_SHOWN) if there are any.
    """
    if not isinstance(entries, list):
        raise ValueError('"annotations" is not a list')
    posts = session.scalars(
        select(Post).join(Post.samples).where(Sample.id == sample.id)
        .options(selectinload(Post.container))).all()
    post_by_external_ids = {(post.container.external_id, post.external_id): post for post in posts}

    annotations = []
    problems = []
    seen = set()
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            problems.append(f'annotation {number} is not an object')
            continue
        external_ids = (entry.get('container_external_id'), entry.get('post_external_id'))
        post = post_by_external_ids.get(external_ids)
        if post is None:
            problems.append(f'annotation {number}: post {external_ids[1]!r} in container '
                            f'{external_ids[0]!r} is not part of sample {sample.name!r}')
            continue
        if post.id in seen:
            problems.append(f'annotation {number}: post {external_ids[1]!r} appears twice')
            continue
        seen.add(post.id)
        try:
            mentions = clean_annotation(post.text, entry.get('stance'), entry.get('criteria'))
        except ValueError as error:
            problems.append(f'annotation {number} (post {external_ids[1]!r}): {error}')
            continue
        annotations.append((post.id, entry['stance'], mentions))

    if problems:
        shown = '\n  '.join(problems[:MAX_PROBLEMS_SHOWN])
        more = f'\n  … and {len(problems) - MAX_PROBLEMS_SHOWN} more' if len(problems) > MAX_PROBLEMS_SHOWN else ''
        raise ValueError(f'{len(problems)} problem(s) in the export, nothing was imported:\n  {shown}{more}')
    return annotations
