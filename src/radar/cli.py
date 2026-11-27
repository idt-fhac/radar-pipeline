# cli.py — command line interface: `uv run radar <command> ...` (see `uv run radar --help`)

import argparse
import logging
import os
import sys
from datetime import datetime, timezone


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Must happen before anything imports radar.db.engine: the engine reads these at import time.
    os.environ['RADAR_DB'] = args.db
    os.environ['RADAR_SQL_ECHO'] = '1' if args.sql else '0'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    # Output redirected into a file or pipe uses the Windows code page, which has no emojis;
    # comments are full of them. Replace such characters there instead of failing.
    sys.stdout.reconfigure(errors='replace')
    if args.handler is not cmd_models:
        from radar.db.engine import DB_PATH
        if not DB_PATH.exists():  # SQLite would silently create an empty file without tables
            sys.exit(f'error: {DB_PATH.name} does not exist; create it first: '
                     f'$env:RADAR_DB = "{args.db}"; uv run alembic upgrade head')
    if args.db == 'prod':
        logging.getLogger('radar').info('using the PRODUCTION database')

    try:
        args.handler(args)
    except (ValueError, RuntimeError) as error:
        sys.exit(f'error: {error}')


def build_parser():
    # neither import touches the engine
    from radar.annotation.llm_client import DEFAULT_PROVIDER, PROVIDERS
    from radar.annotation.run_llm import CONTEXT_STRATEGIES

    parser = argparse.ArgumentParser(prog='radar', description='RADAR pipeline')
    parser.add_argument('--db', choices=['test', 'prod'], default=os.getenv('RADAR_DB', 'test'),
                        help='database to use (default: test, or $RADAR_DB)')
    parser.add_argument('--sql', action='store_true', help='log every SQL statement')
    commands = parser.add_subparsers(dest='command', required=True)

    collect = commands.add_parser('collect', help='search YouTube and store videos and comments')
    collect.add_argument('topic', help='topic name, e.g. "E-Auto"')
    collect.add_argument('search_term', help='YouTube search term, e.g. "Verbrenner-Aus"')
    collect.add_argument('--max-videos', type=int, help='videos taken from the search results')
    collect.add_argument('--max-threads', type=int, help='top-level comments per video')
    collect.add_argument('--order', choices=['relevance', 'time'], help='comment order')
    collect.add_argument('--replies', choices=['all', 'included', 'none'], help='which replies to fetch')
    collect.add_argument('--allow-dirty', action='store_true', help='allow uncommitted code changes')
    collect.set_defaults(handler=cmd_collect)

    sample = commands.add_parser('sample', help='load, filter and freeze a sample of a topic')
    sample.add_argument('name', help='unique sample name, e.g. "e-auto-main"')
    sample.add_argument('topic', help='topic name')
    sample.add_argument('--source', help='only posts from this source, e.g. youtube')
    sample.add_argument('--video', action='append', dest='video_ids', metavar='VIDEO_ID',
                        help='only posts from this video (external id); repeatable')
    sample.add_argument('--after', type=parse_time,
                        help='published at or after (ISO date or time; UTC unless it carries an offset)')
    sample.add_argument('--before', type=parse_time, help='published before (as --after)')
    sample.add_argument('--include-root-posts', action='store_true', help='also take the videos themselves')
    sample.add_argument('--min-words', type=int, help='drop posts with fewer words, not counting links and @-mentions '
                             '(default from the config, 0 = off)')
    sample.add_argument('--keep-empty', action='store_true', help='keep posts without letters')
    sample.add_argument('--keep-duplicates', action='store_true', help='keep posts with duplicate text')
    sample.add_argument('--allow-dirty', action='store_true')
    sample.set_defaults(handler=cmd_sample)

    subset = commands.add_parser('subset', help='draw a reproducible random subset of a sample')
    subset.add_argument('name', help='unique sample name, e.g. "e-auto-300"')
    subset.add_argument('base', help='name of the sample to draw from')
    subset.add_argument('n', type=int, help='number of posts')
    subset.add_argument('--seed', type=int, help='default from the config')
    subset.add_argument('--allow-dirty', action='store_true')
    subset.set_defaults(handler=cmd_subset)

    run = commands.add_parser('run', help='start an LLM annotation run on a sample')
    run.add_argument('sample', help='sample name')
    run.add_argument('prompt', help='prompt name (file stem in prompts/), e.g. extraction_v1')
    run.add_argument('--provider', choices=list(PROVIDERS), help=f'default: {DEFAULT_PROVIDER}')
    run.add_argument('--model', help="model id (default: the provider's default model)")
    run.add_argument('--temperature', type=float)
    run.add_argument('--batch-size', type=int)
    run.add_argument('--context', dest='context_strategy', choices=CONTEXT_STRATEGIES,
                     help='what the LLM gets besides the comment; +parent: for a reply, the comment it stands below')
    thinking = run.add_mutually_exclusive_group()
    thinking.add_argument('--thinking', dest='thinking', action='store_true', default=None,
                          help='reasoning phase on (default)')
    thinking.add_argument('--no-thinking', dest='thinking', action='store_false', help='reasoning phase off')
    json_mode = run.add_mutually_exclusive_group()
    json_mode.add_argument('--json-mode', dest='json_mode', action='store_true', default=None,
                           help='server enforces a valid JSON answer (default from the config)')
    json_mode.add_argument('--no-json-mode', dest='json_mode', action='store_false')
    run.add_argument('--max-rounds', type=int)
    run.add_argument('--workers', type=int, help='parallel requests')
    run.add_argument('--allow-dirty', action='store_true')
    run.set_defaults(handler=cmd_run)

    resume = commands.add_parser('resume', help='continue an aborted or incomplete run')
    resume.add_argument('run_id', type=int)
    resume.add_argument('--max-rounds', type=int)
    resume.add_argument('--workers', type=int)
    resume.set_defaults(handler=cmd_resume)

    annotate = commands.add_parser('annotate', help='annotate a sample by hand in the browser')
    annotate.add_argument('sample', help='sample name')
    annotate.add_argument('prompt', help='prompt name; its text is shown as the annotation guideline')
    annotate.add_argument('--annotator', required=True,
                          help='your name; starting again with the same sample, prompt and name continues the run')
    annotate.add_argument('--port', type=int, help='default from the config')
    annotate.add_argument('--no-browser', action='store_true', help='do not open the browser')
    annotate.add_argument('--allow-dirty', action='store_true')
    annotate.set_defaults(handler=cmd_annotate)

    export_human = commands.add_parser('export-human', help='write a human run to a JSON file')
    export_human.add_argument('run_id', type=int)
    export_human.add_argument('file', help='e.g. exports/run-12.json (exports/ is ignored by Git)')
    export_human.set_defaults(handler=cmd_export_human)

    import_human = commands.add_parser('import-human', help='read a human run from a JSON file')
    import_human.add_argument('file')
    import_human.set_defaults(handler=cmd_import_human)

    show = commands.add_parser('show', help='show the results of a run')
    show.add_argument('run_id', type=int)
    show.add_argument('--full', action='store_true', help='print the full comment texts')
    show.set_defaults(handler=cmd_show)

    runs = commands.add_parser('runs', help='list annotation runs')
    runs.set_defaults(handler=cmd_runs)

    quotes = commands.add_parser('quotes', help='quote fidelity: how many quotes occur word for word in their comment')
    quotes.add_argument('run_ids', type=int, nargs='*', metavar='run_id', help='annotation runs (default: all)')
    quotes.add_argument('--list', dest='list_quotes', action='store_true',
                        help='also print the quotes that do not occur in their comment')
    quotes.set_defaults(handler=cmd_quotes)

    models = commands.add_parser('models', help='list the models a provider offers')
    models.add_argument('provider', nargs='?', default=DEFAULT_PROVIDER, choices=list(PROVIDERS))
    models.set_defaults(handler=cmd_models)

    return parser


def parse_time(value):
    """'2026-01-01' or '2026-01-01T12:00' -> datetime in UTC.

    Values without time zone count as UTC; '2026-01-01T00:00+01:00' becomes 23:00 UTC of the day before.
    """
    moment = datetime.fromisoformat(value)
    return moment.astimezone(timezone.utc) if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def given(args, *names):
    """Only the options the user actually set, so the functions' own defaults apply otherwise."""
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def cmd_collect(args):
    from radar.db.engine import Session
    from radar.collection.collect import collect_youtube
    options = {'max_videos': args.max_videos, 'max_threads_per_video': args.max_threads,
               'comment_order': args.order, 'replies': args.replies}
    with Session() as session:
        collect_youtube(session, args.topic, args.search_term, allow_dirty=args.allow_dirty,
                        **{key: value for key, value in options.items() if value is not None})


def cmd_sample(args):
    from radar.db.engine import Session
    from radar.sampling.sample import create_sample
    with Session() as session:
        sample = create_sample(
            session, args.name, args.topic, source=args.source, video_ids=args.video_ids,
            published_after=args.after, published_before=args.before,
            include_root_posts=args.include_root_posts, drop_empty=not args.keep_empty,
            drop_duplicates=not args.keep_duplicates, allow_dirty=args.allow_dirty,
            **({} if args.min_words is None else {'min_words': args.min_words or None}))
        session.commit()
        print_definition(sample)


def cmd_subset(args):
    from radar.db.engine import Session
    from radar.sampling.sample import create_subset
    with Session() as session:
        subset = create_subset(session, args.name, args.base, args.n, allow_dirty=args.allow_dirty,
                               **given(args, 'seed'))
        session.commit()
        print_definition(subset)


def print_definition(sample):
    definition = sample.definition
    print(f'sample {sample.name!r} (id {sample.id})')
    if 'loaded' in definition:
        print(f'  loaded             {definition["loaded"]:>6}')
        for step in definition['filters']:
            print(f'  {step["name"]:<18} {step["remaining"]:>6}')
    else:
        print(f'  drawn {definition["n"]} of {definition["base_size"]} from {definition["base_sample"]!r}'
              f' (seed {definition["seed"]})')


def cmd_run(args):
    from radar.db.engine import Session
    from radar.annotation.run_llm import run_llm
    with Session() as session:
        annotation_run = run_llm(session, args.sample, args.prompt, allow_dirty=args.allow_dirty,
                                 **given(args, 'provider', 'model', 'temperature', 'batch_size',
                                         'context_strategy', 'thinking', 'json_mode', 'max_rounds', 'workers'))
        print(f'run {annotation_run.id}')


def cmd_resume(args):
    from radar.db.engine import Session
    from radar.annotation.run_llm import resume_run
    with Session() as session:
        resume_run(session, args.run_id, **given(args, 'max_rounds', 'workers'))


def cmd_annotate(args):
    from radar.db.engine import Session
    from radar.annotation.human.app import serve
    from radar.annotation.human.store import start_human_run
    with Session() as session:
        annotation_run, created = start_human_run(session, args.sample, args.prompt, args.annotator,
                                                  allow_dirty=args.allow_dirty)
        run_id = annotation_run.id
    print(f'run {run_id} ({"new" if created else "continued"})')
    serve(Session, run_id, open_browser=not args.no_browser, **given(args, 'port'))


def cmd_export_human(args):
    from radar.db.engine import Session
    from radar.annotation.human.transfer import export_run, save_export
    with Session() as session:
        data = export_run(session, args.run_id)
    save_export(data, args.file)
    print(f'run {args.run_id} of {data["annotator"]!r}: {len(data["annotations"])} annotations -> {args.file}')


def cmd_import_human(args):
    from radar.db.engine import Session
    from radar.annotation.human.transfer import import_run, load_export
    data = load_export(args.file)
    with Session() as session:
        run_id = import_run(session, data)
    print(f'run {run_id}: {len(data["annotations"])} annotations of {data["annotator"]!r} imported')


def cmd_show(args):
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from radar.db.engine import Session
    from radar.db.models.annotation import Annotation, AnnotationRun, Batch, CriterionMention
    from radar.db.models.collection import Post
    with Session() as session:
        annotation_run = session.scalar(select(AnnotationRun).where(AnnotationRun.id == args.run_id).options(
            selectinload(AnnotationRun.llm_run), selectinload(AnnotationRun.human_run),
            selectinload(AnnotationRun.sample), selectinload(AnnotationRun.prompt_version)))
        if annotation_run is None:
            raise ValueError(f'unknown annotation run {args.run_id}')
        llm_run = annotation_run.llm_run
        kind = (f'{llm_run.params.get("provider")} {llm_run.model}, thinking={llm_run.params.get("thinking")}, '
                f'json_mode={llm_run.params.get("json_mode", False)}, '
                f'context={llm_run.context_strategy}') if llm_run else f'human {annotation_run.human_run.annotator!r}'
        print(f'run {annotation_run.id}: sample {annotation_run.sample.name!r}, '
              f'prompt {annotation_run.prompt_version.name!r}, {kind}, code {annotation_run.code_version[:12]}')

        if llm_run:
            batches = session.execute(select(Batch.number, Batch.status)
                                      .where(Batch.run_id == annotation_run.id).order_by(Batch.number)).all()
            print('batches: ' + ', '.join(f'{number}:{status}' for number, status in batches))

        rows = session.execute(select(Post.id, Post.text, Annotation.stance)
                               .join(Annotation, Annotation.post_id == Post.id)
                               .where(Annotation.run_id == annotation_run.id).order_by(Post.id)).all()
        mentions = {}
        for mention in session.scalars(select(CriterionMention)
                                       .where(CriterionMention.run_id == annotation_run.id)
                                       .order_by(CriterionMention.id)):
            mentions.setdefault(mention.post_id, []).append(mention)

        for post_id, text, stance in rows:
            text = text if args.full else shorten(text, 200)
            print(f'\n#{post_id} [{stance}] {text}')
            for mention in mentions.get(post_id, []):
                print(f'    -> {mention.label} ({mention.sentiment}): {mention.quote}')
        print(f'\n{len(rows)} annotated posts, {sum(map(len, mentions.values()))} criteria')


def cmd_runs(args):
    from sqlalchemy import func, select
    from sqlalchemy.orm import selectinload
    from radar.db.engine import Session
    from radar.db.models.annotation import Annotation, AnnotationRun
    with Session() as session:
        runs = session.scalars(select(AnnotationRun).order_by(AnnotationRun.id).options(
            selectinload(AnnotationRun.llm_run), selectinload(AnnotationRun.human_run),
            selectinload(AnnotationRun.sample), selectinload(AnnotationRun.prompt_version))).all()
        counts = dict(session.execute(select(Annotation.run_id, func.count())
                                      .group_by(Annotation.run_id)).all())
        for run in runs:
            kind = (f'llm {run.llm_run.params.get("provider")} {run.llm_run.model} '
                    f'thinking={run.llm_run.params.get("thinking")}') if run.llm_run else f'human {run.human_run.annotator}'
            print(f'{run.id:>4}  {run.started_at:%Y-%m-%d %H:%M}  {run.sample.name:<16} '
                  f'{run.prompt_version.name:<16} {kind}  annotated={counts.get(run.id, 0)}')


def cmd_quotes(args):
    from radar.analysis.quotes import non_verbatim_quotes, quote_fidelity
    from radar.db.engine import Session
    with Session() as session:
        rows = quote_fidelity(session, args.run_ids or None)
        print(' run  mentions  quotes  verbatim    share')
        for row in rows:
            share = '      –' if row['share'] is None else f'{row["share"]:6.1%}'
            print(f'{row["run_id"]:>4}  {row["mentions"]:>8}  {row["quotes"]:>6}  {row["verbatim"]:>8}  {share:>7}')
        for run_id in sorted(set(args.run_ids) - {row['run_id'] for row in rows}):
            print(f'{run_id:>4}  no criterion mentions')
        if args.list_quotes:
            for row in rows:
                for quote in non_verbatim_quotes(session, row['run_id']):
                    print(f'\nrun {row["run_id"]}, post {quote["post_id"]}, {quote["label"]!r}: {quote["quote"]}')
                    print(f'    comment: {shorten(quote["text"], 300)}')


def cmd_models(args):
    from radar.annotation.llm_client import list_models
    for model in list_models(args.provider):
        print(model)


def shorten(text, length):
    text = ' '.join(text.split())
    return text if len(text) <= length else text[:length - 1] + '…'


if __name__ == '__main__':
    main()
