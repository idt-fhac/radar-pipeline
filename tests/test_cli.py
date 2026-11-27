import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone

import pytest

from radar.annotation.llm_client import DEFAULT_PROVIDER
from radar.cli import build_parser, given, parse_time, print_definition, shorten


def parse(*argv):
    return build_parser().parse_args(argv)


# --- helpers ---------------------------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    ('2026-01-01', datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ('2026-01-01T12:30', datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)),
    ('2026-01-01T12:30:00+00:00', datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)),
])
def test_parse_time_counts_values_without_time_zone_as_utc(value, expected):
    assert parse_time(value) == expected
    assert parse_time(value).utcoffset().total_seconds() == 0


@pytest.mark.parametrize('value, expected', [
    ('2026-01-01T00:00+01:00', datetime(2025, 12, 31, 23, 0, tzinfo=timezone.utc)),
    ('2026-01-01T12:30:00-05:00', datetime(2026, 1, 1, 17, 30, tzinfo=timezone.utc)),
])
def test_parse_time_converts_other_time_zones_to_utc(value, expected):
    # the database drops the time zone, so anything but UTC would be compared an offset off
    assert parse_time(value) == expected
    assert parse_time(value).utcoffset().total_seconds() == 0


def test_given_passes_on_only_what_the_user_set():
    args = Namespace(batch_size=50, model=None, thinking=False, temperature=0.0)
    # False and 0.0 are values the user chose; only None means "not set"
    assert given(args, 'batch_size', 'model', 'thinking', 'temperature') == {
        'batch_size': 50, 'thinking': False, 'temperature': 0.0}


@pytest.mark.parametrize('text, expected', [
    ('kurz', 'kurz'),
    ('a\n\nb   c', 'a b c'),  # line breaks and runs of blanks become one blank
    ('abcdefghij', 'abcdefghij'),
    ('abcdefghijk', 'abcdefghi…'),
])
def test_shorten(text, expected):
    assert shorten(text, 10) == expected


def test_print_definition_of_a_sample_and_of_a_subset(capsys):
    print_definition(Namespace(name='main', id=1, definition={
        'loaded': 5, 'filters': [{'name': 'remove_empty', 'remaining': 4},
                                 {'name': 'remove_short', 'min_words': 5, 'remaining': 3}]}))
    print_definition(Namespace(name='sub', id=2, definition={
        'base_sample': 'main', 'n': 2, 'seed': 42, 'base_size': 3}))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "sample 'main' (id 1)"
    assert [line.split() for line in lines[1:4]] == [['loaded', '5'], ['remove_empty', '4'], ['remove_short', '3']]
    assert lines[5].strip() == "drawn 2 of 3 from 'main' (seed 42)"


# --- the parser ------------------------------------------------------------------------------

def test_database_defaults_to_test_and_can_be_chosen(monkeypatch):
    monkeypatch.delenv('RADAR_DB', raising=False)
    assert parse('runs').db == 'test'
    assert parse('--db', 'prod', 'runs').db == 'prod'
    monkeypatch.setenv('RADAR_DB', 'prod')
    assert parse('runs').db == 'prod'
    with pytest.raises(SystemExit):
        parse('--db', 'other', 'runs')


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        parse()


def test_run_leaves_unset_options_to_the_library():
    args = parse('run', 'e-auto-300', 'extraction_v2')
    assert (args.sample, args.prompt) == ('e-auto-300', 'extraction_v2')
    options = ['provider', 'model', 'temperature', 'batch_size', 'context_strategy', 'thinking', 'json_mode',
               'max_rounds', 'workers']
    assert given(args, *options) == {}  # so the defaults of the config apply
    assert args.allow_dirty is False


def test_run_options():
    args = parse('run', 's', 'p', '--provider', DEFAULT_PROVIDER, '--model', 'm', '--temperature', '0.5',
                 '--batch-size', '50', '--context', 'text', '--no-thinking', '--json-mode',
                 '--max-rounds', '2', '--workers', '8', '--allow-dirty')
    assert given(args, 'provider', 'model', 'temperature', 'batch_size', 'context_strategy', 'thinking',
                 'json_mode', 'max_rounds', 'workers') == {
        'provider': DEFAULT_PROVIDER, 'model': 'm', 'temperature': 0.5, 'batch_size': 50,
        'context_strategy': 'text', 'thinking': False, 'json_mode': True, 'max_rounds': 2, 'workers': 8}
    assert args.allow_dirty is True


def test_run_refuses_contradicting_and_unknown_options():
    for argv in [('run', 's', 'p', '--thinking', '--no-thinking'),
                 ('run', 's', 'p', '--json-mode', '--no-json-mode'),
                 ('run', 's', 'p', '--provider', 'nope'),
                 ('run', 's', 'p', '--context', 'text+everything')]:
        with pytest.raises(SystemExit):
            parse(*argv)


def test_context_choices_are_the_strategies_of_the_code():
    from radar.annotation.run_llm import CONTEXT_STRATEGIES
    for strategy in CONTEXT_STRATEGIES:
        assert parse('run', 's', 'p', '--context', strategy).context_strategy == strategy


def test_building_the_parser_does_not_import_the_engine():
    # main() sets RADAR_DB after parsing; the engine reads it at import time, so it must not be loaded yet
    code = ("import sys; from radar.cli import build_parser; build_parser(); "
            "sys.exit('radar.db.engine' in sys.modules)")
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


def test_sample_options():
    args = parse('sample', 'main', 'E-Auto', '--video', 'v1', '--video', 'v2', '--after', '2026-01-01',
                 '--min-words', '0', '--keep-empty', '--include-root-posts')
    assert (args.name, args.topic, args.video_ids) == ('main', 'E-Auto', ['v1', 'v2'])
    assert args.after == datetime(2026, 1, 1, tzinfo=timezone.utc) and args.before is None
    assert args.min_words == 0 and args.keep_empty and not args.keep_duplicates
    assert args.include_root_posts and args.source is None


def test_subset_and_collect_options():
    subset = parse('subset', 'e-auto-300', 'e-auto-main', '300')
    assert (subset.name, subset.base, subset.n, subset.seed) == ('e-auto-300', 'e-auto-main', 300, None)

    collect = parse('collect', 'E-Auto', 'Verbrenner-Aus', '--max-videos', '25', '--replies', 'none')
    assert (collect.topic, collect.search_term) == ('E-Auto', 'Verbrenner-Aus')
    assert (collect.max_videos, collect.max_threads, collect.order, collect.replies) == (25, None, None, 'none')
    with pytest.raises(SystemExit):
        parse('collect', 'E-Auto', 'Verbrenner-Aus', '--order', 'newest')


def test_annotate_needs_an_annotator():
    with pytest.raises(SystemExit):
        parse('annotate', 's', 'p')
    args = parse('annotate', 's', 'p', '--annotator', 'Anna', '--no-browser')
    assert (args.annotator, args.port, args.no_browser) == ('Anna', None, True)


def test_quotes_takes_any_number_of_runs():
    assert (parse('quotes').run_ids, parse('quotes').list_quotes) == ([], False)
    args = parse('quotes', '6', '7', '--list')
    assert (args.run_ids, args.list_quotes) == ([6, 7], True)


def test_every_command_has_a_handler():
    for argv in [('collect', 't', 's'), ('sample', 'n', 't'), ('subset', 'n', 'b', '1'), ('run', 's', 'p'),
                 ('resume', '1'), ('annotate', 's', 'p', '--annotator', 'A'), ('export-human', '1', 'f.json'),
                 ('import-human', 'f.json'), ('show', '1'), ('runs',), ('quotes',), ('models',)]:
        assert callable(parse(*argv).handler)
