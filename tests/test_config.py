import pytest

from radar.config import CONFIG, merge, setting


def test_merge_overrides_single_values_and_keeps_the_rest():
    base = {'annotation': {'batch_size': 20, 'workers': 3}, 'llm': {'providers': {'idt': {'model': 'a'}}}}
    override = {'annotation': {'workers': 8}, 'llm': {'providers': {'own': {'model': 'b'}}}}
    merged = merge(base, override)
    assert merged['annotation'] == {'batch_size': 20, 'workers': 8}
    assert set(merged['llm']['providers']) == {'idt', 'own'}
    assert base['annotation']['workers'] == 3  # the inputs are not changed


def test_every_section_the_code_reads_exists():
    for section in ['youtube', 'sampling', 'annotation', 'llm', 'embedding', 'mapping']:
        assert section in CONFIG


def test_missing_setting_is_an_error():
    with pytest.raises(KeyError, match=r'\[annotation\] nope'):
        setting('annotation', 'nope')
