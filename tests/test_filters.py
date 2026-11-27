from datetime import datetime, timedelta

import pytest

from radar.db.models.collection import Post
from radar.sampling.filters import (is_empty_post, is_short_post, remove_duplicates, remove_empty,
                                    remove_short)

DAY = datetime(2026, 11, 1)


def post(text, post_id=1, day=0):
    """A post that is in no database: the filters only read id, text and published_at."""
    return Post(id=post_id, text=text, published_at=DAY + timedelta(days=day))


def ids(posts):
    return [p.id for p in posts]


# --- remove_empty ----------------------------------------------------------------------------

@pytest.mark.parametrize('text, empty', [
    ('', True),
    ('   \n ', True),
    ('👍👍', True),
    ('!!! ???', True),
    ('100 %', True),
    ('https://youtu.be/abc123', True),
    ('www.example.com/seite', True),
    ('@3fa9c0e1b2d4', True),                            # a pseudonymised mention is hex, not a word
    ('@3fa9c0e1b2d4 👍 https://example.com 12:30', True),
    ('Ja', False),
    ('@3fa9c0e1b2d4 stimmt', False),
    ('Ähm', False),
    ('siehe https://example.com', False),
])
def test_is_empty_post(text, empty):
    assert is_empty_post(post(text)) is empty


def test_remove_empty_keeps_the_order():
    posts = [post('Reichweite', 1), post('👍', 2), post('Preis', 3), post('@3fa9c0e1b2d4', 4)]
    assert ids(remove_empty(posts)) == [1, 3]


# --- remove_short ----------------------------------------------------------------------------

@pytest.mark.parametrize('text, threshold, short', [
    ('eins zwei drei vier', 5, True),
    ('eins zwei drei vier fünf', 5, False),
    ('eins  zwei\n\ndrei\tvier fünf', 5, False),  # any whitespace separates words
    ('eins', 1, False),
    ('', 1, True),
    ('eins zwei', 0, False),
])
def test_is_short_post(text, threshold, short):
    assert is_short_post(post(text), threshold) is short


def test_remove_short_keeps_the_order_and_defaults_to_five_words():
    posts = [post('eins zwei drei vier fünf sechs', 1), post('zu kurz', 2), post('a b c d e', 3)]
    assert ids(remove_short(posts)) == [1, 3]
    assert ids(remove_short(posts, 6)) == [1]


# --- remove_duplicates -----------------------------------------------------------------------

def test_remove_duplicates_keeps_the_earliest_post():
    posts = [post('Verbrenner bleibt', 1, day=2), post('Verbrenner bleibt', 2, day=1),
             post('Verbrenner bleibt', 3, day=3)]
    assert ids(remove_duplicates(posts)) == [2]


def test_remove_duplicates_breaks_ties_by_id():
    posts = [post('Verbrenner bleibt', 7), post('Verbrenner bleibt', 3)]
    assert ids(remove_duplicates(posts)) == [3]


def test_remove_duplicates_ignores_case_and_outer_whitespace():
    posts = [post('Verbrenner bleibt', 1, day=1), post('  verbrenner BLEIBT\n', 2, day=2)]
    assert ids(remove_duplicates(posts)) == [1]


def test_remove_duplicates_keeps_texts_that_differ_inside():
    posts = [post('Verbrenner bleibt', 1), post('Verbrenner  bleibt', 2), post('Verbrenner bleibt!', 3)]
    assert sorted(ids(remove_duplicates(posts))) == [1, 2, 3]


def test_remove_duplicates_returns_the_posts_oldest_first():
    posts = [post('c', 1, day=3), post('a', 2, day=1), post('b', 3, day=2)]
    assert ids(remove_duplicates(posts)) == [2, 3, 1]


def test_filters_do_not_change_their_input():
    posts = [post('👍', 1), post('kurz', 2), post('kurz', 3)]
    for remove in (remove_empty, remove_short, remove_duplicates):
        remove(posts)
    assert ids(posts) == [1, 2, 3]
