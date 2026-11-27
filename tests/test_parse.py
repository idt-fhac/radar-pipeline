import pytest

from radar.annotation.parse import normalise, parse_mentions, parse_response, to_int


def entry(post_id, position='dafuer', criteria=()):
    return {'id': post_id, 'position': position, 'kriterien': list(criteria)}


def criterion(label='Reichweite', sentiment='negativ', quote='zu wenig Reichweite'):
    return {'kriterium': label, 'sentiment': sentiment, 'zitat': quote}


# --- small helpers ---------------------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    ('dafuer', 'dafuer'),
    (' Dafür ', 'dafuer'),
    ('DAGEGEN', 'dagegen'),
    ('Unklär', 'unklaer'),
    ('möglich', 'moeglich'),
    (None, ''),
    (17, ''),
    (['dafuer'], ''),
])
def test_normalise(value, expected):
    assert normalise(value) == expected


@pytest.mark.parametrize('value, expected', [
    (17, 17),
    ('17', 17),
    (' 17 ', 17),
    (True, None),   # bool is an int in Python, but no id
    (17.0, None),
    ('17a', None),
    ('-3', None),
    ('', None),
    (None, None),
])
def test_to_int(value, expected):
    assert to_int(value) == expected


# --- parse_response --------------------------------------------------------------------------

def test_valid_answer_is_translated_to_the_stored_values():
    parsed = {'ergebnisse': [entry(1, 'dagegen', [criterion()]), entry(2, 'unklar')]}
    annotations, problems = parse_response(parsed, [1, 2])
    assert annotations == {
        1: {'stance': 'contra', 'mentions': [
            {'label': 'Reichweite', 'sentiment': 'negative', 'quote': 'zu wenig Reichweite'}]},
        2: {'stance': 'unclear', 'mentions': []},  # no criteria is a result of its own
    }
    assert problems == []


@pytest.mark.parametrize('position, stance', [
    ('dafuer', 'pro'), ('dagegen', 'contra'), ('gemischt', 'mixed'), ('unklar', 'unclear'),
    ('Dafür', 'pro'),  # umlaut and capital despite the prompt
])
def test_every_position_has_a_stance(position, stance):
    annotations, problems = parse_response({'ergebnisse': [entry(1, position)]}, [1])
    assert annotations[1]['stance'] == stance and problems == []


@pytest.mark.parametrize('parsed', [None, [], 'text', {}, {'ergebnisse': None}, {'ergebnisse': {'id': 1}}])
def test_answer_without_result_list_gives_no_annotations(parsed):
    annotations, problems = parse_response(parsed, [1, 2])
    assert annotations == {}
    assert problems == ['no list "ergebnisse" in the answer']


def test_ids_may_come_back_as_strings():
    annotations, problems = parse_response({'ergebnisse': [entry('1'), entry(' 2 ')]}, [1, 2])
    assert set(annotations) == {1, 2} and problems == []


def test_id_that_was_not_sent_is_ignored():
    annotations, problems = parse_response({'ergebnisse': [entry(1), entry(99)]}, [1])
    assert set(annotations) == {1}
    assert problems == ['id 99 was not in the batch']


def test_id_answered_twice_keeps_the_first_answer():
    annotations, problems = parse_response({'ergebnisse': [entry(1, 'dafuer'), entry(1, 'dagegen')]}, [1])
    assert annotations[1]['stance'] == 'pro'
    assert problems == ['id 1 answered twice, kept the first answer']


@pytest.mark.parametrize('broken, message', [
    ({'id': 2, 'position': 'vielleicht', 'kriterien': []}, "id 2: invalid position 'vielleicht'"),
    ({'id': 2, 'kriterien': []}, 'id 2: invalid position None'),
    ({'id': 2, 'position': 'dafuer', 'kriterien': 'keine'}, 'id 2: "kriterien" is not a list'),
    ({'id': 2, 'position': 'dafuer'}, 'id 2: "kriterien" is not a list'),
    ('kein Objekt', "entry is not an object: 'kein Objekt'"),
])
def test_broken_entry_counts_as_missing_and_the_rest_is_kept(broken, message):
    annotations, problems = parse_response({'ergebnisse': [entry(1), broken]}, [1, 2])
    assert set(annotations) == {1}  # post 2 has no annotation, so it is sent again
    assert problems == [message, '1 of 2 ids without valid answer: [2]']


def test_skipped_ids_are_reported():
    annotations, problems = parse_response({'ergebnisse': [entry(2)]}, [1, 2, 3])
    assert set(annotations) == {2}
    assert problems == ['2 of 3 ids without valid answer: [1, 3]']


# --- parse_mentions --------------------------------------------------------------------------

@pytest.mark.parametrize('sentiment, stored', [
    ('positiv', 'positive'), ('negativ', 'negative'), ('neutral', 'neutral'), (' Positiv ', 'positive'),
])
def test_sentiments_are_translated(sentiment, stored):
    problems = []
    assert parse_mentions(1, [criterion(sentiment=sentiment)], problems)[0]['sentiment'] == stored
    assert problems == []


def test_invalid_sentiment_keeps_the_mention_without_sentiment():
    problems = []
    mentions = parse_mentions(1, [criterion(sentiment='super')], problems)
    assert mentions == [{'label': 'Reichweite', 'sentiment': None, 'quote': 'zu wenig Reichweite'}]
    assert problems == ["id 1: invalid sentiment 'super' for 'Reichweite'"]


@pytest.mark.parametrize('label', [None, '', '   ', 17])
def test_criterion_without_label_is_dropped(label):
    problems = []
    assert parse_mentions(1, [criterion(label=label), criterion('Preis')], problems) == [
        {'label': 'Preis', 'sentiment': 'negative', 'quote': 'zu wenig Reichweite'}]
    assert problems == ['id 1: criterion without label']


def test_criterion_that_is_no_object_is_dropped():
    problems = []
    assert parse_mentions(1, ['Reichweite'], problems) == []
    assert problems == ["id 1: criterion is not an object: 'Reichweite'"]


def test_label_twice_in_one_post_keeps_the_first_whatever_the_case():
    problems = []
    mentions = parse_mentions(1, [criterion('Preis', 'positiv'), criterion('preis', 'negativ')], problems)
    assert [(mention['label'], mention['sentiment']) for mention in mentions] == [('Preis', 'positive')]
    assert problems == ["id 1: criterion 'preis' twice, kept the first"]


def test_label_and_quote_are_stripped():
    mentions = parse_mentions(1, [criterion(' Ladezeit ', quote='  dauert ewig\n')], [])
    assert mentions == [{'label': 'Ladezeit', 'sentiment': 'negative', 'quote': 'dauert ewig'}]


@pytest.mark.parametrize('quote', [None, '', '   ', 17])
def test_missing_quote_is_stored_as_none(quote):
    problems = []
    assert parse_mentions(1, [criterion(quote=quote)], problems)[0]['quote'] is None
    assert problems == []  # a quote is optional
