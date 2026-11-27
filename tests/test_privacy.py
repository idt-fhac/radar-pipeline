import hashlib
import hmac
import re

import pytest

from radar import privacy
from radar.privacy import MENTION, pseudonym, pseudonymize_mentions


def test_pseudonym_is_short_hex_and_stable():
    first = pseudonym('youtube', 'UC-anna')
    assert re.fullmatch(r'[0-9a-f]{12}', first)
    assert pseudonym('youtube', 'UC-anna') == first


def test_pseudonym_is_an_hmac_of_source_and_account():
    # Pins the algorithm: changing it would give every author a new pseudonym, like a new key.
    expected = hmac.new(privacy.SALT, b'youtube:UC-anna', hashlib.sha256).hexdigest()[:12]
    assert pseudonym('youtube', 'UC-anna') == expected


def test_pseudonym_differs_by_account_and_source():
    assert pseudonym('youtube', 'UC-anna') != pseudonym('youtube', 'UC-ben')
    assert pseudonym('youtube', 'UC-anna') != pseudonym('reddit', 'UC-anna')


def test_pseudonym_depends_on_the_key(monkeypatch):
    before = pseudonym('youtube', 'UC-anna')
    monkeypatch.setattr(privacy, 'SALT', bytes.fromhex('cd' * 32))
    assert pseudonym('youtube', 'UC-anna') != before


@pytest.mark.parametrize('account_id', [None, '', '[deleted]'])
def test_pseudonym_refuses_a_missing_account(account_id):
    # otherwise all deleted users would become one person
    with pytest.raises(ValueError, match='no account id'):
        pseudonym('reddit', account_id)


def test_known_handle_gets_the_pseudonym_of_its_account():
    handles = {'@anna': pseudonym('youtube', 'UC-anna')}
    text = pseudonymize_mentions('@anna das stimmt nicht', 'youtube', handles)
    assert text == f'@{handles["@anna"]} das stimmt nicht'


def test_unknown_handle_is_pseudonymised_from_the_handle_itself():
    text = pseudonymize_mentions('Frag mal @ben', 'youtube', {})
    assert text == f'Frag mal @{pseudonym("youtube", "@ben")}'
    assert 'ben' not in text


def test_full_stop_after_a_mention_stays_in_the_text():
    handles = {'@anna': 'aaaaaaaaaaaa'}
    assert pseudonymize_mentions('Danke @anna.', 'youtube', handles) == 'Danke @aaaaaaaaaaaa.'
    assert pseudonymize_mentions('Danke @anna...', 'youtube', handles) == 'Danke @aaaaaaaaaaaa...'


def test_handle_with_dot_dash_and_digits_is_one_mention():
    handles = {'@ASSO.24-x_y': 'bbbbbbbbbbbb'}
    assert pseudonymize_mentions('@ASSO.24-x_y: nein', 'youtube', handles) == '@bbbbbbbbbbbb: nein'


def test_every_mention_in_a_text_is_replaced():
    handles = {'@anna': 'aaaaaaaaaaaa', '@ben': 'bbbbbbbbbbbb'}
    text = pseudonymize_mentions('@anna und @ben, hört auf @anna', 'youtube', handles)
    assert text == '@aaaaaaaaaaaa und @bbbbbbbbbbbb, hört auf @aaaaaaaaaaaa'


def test_text_without_mention_is_unchanged():
    text = 'Die Reichweite im Winter ist ein Witz.'
    assert pseudonymize_mentions(text, 'youtube', {'@anna': 'aaaaaaaaaaaa'}) == text


def test_mail_address_loses_its_domain_too():
    # '@example.org' looks like a mention; replacing it removes personal data as well
    text = pseudonymize_mentions('schreib an anna@example.org', 'youtube', {})
    assert 'example' not in text and text.startswith('schreib an anna@')


def test_a_pseudonymised_mention_is_still_recognised_as_a_mention():
    # the sampling filters rely on this to ignore mentions
    text = pseudonymize_mentions('@anna', 'youtube', {})
    assert MENTION.fullmatch(text)
