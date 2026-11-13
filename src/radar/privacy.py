# privacy.py — pseudonymisation of authors and @-mentions (applied before anything is stored)

import hashlib
import hmac
import os
import re

salt_hex = os.getenv('SALT_HEX')
if not salt_hex:
    raise RuntimeError('SALT_HEX is not set (expected in .env)')
SALT = bytes.fromhex(salt_hex)

PSEUDONYM_LENGTH = 12

# '@' followed by letters, digits, '_', '.' or '-', e.g. a YouTube handle like @ASSO.24
MENTION = re.compile(r'@[\w.\-]+')


def pseudonym(source, account_id):
    """Stable pseudonym for an account of a source."""
    if not account_id or account_id == '[deleted]':
        raise ValueError(f'no account id for {source!r}, the user may have been deleted')
    data = f'{source}:{account_id}'.encode('utf-8')
    return hmac.new(SALT, data, hashlib.sha256).hexdigest()[:PSEUDONYM_LENGTH]


def pseudonymize_mentions(text, source, handle_to_pseudonym):
    """Replace every @handle in text by @<pseudonym>.

    Known handles (authors seen in the same batch) get the pseudonym of their account,
    so mentions and authors match. Unknown handles are pseudonymised from the handle
    itself; the '@' keeps them apart from account ids.
    """
    def replace(match):
        found = match.group()
        handle = found.rstrip('.')          # '@anna.' at the end of a sentence
        trailing = found[len(handle):]      # keep the full stop in the text
        replacement = handle_to_pseudonym.get(handle) or pseudonym(source, handle)
        return '@' + replacement + trailing

    return MENTION.sub(replace, text)
