# raw.py — removes personal data from raw YouTube API items before they are stored.
# Every function works on a copy, so the items returned by the API stay unchanged.

import copy

from radar import privacy

SOURCE = 'youtube'

# fields of a comment snippet that identify its author in plain text
COMMENT_AUTHOR_FIELDS = ('authorDisplayName', 'authorProfileImageUrl', 'authorChannelUrl')


def author_pseudonym(comment):
    """Pseudonym of a comment's author, or None if the API gives no channel id (deleted user)."""
    channel_id = comment['snippet'].get('authorChannelId', {}).get('value')
    return privacy.pseudonym(SOURCE, channel_id) if channel_id else None


def build_handle_map(comments):
    """Map the handle of every author in comments (authorDisplayName, e.g. '@anna') to their pseudonym."""
    handles = {}
    for comment in comments:
        pseudonym = author_pseudonym(comment)
        handle = comment['snippet'].get('authorDisplayName')
        if pseudonym and handle:
            handles[handle] = pseudonym
    return handles


def pseudonymize_fields(container, keys, handles):
    """Replace @-mentions in the given text fields of a dict, if present."""
    for key in keys:
        if key in container:
            container[key] = privacy.pseudonymize_mentions(container[key], SOURCE, handles)


def clean_video(item, handles):
    """Copy of a videos.list item without channel name and with pseudonymised channel id and mentions."""
    item = copy.deepcopy(item)
    snippet = item['snippet']
    snippet['channelId'] = privacy.pseudonym(SOURCE, snippet['channelId'])
    snippet.pop('channelTitle', None)
    pseudonymize_fields(snippet, ('title', 'description'), handles)
    pseudonymize_fields(snippet.get('localized', {}), ('title', 'description'), handles)
    return item


def clean_comment(item, handles):
    """Copy of a comment item without author name, profile image and channel URL,
    with pseudonymised author and video channel ids and mentions in the text."""
    pseudonym = author_pseudonym(item)
    item = copy.deepcopy(item)
    snippet = item['snippet']
    for key in COMMENT_AUTHOR_FIELDS:
        snippet.pop(key, None)
    if pseudonym:
        snippet['authorChannelId'] = {'value': pseudonym}
    else:
        snippet.pop('authorChannelId', None)
    if 'channelId' in snippet:
        snippet['channelId'] = privacy.pseudonym(SOURCE, snippet['channelId'])
    pseudonymize_fields(snippet, ('textDisplay', 'textOriginal'), handles)
    return item
