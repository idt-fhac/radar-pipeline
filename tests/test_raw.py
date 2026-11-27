import copy
import json

from radar.collection import raw
from radar.privacy import pseudonym


def comment(comment_id='c1', channel_id='UC-anna', name='@anna', text='Die Reichweite reicht mir.', parent_id=None):
    """A comment as the YouTube API returns it (only the fields the pipeline touches)."""
    snippet = {
        'authorDisplayName': name,
        'authorProfileImageUrl': 'https://yt3.ggpht.com/anna.jpg',
        'authorChannelUrl': 'http://www.youtube.com/@anna',
        'authorChannelId': {'value': channel_id},
        'channelId': 'UC-kanal',
        'videoId': 'v1',
        'textDisplay': text,
        'textOriginal': text,
        'likeCount': 3,
        'publishedAt': '2026-11-01T10:00:00Z',
        'updatedAt': '2026-11-01T10:00:00Z',
    }
    if channel_id is None:  # deleted user
        del snippet['authorChannelId']
    if parent_id is not None:
        snippet['parentId'] = parent_id
    return {'kind': 'youtube#comment', 'id': comment_id, 'snippet': snippet}


def video(description='Heute mit @anna im Test.'):
    return {
        'kind': 'youtube#video',
        'id': 'v1',
        'snippet': {
            'channelId': 'UC-kanal',
            'channelTitle': 'Kanalname',
            'title': 'E-Auto Test | @kanalname',
            'description': description,
            'publishedAt': '2026-10-01T08:00:00Z',
            'localized': {'title': 'E-Auto Test | @kanalname', 'description': description},
        },
        'statistics': {'likeCount': '120', 'commentCount': '45'},
    }


# --- authors and handles ---------------------------------------------------------------------

def test_author_pseudonym_comes_from_the_channel_id():
    assert raw.author_pseudonym(comment()) == pseudonym('youtube', 'UC-anna')


def test_deleted_author_has_no_pseudonym():
    assert raw.author_pseudonym(comment(channel_id=None)) is None


def test_handle_map_links_display_names_to_author_pseudonyms():
    comments = [comment('c1', 'UC-anna', '@anna'), comment('c2', 'UC-ben', '@ben'),
                comment('c3', None, '@geloescht')]
    assert raw.build_handle_map(comments) == {
        '@anna': pseudonym('youtube', 'UC-anna'),
        '@ben': pseudonym('youtube', 'UC-ben'),
    }


# --- clean_comment ---------------------------------------------------------------------------

def test_clean_comment_removes_everything_that_names_the_author():
    item = comment(text='@ben du irrst dich, sagt @fremd')
    handles = raw.build_handle_map([item, comment('c2', 'UC-ben', '@ben')])
    clean = raw.clean_comment(item, handles)
    snippet = clean['snippet']

    assert 'authorDisplayName' not in snippet
    assert 'authorProfileImageUrl' not in snippet
    assert 'authorChannelUrl' not in snippet
    assert snippet['authorChannelId'] == {'value': pseudonym('youtube', 'UC-anna')}
    assert snippet['channelId'] == pseudonym('youtube', 'UC-kanal')  # the channel of the video
    expected = f'@{pseudonym("youtube", "UC-ben")} du irrst dich, sagt @{pseudonym("youtube", "@fremd")}'
    assert snippet['textDisplay'] == snippet['textOriginal'] == expected

    stored = json.dumps(clean)
    for plain in ['anna', 'UC-anna', 'UC-ben', 'UC-kanal', '@ben', 'fremd']:
        assert plain not in stored


def test_clean_comment_keeps_the_other_fields():
    clean = raw.clean_comment(comment(parent_id='c0'), {})
    assert clean['id'] == 'c1' and clean['kind'] == 'youtube#comment'
    assert clean['snippet']['parentId'] == 'c0'
    assert clean['snippet']['likeCount'] == 3
    assert clean['snippet']['publishedAt'] == '2026-11-01T10:00:00Z'


def test_clean_comment_of_a_deleted_author_has_no_author_id():
    clean = raw.clean_comment(comment(channel_id=None), {})
    assert 'authorChannelId' not in clean['snippet']


def test_clean_comment_does_not_change_the_api_item():
    item = comment(text='Hallo @ben')
    before = copy.deepcopy(item)
    raw.clean_comment(item, {})
    assert item == before


# --- clean_video -----------------------------------------------------------------------------

def test_clean_video_removes_the_channel_name_and_pseudonymises_ids_and_mentions():
    handles = {'@anna': pseudonym('youtube', 'UC-anna')}
    clean = raw.clean_video(video(), handles)
    snippet = clean['snippet']

    assert 'channelTitle' not in snippet
    assert snippet['channelId'] == pseudonym('youtube', 'UC-kanal')
    assert snippet['description'] == f'Heute mit @{handles["@anna"]} im Test.'
    assert snippet['title'] == f'E-Auto Test | @{pseudonym("youtube", "@kanalname")}'
    assert snippet['localized'] == {'title': snippet['title'], 'description': snippet['description']}
    assert clean['statistics'] == {'likeCount': '120', 'commentCount': '45'}

    stored = json.dumps(clean)
    for plain in ['Kanalname', 'kanalname', 'UC-kanal', '@anna']:
        assert plain not in stored


def test_clean_video_works_without_localized_texts():
    item = video()
    del item['snippet']['localized']
    assert 'localized' not in raw.clean_video(item, {})['snippet']


def test_clean_video_does_not_change_the_api_item():
    item = video()
    before = copy.deepcopy(item)
    raw.clean_video(item, {})
    assert item == before
