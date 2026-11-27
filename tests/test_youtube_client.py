# The YouTube client with a faked API: no request leaves this computer and no quota is used.

import logging

import pytest
import requests

from radar.sources import youtube


class FakeApi:
    """Replaces youtube.get: `answer(endpoint, params)` returns the JSON answer; calls are recorded."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, endpoint, **params):
        self.calls.append((endpoint, params))
        return self.answer(endpoint, params)


@pytest.fixture
def api(monkeypatch):
    def install(answer):
        fake = FakeApi(answer)
        monkeypatch.setattr(youtube, 'get', fake)
        return fake
    return install


def paged(items, key='items'):
    """An answer function that serves the items page by page, as large as maxResults asks for."""
    def answer(endpoint, params):
        start = int(params.get('pageToken') or 0)
        end = start + params['maxResults']
        page = {key: items[start:end]}
        if end < len(items):
            page['nextPageToken'] = str(end)
        return page
    return answer


def http_error(reason=None):
    """An HTTPError as requests raises it, with the error body of the YouTube API (or none)."""
    response = requests.Response()
    response.status_code = 403
    if reason is not None:
        response._content = ('{"error": {"errors": [{"reason": "%s"}]}}' % reason).encode()
    else:
        response._content = b'<html>Forbidden</html>'
    return requests.HTTPError(response=response)


def thread(comment_id, replies=(), total=None):
    """A commentThreads item: a top-level comment with the replies the API delivers along with it."""
    item = {'id': comment_id, 'snippet': {
        'topLevelComment': {'id': comment_id, 'snippet': {'textOriginal': f'Text {comment_id}'}},
        'totalReplyCount': len(replies) if total is None else total}}
    if replies:
        item['replies'] = {'comments': [reply(reply_id, comment_id) for reply_id in replies]}
    return item


def reply(reply_id, parent_id):
    return {'id': reply_id, 'snippet': {'parentId': parent_id, 'textOriginal': f'Text {reply_id}'}}


def ids(comments):
    return [comment['id'] for comment in comments]


# --- get -------------------------------------------------------------------------------------

def test_get_needs_the_api_key(monkeypatch):
    monkeypatch.delenv('YT_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='YT_API_KEY is not set'):
        youtube.get('videos', part='id', id='abc')


def test_error_reason_reads_the_youtube_error_body():
    assert youtube.error_reason(http_error('commentsDisabled')) == 'commentsDisabled'
    assert youtube.error_reason(http_error()) is None


# --- search ----------------------------------------------------------------------------------

def test_get_vids_pages_until_it_has_enough(api):
    videos = [{'id': {'videoId': f'v{number}'}} for number in range(200)]
    fake = api(paged(videos))
    found = youtube.get_vids('Verbrenner-Aus', max_videos=60, search_params={'regionCode': 'DE'})

    assert found == videos[:60]  # in ranked order
    assert [params['maxResults'] for endpoint, params in fake.calls] == [50, 10]
    assert [params['pageToken'] for endpoint, params in fake.calls] == [None, '50']
    endpoint, params = fake.calls[0]
    assert endpoint == 'search'
    assert (params['q'], params['type'], params['regionCode']) == ('Verbrenner-Aus', 'video', 'DE')


def test_get_vids_warns_if_the_search_finds_fewer(api, caplog):
    api(paged([{'id': {'videoId': 'v1'}}, {'id': {'videoId': 'v2'}}]))
    with caplog.at_level(logging.WARNING, logger='radar.sources.youtube'):
        assert len(youtube.get_vids('seltener Begriff', max_videos=10, search_params={})) == 2
    assert 'requested 10 videos, found only 2' in caplog.text


def test_get_vids_uses_the_search_params_of_the_config_by_default(api):
    fake = api(paged([{'id': {'videoId': 'v1'}}]))
    youtube.get_vids('Verbrenner-Aus', max_videos=1)
    for key, value in youtube.SEARCH_PARAMS.items():
        assert fake.calls[0][1][key] == value


# --- video details ---------------------------------------------------------------------------

def test_get_video_details_asks_for_fifty_ids_per_call(api):
    fake = api(lambda endpoint, params: {'items': [{'id': video_id} for video_id in params['id'].split(',')]})
    video_ids = [f'v{number}' for number in range(120)]
    details = youtube.get_video_details(video_ids)

    assert list(details) == video_ids and details['v7'] == {'id': 'v7'}
    assert [len(params['id'].split(',')) for endpoint, params in fake.calls] == [50, 50, 20]
    assert fake.calls[0][0] == 'videos' and fake.calls[0][1]['part'] == 'snippet,statistics'


def test_get_video_details_warns_about_missing_videos(api, caplog):
    api(lambda endpoint, params: {'items': [{'id': 'v1'}]})
    with caplog.at_level(logging.WARNING, logger='radar.sources.youtube'):
        assert set(youtube.get_video_details(['v1', 'v2'])) == {'v1'}
    assert "no details for videos ['v2']" in caplog.text


def test_get_video_details_of_nothing_calls_nothing(api):
    fake = api(lambda endpoint, params: {'items': []})
    assert youtube.get_video_details([]) == {}
    assert fake.calls == []


# --- threads and comments --------------------------------------------------------------------

def test_get_threads_pages_up_to_the_limit(api):
    threads = [thread(f'c{number}') for number in range(250)]
    fake = api(paged(threads))
    assert youtube.get_threads('v1', max_threads=130, order='relevance') == threads[:130]
    assert [params['maxResults'] for endpoint, params in fake.calls] == [100, 30]
    endpoint, params = fake.calls[0]
    assert endpoint == 'commentThreads'
    assert (params['videoId'], params['order'], params['textFormat']) == ('v1', 'relevance', 'plainText')


def test_get_threads_stops_when_the_video_has_no_more(api):
    fake = api(paged([thread('c1'), thread('c2')]))
    assert len(youtube.get_threads('v1', max_threads=100000, order='time')) == 2
    assert len(fake.calls) == 1


def test_disabled_comments_give_no_threads(api, caplog):
    def answer(endpoint, params):
        raise http_error('commentsDisabled')
    api(answer)
    with caplog.at_level(logging.WARNING, logger='radar.sources.youtube'):
        assert youtube.get_threads('v1', max_threads=10, order='time') == []
    assert 'video v1: comments are disabled' in caplog.text


def test_other_api_errors_abort(api):
    def answer(endpoint, params):
        raise http_error('quotaExceeded')
    api(answer)
    with pytest.raises(requests.HTTPError):
        youtube.get_threads('v1', max_threads=10, order='time')


def test_get_all_replies_pages_through_one_thread(api):
    replies = [reply(f'r{number}', 'c1') for number in range(230)]
    fake = api(paged(replies))
    assert youtube.get_all_replies('c1') == replies
    assert [endpoint for endpoint, params in fake.calls] == ['comments'] * 3
    assert fake.calls[0][1]['parentId'] == 'c1'


def comments_api(api, threads, all_replies):
    """Fake for get_comments: the threads of the video and, per thread, what comments.list returns."""
    def answer(endpoint, params):
        if endpoint == 'commentThreads':
            return {'items': threads}
        return {'items': all_replies[params['parentId']]}
    return api(answer)


def test_get_comments_is_a_flat_list_with_each_top_level_comment_before_its_replies(api):
    comments_api(api, [thread('c1', replies=['r1', 'r2']), thread('c2')], {})
    comments = youtube.get_comments('v1', max_threads=10, order='relevance', replies='all')
    assert ids(comments) == ['c1', 'r1', 'r2', 'c2']
    assert [comment.get('totalReplyCount') for comment in comments] == [2, None, None, 0]
    assert comments[1]['snippet']['parentId'] == 'c1'


def test_replies_all_fetches_a_thread_only_if_the_api_delivered_too_few(api):
    more = [reply(f'r{number}', 'c1') for number in range(1, 8)]
    fake = comments_api(api, [thread('c1', replies=['r1', 'r2'], total=7), thread('c2', replies=['x1'])],
                        {'c1': more})
    comments = youtube.get_comments('v1', max_threads=10, order='relevance', replies='all')
    assert ids(comments) == ['c1', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'c2', 'x1']
    assert [endpoint for endpoint, params in fake.calls] == ['commentThreads', 'comments']  # not for c2


def test_replies_included_takes_what_the_api_delivered(api):
    fake = comments_api(api, [thread('c1', replies=['r1', 'r2'], total=7)], {})
    comments = youtube.get_comments('v1', max_threads=10, order='relevance', replies='included')
    assert ids(comments) == ['c1', 'r1', 'r2']
    assert comments[0]['totalReplyCount'] == 7  # the size of the thread, not what was fetched
    assert len(fake.calls) == 1


def test_replies_none_takes_top_level_comments_only(api):
    comments_api(api, [thread('c1', replies=['r1', 'r2']), thread('c2')], {})
    comments = youtube.get_comments('v1', max_threads=10, order='relevance', replies='none')
    assert ids(comments) == ['c1', 'c2']


def test_get_comments_refuses_unknown_settings(api):
    fake = comments_api(api, [], {})
    with pytest.raises(ValueError, match="unknown order: 'newest'"):
        youtube.get_comments('v1', order='newest')
    with pytest.raises(ValueError, match="unknown replies setting: 'some'"):
        youtube.get_comments('v1', replies='some')
    assert fake.calls == []  # checked before the first request
