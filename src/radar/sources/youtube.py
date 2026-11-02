# youtube.py — fetches data from the YouTube Data API v3.
# Knows nothing about the database: every function returns plain API dicts.

import logging
import os

from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger(__name__)

API_URL = 'https://www.googleapis.com/youtube/v3'

# --- default settings ---------------------------------------------------------
# Search parameters are passed to search.list and define *which* videos are found.
# They are part of the identity of a search query in the database.
SEARCH_PARAMS = {
    'regionCode': 'DE',
    'relevanceLanguage': 'de',
}
MAX_VIDEOS = 10              # videos taken from the search results
MAX_THREADS_PER_VIDEO = 100  # top-level comments (threads) per video
COMMENT_ORDER = 'relevance'       # 'time' (API default) or 'relevance'
REPLIES = 'all'              # 'all', 'included' (up to 5 per thread, as delivered) or 'none'

# --- API limits (fixed by YouTube, not settings) -------------------------------
SEARCH_PAGE_SIZE = 50    # max. maxResults for search.list
VIDEOS_BATCH_SIZE = 50   # max. ids per videos.list call
COMMENTS_PAGE_SIZE = 100 # max. maxResults for commentThreads.list and comments.list


def get(endpoint, **params):
    """GET request to the YouTube API, returns the decoded JSON answer."""
    key = os.getenv('YT_API_KEY')
    if not key:
        raise RuntimeError('YT_API_KEY is not set (expected in .env)')
    params['key'] = key
    answer = requests.get(f'{API_URL}/{endpoint}', params=params, timeout=30)
    answer.raise_for_status()
    return answer.json()


def error_reason(err):
    """Return the reason the YouTube API gives for an HTTP error, e.g. 'commentsDisabled'.

    Returns None if the response has no YouTube error body.
    """
    try:
        return err.response.json()['error']['errors'][0]['reason']
    except (ValueError, KeyError, IndexError):
        return None


def get_vids(search_term, max_videos=MAX_VIDEOS, search_params=None):
    """Search for videos and return up to max_videos search results in ranked order."""
    search_params = SEARCH_PARAMS if search_params is None else search_params
    vids = []
    page_token = None
    while len(vids) < max_videos:
        res = get('search', part='snippet', type='video', q=search_term,
                  maxResults=min(SEARCH_PAGE_SIZE, max_videos - len(vids)),
                  pageToken=page_token, **search_params)
        vids.extend(res['items'])
        page_token = res.get('nextPageToken')
        if page_token is None:
            break
    if len(vids) < max_videos:
        logger.warning('search %r: requested %d videos, found only %d', search_term, max_videos, len(vids))
    return vids


def get_video_details(video_ids):
    """Return full snippet (incl. complete description) and statistics per video.

    The result maps video id -> videos.list item. Up to 50 ids cost one API call.
    """
    details = {}
    for start in range(0, len(video_ids), VIDEOS_BATCH_SIZE):
        batch = video_ids[start:start + VIDEOS_BATCH_SIZE]
        res = get('videos', part='snippet,statistics', id=','.join(batch))
        for item in res['items']:
            details[item['id']] = item
    missing = set(video_ids) - set(details)
    if missing:
        logger.warning('no details for videos %s (deleted or private?)', sorted(missing))
    return details


def get_threads(video_id, max_threads, order):
    """Return up to max_threads comment threads of a video, or [] if comments are disabled."""
    threads = []
    page_token = None
    try:
        while len(threads) < max_threads:
            res = get('commentThreads', part='snippet,replies', videoId=video_id,
                      maxResults=min(COMMENTS_PAGE_SIZE, max_threads - len(threads)),
                      order=order, textFormat='plainText', pageToken=page_token)
            threads.extend(res['items'])
            page_token = res.get('nextPageToken')
            if page_token is None:
                break
    except requests.HTTPError as err:
        if error_reason(err) == 'commentsDisabled':
            logger.warning('video %s: comments are disabled, skipped', video_id)
            return []
        raise
    return threads


def get_all_replies(comment_id):
    """Return all replies to a top-level comment (YouTube has no deeper levels)."""
    replies = []
    page_token = None
    while True:
        res = get('comments', part='snippet', parentId=comment_id,
                  maxResults=COMMENTS_PAGE_SIZE, textFormat='plainText', pageToken=page_token)
        replies.extend(res['items'])
        page_token = res.get('nextPageToken')
        if page_token is None:
            return replies


def get_comments(video_id, max_threads=MAX_THREADS_PER_VIDEO, order=COMMENT_ORDER, replies=REPLIES):
    """Return the comments of a video as one flat list.

    Top-level comments carry an extra key 'totalReplyCount' (taken from their thread);
    replies can be recognised by snippet['parentId'].
    """
    if order not in ('time', 'relevance'):
        raise ValueError(f'unknown order: {order!r}')
    if replies not in ('all', 'included', 'none'):
        raise ValueError(f'unknown replies setting: {replies!r}')

    comments = []
    for thread in get_threads(video_id, max_threads, order):
        top = thread['snippet']['topLevelComment']
        total = thread['snippet']['totalReplyCount']
        top['totalReplyCount'] = total
        comments.append(top)

        included = thread.get('replies', {}).get('comments', [])
        if replies == 'included':
            comments.extend(included)
        elif replies == 'all':
            # the included replies are complete if the thread has no more than those
            comments.extend(included if total <= len(included) else get_all_replies(top['id']))
    return comments
