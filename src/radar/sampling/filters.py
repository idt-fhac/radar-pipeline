# filters.py — content filters on loaded posts; no database access, deterministic

import re

from radar.db.models.collection import Post
from radar.privacy import MENTION

URL = re.compile(r'https?://\S+|www\.\S+')


def remove_duplicates(posts: list[Post]) -> list[Post]:
    """Keep one post per normalised text: the earliest, ties broken by id."""
    seen = set()
    result = []
    for post in sorted(posts, key=lambda p: (p.published_at, p.id)):
        text = post.text.strip().lower()
        if text not in seen:
            seen.add(text)
            result.append(post)
    return result


def without_links_and_mentions(text: str) -> str:
    """The text with every link and @-mention replaced by a blank: neither says anything about the topic."""
    text = URL.sub(' ', text)
    return MENTION.sub(' ', text)  # pseudonymised mentions are hex and would count as letters and words


def is_short_post(post: Post, threshold: int) -> bool:
    """True if the post has fewer words than threshold; links and @-mentions are not words.

    Otherwise a reply, which often starts with '@<pseudonym>', would pass with one word less.
    """
    return len(without_links_and_mentions(post.text).split()) < threshold


def remove_short(posts: list[Post], threshold: int = 5) -> list[Post]:
    return [post for post in posts if not is_short_post(post, threshold)]


def is_empty_post(post: Post) -> bool:
    """True if nothing but links, @-mentions, emojis, digits or punctuation is left."""
    return not any(char.isalpha() for char in without_links_and_mentions(post.text))


def remove_empty(posts: list[Post]) -> list[Post]:
    return [post for post in posts if not is_empty_post(post)]
