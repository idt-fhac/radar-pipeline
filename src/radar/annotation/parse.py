# parse.py — turn an LLM answer into validated annotations. No database, no API: plain data in, plain data out.

import logging

logger = logging.getLogger(__name__)

# The prompt asks in German; the database stores English values (decided 16.11.).
STANCES = {'dafuer': 'pro', 'dagegen': 'contra', 'gemischt': 'mixed', 'unklar': 'unclear'}
SENTIMENTS = {'positiv': 'positive', 'negativ': 'negative', 'neutral': 'neutral'}


def normalise(value: object) -> str:
    """'Dafür ' -> 'dafuer': models sometimes write umlauts or capitals despite the prompt.

    Anything that is not a string becomes '' (no valid key), so the lookup simply fails.
    """
    if not isinstance(value, str):
        return ''
    return value.strip().lower().replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe')


def parse_response(parsed, sent_ids):
    """Validate the parsed JSON answer of one batch against the ids that were sent.

    :param parsed: the answer as dict (llm_client result 'parsed'), may be None
    :param sent_ids: post ids of the batch
    :return: (annotations, problems)
        annotations: {post_id: {'stance': str, 'mentions': [{'label', 'sentiment', 'quote'}]}}
                     only for valid entries; every sent id missing here is resent later
        problems:    list of short descriptions (logged, useful to judge a model)
    """
    annotations = {}
    problems = []
    sent = set(sent_ids)

    results = parsed.get('ergebnisse') if isinstance(parsed, dict) else None
    if not isinstance(results, list):
        return annotations, ['no list "ergebnisse" in the answer']

    for entry in results:
        if not isinstance(entry, dict):
            problems.append(f'entry is not an object: {entry!r}')
            continue

        post_id = to_int(entry.get('id'))
        if post_id not in sent:
            problems.append(f'id {entry.get("id")!r} was not in the batch')
            continue
        if post_id in annotations:
            problems.append(f'id {post_id} answered twice, kept the first answer')
            continue

        stance = STANCES.get(normalise(entry.get('position')))
        if stance is None:
            # Invalid entry counts as missing: the post is resent in the next round.
            problems.append(f'id {post_id}: invalid position {entry.get("position")!r}')
            continue

        criteria = entry.get('kriterien')
        if not isinstance(criteria, list):
            problems.append(f'id {post_id}: "kriterien" is not a list')
            continue

        annotations[post_id] = {'stance': stance,
                                'mentions': parse_mentions(post_id, criteria, problems)}

    missing = sent - annotations.keys()
    if missing:
        problems.append(f'{len(missing)} of {len(sent)} ids without valid answer: {sorted(missing)}')
    for problem in problems:
        logger.warning(problem)
    return annotations, problems


def parse_mentions(post_id, criteria, problems):
    """Validate the criteria of one post; drop broken ones, keep the rest."""
    mentions = []
    seen = set()
    for criterion in criteria:
        if not isinstance(criterion, dict):
            problems.append(f'id {post_id}: criterion is not an object: {criterion!r}')
            continue
        label = criterion.get('kriterium')
        if not isinstance(label, str) or not label.strip():
            problems.append(f'id {post_id}: criterion without label')
            continue
        label = label.strip()
        if label.lower() in seen:  # the prompt allows each criterion once per post
            problems.append(f'id {post_id}: criterion {label!r} twice, kept the first')
            continue
        seen.add(label.lower())

        sentiment = SENTIMENTS.get(normalise(criterion.get('sentiment')))
        if sentiment is None:  # keep the mention: the label is the main result, sentiment is optional
            problems.append(f'id {post_id}: invalid sentiment {criterion.get("sentiment")!r} for {label!r}')
        quote = criterion.get('zitat')
        quote = quote.strip() if isinstance(quote, str) and quote.strip() else None

        mentions.append({'label': label, 'sentiment': sentiment, 'quote': quote})
    return mentions


def to_int(value):
    """Ids may come back as 17 or "17"; anything else is invalid."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None
