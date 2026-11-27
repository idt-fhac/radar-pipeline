# quotes.py — quote fidelity: does a quote stand word for word in its comment? Reads only.
#
# Nothing is stored for this: the answer follows from criterion_mention.quote and post.text,
# so it is computed on request and can never disagree with the data.

from sqlalchemy import case, func, select

from radar.db.models.annotation import CriterionMention
from radar.db.models.collection import Post

# The rule human annotators are held to (clean_annotation: `quote in text`): the exact
# string, case and whitespace included. SQLite's instr() gives the position of the first
# match, 0 if there is none and NULL for a mention without quote.
VERBATIM = func.instr(Post.text, CriterionMention.quote) > 0


def quote_fidelity(session, run_ids=None):
    """Per annotation run: how many of its quotes occur word for word in the text of their post.

    The prompt asks for quotes copied verbatim; an LLM does not always comply (the parser
    cannot check it, it does not know the texts). Human runs reach 1.0 by construction,
    because the annotation tool refuses any other quote.

    The measure only says that a quote was copied, not that it is complete or supports the
    criterion: a quote cut short (e.g. by the JSON mode) still counts as verbatim.
    :param run_ids: only these annotation runs; None means all runs
    :return: list of dicts ordered by run id, one per run that has criterion mentions:
        'run_id', 'mentions' (criterion mentions), 'quotes' (mentions with a quote),
        'verbatim' (quotes found in the text), 'share' (verbatim / quotes; None without quotes)
    """
    stmt = (
        select(CriterionMention.run_id,
               func.count().label('mentions'),
               func.count(CriterionMention.quote).label('quotes'),  # count(column) skips NULL
               func.sum(case((VERBATIM, 1), else_=0)).label('verbatim'))
        .join(Post, Post.id == CriterionMention.post_id)
        .group_by(CriterionMention.run_id)
        .order_by(CriterionMention.run_id)
    )
    if run_ids is not None:
        stmt = stmt.where(CriterionMention.run_id.in_(run_ids))
    return [{'run_id': run_id, 'mentions': mentions, 'quotes': quotes, 'verbatim': verbatim,
             'share': verbatim / quotes if quotes else None}
            for run_id, mentions, quotes, verbatim in session.execute(stmt)]


def non_verbatim_quotes(session, run_id):
    """The quotes of one run that do not occur in the text of their post, to look at them.

    :return: list of dicts ordered by post and mention: 'post_id', 'label', 'quote', 'text'
    """
    stmt = (
        select(CriterionMention.post_id, CriterionMention.label, CriterionMention.quote, Post.text)
        .join(Post, Post.id == CriterionMention.post_id)
        .where(CriterionMention.run_id == run_id, CriterionMention.quote.is_not(None), ~VERBATIM)
        .order_by(CriterionMention.post_id, CriterionMention.id)
    )
    return [{'post_id': post_id, 'label': label, 'quote': quote, 'text': text}
            for post_id, label, quote, text in session.execute(stmt)]
