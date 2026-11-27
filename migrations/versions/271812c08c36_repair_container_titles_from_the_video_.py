"""repair container titles from the video details

Revision ID: 271812c08c36
Revises: 07f0136fad11
Create Date: 2026-11-27 15:09:08.197127

Data migration, no schema change.

container.title used to come from search.list, which escapes HTML ('&amp;' for '&') and
whose @-mentions were stored as they were. The title of the same video from videos.list
is in the raw JSON of the root post: unescaped and with pseudonymised mentions. This
copies it into container.title. Containers whose root post has no such title (other
sources, test data) keep theirs.

Annotation runs made before this migration with the video title as context saw the old
title (test database only; the production database had no runs yet).
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '271812c08c36'
down_revision: Union[str, Sequence[str], None] = '07f0136fad11'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The root post of a container is its post without parent.
ROOT_TITLE = """(SELECT json_extract(post.raw, '$.snippet.title') FROM post
                WHERE post.container_id = container.id AND post.parent_id IS NULL)"""
REPAIR_TITLES = f'UPDATE container SET title = {ROOT_TITLE} WHERE {ROOT_TITLE} IS NOT NULL'


def upgrade() -> None:
    """Copy the title stored with the root post into container.title."""
    op.execute(REPAIR_TITLES)


def downgrade() -> None:
    """Nothing to restore: the escaped titles of the search results are not kept anywhere."""
