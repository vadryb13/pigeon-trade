"""add chat messages table

Revision ID: a1b2c3d4e5f6
Revises: fcb396c4088d
Create Date: 2026-07-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'fcb396c4088d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Создаёт таблицу chat_messages + индекс."""
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.String(length=64), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['session_id'], ['sessions.id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_chat_messages_session_created',
        'chat_messages',
        ['session_id', 'created_at'],
    )


def downgrade() -> None:
    """Удаляет таблицу chat_messages."""
    op.drop_index('ix_chat_messages_session_created', table_name='chat_messages')
    op.drop_table('chat_messages')
