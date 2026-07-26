"""add session_settings table (encrypted per-session credentials)

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Создаёт таблицу session_settings: 1:1 с sessions.id, FK CASCADE.

    Колонки:
    - session_id: PK + FK на sessions.id
    - llm_model: имя модели (e.g. claude-3-5-sonnet-20241022)
    - *_encrypted: Fernet-токены ключей
    - invest_sandbox: bool toggle для INVEST_GRPC_API_SANDBOX
    - updated_at: для аудита
    """
    op.create_table(
        'session_settings',
        sa.Column('session_id', sa.String(length=64), nullable=False),
        sa.Column('llm_model', sa.String(length=120), nullable=False),
        sa.Column('llm_api_key_encrypted', sa.Text(), nullable=False),
        sa.Column('openai_api_key_encrypted', sa.Text(), nullable=False),
        sa.Column('invest_token_encrypted', sa.Text(), nullable=False),
        sa.Column('invest_sandbox', sa.Boolean(),
                  nullable=False, server_default=sa.true()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['session_id'], ['sessions.id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('session_id'),
    )


def downgrade() -> None:
    """Удаляет таблицу session_settings."""
    op.drop_table('session_settings')