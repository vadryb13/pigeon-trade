"""initial: runs + hypotheses + sessions

Revision ID: fcb396c4088d
Revises: 
Create Date: 2026-07-12 00:23:01.972607

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fcb396c4088d'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('sessions',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('runs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('goal', sa.Text(), nullable=False),
    sa.Column('session_id', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('summary_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_runs_session_id'), 'runs', ['session_id'], unique=False)
    op.create_table('hypotheses',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('family', sa.String(length=64), nullable=False),
    sa.Column('ticker', sa.String(length=32), nullable=False),
    sa.Column('config_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('embedding', VECTOR(dim=1536), nullable=True),
    sa.Column('dsr', sa.Float(), nullable=True),
    sa.Column('pbo', sa.Float(), nullable=True),
    sa.Column('cpcv', sa.Float(), nullable=True),
    sa.Column('sharpe', sa.Float(), nullable=True),
    sa.Column('max_drawdown', sa.Float(), nullable=True),
    sa.Column('is_valid', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_hypotheses_run_id'), 'hypotheses', ['run_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_hypotheses_run_id'), table_name='hypotheses')
    op.drop_table('hypotheses')
    op.drop_index(op.f('ix_runs_session_id'), table_name='runs')
    op.drop_table('runs')
    op.drop_table('sessions')
