"""cascade FK for runs/hypotheses on session/run delete

Revision ID: c5f5393682f1
Revises: b2c3d4e5f6a7
Create Date: 2026-07-27 23:21:27.554324

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5f5393682f1'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: cascade FK on runs/hypotheses.

    Run.session_id and Hypothesis.run_id get ondelete="CASCADE" so deleting
    a Session or Run cleans up the dependent rows automatically (B17).
    """
    op.drop_constraint(op.f("hypotheses_run_id_fkey"), "hypotheses", type_="foreignkey")
    op.create_foreign_key(
        op.f("hypotheses_run_id_fkey"),
        "hypotheses", "runs",
        ["run_id"], ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint(op.f("runs_session_id_fkey"), "runs", type_="foreignkey")
    op.create_foreign_key(
        op.f("runs_session_id_fkey"),
        "runs", "sessions",
        ["session_id"], ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f("runs_session_id_fkey"), "runs", type_="foreignkey")
    op.create_foreign_key(
        op.f("runs_session_id_fkey"), "runs", "sessions",
        ["session_id"], ["id"],
    )
    op.drop_constraint(op.f("hypotheses_run_id_fkey"), "hypotheses", type_="foreignkey")
    op.create_foreign_key(
        op.f("hypotheses_run_id_fkey"), "hypotheses", "runs",
        ["run_id"], ["id"],
    )
