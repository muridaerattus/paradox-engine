"""add persisted classpect thread answers

Revision ID: c37a5e214c60
Revises: a81f25f3b912
Create Date: 2026-09-22 13:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c37a5e214c60"
down_revision: Union[str, Sequence[str], None] = "a81f25f3b912"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "classpect_thread_answer",
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("question_key", sa.String(), nullable=False),
        sa.Column("choice", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["classpect_thread.id"]),
        sa.PrimaryKeyConstraint("thread_id", "question_key"),
    )
    op.create_index(
        op.f("ix_classpect_thread_answer_thread_id"),
        "classpect_thread_answer",
        ["thread_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_classpect_thread_answer_thread_id"),
        table_name="classpect_thread_answer",
    )
    op.drop_table("classpect_thread_answer")
