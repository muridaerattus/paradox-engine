"""add classpect threads

Revision ID: a81f25f3b912
Revises: 7b35cf258c85
Create Date: 2026-09-22 12:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a81f25f3b912"
down_revision: Union[str, Sequence[str], None] = "7b35cf258c85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "classpect_thread",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("class_result", sa.String(), nullable=True),
        sa.Column("aspect_result", sa.String(), nullable=True),
        sa.Column("result", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_classpect_thread_status"),
        "classpect_thread",
        ["status"],
        unique=False,
    )
    op.create_table(
        "classpect_thread_message",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["classpect_thread.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_classpect_thread_message_thread_id"),
        "classpect_thread_message",
        ["thread_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_classpect_thread_message_thread_id"),
        table_name="classpect_thread_message",
    )
    op.drop_table("classpect_thread_message")
    op.drop_index(op.f("ix_classpect_thread_status"), table_name="classpect_thread")
    op.drop_table("classpect_thread")
