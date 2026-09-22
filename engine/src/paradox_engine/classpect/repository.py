from datetime import datetime, timezone

from sqlalchemy import update
from sqlmodel import select

from paradox_engine.alchemy.repository import session_maker
from paradox_engine.classpect.models import (
    ClasspectThread,
    ClasspectThreadMessage,
    ClasspectThreadState,
    ParadoxEngineOutput,
)


class ThreadNotFoundError(LookupError):
    pass


class ThreadConflictError(RuntimeError):
    pass


async def create_thread(opening_message: str) -> ClasspectThreadState:
    thread = ClasspectThread()
    message = ClasspectThreadMessage(
        thread_id=thread.id,
        role="assistant",
        content=opening_message,
    )
    async with session_maker.begin() as session:
        session.add(thread)
        session.add(message)
    return ClasspectThreadState(thread=thread, messages=[message])


async def get_thread(thread_id: str) -> ClasspectThreadState | None:
    async with session_maker() as session:
        thread = await session.get(ClasspectThread, thread_id)
        if thread is None:
            return None
        result = await session.exec(
            select(ClasspectThreadMessage)
            .where(ClasspectThreadMessage.thread_id == thread_id)
            .order_by(ClasspectThreadMessage.id)
        )
        return ClasspectThreadState(thread=thread, messages=list(result.all()))


async def append_exchange(
    *,
    thread_id: str,
    expected_version: int,
    user_message: str,
    assistant_message: str,
    result: ParadoxEngineOutput | None = None,
) -> ClasspectThreadState:
    values: dict = {
        "version": expected_version + 1,
        "updated_at": datetime.now(timezone.utc),
    }
    if result is not None:
        values.update(
            status="completed",
            class_result=result.class_result,
            aspect_result=result.aspect_result,
            result=result.llm_response,
        )

    async with session_maker.begin() as session:
        update_result = await session.exec(
            update(ClasspectThread)
            .where(
                ClasspectThread.id == thread_id,
                ClasspectThread.version == expected_version,
                ClasspectThread.status == "active",
            )
            .values(**values)
        )
        if update_result.rowcount != 1:
            exists = await session.get(ClasspectThread, thread_id)
            if exists is None:
                raise ThreadNotFoundError(thread_id)
            raise ThreadConflictError(thread_id)
        session.add(
            ClasspectThreadMessage(
                thread_id=thread_id, role="user", content=user_message
            )
        )
        session.add(
            ClasspectThreadMessage(
                thread_id=thread_id,
                role="assistant",
                content=assistant_message,
            )
        )

    state = await get_thread(thread_id)
    if state is None:
        raise ThreadNotFoundError(thread_id)
    return state
