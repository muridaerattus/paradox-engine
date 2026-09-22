from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from paradox_engine.api.dependencies import resources
from paradox_engine.api.rate_limit import limit_new_threads, limit_thread_messages
from paradox_engine.api.schemas import (
    ClasspectRequest,
    ClasspectResponse,
    ClasspectThreadContinuationRequest,
    ClasspectThreadMessageResponse,
    ClasspectThreadResponse,
)
from paradox_engine.classpect.models import ClasspectThreadState
from paradox_engine.classpect.repository import (
    ThreadConflictError,
    ThreadNotFoundError,
)
from paradox_engine.runtime import RuntimeResources

router = APIRouter(tags=["classpect"])
Resources = Annotated[RuntimeResources, Depends(resources)]
ThreadId = Annotated[UUID, Path(description="ID of the conversation thread")]


def _thread_response(state: ClasspectThreadState) -> ClasspectThreadResponse:
    result = None
    if state.thread.status == "completed":
        result = ClasspectResponse(
            **{
                "class": state.thread.class_result,
                "aspect": state.thread.aspect_result,
                "result": state.thread.result,
            }
        )
    return ClasspectThreadResponse(
        thread_id=UUID(state.thread.id),
        status=state.thread.status,
        messages=[
            ClasspectThreadMessageResponse.model_validate(message, from_attributes=True)
            for message in state.messages
        ],
        result=result,
    )


@router.post("/classpect", response_model=ClasspectResponse)
async def classpect(request: ClasspectRequest, runtime: Resources) -> ClasspectResponse:
    result = await runtime.classpect.calculate_title(
        request.personality,
        runtime.class_quiz,
        runtime.aspect_quiz,
    )
    return ClasspectResponse(
        **{
            "class": result.class_result,
            "aspect": result.aspect_result,
            "result": result.llm_response,
        }
    )


@router.post(
    "/classpect/thread",
    response_model=ClasspectThreadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(limit_new_threads)],
)
async def create_thread(runtime: Resources) -> ClasspectThreadResponse:
    return _thread_response(await runtime.classpect_threads.create_thread())


@router.get("/classpect/thread/{thread_id}", response_model=ClasspectThreadResponse)
async def get_thread_messages(
    thread_id: ThreadId, runtime: Resources
) -> ClasspectThreadResponse:
    try:
        state = await runtime.classpect_threads.get_thread(str(thread_id))
    except ThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc
    return _thread_response(state)


@router.post(
    "/classpect/thread/{thread_id}",
    response_model=ClasspectThreadResponse,
    dependencies=[Depends(limit_thread_messages)],
)
async def send_thread_message(
    thread_id: ThreadId, request: ClasspectThreadContinuationRequest, runtime: Resources
) -> ClasspectThreadResponse:
    try:
        state = await runtime.classpect_threads.continue_thread(
            str(thread_id), request.message
        )
    except ThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc
    except ThreadConflictError as exc:
        raise HTTPException(
            status_code=409, detail="Thread is complete or was concurrently modified"
        ) from exc
    return _thread_response(state)
