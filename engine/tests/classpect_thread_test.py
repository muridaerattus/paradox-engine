import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from paradox_engine.classpect import repository
from paradox_engine.classpect.models import (
    ClasspectInterviewDecision,
    ClasspectThread,
    ClasspectThreadMessage,
    ClasspectThreadState,
    ParadoxEngineOutput,
)
from paradox_engine.classpect.thread_service import ClasspectThreadService
from paradox_engine.config import Settings
from paradox_engine.api.dependencies import resources
from paradox_engine.api.rate_limit import IPRateLimiter
from paradox_engine.api.routers.classpect import router
from paradox_engine.prompts.library import PromptLibrary
from paradox_engine.prompts.messages import classpect_interview_messages

CLASS_QUIZ = [
    {
        "question": "I direct my own path.",
        "answers": [
            {"answer": "Agree", "personality_types": ["seer of <aspect>"]},
            {"answer": "Disagree", "personality_types": ["heir of <aspect>"]},
        ],
    }
]
ASPECT_QUIZ = [
    {
        "question": "I seek knowledge.",
        "answers": [
            {"answer": "Agree", "personality_types": ["light"]},
            {"answer": "Disagree", "personality_types": ["void"]},
        ],
    }
]


def _state(*, status="active", version=0, messages=None):
    thread = ClasspectThread(id=str(uuid4()), status=status, version=version)
    return ClasspectThreadState(thread=thread, messages=messages or [])


class FakeRepository:
    def __init__(self, state):
        self.state = state
        self.exchange = None

    async def create_thread(self, opening_message):
        self.opening_message = opening_message
        return self.state

    async def get_thread(self, thread_id):
        return self.state if self.state.thread.id == thread_id else None

    async def append_exchange(self, **kwargs):
        self.exchange = kwargs
        return self.state


class FakeLLM:
    def __init__(self, decision, choices):
        self.decision = decision
        self.choices = choices
        self.request = None
        self.classifier_request = None

    async def classify_choices(self, **kwargs):
        self.classifier_request = kwargs
        return self.choices

    async def generate_structured(self, **kwargs):
        self.request = kwargs
        return self.decision


class FakeClasspect:
    def __init__(self):
        self.personality = None

    async def calculate_title_from_answers(
        self,
        personality,
        class_quiz,
        aspect_quiz,
        class_answer_indexes,
        aspect_answer_indexes,
    ):
        self.personality = personality
        self.answer_indexes = (class_answer_indexes, aspect_answer_indexes)
        return ParadoxEngineOutput(
            class_result="Seer",
            aspect_result="Light",
            llm_response="Your title is the Seer of Light.",
        )


def _service(state, decision, choices):
    prompts = PromptLibrary.load(Path("prompts"))
    repo = FakeRepository(state)
    llm = FakeLLM(decision, choices)
    classpect = FakeClasspect()
    service = ClasspectThreadService(
        llm=llm,
        prompts=prompts,
        settings=Settings(),
        classpect=classpect,
        class_quiz=CLASS_QUIZ,
        aspect_quiz=ASPECT_QUIZ,
        thread_repository=repo,
    )
    return service, repo, llm, classpect


def test_interview_prompt_uses_paradox_engine_persona_without_final_only_task():
    prompts = PromptLibrary.load(Path("prompts"))
    messages = classpect_interview_messages(
        prompts,
        history=[{"role": "user", "content": "I take risks."}],
        unresolved_questions=["I direct my own path."],
    )
    system_prompt = messages[0]["content"]
    assert "You are the PARADOX ENGINE" in system_prompt
    assert "You do not decide whether the interview is complete" in system_prompt
    assert "There is no follow-up to this response" not in system_prompt
    assert "{coverage_instruction}" not in system_prompt
    assert "1 quiz item(s)" in system_prompt
    assert 'Never call a question "final," "last,"' in system_prompt
    assert messages[-1] == {"role": "user", "content": "I take risks."}


def test_active_interview_appends_in_character_follow_up():
    state = _state(
        messages=[
            ClasspectThreadMessage(
                id=1,
                thread_id="unused",
                role="assistant",
                content="Describe yourself.",
            )
        ]
    )
    decision = ClasspectInterviewDecision(
        response="What do you protect when the cost becomes severe?",
        personality_summary="A premature summary that must not complete the thread.",
    )
    service, repo, llm, classpect = _service(
        state,
        decision,
        {
            "class_question_1": "insufficient_evidence",
            "aspect_question_1": "option_1",
        },
    )

    asyncio.run(service.continue_thread(state.thread.id, "My friends."))

    assert repo.exchange == {
        "thread_id": state.thread.id,
        "expected_version": 0,
        "user_message": "My friends.",
        "assistant_message": "What do you protect when the cost becomes severe?",
        "classifier_answers": {"aspect_question_1": "option_1"},
    }
    assert llm.request["messages"][-1]["content"] == "My friends."
    assert "I direct my own path." in llm.request["messages"][0]["content"]
    assert (
        "Available responses: Agree | Disagree" in llm.request["messages"][0]["content"]
    )
    assert (
        "insufficient_evidence"
        in llm.classifier_request["questions"]["class_question_1"]["criteria"]
    )
    assert classpect.personality is None


def test_ready_interview_runs_existing_classpect_pipeline():
    state = _state()
    state.classifier_answers = {"aspect_question_1": "option_1"}
    decision = ClasspectInterviewDecision(
        response="",
        personality_summary="Protective, analytical, and motivated by discovery.",
    )
    service, repo, _, classpect = _service(
        state,
        decision,
        {"class_question_1": "option_1"},
    )

    asyncio.run(service.continue_thread(state.thread.id, "I seek hidden patterns."))

    assert classpect.personality == decision.personality_summary
    assert classpect.answer_indexes == ([0], [0])
    assert set(service.llm.classifier_request["questions"]) == {"class_question_1"}
    assert repo.exchange["classifier_answers"] == {"class_question_1": "option_1"}
    assert repo.exchange["result"].class_result == "Seer"
    assert repo.exchange["assistant_message"] == "Your title is the Seer of Light."


def test_repository_persists_and_completes_thread(tmp_path, monkeypatch):
    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'threads.db'}")
        session_maker = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        monkeypatch.setattr(repository, "session_maker", session_maker)
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)

        created = await repository.create_thread("The Engine awakens.")
        completed = await repository.append_exchange(
            thread_id=created.thread.id,
            expected_version=0,
            user_message="I persist.",
            assistant_message="You are the Heir of Time.",
            classifier_answers={"class_question_1": "option_2"},
            result=ParadoxEngineOutput(
                class_result="Heir",
                aspect_result="Time",
                llm_response="You are the Heir of Time.",
            ),
        )
        reloaded = await repository.get_thread(created.thread.id)
        await engine.dispose()
        return created, completed, reloaded

    created, completed, reloaded = asyncio.run(scenario())
    assert created.messages[0].id is not None
    assert completed.thread.status == "completed"
    assert completed.thread.version == 1
    assert [message.role for message in reloaded.messages] == [
        "assistant",
        "user",
        "assistant",
    ]
    assert reloaded.thread.class_result == "Heir"
    assert reloaded.classifier_answers == {"class_question_1": "option_2"}


def test_thread_endpoints_return_full_state_and_validate_messages():
    state = _state(
        messages=[
            ClasspectThreadMessage(
                id=1,
                thread_id="unused",
                role="assistant",
                content="The Engine awakens.",
            )
        ]
    )

    class FakeThreadService:
        async def create_thread(self):
            return state

        async def get_thread(self, thread_id):
            return state

        async def continue_thread(self, thread_id, message):
            return state

    app = FastAPI()
    app.state.rate_limiter = IPRateLimiter()
    app.include_router(router)
    app.dependency_overrides[resources] = lambda: SimpleNamespace(
        classpect_threads=FakeThreadService()
    )

    with TestClient(app) as client:
        created = client.post("/classpect/thread")
        create_limited = client.post("/classpect/thread")
        fetched = client.get(f"/classpect/thread/{state.thread.id}")
        posted = client.post(
            f"/classpect/thread/{state.thread.id}", json={"message": "First"}
        )
        message_limited = client.post(
            f"/classpect/thread/{state.thread.id}", json={"message": "Second"}
        )
        app.state.rate_limiter = IPRateLimiter()
        invalid = client.post(
            f"/classpect/thread/{state.thread.id}", json={"message": ""}
        )

    assert created.status_code == 201
    assert created.json()["thread_id"] == state.thread.id
    assert created.json()["messages"][0]["content"] == "The Engine awakens."
    assert create_limited.status_code == 429
    assert create_limited.headers["retry-after"] == "60"
    assert fetched.status_code == 200
    assert posted.status_code == 200
    assert message_limited.status_code == 429
    assert message_limited.headers["retry-after"] == "3"
    assert invalid.status_code == 422
