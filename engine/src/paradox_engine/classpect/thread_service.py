from paradox_engine.classpect import repository
from paradox_engine.classpect.models import (
    ClasspectInterviewDecision,
    ClasspectThreadState,
)
from paradox_engine.classpect.repository import (
    ThreadConflictError,
    ThreadNotFoundError,
)
from paradox_engine.classpect.service import ClasspectService
from paradox_engine.classpect.utils import quiz_to_classifier_questions
from paradox_engine.config import Settings
from paradox_engine.llm import LLMClient
from paradox_engine.prompts.library import PromptLibrary
from paradox_engine.prompts.messages import classpect_interview_messages

INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _coverage_questions(
    class_quiz: list[dict], aspect_quiz: list[dict]
) -> tuple[dict[str, dict], dict[str, dict[str, int]], dict[str, str]]:
    questions = {}
    answer_lookups = {}
    labels = {}
    for quiz_name, quiz in (("class", class_quiz), ("aspect", aspect_quiz)):
        quiz_questions, quiz_lookups = quiz_to_classifier_questions(quiz)
        for question_name, question in quiz_questions.items():
            key = f"{quiz_name}_{question_name}"
            question["instructions"] += (
                " Choose insufficient_evidence unless the conversation clearly "
                "supports one of the substantive responses. Do not guess."
            )
            question["criteria"][INSUFFICIENT_EVIDENCE] = (
                "The conversation does not contain enough evidence to confidently "
                "choose another response."
            )
            questions[key] = question
            answer_lookups[key] = quiz_lookups[question_name]
            question_index = int(question_name.removeprefix("question_")) - 1
            source_question = quiz[question_index]
            answer_options = " | ".join(
                answer["answer"] for answer in source_question["answers"]
            )
            labels[key] = (
                f"{source_question['question']} Available responses: {answer_options}"
            )
    return questions, answer_lookups, labels


class ClasspectThreadService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptLibrary,
        settings: Settings,
        classpect: ClasspectService,
        class_quiz: list[dict],
        aspect_quiz: list[dict],
        thread_repository=repository,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.settings = settings
        self.classpect = classpect
        self.class_quiz = class_quiz
        self.aspect_quiz = aspect_quiz
        self.repository = thread_repository

    async def create_thread(self) -> ClasspectThreadState:
        return await self.repository.create_thread(
            self.prompts.classpect_thread_opening.strip()
        )

    async def get_thread(self, thread_id: str) -> ClasspectThreadState:
        state = await self.repository.get_thread(thread_id)
        if state is None:
            raise ThreadNotFoundError(thread_id)
        return state

    async def continue_thread(
        self, thread_id: str, user_message: str
    ) -> ClasspectThreadState:
        state = await self.get_thread(thread_id)
        if state.thread.status != "active":
            raise ThreadConflictError(thread_id)

        history = [
            {"role": message.role, "content": message.content}
            for message in state.messages
        ]
        history.append({"role": "user", "content": user_message})
        questions, answer_lookups, labels = _coverage_questions(
            self.class_quiz, self.aspect_quiz
        )
        choices = await self.llm.classify_choices(
            model=self.settings.classpect_classifier_model,
            state={"conversation": history},
            questions=questions,
        )
        unresolved_questions = [
            labels[name] for name in questions if choices[name] == INSUFFICIENT_EVIDENCE
        ]
        decision = await self.llm.generate_structured(
            model=self.settings.classpect_model,
            messages=classpect_interview_messages(
                self.prompts,
                history=history,
                unresolved_questions=unresolved_questions,
            ),
            output_type=ClasspectInterviewDecision,
            max_tokens=4096,
            reasoning_effort="minimal",
        )

        if unresolved_questions:
            if not decision.response.strip():
                raise ValueError("The interviewer returned an empty follow-up question")
            return await self.repository.append_exchange(
                thread_id=thread_id,
                expected_version=state.thread.version,
                user_message=user_message,
                assistant_message=decision.response,
            )

        if not decision.personality_summary.strip():
            raise ValueError("The interviewer returned an empty personality summary")
        class_answer_indexes = [
            answer_lookups[name][choices[name]]
            for name in questions
            if name.startswith("class_")
        ]
        aspect_answer_indexes = [
            answer_lookups[name][choices[name]]
            for name in questions
            if name.startswith("aspect_")
        ]
        result = await self.classpect.calculate_title_from_answers(
            decision.personality_summary,
            self.class_quiz,
            self.aspect_quiz,
            class_answer_indexes,
            aspect_answer_indexes,
        )
        return await self.repository.append_exchange(
            thread_id=thread_id,
            expected_version=state.thread.version,
            user_message=user_message,
            assistant_message=result.llm_response,
            result=result,
        )
