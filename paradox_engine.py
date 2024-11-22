import aiofiles
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def to_snake_case(string):
    letters_and_spaces = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    string = ''.join(filter(letters_and_spaces.__contains__, string))
    return ('_'.join(string.split(' '))).upper()

async def create_enum_from_json(question):
    answer_list = [a['answer'] for a in question['answers']]
    return Enum('AnswerField', {to_snake_case(a): a for a in answer_list})

async def create_model_from_enum(enum):
    class AnswerToQuizQuestion(BaseModel):
        ThoughtProcess: str = Field(description='About 30 words or less to justify an answer before you write it down.')
        AnswerField: enum
    return AnswerToQuizQuestion

async def answer_questions(quiz_json, llm, prompt, character_description):
    results = set()
    for question in quiz_json:
        for answer in question['answers']:
            results.update(answer['personality_types'])

    result_scores = {result: 0 for result in results}
    for question in quiz_json:
        answer_list = question['answers']
        answers_by_text = {a['answer']: a['personality_types'] for a in answer_list}

        answer_enum = await create_enum_from_json(question)
        answer_model = await create_model_from_enum(answer_enum)

        structured_llm = llm.with_structured_output(answer_model)
        parser = PydanticOutputParser(pydantic_object=answer_model)
        prompted_llm = prompt | structured_llm
        try:
            llm_response = await prompted_llm.ainvoke(
                {
                    'character_description': character_description, 
                    'format_instructions': parser.get_format_instructions(),
                    'question': question['question']
                }
            )
            
            answer = llm_response.AnswerField.value
            personality_types = answers_by_text[answer]
            for result in personality_types:
                result_scores[result] += 1

            print(f"{question['question']}: {answer}")
        except ValidationError as e:
            print('Question skipped due to validation error.')
        
    return max(result_scores, key=result_scores.get)

async def calculate_title(character_description, class_quiz_json, aspect_quiz_json):
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.0)
    prompt = ChatPromptTemplate([
        ('system', 
         """You are answering a personality quiz. Your personality is described in a few paragraphs below.
         <PERSONALITY>
         {character_description}
         </PERSONALITY>
         You are about to be given the next question in the quiz. Pick the answer that most fits with your personality.
         {format_instructions}"""),
        ('user', '{question}')
    ])

    class_result = await answer_questions(class_quiz_json, llm, prompt, character_description)
    aspect_result = await answer_questions(aspect_quiz_json, llm, prompt, character_description)

    class_result = class_result.split(' ')[0].capitalize()
    aspect_result = aspect_result.capitalize()

    print(f"Final answer: {class_result} of {aspect_result}")

    # retrieve class and aspect summaries
    class_prompt = None
    aspect_prompt = None
    paradox_engine_prompt = None

    async with aiofiles.open(f'prompts/classes/{class_result.lower()}.md') as f:
        class_prompt = await f.read()
    async with aiofiles.open(f'prompts/aspects/{aspect_result.lower()}.md') as f:
        aspect_prompt = await f.read()
    async with aiofiles.open('prompts/paradox_engine.md') as f:
        paradox_engine_prompt = await f.read()

    llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')
    prompt = ChatPromptTemplate([
        ('system', paradox_engine_prompt),
        ('user', f'{class_result} of {aspect_result}')
    ])
    prompted_llm = prompt | llm
    llm_response = await prompted_llm.ainvoke({
        'character_description': character_description,
        'class_data': class_prompt,
        'aspect_data': aspect_prompt
    })

    return llm_response.content