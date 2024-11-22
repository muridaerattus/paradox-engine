from dotenv import load_dotenv, find_dotenv
import os
from typing import Literal
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from langchain_anthropic import ChatAnthropic


load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

class ResponseModelGreeting(BaseModel):
    greeting: Literal['hi', 'hello', 'sup'] = Field(description='random greeting, pick something different than the user')

llm = ChatAnthropic(model="claude-3-5-haiku-20241022").with_structured_output(ResponseModelGreeting)
print(llm.invoke('hello'))
