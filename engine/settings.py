from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLASS_QUIZ_FILENAME = os.environ.get("CLASS_QUIZ_FILENAME")
ASPECT_QUIZ_FILENAME = os.environ.get("ASPECT_QUIZ_FILENAME")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./paradox.db")
TERMINAL_URL = os.environ.get("TERMINAL_URL", "http://localhost:3000")
API_ROOT_PATH = os.environ.get("API_ROOT_PATH", "/api")
PROMPTS_DIRECTORY = os.environ.get("PROMPTS_DIRECTORY", "prompts")
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "false")