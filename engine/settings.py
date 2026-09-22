from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
CLASS_QUIZ_FILENAME = os.environ.get("CLASS_QUIZ_FILENAME")
ASPECT_QUIZ_FILENAME = os.environ.get("ASPECT_QUIZ_FILENAME")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ALCHEMY_MODEL = os.environ.get(
    "ALCHEMY_MODEL", "meta-llama/llama-3.3-70b-instruct"
)
CLASSPECT_MODEL = os.environ.get("CLASSPECT_MODEL", "z-ai/glm-5.2")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./paradox.db")
TERMINAL_URL = os.environ.get("TERMINAL_URL", "http://localhost:3000")
API_ROOT_PATH = os.environ.get("API_ROOT_PATH", "/api")
PROMPTS_DIRECTORY = os.environ.get("PROMPTS_DIRECTORY", "prompts")
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "false")
