from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
CLASS_QUIZ_FILENAME = os.environ.get("CLASS_QUIZ_FILENAME")
ASPECT_QUIZ_FILENAME = os.environ.get("ASPECT_QUIZ_FILENAME")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ALCHEMY_MODEL = os.environ.get("ALCHEMY_MODEL", "meta-llama/llama-3.3-70b-instruct")
CLASSPECT_MODE = os.environ.get("CLASSPECT_MODE", "classifier").lower()
if CLASSPECT_MODE not in {"classifier", "llm"}:
    raise ValueError("CLASSPECT_MODE must be either 'classifier' or 'llm'")
CLASSPECT_CLASSIFIER_MODEL = os.environ.get("CLASSPECT_CLASSIFIER_MODEL", "jev-latest")
CLASSPECT_MODEL = os.environ.get("CLASSPECT_MODEL", "z-ai/glm-5.3-flash")
FRAYMOTIF_MODEL = os.environ.get("FRAYMOTIF_MODEL", "z-ai/glm-5.3")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./paradox.db")
TERMINAL_URL = os.environ.get("TERMINAL_URL", "http://localhost:3000")
API_ROOT_PATH = os.environ.get("API_ROOT_PATH", "/api")
PROMPTS_DIRECTORY = os.environ.get("PROMPTS_DIRECTORY", "prompts")
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "false")
