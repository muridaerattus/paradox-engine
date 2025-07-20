from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
CLASS_QUIZ_FILENAME = os.environ.get('CLASS_QUIZ_FILENAME')
ASPECT_QUIZ_FILENAME = os.environ.get('ASPECT_QUIZ_FILENAME')
DISCORD_BOT_TOKEN = os.environ.get('DISCORD_BOT_TOKEN')
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')
GUILD_ID = int(os.environ.get('GUILD_ID'))
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite+aiosqlite:///./paradox.db')
PROMPTS_DIRECTORY = os.environ.get('PROMPTS_DIRECTORY', 'prompts')
