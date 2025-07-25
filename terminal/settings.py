from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
GUILD_ID = int(os.environ.get("GUILD_ID"))
API_URL = os.environ.get(
    "API_URL", "http://localhost:8000"
)  # Default to local FastAPI server
