import discord
from discord import app_commands
import os
import json
from dotenv import load_dotenv, find_dotenv
from paradox_engine import calculate_title

load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
CLASS_QUIZ_FILENAME = os.environ.get('CLASS_QUIZ_FILENAME')
ASPECT_QUIZ_FILENAME = os.environ.get('ASPECT_QUIZ_FILENAME')
DISCORD_BOT_TOKEN = os.environ.get('DISCORD_BOT_TOKEN')
GUILD_ID = os.environ.get('GUILD_ID')

class_quiz_json = json.load(open(CLASS_QUIZ_FILENAME, 'r'))
aspect_quiz_json = json.load(open(ASPECT_QUIZ_FILENAME, 'r'))

MY_GUILD = discord.Object(id=GUILD_ID)

class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # This copies the global commands over to your guild.
        #self.tree.copy_global_to(guild=MY_GUILD)
        #await self.tree.sync(guild=MY_GUILD)
        #self.tree.clear_commands(guild=MY_GUILD)
        await self.tree.sync()
        #await self.tree.sync(guild=MY_GUILD)

intents = discord.Intents.default()
client = MyClient(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('------')

@client.tree.command()
async def credits(interaction: discord.Interaction):
    """Displays credits for the bot."""
    credits_text = """```
    Discord bot made by @murida.
    Classpect knowledge given by her good friends @reachartwork and Tamago, used with permission.
    Thank you for being part of our fandom.
    ```"""
    await interaction.response.send_message(credits_text)

@client.tree.command()
@app_commands.describe(personality='the personality of your character')
async def classpect(interaction: discord.Interaction, personality: str):
    """Given a personality, decide a Class and Aspect, along with corresponding powers in battle."""
    await interaction.response.defer(thinking=True)
    result = await calculate_title(personality, class_quiz_json, aspect_quiz_json)
    if len(result) > 1800:
        i = 1799
        while i > 0:
            if result[i] == ' ':
                break
            i -= 1
        result_1 = result[:i]
        result_2 = result[i:]
        await interaction.followup.send(f'```{result_1}```')
        await interaction.followup.send(f'```{result_2}```')
    else:
        await interaction.followup.send(f'```{result}```')

client.run(DISCORD_BOT_TOKEN)
