import discord
from discord import app_commands
import json
from paradox_engine import calculate_title
from alchemy.service import alchemize_items
from alchemy.models import Operation
from database.alchemy_database import get_item_by_code
from settings import CLASS_QUIZ_FILENAME, ASPECT_QUIZ_FILENAME, GUILD_ID, DISCORD_BOT_TOKEN

class_quiz_json = json.load(open(CLASS_QUIZ_FILENAME, 'r'))
aspect_quiz_json = json.load(open(ASPECT_QUIZ_FILENAME, 'r'))

MY_GUILD = discord.Object(id=GUILD_ID)

CREDITS_TEXT = """```--- CREDITS ---
Discord bot made by @murida.
Classpect knowledge given by her good friends @reachartwork and Tamago, used with permission.
Example answers provided by her good friend NeoUndying, used with permission.
This is a Homestuck fan project. We are not affiliated with What Pumpkin.
Thank you for being part of our shared fandom.

"Real paradise lies eternally in the person who dreams of it. Why don't you venture forth in search of your own utopia?"```"""

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
    await interaction.response.send_message(CREDITS_TEXT)

@client.tree.command()
@app_commands.describe(personality='the personality of your character')
async def classpect(interaction: discord.Interaction, personality: str):
    """Given a personality, decide a Class and Aspect, along with corresponding powers in battle."""
    await interaction.response.defer(thinking=True)
    try:
        result = await calculate_title(personality, class_quiz_json, aspect_quiz_json)
    except Exception as e:
        print(e)
        await interaction.followup.send('```[ERROR] Skaian link temporarily disconnected. Please try again later.```')
        return
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

@client.tree.command()
@app_commands.describe(item_one="the first item's name or code", item_two="the second item's name or code", operation='the operation to perform')
async def alchemy(interaction: discord.Interaction, item_one: str, item_two: str, operation: Operation):
    """Combine two items using their alchemy codes."""
    await interaction.response.defer(thinking=True)
    try:
        operation_text = '&&' if operation == 'and' else '||'
        combined_item = await alchemize_items(item_one, item_two, operation)
        await interaction.followup.send(f"""```{item_one} {operation_text} {item_two}\nITEM: {combined_item.name}\nCODE: {combined_item.code}\nDESCRIPTION: {combined_item.description}```""")
    except Exception as e:
        print(e)
        await interaction.followup.send('```[ERROR] Skaian link temporarily disconnected. Please try again later.```')
        return
    
@client.tree.command()
@app_commands.describe(code="item's alchemy code")
async def captchalogue(interaction: discord.Interaction, code: str):
    """Get an exicting item by its alchemy code."""
    await interaction.response.defer(thinking=True)
    try:
        item = await get_item_by_code(code)
        if not item:
            await interaction.followup.send('```[WARNING] Item not found. Please check the code and try again.```')
            return
        await interaction.followup.send(f"""```ITEM: {item.name}\nCODE: {item.code}\nDESCRIPTION: {item.description}```""")
    except Exception as e:
        print(e)
        await interaction.followup.send('```[ERROR] Skaian link temporarily disconnected. Please try again later.```')
        return

client.run(DISCORD_BOT_TOKEN)
