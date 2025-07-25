import discord
from discord import app_commands
import httpx
import logging
from settings import GUILD_ID, DISCORD_BOT_TOKEN, API_URL
from utils import split_message

MY_GUILD = discord.Object(id=GUILD_ID)

CREDITS_TEXT = """```--- CREDITS ---
Discord bot made by @murida.
Classpect knowledge given by her good friends @reachartwork and Tamago, used with permission.
Example classpecting answers provided by her good friend NeoUndying, used with permission.
Example alchemy and fraymotifs written by @murida.
This is a Homestuck fan project. We are not affiliated with What Pumpkin.
Thank you for being part of our shared fandom.

"Real paradise lies eternally in the person who dreams of it. Why don't you venture forth in search of your own utopia?"```"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()


intents = discord.Intents.default()
client = MyClient(intents=intents)


@client.event
async def on_ready():
    logger.info(f"Logged in as {client.user} (ID: {client.user.id})")
    logger.info("------")


@client.tree.command()
async def credits(interaction: discord.Interaction):
    """Displays credits for the bot."""
    await interaction.response.send_message(CREDITS_TEXT)


@client.tree.command()
@app_commands.describe(personality="the personality of your character")
async def classpect(interaction: discord.Interaction, personality: str):
    """Given a personality, decide a Class and Aspect, along with corresponding powers in battle."""
    await interaction.response.defer(thinking=True)
    async with httpx.AsyncClient() as client_http:
        try:
            resp = await client_http.post(
                f"{API_URL}/classpect", json={"personality": personality}
            )
            resp.raise_for_status()
            result = resp.json()["result"]
        except Exception as e:
            logger.error(e)
            await interaction.followup.send(
                "```[ERROR] Skaian link temporarily disconnected. Please try again later.```"
            )
            return
    for chunk in split_message(result):
        await interaction.followup.send(f"```{chunk}```")


@client.tree.command()
@app_commands.describe(
    item_one="the first item's name or code",
    item_two="the second item's name or code",
    operation="the operation to perform",
)
async def alchemy(
    interaction: discord.Interaction, item_one: str, item_two: str, operation: str
):
    """Combine two items using their alchemy codes."""
    await interaction.response.defer(thinking=True)
    async with httpx.AsyncClient() as client_http:
        try:
            resp = await client_http.post(
                f"{API_URL}/alchemy/alchemize",
                json={
                    "item_one": item_one,
                    "item_two": item_two,
                    "operation": operation,
                },
            )
            resp.raise_for_status()
            combined_item = resp.json()
            operation_text = "&&" if operation == "and" else "||"
            output = f"{item_one} {operation_text} {item_two}\nITEM: {combined_item['name']}\nCODE: {combined_item['code']}\nDESCRIPTION: {combined_item['description']}"
            for chunk in split_message(output):
                await interaction.followup.send(f"```{chunk}```")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                await interaction.followup.send(
                    "```[WARNING] Item not found. Please check the code and try again.```"
                )
            else:
                logger.error(e)
                await interaction.followup.send(
                    "```[ERROR] Skaian link temporarily disconnected. Please try again later.```"
                )
        except Exception as e:
            logger.error(e)
            await interaction.followup.send(
                "```[ERROR] Skaian link temporarily disconnected. Please try again later.```"
            )


@client.tree.command()
@app_commands.describe(code="item's alchemy code")
async def captchalogue(interaction: discord.Interaction, code: str):
    """Get an existing item by its alchemy code."""
    await interaction.response.defer(thinking=True)
    async with httpx.AsyncClient() as client_http:
        try:
            resp = await client_http.get(
                f"{API_URL}/alchemy/captchalogue", params={"code": code}
            )
            resp.raise_for_status()
            item = resp.json()
            output = f"ITEM: {item['name']}\nCODE: {item['code']}\nDESCRIPTION: {item['description']}"
            for chunk in split_message(output):
                await interaction.followup.send(f"```{chunk}```")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                await interaction.followup.send(
                    "```[WARNING] Item not found. Please check the code and try again.```"
                )
            else:
                logger.error(e)
                await interaction.followup.send(
                    "```[ERROR] Skaian link temporarily disconnected. Please try again later.```"
                )
        except Exception as e:
            logger.error(e)
            await interaction.followup.send(
                "```[ERROR] Skaian link temporarily disconnected. Please try again later.```"
            )


@client.tree.command()
@app_commands.describe(
    players="list of titles in the form 'Class of Aspect, Class of Aspect'",
    memory="memory the fraymotif crystallizes",
    additional_info="additional information about the fraymotif",
)
async def fraymotif(
    interaction: discord.Interaction, players: str, memory: str, additional_info: str
):
    """Crystallize a memory into a fraymotif suited to a player or group of players."""
    await interaction.response.defer(thinking=True)
    async with httpx.AsyncClient() as client_http:
        try:
            resp = await client_http.post(
                f"{API_URL}/fraymotif",
                json={
                    "players": players,
                    "memory": memory,
                    "additional_info": additional_info,
                },
            )
            resp.raise_for_status()
            fraymotif = resp.json()
            output = f"{fraymotif['visual_description']}\n\n{fraymotif['name'].upper()}\n\n{fraymotif['mechanical_description']}"
            for chunk in split_message(output):
                await interaction.followup.send(f"```{chunk}```")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                await interaction.followup.send(
                    '```[ERROR] Valid titles not detected. Please use the form "Class of Aspect, Class of Aspect".```'
                )
            else:
                logger.error(e)
                await interaction.followup.send(
                    "```[ERROR] Skaian link temporarily disconnected. Please try again later.```"
                )
        except Exception as e:
            logger.error(e)
            await interaction.followup.send(
                "```[ERROR] Skaian link temporarily disconnected. Please try again later.```"
            )


client.run(DISCORD_BOT_TOKEN)
