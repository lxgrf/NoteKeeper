import discord
from discord import app_commands
import os
from dotenv import load_dotenv
from functools import wraps

# Load environment variables
load_dotenv()

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# List of approved guild IDs
APPROVED_GUILDS = [1114617197931790376]  # Replace with your actual approved guild IDs

def guild_check():
    def predicate(interaction: discord.Interaction):
        if interaction.guild_id not in APPROVED_GUILDS:
            return False
        return True
    return app_commands.check(predicate)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    try:
        await tree.sync()
        print("Synced command tree")
    except Exception as e:
        print(e)

@tree.command(name="hello", description="Get a friendly greeting from the bot")
@guild_check()
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message(f'Hello, {interaction.user.name}!')

@tree.command(name="ask", description="Ask the bot a question")
@guild_check()
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.send_message(f"You asked: {question}")

@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.errors.CheckFailure):
        await interaction.response.send_message("Sorry, this bot is not authorized to respond in this server.", ephemeral=True)
    else:
        # Handle other types of errors
        await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)

# Run the bot
bot.run(os.getenv('DISCORD_NOTEKEEPER_KEY'))