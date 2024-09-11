import discord
from discord import app_commands
import os
from dotenv import load_dotenv
from functools import wraps
import sys
from pathlib import Path
from src.ollama.answer import answer_question
from src.notion.download import process_notion_databases

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

# Load environment variables
config_path = project_root / "config" / ".env"
load_dotenv(dotenv_path=config_path)

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
    await interaction.response.defer(thinking=True)
    
    try:
        answer = answer_question(question, [])  # Passing an empty list for database_ids
        await interaction.followup.send(f"Question: {question}\n\nAnswer: {answer}")
    except Exception as e:
        await interaction.followup.send(f"An error occurred while processing your question: {str(e)}")

@tree.command(name="update", description="Update the database from Notion")
@guild_check()
async def update(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    
    try:
        process_notion_databases()
        await interaction.followup.send("Successfully updated the database from Notion.")
    except Exception as e:
        await interaction.followup.send(f"An error occurred while updating the database: {str(e)}")

@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.errors.CheckFailure):
        await interaction.response.send_message("Sorry, this bot is not authorized to respond in this server.", ephemeral=True)
    else:
        # Handle other types of errors
        await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)



# Run the bot
if __name__ == "__main__":
    bot.run(os.getenv('DISCORD_NOTEKEEPER_KEY'))