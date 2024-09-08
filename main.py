import os
from dotenv import load_dotenv
from pathlib import Path
from src.discord.discord_bot import bot

def main():
    # Get the project root directory
    project_root = Path(__file__).parent

    # Load environment variables
    config_path = project_root / "config" / ".env"
    load_dotenv(dotenv_path=config_path)

    # Run the bot
    bot_token = os.getenv('DISCORD_NOTEKEEPER_KEY')
    if bot_token is None:
        raise ValueError("DISCORD_NOTEKEEPER_KEY not found in environment variables")
    
    bot.run(bot_token)

if __name__ == "__main__":
    main()