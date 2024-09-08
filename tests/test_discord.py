import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import discord
from discord import app_commands
from src.discord.discord_bot import bot, tree, guild_check, APPROVED_GUILDS

@pytest.fixture
def mock_interaction():
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.response.send_message = AsyncMock()
    interaction.user.name = "TestUser"
    return interaction

@pytest.mark.asyncio
async def test_on_ready():
    with patch.object(tree, 'sync', new_callable=AsyncMock) as mock_sync:
        await bot.on_ready()
        mock_sync.assert_called_once()

@pytest.mark.asyncio
async def test_hello_command(mock_interaction):
    await tree.get_command("hello").callback(mock_interaction)
    mock_interaction.response.send_message.assert_called_once_with("Hello, TestUser!")

@pytest.mark.asyncio
async def test_ask_command(mock_interaction):
    question = "What is the meaning of life?"
    await tree.get_command("ask").callback(mock_interaction, question)
    mock_interaction.response.send_message.assert_called_once_with(f"You asked: {question}")

@pytest.mark.parametrize("guild_id,expected", [
    (APPROVED_GUILDS[0], True),
    (9999, False),
])
def test_guild_check(guild_id, expected):
    mock_interaction = MagicMock(spec=discord.Interaction)
    mock_interaction.guild_id = guild_id
    result = guild_check()(lambda x: True)(mock_interaction)
    assert result == expected

@pytest.mark.asyncio
async def test_on_app_command_error_check_failure(mock_interaction):
    error = app_commands.errors.CheckFailure()
    await tree.on_error(mock_interaction, error)
    mock_interaction.response.send_message.assert_called_once_with(
        "Sorry, this bot is not authorized to respond in this server.",
        ephemeral=True
    )

@pytest.mark.asyncio
async def test_on_app_command_error_generic(mock_interaction):
    error = Exception("Generic error")
    await tree.on_error(mock_interaction, error)
    mock_interaction.response.send_message.assert_called_once_with(
        "An error occurred while processing the command.",
        ephemeral=True
    )