import importlib

import discord
import pytest


def test_import_does_not_connect_and_mentions_are_disabled(monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("Importing the bot must not connect to Discord")

    monkeypatch.setenv("GUILD_ID", "1")
    monkeypatch.setattr(discord.Client, "run", unexpected_run)
    main = importlib.import_module("main")
    mentions = main.client.allowed_mentions
    assert not mentions.everyone
    assert not mentions.users
    assert not mentions.roles
    assert not mentions.replied_user
