# Paradox Engine

This is a Discord bot for "classpecting" personalities, or giving them Titles from the webcomic _Homestuck_. Here's the code for people who want to peek inside, though it's kind of useless without the prompts. Though all the data in the prompts was used with permission, out of respect to the original authors, I chose not to make them public. I preserved the directory structure so you could theoretically write your own prompts for each file.

## Prompt Directory

A Title is composed of a class and an aspect. `aspects/` contains one Markdown file for each aspect:

```
blood.md
breath.md
doom.md
heart.md
hope.md
life.md
light.md
mind.md
rage.md
space.md
time.md
void.md
```

`classes/` contains one Markdown file for each class:

```
bard.md
heir.md
knight.md
mage.md
maid.md
page.md
prince.md
rogue.md
seer.md
sylph.md
thief.md
witch.md
```

Each of these files contains a description of the respective class or aspect. You can substitute whatever theories you find online for each file, though the author gave me written permission to republish them.

`quiz_answerer.md` answers a personality quiz as a given personality. This is the prompt that has been made public.

The `quiz_answerer` uses example answers from `aspect_example.md` and `class_example.md` that contain a written personality description along with a set of answers corresponding to that personality.

Finally, `paradox_engine.md` describes the personality of the Paradox Engine, along with the instruction to write a justification for the Title along with the powers it may have, using data from the respective class and aspect .md files, and the personality given by the user.

## LLM opinions

I personally like Claude's interpretation of the Paradox Engine the most, just because other LLMs manage to somehow be simultaneously too unstructured and too boring. However, Llama-70B produces the most variation in quiz-answering, leading to more interesting classpect results. Opus or the 403B/405B Llama models are overkill, Sonnet works well enough for both.

## Instructions

`pip install uv`, or however you want to install uv.

`uv lock`, then `uv sync`.

`uv run alembic upgrade head` to initialize the local database.

`source .venv/bin/activate` to activate the virtual environment.

`nohup python3 main.py > log.txt & disown -h` to run the bot in the background.

## Credits

Classpect knowledge given by my good friends @reachartwork and Tamago, used with permission.

Example answers provided by my good friend NeoUndying, used with permission.

This is a Homestuck fan project. We are not affiliated with What Pumpkin.

Thank you for being part of our shared fandom.

"Real paradise lies eternally in the person who dreams of it. Why don't you venture forth in search of your own utopia?"