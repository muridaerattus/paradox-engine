# Paradox Engine

This is a Discord bot for "classpecting" personalities, or giving them Titles from the webcomic _Homestuck_. Here's the code for people who want to peek inside, though it's kind of useless without the prompts. Though all the data in the prompts was used with permission, out of respect to the original authors, I chose not to make the character example prompts public. I preserved the directory structure so you could theoretically write your own prompts for each file.

## Prompt Directory

### Classpecting

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

The `quiz_answerer` uses example answers from `aspect_example.md` and `class_example.md` that contain a written personality description along with a set of answers corresponding to that personality. These two prompts are kept private, out of respect to the original authors.

Finally, `paradox_engine.md` describes the personality of the Paradox Engine, along with the instruction to write a justification for the Title along with the powers it may have, using data from the respective class and aspect .md files, and the personality given by the user. I have made the structure and instructions for the prompt public, but kept the "personality" section for the Paradox Engine private, because I encourage you to make your own.

## LLM opinions

I personally like Claude's interpretation of the Paradox Engine the most, just because other LLMs manage to somehow be simultaneously too unstructured and too boring. However, Llama-70B produces the most variation in quiz-answering, leading to more interesting classpect results. Opus or the 403B/405B Llama models are overkill, Sonnet works well enough for both.

## Instructions

### Docker (deprecated for now)

```
docker run --env-file path/to/your/env-file \
-v /path/to/your/prompts:/app/prompts \
-v /path/to/quiz/class_quiz.json:/app/class_quiz.json \
-v /path/to/quiz/aspect_quiz.json:/app/aspect_quiz.json \
-v ./paradox.db:/app/paradox.db \
muridaerattus/paradox-engine
```

It's about the same amount of work to update, but at least I can start and stop it in the background.

### Manual

I manage dependencies with `uv`, because it's fast. Before all this, use `pip install uv`, or however you want to install uv.

#### Running the Engine (FastAPI backend)

1. Install dependencies (from the `engine` directory):
    ```bash
    cd engine
    uv venv
    source .venv/bin/activate
    uv sync
    ```
2. Update the database. Replace `sqlalchemy.url` in `alembic.ini.example` with the name of your database connection. Try `sqlite+aiosqlite:///./paradox.db`. Then run:
    ```bash
    cp alembic.ini.example cp alembic.ini
    uv run alembic upgrade head
    uv run -m scripts.preload_objects
    ```
3. Set up your `.env` by copying over the `.env.example`.
4. Start the FastAPI server in the background:
    ```bash
    nohup uvicorn main:app --reload > log.txt & disown -h
    ```
   By default, the API will be available at http://localhost:8000

#### Running the Terminal (Discord bot)

1. Install dependencies (from the `terminal` directory):
    ```bash
    cd terminal
    uv sync
    ```
2. Set up your `.env` by copying over the `.env.example`.
3. Start the Discord bot in the background:
    ```bash
    nohup uv run main.py > log.txt & disown -h
    ```

**Note:** The Discord bot requires the FastAPI engine to be running and accessible at the configured API URL (default: http://localhost:8000).

## Credits

Classpect knowledge given by my good friends @reachartwork and Tamago, used with permission.

Example answers provided by my good friend NeoUndying, used with permission.

This is a Homestuck fan project. We are not affiliated with What Pumpkin.

Thank you for being part of our shared fandom.

"Real paradise lies eternally in the person who dreams of it. Why don't you venture forth in search of your own utopia?"