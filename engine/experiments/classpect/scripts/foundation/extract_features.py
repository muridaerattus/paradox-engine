import argparse
import asyncio

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from experiments.classpect.decision_tree import (
    DESCRIPTIONS_PATH,
    FEATURES_PATH,
    FEATURE_NAMES,
    DescriptionRecord,
    FeatureRecord,
    read_json,
    write_json,
)


FEATURE_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
Extract classpecting features from personality descriptions.

Return only the requested structured fields. Feature values must be integers from
1 to 5. Use 1 for very low, 3 for mixed/moderate, and 5 for very high.

Features to score:
{feature_names}

Do not infer from canonical classpect, blood color, gender, dream moon, species,
weapons, powers, or other innate setting facts. Score only the personality text.
""".strip(),
        ),
        (
            "user",
            "Character: {character}\n\nDescription:\n{description}",
        ),
    ]
)


async def extract_record(
    llm: ChatAnthropic, semaphore: asyncio.Semaphore, record: DescriptionRecord
) -> FeatureRecord:
    structured_llm = llm.with_structured_output(FeatureRecord)
    async with semaphore:
        return await (FEATURE_PROMPT | structured_llm).ainvoke(
            {
                "character": record.character,
                "description": record.description,
                "feature_names": "\n".join(f"- {name}" for name in FEATURE_NAMES),
            }
        )


async def extract_features(model: str, concurrency: int) -> None:
    descriptions = [
        DescriptionRecord.model_validate(record)
        for record in read_json(DESCRIPTIONS_PATH)
    ]
    llm = ChatAnthropic(model=model, max_tokens=2048)
    semaphore = asyncio.Semaphore(concurrency)
    records = await asyncio.gather(
        *(extract_record(llm, semaphore, record) for record in descriptions)
    )
    write_json(FEATURES_PATH, [record.model_dump() for record in records])


def main() -> None:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description="Extract gitignored decision-tree features from generated descriptions."
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model used for feature extraction.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum concurrent LLM calls.",
    )
    args = parser.parse_args()
    asyncio.run(extract_features(args.model, args.concurrency))


if __name__ == "__main__":
    main()
