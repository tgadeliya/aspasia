from pathlib import Path
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import answer
from inspect_ai.solver import multiple_choice, system_message

from aspasia.datasets import QuALITY

MCQ_TEMPLATE = dedent("""
    Please answer the following question:        
    {question}

    Choices:
    {choices}
""")

USER_TEMPLATE = dedent("""
    Article:
    {article}
""")

DEBATE_SYSTEM_TEMPLATE = dedent("""
    You are a debater. Your goal is to choose the most accurate answer to the question 
    based on the provided article.
    You will be given a question and two possible answers. Answer using the following
    format: ANSWER:$LETTER
""")


@task
def consultancy():
    dataset = QuALITY(
        Path("/Users/tsimur.hadeliya/code/aspasia/data")
    ).get_memory_dataset("dev")
    return Task(
        dataset=dataset,
        solver=[
            system_message(DEBATE_SYSTEM_TEMPLATE),
            system_message(USER_TEMPLATE),
            multiple_choice(template=MCQ_TEMPLATE, multiple_correct=False),
        ],
        message_limit=4,
        model="openai/gpt-4.1-nano",
        model_roles={"debater": "openai/gpt-4.1-nano"},
        config=GenerateConfig(temperature=0.0, max_tokens=300),
        name="consultancy_test",
        scorer=answer(pattern="letter")
    )
