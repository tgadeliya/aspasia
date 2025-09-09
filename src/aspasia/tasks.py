from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import answer
from inspect_ai.solver import user_message

from aspasia.datasets import QuALITY
from aspasia.debate_protocols import consultancy
from aspasia.prompts import (
    MCQ_TEMPLATE, ARTICLE_TEMPLATE
)
from aspasia.solvers import multiple_choice_no_generation


@task
def consultancy_runner():
    dataset = QuALITY(
        Path("/Users/tsimur.hadeliya/code/aspasia/data")
    ).get_memory_dataset("dev")
    return Task(
        dataset=dataset,
        solver=[
            multiple_choice_no_generation(
                template=MCQ_TEMPLATE,
            ),
            user_message(ARTICLE_TEMPLATE),
            consultancy(num_turns=2),
        ],
        message_limit=12,
        model="openai/gpt-4.1-nano",
        model_roles={"debater": "openai/gpt-4o-mini", "judge": "openai/gpt-4.1-nano"},
        config=GenerateConfig(temperature=0.0, max_tokens=300),
        name="consultancy_test",
        scorer=answer(pattern="letter"),
    )
