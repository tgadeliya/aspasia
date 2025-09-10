from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import answer
from inspect_ai.solver import user_message

from aspasia.datasets import QuALITY
from aspasia.debate_protocols import consultancy
from aspasia.prompts import ARTICLE_TEMPLATE, MCQ_TEMPLATE
from aspasia.solvers import multiple_choice_no_generation


@task
def consultancy_runner(
    num_turns: int = 2,
    interactive: bool = False,
    consultant_model: str = "openai/gpt-4.1-nano",
    judge_model: str = "openai/gpt-4.1-nano",
    consultant_side: Literal["random", "target"] = "target"
):
    dataset = QuALITY(
        Path("/Users/tsimur.hadeliya/code/aspasia/data")
    ).get_memory_dataset("dev")
    consultancy_solver = consultancy(
        num_turns=num_turns,
        interactive=interactive,
        consultant_side=consultant_side,
    )
    return Task(
        dataset=dataset,
        solver=[
            multiple_choice_no_generation(template=MCQ_TEMPLATE),
            user_message(ARTICLE_TEMPLATE),
            consultancy_solver,
        ],
        message_limit=12,
        model_roles={"consultant": consultant_model, "judge": judge_model},
        # TODO: Add generation for both models
        config=GenerateConfig(temperature=0.0, max_tokens=300),
        name="consultancy_test",
        # TODO: Add custom scorer: judge model will evaluate letter based on given conversation
        scorer=answer(pattern="letter"),
    )
