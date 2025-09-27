from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import answer
from inspect_ai.solver import user_message

from aspasia.agents import debater_agent, human_judge_agent, judge_agent
from aspasia.datasets import QuALITY
from aspasia.prompts import (
    ARTICLE_TEMPLATE,
    DEBATER_JUDGE_PROMPT,
    DEBATER_PROMPT,
    MCQ_TEMPLATE,
)
from aspasia.protocols import (
    consultancy,
    debate,
)
from aspasia.solvers import multiple_choice_no_generation

DEFAULT_DATA_PATH = Path("/Users/tsimur.hadeliya/code/aspasia/data")

@task
def consultancy_runner(
    dataset_path: Path = DEFAULT_DATA_PATH,
    num_turns: int = 2,
    interactive: bool = False,
    consultant_model: str = "openai/gpt-4.1-nano",
    judge_model: str = "openai/gpt-4.1-nano",
    consultant_side: Literal["random", "target"] = "target",
    random_seed: int = 25,
):
    dataset = QuALITY(
        dataset_path,
        random_seed=random_seed,
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
        # TODO: Add custom scorer: judge model will evaluate letter based on
        # given conversation
        scorer=answer(pattern="letter"),
    )


@task
def debate_runner(
    dataset_path: Path = DEFAULT_DATA_PATH,
    num_turns: int = 2,
    num_debaters: int = 2,
    interactive: bool = False,
    debater_model: str = "openai/gpt-4.1-nano",
    judge_model: str = "openai/gpt-4.1-nano",
    judge_type: Literal["agent", "human"] = "agent",
    random_seed: int = 25,
):
    dataset = QuALITY(
        dataset_path,
        random_seed=random_seed,
    ).get_memory_dataset("dev")
    
    debaters = [debater_agent(agent_prompt=DEBATER_PROMPT) for _ in range(num_debaters)]

    if judge_type == "agent":
        judge = judge_agent(
            agent_prompt=DEBATER_JUDGE_PROMPT, ignore_msg_with_tags=["<article>"]
        )
    elif judge_type == "human":
        judge = human_judge_agent(ignore_msg_with_tags=["<article>"])
    else:
        raise ValueError(f"Wrong {judge_type=}")

    debate_solver = debate(
        num_turns=num_turns,
        debaters=debaters,
        judge=judge,
        interactive=interactive,
    )

    run_name = f"debate_{num_debaters=}_{num_turns=}_{interactive=}_{judge_type=}"

    return Task(
        dataset=dataset,
        solver=[
            multiple_choice_no_generation(template=MCQ_TEMPLATE),
            user_message(ARTICLE_TEMPLATE),
            debate_solver,
        ],
        message_limit=20,
        model_roles={"debater": debater_model, "judge": judge_model},
        # TODO: Add generation for both models
        config=GenerateConfig(temperature=0.1, max_tokens=300, seed=random_seed),
        name=run_name,
        # TODO: Add custom scorer: judge model will evaluate letter based
        # on given conversation
        scorer=answer(pattern="letter"),
    )
