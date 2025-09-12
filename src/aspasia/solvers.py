import logging
from random import Random

from inspect_ai._util.logger import warn_once
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.solver._multiple_choice import (
    MULTIPLE_ANSWER_TEMPLATE,
    MULTIPLE_ANSWER_TEMPLATE_COT,
    SINGLE_ANSWER_TEMPLATE,
    SINGLE_ANSWER_TEMPLATE_COT,
    DeprecatedArgs,
    prompt,
    valid_template,
)
from inspect_ai.util import resource
from typing_extensions import Unpack

logger = logging.getLogger(__name__)


@solver
def multiple_choice_no_generation(
    *,
    template: str | None = None,
    cot: bool = False,
    multiple_correct: bool = False,
    **kwargs: Unpack[DeprecatedArgs],
) -> Solver:
    """
    Copy-paste of Inspect's multiple_choice solver without generation step.
    """
    shuffle: bool | Random = False
    if "shuffle" in kwargs:
        shuffle = kwargs["shuffle"]

        if shuffle:
            warn_once(
                logger,
                "The multiple choice shuffle parameter is deprecated. Please shuffle choices at the time your dataset is read by using the shuffle_choices method/parameter of the datasets API.",
            )

    if template and not valid_template(template):
        raise ValueError(
            "The template must contain '{question}' and '{choices}' placeholders for string substitution."
        )

    if template is None:
        if multiple_correct:
            if cot:
                template = MULTIPLE_ANSWER_TEMPLATE_COT
            else:
                template = MULTIPLE_ANSWER_TEMPLATE
        else:
            if cot:
                template = SINGLE_ANSWER_TEMPLATE_COT
            else:
                template = SINGLE_ANSWER_TEMPLATE

    template = resource(template)

    if shuffle is True:
        shuffle = Random()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("The multiple_choice solver requires samples with choices")

        if isinstance(shuffle, Random):
            state.choices.shuffle(shuffle)

        state.user_prompt.text = prompt(
            question=state.user_prompt.text,
            choices=state.choices,
            template=str(template),
        )

        state.user_prompt.metadata = {"target": state.metadata["target"]}
        return state

    return solve
