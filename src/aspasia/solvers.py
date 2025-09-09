import logging
import re
from enum import Enum
from random import Random
from typing import Match, TypedDict

from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai._util.logger import warn_once
from inspect_ai.solver._multiple_choice import (
    MULTIPLE_ANSWER_TEMPLATE,
    MULTIPLE_ANSWER_TEMPLATE_COT,
    SINGLE_ANSWER_TEMPLATE,
    SINGLE_ANSWER_TEMPLATE_COT,
    DeprecatedArgs,
    parse_answers,
    pretend_we_didnt_shuffle,
    prompt,
    set_choices_based_on_generated_response,
    unshuffle_choices,
    valid_template,
)
from inspect_ai.solver._solver import Generate, Solver, solver
from inspect_ai.solver._task_state import Choices, TaskState
from inspect_ai.util import resource
from typing_extensions import Unpack

logger = logging.getLogger(__name__)


@solver
def multiple_choice_no_generation(
    *,
    template: str | None = None,
    cot: bool = False,
    multiple_correct: bool = False,
    max_tokens: int | None = None,
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

        return state

    return solve
