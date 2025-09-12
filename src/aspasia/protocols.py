from typing import Literal

from inspect_ai import solver
from inspect_ai.agent import Agent, AgentState, agent, run
from inspect_ai.solver import Generate, Solver, TaskState, solver

from aspasia.agents import consultant_agent, judge_agent
from aspasia.prompts import CONSULTANT_JUDGE_PROMPT, CONSULTANT_PROMPT


@agent
def consultancy(
    num_turns: int,
    interactive: bool = False,
    symmetric: bool = False,
    consultant_side: Literal["target", "random"] = "target",
) -> Agent:
    """

    Args:
    interactive: bool - Use human as a judge. When False, using LLM as a judge.
    symmetric: bool - When True judge has access to all information that consultant,
                except consultant's thinking (text in <thinking> </thinking> tags)
    """

    consultant = consultant_agent(
        agent_prompt=CONSULTANT_PROMPT,
    )

    judge = judge_agent(
        agent_prompt=CONSULTANT_JUDGE_PROMPT,
        ignore_msg_with_tags=[] if symmetric else ["<article>"],
    )

    async def execute(state: AgentState) -> AgentState:
        for turn in range(num_turns):
            state = await run(consultant, state)
            state = await run(judge, state)
        return state

    return execute


@agent
def debate(
    debaters: list[Agent],
    judge: Agent,
    num_turns: int,
    interactive: bool = False,
) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        for turn in range(num_turns):
            for debater in debaters:
                state = await run(debater, state)
            if interactive:
                state = await run(judge, state)

        if not interactive:
            state = await run(judge, state)

        return state

    return execute
