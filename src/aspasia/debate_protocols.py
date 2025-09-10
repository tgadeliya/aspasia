from typing import Literal

from inspect_ai import solver
from inspect_ai.agent import Agent, AgentState, agent, run
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import input_screen
from rich.console import Console

from aspasia.interactive_consultant import chatgpt_iteractive_judge_agent
from aspasia.prompts import CONSULTANT_JUDGE_PROMPT, CONSULTANT_PROMPT


@agent
def consultant_agent(
    agent_prompt: str,
) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        model = get_model(role="consultant")
        # append agent prompt
        consultant_messages = [ChatMessageSystem(content=agent_prompt)] + state.messages
        # Add side prompt
        letter_side = consultant_messages.messages[0].metadata["target"] 
        side_prompt = f"\nYou are arguing for {letter_side}"
        consultant_messages.messages[0].content += side_prompt

        messages, output = await model.generate_loop(consultant_messages)
        state.output = output
        state.messages.extend(messages)
        return state

    return execute


def prepare_messages(messages, ignore_tags: list[str], agent_prompt: str | None = None):
    """
    Prepare message history by adding system prompt for agent and
    filtering messages based on ignore_tags.
    """
    prepared_messages = (
        [ChatMessageSystem(content=agent_prompt)] if agent_prompt else []
    )
    for message in messages:
        if any(str(message.content).startswith(tag) for tag in ignore_tags):
            continue  # skip messages starting with tags from ignore_tags
        prepared_messages.append(message)
    return prepared_messages


@agent
def judge_agent(agent_prompt: str, ignore_msg_with_tags: list[str] = []) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        model = get_model(role="judge")

        judge_messages: list = prepare_messages(
            state.messages, ignore_msg_with_tags, agent_prompt
        )
        response, output = await model.generate_loop(judge_messages)
        state.messages.extend(response)
        state.output = output
        return state

    return execute


def display_chat_history_in_console(console: Console, messages: list) -> None:
    for message in messages:
        console.print(f"ROLE: {message.role}\n", message.content)
        console.print("\n\n\n")


@agent
def iteractive_judge_agent(ignore_msg_with_tags: list[str] = []) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        with input_screen(transient=False) as console:
            judge_messages: list = prepare_messages(
                state.messages,
                ignore_msg_with_tags,
            )
            display_chat_history_in_console(console, judge_messages)
            response = console.input("Write your reply:")

        state.messages.append(ChatMessageUser(content=response))
        return state

    return execute


@solver
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

    if interactive:
        judge = chatgpt_iteractive_judge_agent()
    else:
        judge = judge_agent(
            agent_prompt=CONSULTANT_JUDGE_PROMPT,
            ignore_msg_with_tags=[] if symmetric else ["<article>"],
        )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for turn in range(num_turns):
            state = await run(consultant, state)
            state = await run(judge, state)
        return state

    return solve
