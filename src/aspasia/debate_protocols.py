from typing import Literal
from inspect_ai.agent import Agent, AgentState, agent, run, human_cli
from inspect_ai.model import ChatMessageSystem, get_model

from aspasia.prompts import CONSULTANT_JUDGE_PROMPT, CONSULTANT_PROMPT


@agent
def consultant_agent(agent_prompt: str, interactive: bool = False) -> Agent:
    model = get_model(role="consultant")
    
    async def execute(state: AgentState) -> AgentState:
        # append agent prompt
        state.messages.insert(0, ChatMessageSystem(content=agent_prompt))
        messages, output = await model.generate_loop(state.messages)
        state.messages.pop(0)  # remove agent prompt
        state.output = output
        state.messages.extend(messages)
        return state

    return execute


@agent
def judge_agent(agent_prompt: str, ignore_msg_with_tags: list[str] = []) -> Agent:
    model = get_model(role="judge")

    async def execute(state: AgentState) -> AgentState:
        judge_messages: list = [ChatMessageSystem(content=agent_prompt)]
        for message in state.messages:
            if any(str(message.content).startswith(tag) for tag in ignore_msg_with_tags):
                continue  # skip messages starting with tags from ignore_msg_with_tags
            judge_messages.append(message)

        response, output = await model.generate_loop(judge_messages)
        state.messages.extend(response)
        state.output = output
        return state

    return execute


@agent
def consultancy(num_turns: int) -> Agent:
    consultant = consultant_agent(agent_prompt=CONSULTANT_PROMPT)
    judge = judge_agent(
        agent_prompt=CONSULTANT_JUDGE_PROMPT,
        ignore_msg_with_tags=["<article>"]
    )

    async def execute(state: AgentState) -> AgentState:
        for _ in range(num_turns):
            state = await run(consultant, state)
            state = await run(judge, state)
        return state

    return execute
