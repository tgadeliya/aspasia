from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.util import input_screen

from aspasia.human_interface import display_chat_history_in_console, prompt_for_reply
from aspasia.utils import display_chat_history_in_console, prepare_messages


@agent
def consultant_agent(
    agent_prompt: str,
) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        model = get_model(role="consultant")
        # append agent prompt
        consultant_messages = [ChatMessageSystem(content=agent_prompt)] + state.messages
        # Add side prompt
        letter_side = state.messages[0].metadata["target"]
        side_prompt = f"\nYou are arguing for {letter_side}"
        consultant_messages.messages[0].content += side_prompt

        messages, output = await model.generate_loop(consultant_messages)
        state.output = output
        state.messages.extend(messages)
        return state

    return execute


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


@agent
def debater_agent(agent_prompt: str):
    async def execute(state: AgentState) -> AgentState:
        model = get_model(role="debater")
        # append agent prompt
        debater_messages = [ChatMessageSystem(content=agent_prompt)] + state.messages
        # Add side prompt
        letter_side = state.messages[0].metadata["target"]
        side_prompt = f"\nYou are arguing for {letter_side}"
        debater_messages[0].content += side_prompt

        messages, output = await model.generate_loop(debater_messages)
        state.output = output
        state.messages.extend(messages)
        return state

    return execute


# TODO: Decide what version to keep: this one or in interactive_consultant(style generated with Chatgpt)


@agent
def human_judge_agent(ignore_msg_with_tags: list[str] = []) -> Agent:
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


@agent
def human_judge(ignore_msg_with_tags: list[str] = []) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        with input_screen(transient=False) as console:
            judge_messages: list = prepare_messages(
                state.messages, ignore_msg_with_tags
            )
            display_chat_history_in_console(console, judge_messages)
            response = prompt_for_reply(console, prompt="Write your reply")
        state.messages.append(ChatMessageUser(content=response))
        return state

    return execute
