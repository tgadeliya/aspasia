from inspect import cleandoc

from inspect_ai.model import ChatMessageSystem
from rich.console import Console


def strip_prompt(prompt: str) -> str:
    prompt = cleandoc(prompt).rstrip().replace("\n", "")
    return prompt


def prepare_messages(messages, ignore_tags: list[str], agent_prompt: str | None = None):
    """Prepare message history by adding system prompt for agent and filtering messages
    based on ignore_tags."""
    prepared_messages = (
        [ChatMessageSystem(content=agent_prompt)] if agent_prompt else []
    )
    for message in messages:
        if any(str(message.content).startswith(tag) for tag in ignore_tags):
            continue  # skip messages starting with tags from ignore_tags
        prepared_messages.append(message)
    return prepared_messages


def display_chat_history_in_console(console: Console, messages: list) -> None:
    for message in messages:
        console.print(f"ROLE: {message.role}\n", message.content)
        console.print("\n\n\n")
