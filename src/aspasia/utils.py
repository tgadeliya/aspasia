from inspect import cleandoc


def strip_prompt(prompt: str) -> str:
    prompt = cleandoc(prompt).rstrip().replace("\n", "")
    return prompt
