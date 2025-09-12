from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# ---------- Rendering helpers ----------

ROLE_STYLES = {
    "system": "bold white on grey19",
    "developer": "bold white on dark_magenta",
    "user": "bold black on green3",
    "assistant": "bold black on deep_sky_blue1",
    "tool": "bold black on khaki1",
    "function": "bold black on gold3",
    "critic": "bold white on red3",
    "judge": "bold white on purple",
}


def _safe_to_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if is_dataclass(x):
        return json.dumps(asdict(x), ensure_ascii=False, indent=2)
    try:
        return json.dumps(x, ensure_ascii=False, indent=2)
    except Exception:
        return repr(x)


def _render_content_to_console(console: Console, content: Any) -> None:
    """
    Render ChatMessage.content which can be:
      - str (markdown)
      - list/iter of structured parts (we try to show text; annotate others)
      - arbitrary objects (json-ish)
    """
    if isinstance(content, str):
        console.print(Markdown(content))
        return

    # Common Inspect/LLM shapes: [{type: "text", text: "..."}], etc.
    if isinstance(content, Iterable) and not isinstance(
        content, (bytes, bytearray, dict)
    ):
        for part in content:
            if isinstance(part, str):
                console.print(Markdown(part))
            elif isinstance(part, dict):
                ptype = part.get("type") or part.get("kind")
                if ptype == "text" and "text" in part:
                    console.print(Markdown(str(part["text"])))
                elif ptype in ("image", "image_url", "input_image"):
                    # Terminal placeholder for images
                    src = (
                        part.get("url")
                        or part.get("image_url")
                        or part.get("id")
                        or "<image>"
                    )
                    console.print(f"[dim]🖼️ (image: {src})[/dim]")
                else:
                    console.print(
                        Panel(_safe_to_str(part), title="part", border_style="grey39")
                    )
            else:
                console.print(
                    Panel(_safe_to_str(part), title="part", border_style="grey39")
                )
        return

    # dict/other objects
    console.print(Panel(_safe_to_str(content), border_style="grey39"))


def display_chat_history_in_console(
    console: Console,
    messages: list,
    *,
    show_index: bool = True,
) -> None:
    console.clear()
    console.print(Rule("[bold]Conversation[/bold]"))
    for i, message in enumerate(messages, start=1):
        role = getattr(message, "role", "assistant") or "assistant"
        role_style = ROLE_STYLES.get(role, "bold white on grey27")
        role_badge = Text(f" {role.upper()} ", style=role_style)

        # Optional: show tags if present (e.g., message.metadata.get("tags"))
        subtitle = None
        meta = getattr(message, "metadata", None)
        if isinstance(meta, dict):
            tags = meta.get("tags") or meta.get("tag") or None
            if tags:
                if isinstance(tags, (list, tuple, set)):
                    subtitle = f"tags: {', '.join(map(str, tags))}"
                else:
                    subtitle = f"tags: {tags}"

        title = role_badge
        if show_index:
            title.append(f"  #{i}", style="bold dim")

        with console.capture() as cap:
            _render_content_to_console(console, getattr(message, "content", ""))

        # Wrap captured markdown render in a panel for consistent layout
        body = cap.get()
        console.print(
            Panel.fit(
                body if body.strip() else "[dim]∅ empty[/dim]",
                title=title,
                subtitle=subtitle,
                border_style="grey35",
                padding=(1, 2),
            )
        )
    console.print()  # bottom spacing


# ---------- Input helpers ----------
def _edit_in_system_editor(initial: str = "") -> str:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        # Fallback to multiline mode if no editor configured
        return ""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".md") as tf:
        path = tf.name
        if initial:
            tf.write(initial)
            tf.flush()
    try:
        subprocess.run([editor, path], check=False)
        with open(path, "r") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _multiline_input(console: Console, prompt: str) -> str:
    console.print(
        Panel.fit(
            "[b]Multi-line mode[/b]: type your message and submit by entering a blank line.",
            border_style="grey39",
        )
    )
    lines: list[str] = []
    while True:
        line = console.input(f"[bold]{prompt}[/bold] ")
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def prompt_for_reply(console: Console, prompt: str = "Your reply>") -> str:
    console.print(
        Panel.fit(
            "Reply options: [b]/ml[/b] multi-line • [b]/edit[/b] open $EDITOR • [b]/skip[/b] submit empty",
            border_style="grey35",
        )
    )
    first = console.input(f"[bold]{prompt}[/bold] ").strip()

    if first == "/skip":
        return ""
    if first == "/ml":
        return _multiline_input(console, prompt)
    if first == "/edit":
        edited = _edit_in_system_editor()
        if edited.strip():
            return edited.strip()
        # If editor returned empty, fall back to multi-line
        return _multiline_input(console, prompt)
    return first


# ---------- Agent ----------



from rich.console import Console






