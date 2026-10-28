"""Show the course request's tokens with and without the chat template."""

import io
import sys

from token_viewer.controller import CommandError
from token_viewer.view_models import Message, ViewModel
from token_viewer.wiring import build_token_controller

# The same repository, pinned revision and request as the course's token_lab.
REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
REVISION = "bee7596271d1495f6992ae224aefde4410e816b8"


def main() -> None:
    """Create the controller, submit the course request and display its result."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    controller = build_token_controller()
    try:
        controller.load_model(REPOSITORY, REVISION)
        controller.set_messages(
            [Message(role="user", content="Explain tokenization in one sentence.")]
        )
        chat = controller.select_view("chat")
        plain = controller.select_view("text")
    except CommandError as error:
        raise SystemExit(str(error)) from error

    print_tokens("Chat, after the template", chat)
    print()
    print_tokens("Text alone", plain)


def print_tokens(title: str, snapshot: ViewModel) -> None:
    """Print one sequence supplied by the controller, without tokenizing it."""
    print(f"{title}:")
    print(snapshot.text)
    print(
        f"\n{'Index':>5}  {'ID':>6}  {'Vocabulary entry':<24}  "
        f"{'Decoded text':<24}  Special"
    )
    for token in snapshot.tokens:
        marker = "*" if token.special else ""
        print(
            f"{token.index:>5}  {token.id:>6}  "
            f"{token.piece!r:<24}  {token.text!r:<24}  {marker}"
        )
    print("\n* = special token")
    print(f"\nToken count: {len(snapshot.tokens)}")


if __name__ == "__main__":
    main()
