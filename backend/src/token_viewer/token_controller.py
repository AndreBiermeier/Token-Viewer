"""Controller for the tokenizer."""

from token_viewer.controller import Controller
from token_viewer.model import TokenizerModel
from token_viewer.view_models import Message, ViewModel


class TokenController(Controller):
    """Controller for the token model."""
    tokenizer: TokenizerModel

    def __init__(self, tokenizer: TokenizerModel) -> None:
        """Initialize the tokenizer."""
        self.tokenizer = tokenizer

    def view_model(self) -> ViewModel:
        """Return what the views show now, without changing anything."""
        ...

    def load_model(self, repository: str, revision: str) -> ViewModel:
        """Use the tokenizer and chat template of a model repository.

        Args:
            repository: The model repository on Hugging Face, as owner/name.
            revision: The revision to take the files from.
        """
        ...
    def set_messages(self, messages: list[Message]) -> ViewModel:
        """Replace the chat messages whose tokens the views show."""
        ...

    def load_capture(self, name: str, content: str) -> ViewModel:
        """Read a captured provider response.

        Args:
            name: The capture file's name, for display.
            content: The file's content: JSON text, as capture-nvidia-response
                writes it.
        """
        ...

    def select_view(self, key: str) -> ViewModel:
        """Show the tokens of one view, e.g. "chat"."""
        ...

    def select_token(self, index: int) -> ViewModel:
        """Select one token of the current view."""
        ...
