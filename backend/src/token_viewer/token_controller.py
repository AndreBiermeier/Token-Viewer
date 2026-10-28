"""Controller for the tokenizer."""

from token_viewer.controller import CommandError, Controller
from token_viewer.model import TokenizerModel
from token_viewer.view_models import (
    Message,
    TokenizerView,
    TokenView,
    ViewModel,
    ViewOption,
)


class TokenController(Controller):
    """Controller for the token model."""

    tokenizer: TokenizerModel

    def __init__(self, tokenizer: TokenizerModel) -> None:
        """Initialize the tokenizer."""
        self.tokenizer = tokenizer

    def view_model(self) -> ViewModel:
        """Return what the views show now, without changing anything."""
        repository = self.tokenizer.repository
        revision = self.tokenizer.revision
        tokenizer = None
        if repository is not None and revision is not None:
            tokenizer = TokenizerView(repository=repository, revision=revision)

        encoding = self.tokenizer.chat_encoding
        tokens: list[TokenView] = []
        if encoding is not None:
            pieces = encoding.tokens
            offsets = encoding.offsets
            for index, token_id in enumerate(encoding.ids):
                start, end = offsets[index]
                tokens.append(
                    TokenView(
                        index=index,
                        id=token_id,
                        piece=pieces[index],
                        text=self.tokenizer.decode([token_id]),
                        start=start,
                        end=end,
                        special=self.tokenizer.is_special_token(token_id),
                    )
                )

        rendered_text = self.tokenizer.rendered_text
        return ViewModel(
            tokenizer=tokenizer,
            notice=None,
            messages=[Message(**message) for message in self.tokenizer.messages],
            views=[ViewOption(key="chat", title="Chat, after the template")]
            if rendered_text is not None
            else [],
            current_view="chat" if rendered_text is not None else None,
            text=rendered_text if rendered_text is not None else "",
            tokens=tokens,
            selected=None,
            capture=None,
        )

    def load_model(self, repository: str, revision: str) -> ViewModel:
        """Use the tokenizer and chat template of a model repository.

        Args:
            repository: The model repository on Hugging Face, as owner/name.
            revision: The revision to take the files from.

        Raises:
            CommandError: If the input is invalid or the files cannot be loaded.
        """
        try:
            self.tokenizer.load_tokenizer(repository, revision)
        except (OSError, ValueError) as refusal:
            raise CommandError(str(refusal)) from refusal
        return self.view_model()

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
