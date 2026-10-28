"""Controller for the tokenizer."""

from jinja2 import TemplateError

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

        if self.tokenizer.current_sequence == "text":
            text = self.tokenizer.plain_text
            encoding = self.tokenizer.plain_encoding
        else:
            text = self.tokenizer.rendered_text
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

        views: list[ViewOption] = []
        if self.tokenizer.rendered_text is not None:
            views.append(ViewOption(key="chat", title="Chat, after the template"))
        if self.tokenizer.plain_text is not None:
            views.append(ViewOption(key="text", title="Text alone"))
        return ViewModel(
            tokenizer=tokenizer,
            notice=None,
            messages=[Message(**message) for message in self.tokenizer.messages],
            views=views,
            current_view=self.tokenizer.current_sequence if text is not None else None,
            text=text if text is not None else "",
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
        """Process new messages and return their tokenization for display."""
        message_data = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        try:
            self.tokenizer.process_messages(message_data)
        except (ValueError, TemplateError) as refusal:
            raise CommandError(str(refusal)) from refusal
        return self.view_model()

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
        try:
            self.tokenizer.select_sequence(key)
        except ValueError as refusal:
            raise CommandError(str(refusal)) from refusal
        return self.view_model()

    def select_token(self, index: int) -> ViewModel:
        """Select one token of the current view."""
        ...
