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
            tokens = self._token_views(encoding.ids, encoding.tokens, encoding.offsets)

        views: list[ViewOption] = []
        if self.tokenizer.rendered_text is not None:
            views.append(ViewOption(key="chat", title="Chat, after the template"))
        if self.tokenizer.plain_text is not None:
            views.append(ViewOption(key="text", title="Text alone"))
        selected_index = self.tokenizer.selected_index
        return ViewModel(
            tokenizer=tokenizer,
            notice=None,
            messages=[Message(**message) for message in self.tokenizer.messages],
            views=views,
            current_view=self.tokenizer.current_sequence if text is not None else None,
            text=text if text is not None else "",
            tokens=tokens,
            selected=tokens[selected_index] if selected_index is not None else None,
            capture=None,
        )

    def _token_views(
        self, ids: list[int], pieces: list[str], offsets: list[tuple[int, int]]
    ) -> list[TokenView]:
        """Explain overlapping source ranges while preserving every token."""
        tokens: list[TokenView] = []
        group_start = 0
        while group_start < len(ids):
            group_end = group_start + 1
            source_start, source_end = offsets[group_start]
            while group_end < len(ids):
                next_start, next_end = offsets[group_end]
                # Join overlapping ranges, including overlaps that extend the
                # group.
                # Example of a possible byte-level split of "éö":
                #   Characters:       é        ö
                #   UTF-8 bytes:      C3 A9    C3 B6
                #   Token bytes:     [C3] [A9 C3] [B6]
                #   Source ranges:   (0,1) (0,2) (1,2)
                # The middle token overlaps both characters. Adding it grows
                # the group's range to (0,2), so the last token joins too.
                if not (
                    source_start < source_end
                    and next_start < next_end
                    and next_start < source_end
                    and source_start < next_end
                ):
                    break
                source_start = min(source_start, next_start)
                source_end = max(source_end, next_end)
                group_end += 1

            group_text = self.tokenizer.decode(ids[group_start:group_end])
            group_size = group_end - group_start
            for index in range(group_start, group_end):
                decoded = (
                    group_text
                    if group_size == 1
                    else self.tokenizer.decode([ids[index]])
                )
                label = decoded
                if group_size > 1:
                    part = index - group_start + 1
                    label = f"[part {part}/{group_size} of {group_text!r}]"
                start, end = offsets[index]
                tokens.append(
                    TokenView(
                        index=index,
                        id=ids[index],
                        piece=pieces[index],
                        text=label,
                        decoded_text=decoded,
                        start=start,
                        end=end,
                        special=self.tokenizer.is_special_token(ids[index]),
                        group_start=group_start,
                        group_end=group_end,
                        group_text=group_text,
                    )
                )
            group_start = group_end
        return tokens

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
        """Refuse capture loading until the capture deliverable is implemented.

        Args:
            name: The capture file's name, for display.
            content: The file's content: JSON text, as capture-nvidia-response
                writes it.
        """
        raise CommandError("Capture loading is not implemented yet.")

    def select_view(self, key: str) -> ViewModel:
        """Show the tokens of one view, e.g. "chat"."""
        try:
            self.tokenizer.select_sequence(key)
        except ValueError as refusal:
            raise CommandError(str(refusal)) from refusal
        return self.view_model()

    def select_token(self, index: int) -> ViewModel:
        """Select one token of the current view."""
        try:
            self.tokenizer.select_token(index)
        except ValueError as refusal:
            raise CommandError(str(refusal)) from refusal
        return self.view_model()
