"""Render chat messages and tokenize text, as in the course's chat example."""

from pathlib import Path

from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Encoding, Tokenizer


class TokenizerModel:
    """In-memory tokenizer."""

    def __init__(self) -> None:
        """Init without loadin the tokenizer."""
        self._tokenizer: Tokenizer | None = None
        self._chat_template: str | None = None
        self.repository: str | None = None
        self.revision: str | None = None
        self.messages: list[dict[str, str]] = []
        self.rendered_text: str | None = None
        self.chat_encoding: Encoding | None = None
        self.plain_text: str | None = None
        self.plain_encoding: Encoding | None = None
        self.current_sequence: str = "chat"

    def set_messages(self, messages: list[dict[str, str]]) -> None:
        """Replace the stored messages without rendering or tokenizing them."""
        if not messages:
            raise ValueError("Give at least one chat message.")
        self.messages = [message.copy() for message in messages]
        self.rendered_text = None
        self.chat_encoding = None
        self.plain_text = None
        self.plain_encoding = None

    def process_messages(self, messages: list[dict[str, str]]) -> None:
        """Store, render and encode messages, restoring state if any step fails."""
        previous_messages = self.messages
        previous_text = self.rendered_text
        previous_encoding = self.chat_encoding
        previous_plain_text = self.plain_text
        previous_plain_encoding = self.plain_encoding
        try:
            self.set_messages(messages)
            self.render_chat()
            self.encode_chat()
            self.encode_plain_text()
        except Exception:
            self.messages = previous_messages
            self.rendered_text = previous_text
            self.chat_encoding = previous_encoding
            self.plain_text = previous_plain_text
            self.plain_encoding = previous_plain_encoding
            raise

    def load_tokenizer(
        self,
        repository: str,
        revision: str,
        *,
        local_files_only: bool = False,
    ) -> None:
        """Load tokenizer.json and chat_template.jinja from the same revision.

        Args:
            repository: The Hugging Face repository, as owner/name.
            revision: The pinned commit to read the tokenizer from.
            local_files_only: Refuse to download a missing file when True.

        """
        if not repository.strip() or not revision.strip():
            raise ValueError("Give both a repository and a revision.")

        tokenizer_file = hf_hub_download(
            repo_id=repository,
            filename="tokenizer.json",
            revision=revision,
            local_files_only=local_files_only,
            token=False,
        )
        template_file = hf_hub_download(
            repo_id=repository,
            filename="chat_template.jinja",
            revision=revision,
            local_files_only=local_files_only,
            token=False,
        )
        tokenizer = Tokenizer.from_file(tokenizer_file)
        chat_template = Path(template_file).read_text(encoding="utf-8")
        self._tokenizer = tokenizer
        self._chat_template = chat_template
        self.repository = repository
        self.revision = revision
        self.rendered_text = None
        self.chat_encoding = None
        self.plain_text = None
        self.plain_encoding = None

    def render_chat(self) -> str:
        """Store and return the chat text rendered from the current messages."""
        if self._chat_template is None:
            raise ValueError("Load a tokenizer before rendering chat messages.")
        if not self.messages:
            raise ValueError("Give at least one chat message.")

        sandbox = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        text = sandbox.from_string(self._chat_template).render(
            messages=self.messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.rendered_text = text
        self.chat_encoding = None
        return text

    def encode_chat(self) -> Encoding:
        """Store and return the tokenization of the rendered chat text."""
        if self.rendered_text is None:
            raise ValueError("Render the chat before encoding it.")
        encoding = self.encode(self.rendered_text)
        self.chat_encoding = encoding
        return encoding

    def encode(self, text: str) -> Encoding:
        """Return token IDs, vocabulary entries and offsets for plain text."""
        return self._require_tokenizer().encode(text, add_special_tokens=False)

    def encode_plain_text(self) -> Encoding:
        """Store content-only tokens, joining multiple messages with newlines."""
        if not self.messages:
            raise ValueError("Give at least one chat message.")
        text = "\n".join(message["content"] for message in self.messages)
        encoding = self.encode(text)
        self.plain_text = text
        self.plain_encoding = encoding
        return encoding

    def select_sequence(self, key: str) -> None:
        """Choose an existing chat or plain-text tokenization result."""
        if key not in {"chat", "text"}:
            raise ValueError(f"There is no sequence {key!r}.")
        encoding = self.chat_encoding if key == "chat" else self.plain_encoding
        if encoding is None:
            raise ValueError(f"Tokenize messages before selecting {key!r}.")
        self.current_sequence = key

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into text, keeping any special tokens visible."""
        return self._require_tokenizer().decode(token_ids, skip_special_tokens=False)

    def is_special_token(self, token_id: int) -> bool:
        """Check whether the loaded tokenizer marks this ID as special."""
        token = self._require_tokenizer().get_added_tokens_decoder().get(token_id)
        return token is not None and token.special

    def _require_tokenizer(self) -> Tokenizer:
        """Return the tokenizer, or explain that it has not been loaded."""
        if self._tokenizer is None:
            raise ValueError("Load a tokenizer before encoding or decoding text.")
        return self._tokenizer
