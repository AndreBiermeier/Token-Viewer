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

    def render_chat(self, messages: list[dict[str, str]]) -> str:
        """Render messages with the same chat template."""
        if self._chat_template is None:
            raise ValueError("Load a tokenizer before rendering chat messages.")
        if not messages:
            raise ValueError("Give at least one chat message.")

        sandbox = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        return sandbox.from_string(self._chat_template).render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode(self, text: str) -> Encoding:
        """Return token IDs, vocabulary entries and offsets for plain text."""
        return self._require_tokenizer().encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into text, keeping any special tokens visible."""
        return self._require_tokenizer().decode(token_ids, skip_special_tokens=False)

    def _require_tokenizer(self) -> Tokenizer:
        """Return the tokenizer, or explain that it has not been loaded."""
        if self._tokenizer is None:
            raise ValueError("Load a tokenizer before encoding or decoding text.")
        return self._tokenizer
