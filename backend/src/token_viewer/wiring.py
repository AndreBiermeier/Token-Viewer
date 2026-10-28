"""Assemble controllers and models for the application's entry points.

The web back end and terminal use the same model and controller classes.
They run in separate processes, each with its own instances and state.
"""

from token_viewer.controller import Controller
from token_viewer.model import TokenizerModel
from token_viewer.token_controller import TokenController


def build_token_controller() -> Controller:
    """Create a real controller with a fresh, unloaded tokenizer model."""
    model = TokenizerModel()
    return TokenController(model)


def build_controller() -> Controller:
    """Return a new controller, with its model."""
    # Importing api.py calls this, and the route tests import api.py: keep it
    # quick and offline. Download files when the load_model command runs, not
    # here.
    return build_token_controller()
