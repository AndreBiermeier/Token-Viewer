"""Assemble controllers and models for the application's entry points.

The web back end and terminal run in separate processes with separate state.
The terminal uses the real controller; the page still uses the fake until
the remaining browser commands are implemented.
"""

from token_viewer.controller import Controller
from token_viewer.fake_controller import FakeController
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
    # Yours: replace FakeController() with your own controller, built on your
    # model, both in modules of your own in this package.
    return FakeController()
