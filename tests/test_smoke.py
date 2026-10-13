"""Smoke tests to verify testing setup and package structure."""

import sys


def test_python_version() -> None:
    """Verify that Python version meets the project's Python 3.10+ requirement."""
    assert sys.version_info >= (3, 10), "Python 3.10 or higher is required."


def test_package_structure_imports() -> None:
    """Verify that core src packages can be imported properly."""
    import src
    import src.physics
    import src.signal
    import src.models
    import src.dashboard

    assert src is not None
    assert src.physics is not None
    assert src.signal is not None
    assert src.models is not None
    assert src.dashboard is not None
