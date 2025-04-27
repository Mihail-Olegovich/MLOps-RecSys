"""Basic tests to verify the project setup."""


def test_import_mloprec() -> None:
    """Test that we can import the main package."""
    import mloprec

    assert mloprec.__version__ == "0.1.0"


def test_true() -> None:
    """Sanity check to ensure pytest is working."""
    assert True
