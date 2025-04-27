"""Command line interface for the MLOps-RecSys project."""

import argparse
import sys

from mloprec import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the MLOps-RecSys CLI application.

    Args:
        argv: Command line arguments. Defaults to sys.argv.

    Returns:
        Exit code.
    """
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="mloprec",
        description="MLOps project for recommender systems",
    )

    parser.add_argument("--version", action="store_true", help="Print version and exit")

    args = parser.parse_args(argv)

    if args.version:
        print(f"MLOps-RecSys v{__version__}")
        return 0

    print("Welcome to MLOps-RecSys project!")
    print("This is a placeholder for the actual application.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
