"""Auto LLM Innovator framework."""


def main(argv=None):
    from .cli import main as cli_main

    return cli_main(argv)


__all__ = ["main"]
