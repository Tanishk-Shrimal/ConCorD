"""Allow ``python -m concord_lqcd`` to invoke the production CLI."""

from .cli import main


if __name__ == "__main__":
    main()
