"""
Command-line entry point for the automatic evaluation harness.

Delegates to the Helox SDK CLI (``deepiri_helox_sdk.evaluation.__main__``) so
the same commands run from the in-repo package or the installable SDK.

Usage::

    python -m evaluation run --suite-dir evaluation/suites --suite gen \\
        --model-path path/to/model [--max-new-tokens 100] [--eval-dir evaluation]
    python -m evaluation benchmark --model-path path/to/model --prompt "hello"
    python -m evaluation summary [--eval-dir evaluation]
    python -m evaluation history [--eval-dir evaluation] [--suite gen]
"""

from deepiri_helox_sdk.evaluation.__main__ import main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
