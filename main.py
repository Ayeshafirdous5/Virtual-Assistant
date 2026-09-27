"""Entry point for the Virtual Assistant.

This file is intentionally tiny: it only starts the application. All runtime
behaviour lives in :mod:`assistant.app`.

Usage:
    python main.py            speak to the assistant (default)
    python main.py --text     type commands instead
    python main.py --help     show the options
    python main.py --text --debug-nlu  type commands, showing NLU diagnostics

"""

import sys

from assistant.app import main

if __name__ == "__main__":
    sys.exit(main())
