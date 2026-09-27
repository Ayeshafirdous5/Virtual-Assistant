"""Virtual Assistant application package.

This package holds the assistant's modular implementation. Each subpackage has
a single responsibility:

    config           configuration read from the environment / .env
    logging_config   console and rotating-file logging setup

Submodules are deliberately NOT imported here. Importing them is left to the
caller so that importing ``assistant`` stays cheap and free of side effects.
"""

__version__ = "2.0.0"
