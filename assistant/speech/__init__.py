"""Speech input and output.

Two interchangeable implementations of the same two protocols:

    base   Speaker / Listener protocols (no heavy imports)
    voice  pyttsx3 + SpeechRecognition, the original behaviour
    text   console input/output, for --text mode and for tests

Because both satisfy the same contracts, a tool or the app loop can be
written once and driven by either. Importing this package does not start a
microphone or a speech engine.
"""
