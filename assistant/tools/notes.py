"""Keep short text notes for the user.

Supports three actions, told apart by :meth:`NoteTool.extract_slots`:

* ``note buy milk``        add a note
* ``list notes``           read them back
* ``delete note 3``        remove one by its number

The tool never touches ``ctx.speaker`` or ``ctx.listener``. It returns the
lines and the app speaks them. Every database problem is reported as a
sentence, so notes can never stop the assistant.
"""

from __future__ import annotations

from typing import Any

from assistant.core.context import AppContext
from assistant.core.tool import Tool
from assistant.storage.notes import NoteRepository

#: Actions understood by the tool.
ADD = "add"
LIST = "list"
DELETE = "delete"

#: How many notes to read back when the user does not say.
DEFAULT_LIST_LIMIT = 10

#: Words that mean "show me everything". "me" and "my" are included so
#: phrasings like "show me my notes" read as a list request.
_LIST_WORDS = ("list", "show", "read", "what", "me", "my", "all", "notes", "note")

#: Question forms that ask for notes rather than to create one. Checked
#: together with a trailing "note"/"notes", so "note what are the symptoms"
#: is still understood as a new note.
_LIST_LEADS = (
    "what are",
    "what is",
    "what's",
    "whats",
    "what was",
    "show me",
    "list",
    "read",
    "tell me",
)

#: Phrases that introduce a note to delete, longest first.
_DELETE_PREFIXES = ("delete note", "remove note", "delete", "remove")

#: Phrases that introduce a note to add, longest first.
_ADD_PREFIXES = (
    "add a note",
    "add note",
    "create a note",
    "create note",
    "new note",
    "make a note",
    "write a note",
    "write down",
    "note down",
    "remember that",
    "note",
)


class NoteTool(Tool):
    """Add, list and delete notes."""

    name = "notes"
    description = "Save, read back and delete short notes."

    def patterns(self) -> list[str]:
        return ["note", "notes"]

    def extract_slots(self, utterance: str, ctx: AppContext) -> dict[str, Any]:
        """Work out which action was asked for and with what argument.

        Args:
            utterance: the lowercased request.
            ctx: the shared application context, unused here.

        Returns:
            A dict with ``action`` and, where relevant, ``body`` or
            ``note_id``.
        """
        text = utterance.lower().strip()

        for prefix in _DELETE_PREFIXES:
            if text.startswith(prefix):
                remainder = text[len(prefix):].strip()
                digits = "".join(ch for ch in remainder if ch.isdigit())
                if digits:
                    return {"action": DELETE, "note_id": int(digits)}
                # An explicit delete with no number is still a delete; the
                # app asks which note. It must not fall through and be
                # stored as a new note called "delete note".
                return {"action": DELETE, "note_id": None}

        words = text.split()
        if words and all(word in _LIST_WORDS for word in words):
            return {"action": LIST}

        # Question forms: "what are my notes", "what's my notes". The
        # trailing "note"/"notes" keeps these from swallowing ordinary
        # sentences such as "note what are the symptoms".
        if words and words[-1] in ("note", "notes"):
            if any(text.startswith(lead) for lead in _LIST_LEADS):
                return {"action": LIST}

        for prefix in _ADD_PREFIXES:
            if text.startswith(prefix):
                body = text[len(prefix):].strip()
                if body:
                    return {"action": ADD, "body": body}
                break

        # A bare "note" or "notes" reads the notes back rather than storing
        # an empty one, which is the more useful reading.
        if text in ("note", "notes", "my notes"):
            return {"action": LIST}

        return {"action": ADD, "body": text}

    def execute(
        self, ctx: AppContext, slots: dict[str, Any] | None = None
    ) -> list[str]:
        slots = slots or {}
        action = slots.get("action", LIST)

        if ctx.db is None:
            return ["Sorry, I have no place to keep notes right now."]

        try:
            repo = NoteRepository(ctx.db)
            if action == ADD:
                return self._add(repo, slots.get("body", ""))
            if action == DELETE:
                return self._delete(repo, slots.get("note_id"))
            return self._list(repo)
        except ValueError:
            return ["I need some text before I can save a note."]
        except Exception as exc:  # noqa: BLE001 - notes are never critical
            return [f"Sorry, I could not do that with your notes. {exc}"]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _add(self, repo: NoteRepository, body: str) -> list[str]:
        note = repo.add(body)
        return [f"Saved note {note.id}."]

    def _list(self, repo: NoteRepository) -> list[str]:
        notes = repo.list_notes(limit=DEFAULT_LIST_LIMIT)
        if not notes:
            return ["You have not saved any notes yet."]

        lines = [f"You have {repo.count()} notes:"]
        for note in notes:
            lines.append(f"{note.id}. {note.body}")
        return lines

    def _delete(self, repo: NoteRepository, note_id: Any) -> list[str]:
        if note_id is None:
            return ["Tell me which note to delete, for example delete note 2."]

        if repo.delete(int(note_id)):
            return [f"Deleted note {int(note_id)}."]
        return [f"I could not find a note number {int(note_id)}."]
