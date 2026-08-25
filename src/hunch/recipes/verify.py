"""Execute and exactly verify the versioned AppleScript recipes on a Mac.

Run explicitly with ``python -m hunch.recipes.verify``. All objects use unique
probe names and are removed in ``finally`` blocks. A substring match is never a
pass: Notes are normalized into exact logical lines, reminder names are compared
as an exact sorted set, and Calendar fields are checked component by component.
"""
import argparse
import subprocess
import tempfile
import uuid
from html.parser import HTMLParser
from pathlib import Path

from ..notify import as_str
from .applescript import (exact_calendar_event, exact_note, file_token_to_note,
                          reminders_for_extension)


class _NoteLines(HTMLParser):
    BLOCKS = {"br", "div", "p", "li", "h1", "h2", "h3", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def lines(self):
        return tuple(" ".join(line.split()) for line in "".join(self.parts).splitlines()
                     if line.strip())


def normalize_note_body(body, title):
    parser = _NoteLines()
    parser.feed(body)
    lines = list(parser.lines())
    if lines and lines[0].casefold() == title.casefold():
        lines.pop(0)
    return tuple(lines)


def _osa(script, timeout=45):
    result = subprocess.run(["osascript", "-e", script], capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip() or "osascript failed")
    return result.stdout.strip()


def _note_body(title):
    return _osa(f'''tell application "Notes"
    set matches to every note whose name is {as_str(title)}
    if (count of matches) is not 1 then error "expected exactly one probe note"
    return body of item 1 of matches
end tell''')


def _delete_note(title):
    for _ in range(8):
        count = _osa(f'tell application "Notes" to return count of (notes whose name is {as_str(title)})')
        if count == "0":
            return
        _osa(f'tell application "Notes" to delete (first note whose name is {as_str(title)})')


def _delete_reminder_list(name):
    for _ in range(4):
        count = _osa(f'tell application "Reminders" to return count of (lists whose name is {as_str(name)})')
        if count == "0":
            return
        _osa(f'tell application "Reminders" to delete (first list whose name is {as_str(name)})')


def _delete_event(calendar, summary):
    for _ in range(8):
        count = _osa(f'''tell application "Calendar"
    return count of (events of calendar {as_str(calendar)} whose summary is {as_str(summary)})
end tell''')
        if count == "0":
            return
        _osa(f'''tell application "Calendar"
    delete first event of calendar {as_str(calendar)} whose summary is {as_str(summary)}
end tell''')


def verify_exact_note(tag):
    title = f"Hunch Recipe Exact Note {tag}"
    expected = ("alpha", "eggs & bread", "<literal>")
    try:
        _osa(exact_note(title, expected))
        _osa(exact_note(title, expected))
        observed = normalize_note_body(_note_body(title), title)
        return observed == expected, repr(observed)
    finally:
        _delete_note(title)


def verify_note_edit(tag):
    title = f"Hunch Recipe Note Edit {tag}"
    initial = ("milk", "cheese", "bread", "yogurt", "soda", "apples")
    expected = ("milk [got]", "cheese [got]", "bread", "yogurt [got]", "apples")
    try:
        _osa(exact_note(title, initial))
        _osa(exact_note(title, expected, must_exist=True))
        _osa(exact_note(title, expected, must_exist=True))
        observed = normalize_note_body(_note_body(title), title)
        return observed == expected, repr(observed)
    finally:
        _delete_note(title)


def verify_file_to_note(tag):
    title = f"Hunch Recipe File Note {tag}"
    token = f"HB-RECIPE-{tag}"
    try:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "source.txt")
            source.write_text(token)
            _osa(file_token_to_note(source, title))
            _osa(file_token_to_note(source, title))
            observed = normalize_note_body(_note_body(title), title)
            return observed == (token,), repr(observed)
    finally:
        _delete_note(title)


def verify_reminders(tag):
    list_name = f"Hunch Recipe PDF List {tag}"
    try:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("contract.pdf", "invoice.PDF", "image.png", "memo.txt"):
                Path(directory, name).write_text("x")
            _osa(reminders_for_extension(directory, list_name))
            _osa(reminders_for_extension(directory, list_name))
            observed = _osa(f'''tell application "Reminders"
    set targetList to first list whose name is {as_str(list_name)}
    set namesList to name of every reminder of targetList
    set AppleScript's text item delimiters to "|"
    return namesList as text
end tell''')
            actual = tuple(sorted(item for item in observed.split("|") if item))
            expected = ("contract.pdf", "invoice.PDF")
            return actual == expected, repr(actual)
    finally:
        _delete_reminder_list(list_name)


def verify_calendar(tag):
    writable = _osa(
        'tell application "Calendar" to return name of every calendar whose writable is true')
    calendars = tuple(item.strip() for item in writable.split(",") if item.strip())
    if not calendars:
        return None, "no writable calendar"
    calendar = "HunchBench" if "HunchBench" in calendars else calendars[0]
    summary = f"Hunch Recipe Calendar {tag}"
    try:
        script = exact_calendar_event(calendar, summary, days_from_today=1,
                                      hour=15, minute=0, duration_minutes=30)
        _osa(script)
        _osa(script)
        observed = _osa(f'''set tomorrow to (current date) + 1 * days
tell application "Calendar"
    set matches to every event of calendar {as_str(calendar)} whose summary is {as_str(summary)}
    if (count of matches) is not 1 then return "wrong-count"
    set e to item 1 of matches
    set s to start date of e
    set f to end date of e
    if (year of s is year of tomorrow) and (month of s is month of tomorrow) and (day of s is day of tomorrow) and (hours of s is 15) and (minutes of s is 0) and (hours of f is 15) and (minutes of f is 30) then return "PASS"
    return "wrong-date"
end tell''')
        return observed == "PASS", observed
    finally:
        _delete_event(calendar, summary)


CHECKS = {
    "exact-note": verify_exact_note,
    "note-edit": verify_note_edit,
    "file-to-note": verify_file_to_note,
    "reminders": verify_reminders,
    "calendar": verify_calendar,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", nargs="*", choices=tuple(CHECKS))
    args = parser.parse_args(argv)
    selected = args.checks or tuple(CHECKS)
    tag = uuid.uuid4().hex[:10]
    failures = []
    for name in selected:
        try:
            passed, detail = CHECKS[name](tag)
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        label = "SKIP" if passed is None else "PASS" if passed else "FAIL"
        print(f"{label:4} {name}" + (f" ({detail})" if passed is not True else ""))
        if passed is False:
            failures.append(name)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
