"""Render the executable forms of the AppleScript recipes.

The Markdown pack teaches a model the same patterns. These builders are used by
the real-app verifier so recipe verification cannot silently drift from the
documented quoting, idempotency, or read-back behavior.
"""
from html import escape
from pathlib import Path

from ..notify import as_str


def _html_lines(lines):
    values = tuple(lines)
    if not values or any(not isinstance(line, str) for line in values):
        raise ValueError("note lines must be a non-empty sequence of strings")
    return "".join(f"<div>{escape(line)}</div>" for line in values)


def exact_note(title, lines, *, must_exist=False):
    """Create or reconcile one exact-line note and return its body.

    Notes prepends the title when a note is first created with ``name``. On an
    existing note, replacing ``body`` without the title can rename it. The two
    HTML values are intentionally different for those two object-model paths.
    """
    if not isinstance(title, str) or not title:
        raise ValueError("note title cannot be empty")
    body = _html_lines(lines)
    update_body = f"<div>{escape(title)}</div>" + body
    missing = 'error "required note does not exist"' if must_exist else (
        f"set n to make new note with properties {{name:noteTitle, body:{as_str(body)}}}")
    return f'''set noteTitle to {as_str(title)}
tell application "Notes"
    set matches to every note whose name is noteTitle
    if (count of matches) > 1 then error "duplicate notes with requested title"
    if (count of matches) = 0 then
        {missing}
    else
        set n to item 1 of matches
        set body of n to {as_str(update_body)}
    end if
    return body of n
end tell'''


def file_token_to_note(path, title):
    """Create or reconcile a note from one scalar UTF-8 token file."""
    absolute = str(Path(path).expanduser().resolve())
    if not title:
        raise ValueError("note title cannot be empty")
    return f'''set noteTitle to {as_str(title)}
set tokenValue to read POSIX file {as_str(absolute)} as «class utf8»
tell application "Notes"
    set matches to every note whose name is noteTitle
    if (count of matches) > 1 then error "duplicate notes with requested title"
    if (count of matches) = 0 then
        set n to make new note with properties {{name:noteTitle, body:tokenValue}}
    else
        set n to item 1 of matches
        set body of n to "<div>" & noteTitle & "</div><div>" & tokenValue & "</div>"
    end if
    return body of n
end tell'''


def reminders_for_extension(folder, list_name, extension="pdf"):
    """Reconcile a task-owned reminder list to sorted matching file names."""
    absolute = str(Path(folder).expanduser().resolve())
    extension = extension.lstrip(".").lower()
    if not list_name or not extension or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789" for c in extension):
        raise ValueError("list name and alphanumeric extension are required")
    return f'''set folderPath to {as_str(absolute + "/")}
set commandText to "/usr/bin/find " & quoted form of folderPath & " -maxdepth 1 -type f -iname '*.{extension}' -print | /usr/bin/sed 's#.*/##' | /usr/bin/sort"
set itemNames to paragraphs of (do shell script commandText)
tell application "Reminders"
    set matches to every list whose name is {as_str(list_name)}
    if (count of matches) > 1 then error "duplicate reminder lists with requested name"
    if (count of matches) = 0 then
        set targetList to make new list with properties {{name:{as_str(list_name)}}}
    else
        set targetList to item 1 of matches
        repeat while (count of reminders of targetList) > 0
            delete first reminder of targetList
        end repeat
    end if
    repeat with itemName in itemNames
        if (contents of itemName) is not "" then
            make new reminder at end of reminders of targetList with properties {{name:(contents of itemName)}}
        end if
    end repeat
    return name of every reminder of targetList
end tell'''


def exact_calendar_event(calendar, summary, *, days_from_today, hour, minute,
                         duration_minutes):
    """Create or reconcile one named event and return its observed dates."""
    for value, name in ((days_from_today, "days_from_today"), (hour, "hour"),
                        (minute, "minute"), (duration_minutes, "duration_minutes")):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and duration_minutes > 0):
        raise ValueError("invalid calendar time")
    return f'''tell application "Calendar"
    set targetCalendar to calendar {as_str(calendar)}
    set startTime to (current date) + {days_from_today} * days
    set hours of startTime to {hour}
    set minutes of startTime to {minute}
    set seconds of startTime to 0
    set endTime to startTime + {duration_minutes} * minutes
    set matches to every event of targetCalendar whose summary is {as_str(summary)}
    if (count of matches) > 1 then error "duplicate events with requested summary"
    if (count of matches) = 0 then
        set targetEvent to make new event at end of events of targetCalendar with properties {{summary:{as_str(summary)}, start date:startTime, end date:endTime}}
    else
        set targetEvent to item 1 of matches
        set start date of targetEvent to startTime
        set end date of targetEvent to endTime
    end if
    return {{start date of targetEvent, end date of targetEvent}}
end tell'''


__all__ = ["exact_calendar_event", "exact_note", "file_token_to_note",
           "reminders_for_extension"]
