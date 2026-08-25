# Verified operation recipes for Qwen

Use these recipes exactly, substituting only the ALL-CAPS placeholders. Never
add `activate` or `visible:true`. Do not improvise AppleScript control flow. If a
recipe returns an error, check placeholder values once; after a syntax/compile
error, copy the recipe again instead of rewriting it.

Every mutating recipe is idempotent: it reconciles one task-owned object rather
than creating duplicates. A read-back is part of the same call. After a successful
read-back, stop and give the final answer.

## Notes: create or reconcile an exact logical line list

Notes stores `body` as HTML. Never quote `& return &` as text, and do not rely on
plain-text returns: Notes may normalize them to spaces. On creation, `name` causes
Notes to supply the title; do not also put the title in the creation body. On
update, include the title as the first `<div>` so replacing the body cannot rename
the note.

    set noteTitle to "TITLE"
    set createBody to "<div>LINE1</div><div>LINE2</div><div>LINE3</div>"
    set updateBody to "<div>TITLE</div><div>LINE1</div><div>LINE2</div><div>LINE3</div>"
    tell application "Notes"
        set matches to every note whose name is noteTitle
        if (count of matches) > 1 then error "duplicate notes with requested title"
        if (count of matches) = 0 then
            set n to make new note with properties {name:noteTitle, body:createBody}
        else
            set n to item 1 of matches
            set body of n to updateBody
        end if
        return body of n
    end tell

Use one `<div>...</div>` per requested logical line. Escape literal `&`, `<`, and
`>` inside values as `&amp;`, `&lt;`, and `&gt;`.

## Notes: edit an existing note when the complete final sequence is known

Use the exact-line recipe above, but require `(count of matches) = 1`; error if it
is zero. Set `updateBody` to the title div followed by the complete final lines.
This is safer than parsing Notes HTML. Never use it when unspecified content must
be preserved.

## File token to Notes

Resolve the path first. This recipe is for a scalar UTF-8 token with no HTML
markup, where the task requires the body to contain only that token.

    set noteTitle to "TITLE"
    set tokenValue to read POSIX file "/ABSOLUTE/PATH" as «class utf8»
    tell application "Notes"
        set matches to every note whose name is noteTitle
        if (count of matches) > 1 then error "duplicate notes with requested title"
        if (count of matches) = 0 then
            set n to make new note with properties {name:noteTitle, body:tokenValue}
        else
            set n to item 1 of matches
            set body of n to "<div>" & noteTitle & "</div><div>" & tokenValue & "</div>"
        end if
        return body of n
    end tell

## Directory files to one exact Reminders list

Use only for a new or task-owned list whose required final contents are exact.
The re-query/delete-one-at-a-time loop is deliberate; deleting inside a `whose`
loop can silently leave reminders behind.

    set folderPath to "/ABSOLUTE/FOLDER/"
    set commandText to "/usr/bin/find " & quoted form of folderPath & " -maxdepth 1 -type f -iname '*.pdf' -print | /usr/bin/sed 's#.*/##' | /usr/bin/sort"
    set itemNames to paragraphs of (do shell script commandText)
    tell application "Reminders"
        set matches to every list whose name is "LIST_NAME"
        if (count of matches) > 1 then error "duplicate reminder lists with requested name"
        if (count of matches) = 0 then
            set targetList to make new list with properties {name:"LIST_NAME"}
        else
            set targetList to item 1 of matches
            repeat while (count of reminders of targetList) > 0
                delete first reminder of targetList
            end repeat
        end if
        repeat with itemName in itemNames
            if (contents of itemName) is not "" then
                make new reminder at end of reminders of targetList with properties {name:(contents of itemName)}
            end if
        end repeat
        return name of every reminder of targetList
    end tell

## Calendar: create or reconcile one exact event

    tell application "Calendar"
        set targetCalendar to calendar "CALENDAR_NAME"
        set startTime to (current date) + DAYS_FROM_TODAY * days
        set hours of startTime to HOUR_24
        set minutes of startTime to MINUTE
        set seconds of startTime to 0
        set endTime to startTime + DURATION_MINUTES * minutes
        set matches to every event of targetCalendar whose summary is "EVENT_TITLE"
        if (count of matches) > 1 then error "duplicate events with requested summary"
        if (count of matches) = 0 then
            set targetEvent to make new event at end of events of targetCalendar with properties {summary:"EVENT_TITLE", start date:startTime, end date:endTime}
        else
            set targetEvent to item 1 of matches
            set start date of targetEvent to startTime
            set end date of targetEvent to endTime
        end if
        return {start date of targetEvent, end date of targetEvent}
    end tell

`DAYS_FROM_TODAY=1` means tomorrow; `HOUR_24=15` means 3 PM.

## Mail: compose and save a draft, never send

    tell application "Mail"
        set m to make new outgoing message with properties {subject:"SUBJECT", content:"BODY", visible:false}
        tell m
            make new to recipient at end of to recipients with properties {address:"EMAIL"}
            save
        end tell
    end tell
    return "draft saved"

## Paths and direct file operations

Resolve home without inventing a username:

    do shell script "echo $HOME"

Read UTF-8 text:

    read POSIX file "/ABSOLUTE/PATH" as «class utf8»

List a folder:

    do shell script "/bin/ls -1 '/ABSOLUTE/PATH'"

For `file_op`, paths must be absolute and literal: no `~` and no globs. Create a
destination directory before moving files into it. A batch contains one explicit
item per file. Never overwrite unless the task provides a collision rule.
