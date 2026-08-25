from hunch.recipes import qwen_recipe_bytes, qwen_recipe_sha256, qwen_recipe_text
from hunch.recipes.applescript import (exact_calendar_event, exact_note,
                                        file_token_to_note,
                                        reminders_for_extension)
from hunch.recipes import verify as verify_mod
from hunch.recipes.verify import normalize_note_body


def test_versioned_recipe_pack_is_complete_and_stable():
    text = qwen_recipe_text()
    assert "Notes: create or reconcile" in text
    assert "Directory files to one exact Reminders list" in text
    assert "Calendar: create or reconcile" in text
    assert qwen_recipe_bytes().decode() == text + "\n"
    assert len(qwen_recipe_sha256()) == 64


def test_exact_note_has_distinct_idempotent_create_and_update_html():
    script = exact_note('A & <B> "C"', ["milk", "eggs & bread", "<literal>"])
    assert 'body:"<div>milk</div><div>eggs &amp; bread</div><div>&lt;literal&gt;</div>"' in script
    assert '<div>A &amp; &lt;B&gt; &quot;C&quot;</div><div>milk</div>' in script
    assert "duplicate notes with requested title" in script
    required = exact_note("Existing", ["done"], must_exist=True)
    assert "required note does not exist" in required
    assert "make new note" not in required


def test_cross_app_and_calendar_builders_are_bounded_and_idempotent(tmp_path):
    reminders = reminders_for_extension(tmp_path, "Probe List", ".pdf")
    assert str(tmp_path.resolve()) in reminders
    assert "-iname '*.pdf'" in reminders
    assert "repeat while (count of reminders of targetList) > 0" in reminders
    event = exact_calendar_event("HunchBench", "Bench Sync", days_from_today=1,
                                 hour=15, minute=0, duration_minutes=30)
    assert "set start date of targetEvent to startTime" in event
    assert "duplicate events with requested summary" in event


def test_file_token_recipe_uses_absolute_path_and_readback(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("TOKEN")
    script = file_token_to_note(path, "Relay")
    assert str(path.resolve()) in script
    assert "as «class utf8»" in script and "return body of n" in script


def test_note_verifier_compares_exact_normalized_lines():
    body = "<div>Title</div><div>milk</div><div>eggs &amp; bread</div>"
    assert normalize_note_body(body, "Title") == ("milk", "eggs & bread")
    assert normalize_note_body(body, "Other") == ("Title", "milk", "eggs & bread")


def test_verifier_cli_defaults_to_every_check(monkeypatch):
    called = []
    monkeypatch.setattr(verify_mod, "CHECKS", {
        "one": lambda tag: (called.append("one") or (True, "")),
        "two": lambda tag: (called.append("two") or (True, "")),
    })
    assert verify_mod.main([]) == 0
    assert called == ["one", "two"]
