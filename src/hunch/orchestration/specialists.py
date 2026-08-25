"""Immutable specialist registry and canonical recipe loading."""
import hashlib
from dataclasses import dataclass
from importlib.resources import files

from ..agent import _DISPATCH
from .contracts import SpecialistKind


@dataclass(frozen=True)
class SpecialistSpec:
    kind: SpecialistKind
    tool_names: frozenset[str]
    recipe_resource: str
    allowed_context_keys: frozenset[str]

    def prompt(self):
        return files("hunch.orchestration.recipes").joinpath(self.recipe_resource).read_text()


SPECIALISTS = {
    SpecialistKind.APPLESCRIPT: SpecialistSpec(
        SpecialistKind.APPLESCRIPT,
        frozenset({"applescript", "clipboard_get", "clipboard_set", "list_apps",
                   "notify_user"}), "applescript.md",
        frozenset({"app", "account", "target", "source", "note_title",
                   "list_name", "source_directory", "folder_path"})),
    SpecialistKind.NATIVE: SpecialistSpec(
        SpecialistKind.NATIVE,
        frozenset({"file_op", "trash", "open_file", "reveal_in_finder",
                   "clipboard_get", "clipboard_set", "list_apps", "launch_app",
                   "quit_app", "notify_user", "applescript"}), "native.md",
        frozenset({"app", "source", "destination", "target",
                   "source_directory", "folder_path"})),
    SpecialistKind.AX: SpecialistSpec(
        SpecialistKind.AX,
        frozenset({"snapshot", "find", "act", "list_apps", "launch_app",
                   "quit_app", "notify_user"}), "ax.md",
        frozenset({"app", "target", "window"})),
    SpecialistKind.WEB: SpecialistSpec(
        SpecialistKind.WEB,
        frozenset({"web_open", "web_login", "web_snapshot", "web_screenshot",
                   "web_act", "web_restart", "web_tabs", "web_switch_tab",
                   "list_credentials", "web_fill_login", "web_fill_secret",
                   "clipboard_get", "clipboard_set", "notify_user"}), "web.md",
        frozenset({"app", "url", "service", "target"})),
}


def validate_registry():
    unknown = {name for spec in SPECIALISTS.values() for name in spec.tool_names
               if name not in _DISPATCH}
    if unknown:
        raise RuntimeError(f"orchestration registry contains unknown tools: {sorted(unknown)}")
    if "delegate" in _DISPATCH:
        raise RuntimeError("private delegate tool leaked into Hunch dispatch")


def canonical_recipe_bytes():
    parts = []
    for kind in SpecialistKind:
        spec = SPECIALISTS[kind]
        parts.append(f"## {kind.value}\n".encode())
        parts.append(spec.prompt().encode())
        parts.append(b"\n")
    return b"".join(parts)


def recipe_sha256():
    return hashlib.sha256(canonical_recipe_bytes()).hexdigest()


validate_registry()
