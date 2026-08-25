"""Versioned, model-facing recipes for small local agents."""
import hashlib
from importlib.resources import files


def qwen_recipe_text():
    return files(__package__).joinpath("qwen_static.md").read_text().strip()


def qwen_recipe_bytes():
    return (qwen_recipe_text() + "\n").encode()


def qwen_recipe_sha256():
    return hashlib.sha256(qwen_recipe_bytes()).hexdigest()


__all__ = ["qwen_recipe_bytes", "qwen_recipe_sha256", "qwen_recipe_text"]
