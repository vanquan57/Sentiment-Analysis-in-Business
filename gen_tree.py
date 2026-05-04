# gen_tree.py — dùng ASCII, không bị lỗi encoding
import sys
from pathlib import Path

IGNORE = {
    "venv", "venv310", ".git", "__pycache__", ".idea", ".vscode",
    "node_modules", ".env", "*.pyc", "*.pyd"
}

def should_ignore(name):
    return any(
        name == ig or name.endswith(ig.replace("*", ""))
        for ig in IGNORE
    )

def tree(path: Path, prefix: str = ""):
    entries = sorted(
        [e for e in path.iterdir() if not should_ignore(e.name)],
        key=lambda e: (e.is_file(), e.name)
    )
    for i, entry in enumerate(entries):
        connector = "+-- " if i < len(entries) - 1 else "\\-- "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "|   " if i < len(entries) - 1 else "    "
            tree(entry, prefix + extension)

root = Path(".")
print(root.resolve().name + "/")
tree(root)