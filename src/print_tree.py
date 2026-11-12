
from pathlib import Path
import os

def print_tree(root=".", max_depth=3, prefix=""):
    root = Path(root)
    def _walk(p, depth, pref):
        if depth > max_depth:
            return
        items = sorted(list(p.iterdir()), key=lambda x: (x.is_file(), x.name))
        for i, it in enumerate(items):
            connector = "└── " if i == len(items)-1 else "├── "
            print(pref + connector + it.name)
            if it.is_dir():
                _walk(it, depth+1, pref + ("    " if i == len(items)-1 else "│   "))
    print(root.resolve().name)
    _walk(root, 0, "")

if __name__ == "__main__":
    print_tree(".", max_depth=4)
