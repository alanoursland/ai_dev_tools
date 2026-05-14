import ast
import os
import sys
from typing import Optional

IMPORT_LINE = (
    "from kinopulse.core import DynamicalSystem, Time, State, Input, Output, Parameters"
)


def find_insertion_line(tree: ast.Module) -> int:
    """
    Return the 1-based line number *after* the last import statement.
    If there are no imports, return line after the module docstring (if any),
    otherwise return 1 (top of file).
    """
    import_lines = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # end_lineno is available in Py3.8+
            end_lineno = getattr(node, "end_lineno", node.lineno)
            import_lines.append(end_lineno)

    if import_lines:
        return max(import_lines) + 1

    # No imports: place after docstring if present
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        end_lineno = getattr(tree.body[0], "end_lineno", tree.body[0].lineno)
        return end_lineno + 1

    # No imports, no docstring: very top
    return 1


def already_has_kinopulse_import(tree: ast.Module) -> bool:
    """
    Return True if there is already any 'from kinopulse.core import ...' in the module.
    (Keeps things idempotent without getting fancy about merging names.)
    """
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "kinopulse.core":
            return True
    return False


def process_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        print(f"Skipping {path}: syntax error -> {e}")
        return

    if already_has_kinopulse_import(tree):
        print(f"Skipping {path}: kinopulse import already present")
        return

    insert_line = find_insertion_line(tree)

    lines = source.splitlines(keepends=True)

    # Convert 1-based line number to 0-based index. If insert_line is beyond EOF,
    # just append at the end.
    index = min(max(insert_line - 1, 0), len(lines))

    # Ensure we end the inserted line with a newline
    new_line = IMPORT_LINE + "\n"

    lines.insert(index, new_line)

    new_source = "".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_source)

    print(f"Updated {path}")


def walk_and_process(root: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            if filename == "__init__.py":
                continue
            full_path = os.path.join(dirpath, filename)
            process_file(full_path)


def main(argv: Optional[list] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python insert_imports.py /path/to/root")
        sys.exit(1)

    root = argv[0]
    if not os.path.isdir(root):
        print(f"Error: {root!r} is not a directory")
        sys.exit(1)

    walk_and_process(root)


if __name__ == "__main__":
    main()
