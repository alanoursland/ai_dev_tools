import ast
import os
import sys
import argparse

def get_file_content(filepath):
    """Reads file content safely."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def extract_class_source(content, node):
    """
    Extracts the source code for a specific AST class node.
    Handles decorators which appear before the class definition line.
    """
    lines = content.splitlines(keepends=True)
    
    # Determine start line (min of decorators or class def)
    start_lineno = node.lineno
    if node.decorator_list:
        start_lineno = min(d.lineno for d in node.decorator_list)
    
    end_lineno = node.end_lineno
    
    if start_lineno is None or end_lineno is None:
        return None

    # Slice the lines (1-based index to 0-based)
    return "".join(lines[start_lineno-1 : end_lineno])

def analyze_and_merge(root_file, src_file, execute=False):
    """
    Compares two files. If root_file has classes missing from src_file,
    it extracts them, appends them to src_file, and REMOVES them from root_file.
    """
    root_content = get_file_content(root_file)
    src_content = get_file_content(src_file)

    if root_content is None or src_content is None:
        return

    try:
        root_tree = ast.parse(root_content, filename=root_file)
        src_tree = ast.parse(src_content, filename=src_file)
    except SyntaxError as e:
        print(f"Syntax Error parsing files: {e}")
        return

    # Map class names to AST nodes
    root_classes = {node.name: node for node in ast.walk(root_tree) if isinstance(node, ast.ClassDef)}
    src_classes = {node.name: node for node in ast.walk(src_tree) if isinstance(node, ast.ClassDef)}

    # Identify missing classes
    missing_classes = sorted(list(set(root_classes.keys()) - set(src_classes.keys())))

    # Relative path for display
    rel_path = os.path.basename(root_file) 
    if "tests" in root_file:
        try:
            rel_path = root_file.split("tests")[1].lstrip(os.sep)
        except:
            pass

    if not missing_classes:
        return

    print(f"FILE: {rel_path}")
    print(f"  Missing in src/:  \033[91m{', '.join(missing_classes)}\033[0m")

    if execute:
        # 1. APPEND to src/tests
        with open(src_file, "a", encoding="utf-8") as f:
            f.write("\n\n# =============================================================================\n")
            f.write(f"# AUTOMATICALLY MERGED FROM tests/{rel_path.replace(os.sep, '/')}\n")
            f.write("# TODO: CHECK IMPORTS! These classes were moved but imports might be missing.\n")
            f.write("# =============================================================================\n\n")
            
            for class_name in missing_classes:
                node = root_classes[class_name]
                source_code = extract_class_source(root_content, node)
                
                if source_code:
                    print(f"  -> Copying class '{class_name}' to dest...")
                    f.write(source_code)
                    f.write("\n\n")
                else:
                    print(f"  -> Error: Could not extract source for '{class_name}'")
        
        # 2. REMOVE from tests/ (in-place modification)
        # We must read lines again to perform deletion
        root_lines = root_content.splitlines(keepends=True)
        
        # Get nodes to remove
        nodes_to_remove = [root_classes[name] for name in missing_classes]
        
        # SORT DESCENDING by line number! 
        # Crucial: Deleting from bottom up preserves line numbers of items above.
        nodes_to_remove.sort(key=lambda n: n.lineno, reverse=True)
        
        for node in nodes_to_remove:
            # Determine start line (min of decorators or class def)
            start_lineno = node.lineno
            if node.decorator_list:
                start_lineno = min(d.lineno for d in node.decorator_list)
            end_lineno = node.end_lineno

            if start_lineno and end_lineno:
                # Delete the slice (convert 1-based AST line numbers to 0-based list indices)
                # Slice: start_lineno-1 (inclusive) to end_lineno (exclusive in slice, but end_lineno is inclusive in AST)
                # Actually, AST end_lineno is the last line of the class.
                # Python list slice [start:end] excludes end. So we just use end_lineno.
                del root_lines[start_lineno-1 : end_lineno]

        # Write trimmed content back
        with open(root_file, "w", encoding="utf-8") as f:
            f.writelines(root_lines)

        print("  [SUCCESS] Merged to src/ and removed from tests/.")
    else:
        print(f"  [DRY RUN] Would move {len(missing_classes)} classes and delete them from source.")
    
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Merge missing test classes from root tests/ to src/tests/")
    parser.add_argument("--execute", action="store_true", help="Perform the actual file writes (Append + Delete).")
    args = parser.parse_args()

    # Determine directories
    if os.path.exists("tests") and os.path.exists("src/tests"):
        root_dir = "tests"
        src_dir = "src/tests"
    elif os.path.exists("../tests") and os.path.exists("../src/tests"):
        root_dir = "../tests"
        src_dir = "../src/tests"
    else:
        print("Error: Could not locate 'tests' and 'src/tests' directories.")
        sys.exit(1)

    print(f"Scanning for missing classes from '{root_dir}' to '{src_dir}'...")
    if args.execute:
        print("\033[93m>>> EXECUTION MODE ENABLED (Files will be modified) <<<\033[0m")
    else:
        print("\033[96m>>> DRY RUN (Use --execute to perform merge) <<<\033[0m")
    print("="*60)

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".py"):
                continue

            path_in_root = os.path.join(root, file)
            rel_path = os.path.relpath(path_in_root, root_dir)
            path_in_src = os.path.join(src_dir, rel_path)

            if os.path.exists(path_in_src):
                analyze_and_merge(path_in_root, path_in_src, execute=args.execute)

if __name__ == "__main__":
    main()