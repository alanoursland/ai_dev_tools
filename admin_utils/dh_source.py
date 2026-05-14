import argparse
import sys
import configparser
from pathlib import Path

# Try to import the library we just created.
try:
    from code_dehydrator import Dehydrator
except ImportError:
    print("Error: Could not import 'code_dehydrator'.", file=sys.stderr)
    print("Please ensure 'code_dehydrator.py' is in the same directory.", file=sys.stderr)
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively dehydrate Python files based on an INI configuration file."
    )
    parser.add_argument(
        "config_name", 
        type=str, 
        help="Name of the configuration (e.g. 'src' will load 'src.ini')."
    )
    parser.add_argument(
        "--pretty", 
        action="store_true",
        help="Format output with indentation and newlines. Default is compact (single line)."
    )
    return parser.parse_args()

def resolve_path(base_dir: Path, path_str: str) -> Path:
    """
    Resolves a path string. 
    If absolute, returns it. 
    If relative, resolves it relative to base_dir.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()

def main():
    args = parse_args()

    # 1. Load Configuration
    config_path = Path(args.config_name)
    if not config_path.suffix:
        config_path = config_path.with_suffix(".ini")

    # Resolve config path to absolute so we can find its parent
    config_path = config_path.resolve()

    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Calculate the "Anchor" directory (where the INI lives)
    config_base_dir = config_path.parent

    config = configparser.ConfigParser()
    
    # Robust loading: Handle missing headers automatically
    try:
        config.read(config_path, encoding='utf-8')
    except configparser.MissingSectionHeaderError:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        config.read_string(f"[dehydrator]\n{content}")
    except Exception as e:
        print(f"Error reading config file: {e}", file=sys.stderr)
        sys.exit(1)

    if 'dehydrator' in config:
        settings = config['dehydrator']
    else:
        print(f"Error: Could not find [dehydrator] section in '{config_path}'.", file=sys.stderr)
        sys.exit(1)

    if 'source_dir' not in settings or 'output_dir' not in settings:
        print(f"Error: Config file must contain 'source_dir' and 'output_dir'.", file=sys.stderr)
        sys.exit(1)

    # 2. Setup Paths (Relative to the INI file location)
    source_dir = resolve_path(config_base_dir, settings['source_dir'])
    output_dir = resolve_path(config_base_dir, settings['output_dir'])
    
    raw_repo_root = settings.get('repo_root')
    if raw_repo_root:
        repo_root = resolve_path(config_base_dir, raw_repo_root)
    else:
        repo_root = source_dir.parent

    # 3. Validate Inputs
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Dehydration Configuration ---")
    print(f"Config File: {config_path}")
    print(f"Config Base: {config_base_dir} (Paths resolved relative to this)")
    print(f"Source:      {source_dir}")
    print(f"Output:      {output_dir}")
    print(f"Repo Root:   {repo_root}")
    print(f"Format:      {'Pretty' if args.pretty else 'Compact'}")
    print(f"---------------------------------")

    # 4. Initialize Dehydrator
    dehydrator = Dehydrator()
    files_processed = 0
    files_skipped = 0

    # 5. Walk the directory
    for py_file in source_dir.rglob("*.py"):
        
        # --- A. Determine Module Name ---
        try:
            rel_path = py_file.relative_to(source_dir.parent)
        except ValueError:
            continue

        parts = list(rel_path.with_suffix("").parts)

        # Handle __init__.py convention
        if parts[-1] == "__init__":
            parts.pop()
            if not parts:
                module_name = source_dir.name
            else:
                module_name = ".".join(parts)
        else:
            module_name = ".".join(parts)

        if not module_name:
            module_name = "root"

        # --- B. Determine Display Path ---
        try:
            file_display = str(py_file.relative_to(repo_root))
        except ValueError:
            file_display = str(py_file)

        # --- C. Process File ---
        try:
            source_code = py_file.read_text(encoding="utf-8", errors="replace")
            
            blocks = dehydrator.dehydrate_module(
                source_code, 
                file_display, 
                module_name,
                pretty=args.pretty
            )
            
            if not blocks:
                files_skipped += 1
                continue

            # --- D. Write Output ---
            output_filename = f"{module_name}.txt"
            output_path = output_dir / output_filename
            
            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"{{ FILE: {file_display}\n")
                
                for block in blocks:
                    # Indent the function block content for hierarchy
                    content = block.content
                    # Indent every line by 2 spaces
                    indented_content = "\n".join("  " + line for line in content.splitlines())
                    f.write(indented_content)
                    f.write("\n")
                
                f.write("}\n")
            
            print(f"Generated: {output_filename}")
            files_processed += 1

        except Exception as e:
            print(f"FAILED to process {py_file.name}: {e}", file=sys.stderr)

    # 6. Summary
    print(f"---------------------------------")
    print(f"Complete.")
    print(f"Files Dehydrated: {files_processed}")
    print(f"Files Skipped:    {files_skipped} (no definitions found)")

if __name__ == "__main__":
    main()