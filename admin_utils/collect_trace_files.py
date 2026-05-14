import sys
import shutil
import argparse
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Set, Dict, Optional

def parse_cli_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect trace files and resolve filename collisions."
    )
    parser.add_argument(
        "node_id",
        nargs="?",
        help="Pytest-style node id (optional). If provided, -n is ignored."
    )
    # NEW: Argument to specify the number of recent traces
    parser.add_argument(
        "-n", "--last-n",
        type=int,
        default=1,
        help="Process the last N generated traces (default: 1)."
    )
    return parser.parse_args(argv)

def node_id_to_trace_filename(node_id: str) -> str:
    raw_name = node_id.replace(os.sep, "_").replace("::", "__")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_name)
    return f"{safe_name}.txt"

def find_trace_files(script_dir: Path, node_id: Optional[str], n: int) -> List[Path]:
    trace_history_dir = script_dir / "trace_history"
    
    if not trace_history_dir.is_dir():
        print(f"Error: trace_history directory not found at {trace_history_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Specific Node ID (Takes priority)
    if node_id:
        filename = node_id_to_trace_filename(node_id)
        trace_path = trace_history_dir / filename
        if not trace_path.is_file():
            print(f"Error: Trace file for {node_id} not found at {trace_path}", file=sys.stderr)
            sys.exit(1)
        return [trace_path]

    # 2. Find last N files
    candidates = list(trace_history_dir.glob("*.txt"))
    if not candidates:
        print(f"Error: No trace files found in {trace_history_dir}", file=sys.stderr)
        sys.exit(1)
        
    # Sort by time descending (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Return top N
    return candidates[:n]

def extract_paths_from_trace(trace_path: Path) -> Set[str]:
    paths = set()
    # Added encoding safety
    try:
        content = trace_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Warning: Could not read {trace_path.name}, skipping.")
        return set()

    for line in content.splitlines():
        if "FILE:" in line:
            _, _, rest = line.partition("FILE:")
            path_str = rest.strip().split()[0].strip("{}")
            if path_str:
                paths.add(path_str)
    return paths

def resolve_filename_collisions(paths: Set[str], repo_root: Path) -> Dict[str, str]:
    """
    Resolve filename collisions by prepending parent directories.
    Returns a dict mapping original relative path string -> new unique filename.
    """
    # Group paths by their base filename
    by_filename = defaultdict(list)
    for p_str in paths:
        p = Path(p_str)
        by_filename[p.name].append(p_str)
    
    mapping = {}
    
    for filename, path_list in by_filename.items():
        if len(path_list) == 1:
            # No collision
            mapping[path_list[0]] = filename
        else:
            # Collision detected, resolve for each
            for p_str in path_list:
                p = Path(p_str)
                parts = list(p.parts)
                # Start with just the filename
                new_name = parts[-1]
                depth = 2
                
                # Check if this new_name is unique among the *current set of colliding paths*
                # We need to make sure the generated name doesn't conflict with OTHERS in this group
                # OR with others in the global set (though simpler to just resolve locally first)
                
                while True:
                    # Check against all other paths in this specific collision group
                    is_unique = True
                    for other_p_str in path_list:
                        if other_p_str == p_str:
                            continue
                        
                        # Generate the candidate name for the OTHER path at same depth
                        other_parts = list(Path(other_p_str).parts)
                        if len(other_parts) >= depth:
                            other_candidate = "-".join(other_parts[-depth:])
                        else:
                            other_candidate = "-".join(other_parts)
                            
                        if new_name == other_candidate:
                            is_unique = False
                            break
                    
                    if is_unique:
                        break
                        
                    # Go up one level
                    if depth <= len(parts):
                        new_name = "-".join(parts[-depth:])
                        depth += 1
                    else:
                        # Should not happen if paths are unique inputs
                        break
                
                mapping[p_str] = new_name
    return mapping

def main(argv=None):
    args = parse_cli_args(argv)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    # 1. Get List of Trace Files
    trace_files = find_trace_files(script_dir, args.node_id, args.last_n)
    
    if args.node_id:
        print(f"Processing specific trace: {trace_files[0].name}")
    else:
        print(f"Processing last {len(trace_files)} trace(s)...")

    # 2. Accumulate paths from ALL traces
    all_paths = set()
    for tf in trace_files:
        print(f"  - Reading {tf.name}")
        paths = extract_paths_from_trace(tf)
        all_paths.update(paths)

    if not all_paths:
        print("No files found in selected traces.")
        return

    # Filter for existing files
    existing_paths = {p for p in all_paths if (repo_root / p).is_file()}
    missing_paths = all_paths - existing_paths
    
    if missing_paths:
        print(f"Warning: {len(missing_paths)} files referenced in traces not found on disk.")

    # Resolve names (Union of all files)
    mapping = resolve_filename_collisions(existing_paths, repo_root)
    
    # Reset Pool
    pool_dir = script_dir / "trace_pool"
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    pool_dir.mkdir()
    
    # Copy Source Files
    for original_path, new_name in mapping.items():
        src = repo_root / original_path
        dst = pool_dir / new_name
        shutil.copy2(src, dst)
        
    print(f"Copied {len(mapping)} source files.")

    # Copy ALL Trace Files used
    for tf in trace_files:
        shutil.copy2(tf, pool_dir / tf.name)

    print(f"\nTrace pool populated at {pool_dir}")

if __name__ == "__main__":
    main()