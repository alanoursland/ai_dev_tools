#!/usr/bin/env python
import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set, Dict


# =============================================================================
#  TRON THEME HELPERS
# =============================================================================

def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


USE_COLOR = supports_color()

COLOR_CYAN = "\033[96m" if USE_COLOR else ""
COLOR_YELLOW = "\033[93m" if USE_COLOR else ""
COLOR_RED = "\033[91m" if USE_COLOR else ""
COLOR_RESET = "\033[0m" if USE_COLOR else ""


def tron_tag(kind: str = "info") -> str:
    if not USE_COLOR:
        return "[TRON]"
    if kind == "warn":
        return f"{COLOR_YELLOW}[TRON]{COLOR_RESET}"
    if kind == "error":
        return f"{COLOR_RED}[TRON]{COLOR_RESET}"
    return f"{COLOR_CYAN}[TRON]{COLOR_RESET}"


def tron_say(message: str, kind: str = "info", file=None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{tron_tag(kind)} {message}", file=file)


_MESSAGES = {
    "intro": [
        "Identity disc online. You fight for the users.",
        "Boot sequence complete. TRON standing by.",
        "Grid link established. Preparing identity disc.",
    ],
    "target_lock": [
        "Locking onto red program: {target}",
        "Target acquired on the game grid: {target}",
        "Tracing hostile routine: {target}",
    ],
    "prepare_trail": [
        "Clearing old light trails. Making room for fresh data.",
        "Purging previous sectors from light_trail/.",
        "Resetting the battlefield in tron/light_trail/.",
    ],
    "trace_start": [
        "Throwing identity disc at target...",
        "Disc away. Following the call stack into the grid.",
        "Casting trace beam through the program sectors.",
    ],
    "trace_done": [
        "Trace complete. Call stack captured as {trace_name}.",
        "Identity disc returned. Trace file: {trace_name}.",
        "Call tree dehydrated. Manifest stored as {trace_name}.",
    ],
    "copy_start": [
        "Harvesting sector files from the light trail manifest.",
        "Gathering referenced code sectors for investigation.",
        "Collecting all FILE: sectors into tron/light_trail/.",
    ],
    "copy_done": [
        "Sector transfer complete. {count} files copied.",
        "All referenced sectors secured. {count} files in the light trail.",
        "File harvest finished. {count} sector files ready for inspection.",
    ],
    "copy_none": [
        "No FILE: sectors found in trace. Only the manifest is available.",
        "Trace contained no sector references. Study the call stack manifest.",
    ],
    "missing_files": [
        "Warning: some sectors referenced in the trace could not be located.",
        "Grid desync detected: not all referenced sector files exist on disk.",
    ],
    "finished": [
        "Mission step complete. Inspect tron/light_trail/ and plan your fix.",
        "Light trail is ready. Study the sectors, then strike once.",
        "TRACE ON. The rest is up to you.",
    ],
}


def tron_random(key: str, kind: str = "info", **fields) -> None:
    templates = _MESSAGES.get(key, ["{msg}"])
    template = random.choice(templates)
    message = template.format(**fields)
    tron_say(message, kind=kind)


# =============================================================================
#  CLI PARSING
# =============================================================================

def parse_cli_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRON Identity Disc\n"
            "Trace a pytest node id and collect its sector files into tron/light_trail/.\n"
            "You are TRON. You fight for the users."
        )
    )
    parser.add_argument(
        "node_id",
        help=(
            "Pytest-style node id, e.g. "
            "tests/foo/test_bar.py::test_something or "
            "tests/foo/test_bar.py::TestClass::test_method"
        ),
    )
    parser.add_argument(
        "-p",
        "--pretty",
        action="store_true",
        help="Pretty-print the dehydrated call tree in the trace file.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable extra debug tracing (written into the trace file).",
    )
    return parser.parse_args(argv)


# =============================================================================
#  PATHS & ENVIRONMENT
# =============================================================================

def add_admin_tools_to_sys_path(repo_root: Path) -> None:
    admin_tools = repo_root / "admin_utils"
    if str(admin_tools) not in sys.path:
        sys.path.insert(0, str(admin_tools))


def prepare_light_trail_dir(script_path: Path) -> Path:
    """
    Ensure tron/light_trail/ exists and is empty.
    """
    tron_dir = script_path.parent
    dest_dir = tron_dir / "light_trail"

    tron_random("prepare_trail")

    if dest_dir.exists():
        if dest_dir.is_dir():
            shutil.rmtree(dest_dir)
        else:
            dest_dir.unlink()

    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir


# =============================================================================
#  TRACE GENERATION (USING call_tree.py)
# =============================================================================

def generate_trace_to_dir(
    node_id: str,
    dest_dir: Path,
    repo_root: Path,
    pretty: bool,
    debug: bool,
) -> Path:
    """
    Use call_tree's logic to generate a trace for `node_id`.
    Adapted for the new batch-collection architecture of call_tree.py.
    """
    import call_tree as ct  # type: ignore

    ct.DEBUG = debug
    roots = ct.build_roots(repo_root)

    raw_name = node_id.replace(os.sep, "_").replace("::", "__")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_name)
    trace_path = dest_dir / f"{safe_name}.txt"

    # Override tee_print so it only writes to the trace file, not stdout.
    def file_only_tee_print(*args, **kwargs):
        if ct.TRACE_FILE is not None:
            # Ensure we don't accidentally honor a 'file=' kwarg from internal calls
            kwargs = {k: v for k, v in kwargs.items() if k != "file"}
            print(*args, file=ct.TRACE_FILE, **kwargs)

    ct.tee_print = file_only_tee_print

    ct.TRACE_FILE = trace_path.open("w", encoding="utf-8")
    
    # New architecture: Collect results in a dict, then print at the end
    grouped_results: Dict[str, List[str]] = {}

    try:
        # Resolve entry point
        ctx, func_node, function_name = ct.resolve_entry_point(
            node_id=node_id,
            roots=roots,
            repo_root=repo_root,
        )

        # 1. Dehydrate the entry function
        # UPDATED: Use dehydrate_node_wrapper instead of dehydrate_node
        results = ct.dehydrate_node_wrapper(func_node, ctx, function_name, pretty)

        f = ct.TRACE_FILE
        # Header & legend, written directly to the file
        print("DEHYDRATED CALL TREE", file=f)
        print("Legend:", file=f)
        print("  []  : block executed inside a loop", file=f)
        print("  ()  : block executed conditionally", file=f)
        print("  <>  : instance call on a known type", file=f)
        print("  <<>>: variable/attr with unknown type", file=f)
        print("", file=f)

        if not results:
            print(f"No logical calls found in: {function_name}", file=f)
            print(
                "(This usually means it is a class with no explicit constructor)",
                file=f,
            )
            return trace_path

        all_targets: Set[str] = set()

        # 2. Store initial results and gather targets
        for block_content, calls_structure in results:
            # Add to grouped results for final reporting
            file_display = ctx.dehydration_ctx.file_display
            if file_display not in grouped_results:
                grouped_results[file_display] = []
            grouped_results[file_display].append(block_content)

            # Gather next targets
            targets = ct.gather_call_targets(calls_structure)
            all_targets.update(targets)

        # 3. Recursively follow call targets
        # UPDATED: Pass grouped_results to the loop
        if all_targets:
            ct.run_recursion_loop(
                initial_targets=all_targets,
                ctx=ctx,
                roots=roots,
                repo_root=repo_root,
                pretty=pretty,
                debug=debug,
                grouped_results=grouped_results
            )
            
        # 4. Print the final report
        # This will write all the collected blocks to the file via tee_print
        ct.print_final_report(grouped_results, pretty)

    finally:
        if ct.TRACE_FILE is not None:
            ct.TRACE_FILE.close()
            ct.TRACE_FILE = None

    return trace_path


# =============================================================================
#  COPY REFERENCED FILES (USING collect_trace_files.py)
# =============================================================================

def copy_files_from_trace(
    trace_path: Path,
    repo_root: Path,
    dest_dir: Path,
) -> None:
    """
    Parse FILE: lines from the trace and copy all referenced files
    into dest_dir, flattening by filename.
    """
    # FIX: Updated import name to match collect_trace_files.py
    from collect_trace_files import extract_paths_from_trace  # type: ignore

    tron_random("copy_start")

    # FIX: Updated function call
    rel_paths = extract_paths_from_trace(trace_path)

    if not rel_paths:
        tron_random("copy_none", kind="warn")
        return

    missing = []
    copied = 0

    for rel in sorted(rel_paths):
        src_path = (repo_root / rel).resolve()
        if not src_path.is_file():
            missing.append(rel)
            continue

        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)
        copied += 1

    tron_random("copy_done", count=copied)

    if missing:
        tron_random("missing_files", kind="warn")
        for m in missing:
            tron_say(f"  missing: {m}", kind="warn", file=sys.stderr)
            
# =============================================================================
#  MAIN
# =============================================================================

def main(argv=None) -> None:
    args = parse_cli_args(argv)

    tron_random("intro")
    tron_random("target_lock", target=args.node_id)

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    # Ensure admin_utils is in path so we can import call_tree
    add_admin_tools_to_sys_path(repo_root)

    # 1. Prepare tron/light_trail/
    dest_dir = prepare_light_trail_dir(script_path)

    # 2. Generate trace into tron/light_trail/
    tron_random("trace_start")
    trace_path = generate_trace_to_dir(
        node_id=args.node_id,
        dest_dir=dest_dir,
        repo_root=repo_root,
        pretty=args.pretty,
        debug=args.debug,
    )
    tron_random("trace_done", trace_name=trace_path.name)

    # 3. Copy all referenced files into tron/light_trail/
    copy_files_from_trace(trace_path, repo_root, dest_dir)

    # 4. Final message
    tron_random("finished")


if __name__ == "__main__":
    main()