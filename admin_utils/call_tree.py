#!/usr/bin/env python
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import ast
from typing import Dict, List, Tuple, Optional, Any, Set
from typing import TextIO
import os

# Import the new library
try:
    from code_dehydrator import (
        Dehydrator, 
        DehydrationContext, 
        DehydratedBlock, 
        build_import_map, 
        build_local_defs,
        parse_module,            # Moved logic
        find_ast_node,           # Moved logic
        get_test_function_node,  # Moved logic
        get_raw_call_name        # Moved logic
    )
except ImportError:
    print("Error: Could not import 'code_dehydrator'.", file=sys.stderr)
    print("Please ensure 'code_dehydrator.py' is in the same directory.", file=sys.stderr)
    sys.exit(1)

# =======================
# Global Debug Flag
# =======================
DEBUG = False
TRACE_FILE: Optional[TextIO] = None

# =======================
# Root configuration
# =======================

@dataclass(frozen=True)
class Root:
    kind: str
    prefix: str
    path: Path


def build_roots(repo_root: Path) -> Dict[str, Root]:
    src_root = repo_root / "src"
    admin_path = Path(__file__).resolve().parent
    
    return {
        "kinopulse": Root(
            kind="package",
            prefix="kinopulse.",
            path=src_root / "kinopulse",
        ),
        "src": Root(
            kind="dir",
            prefix="",
            path=src_root,
        ),
        "examples": Root(
            kind="dir",
            prefix="",
            path=src_root / "examples",
        ),
        "admin_tools_dir": Root(
            kind="dir",
            prefix="",
            path=admin_path,
        ),
        "admin_tools_pkg": Root(
            kind="package",
            prefix="",
            path=admin_path,
        ),
        "tokamak": Root(
            kind="dir",
            prefix="",
            path=repo_root / "projects" / "tokamak" / "src",
        ),
    }


# =======================
# Node id / file handling
# =======================

def parse_node_id(node_id: str) -> Tuple[Path, Optional[str], str]:
    parts = node_id.split("::")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(f"Unsupported node id format: {node_id!r}")

    file_rel = parts[0]
    if len(parts) == 2:
        cls_name = None
        func_name = parts[1]
    else:
        cls_name = parts[1]
        func_name = parts[2]

    return Path(file_rel), cls_name, func_name


def resolve_node_file(file_rel: Path, roots: Dict[str, Root]) -> Optional[Path]:
    for root in roots.values():
        if root.kind != "dir":
            continue
        candidate = root.path / file_rel
        if candidate.is_file():
            return candidate
    return None


def build_function_qualified_name_from_module(
    module_name: Optional[str],
    cls_name: Optional[str],
    func_name: str,
) -> str:
    if module_name:
        if cls_name is None:
            return f"{module_name}.{func_name}"
        return f"{module_name}.{cls_name}.{func_name}"
    else:
        if cls_name is None:
            return func_name
        return f"{cls_name}.{func_name}"


# =======================
# Module context
# =======================

@dataclass
class ModuleContext:
    file_path: Path
    tree: ast.Module
    top_level_defs: Dict[str, ast.AST]
    dehydration_ctx: DehydrationContext


def infer_module_name(file_path: Path, roots: Dict[str, Root]) -> Optional[str]:
    for root in roots.values():
        if root.kind != "package":
            continue

        try:
            rel = file_path.relative_to(root.path)
        except ValueError:
            continue

        if rel.name == "__init__.py":
            rel_clean = rel.parent
        else:
            rel_clean = rel.with_suffix("")
            
        if rel_clean == Path("."):
            return root.prefix.rstrip(".")

        dotted = ".".join(rel_clean.parts)
        return root.prefix + dotted

    return None


def build_module_context(
    file_path: Path,
    repo_root: Path,
    roots: Dict[str, Root],
) -> Optional[ModuleContext]:
    # Delegated to code_dehydrator
    tree = parse_module(file_path)
    if tree is None:
        return None

    module_name = infer_module_name(file_path, roots)

    try:
        file_rel_display = file_path.relative_to(repo_root)
    except ValueError:
        file_rel_display = file_path

    file_display = str(file_rel_display)

    import_map = build_import_map(tree, module_name)
    local_defs = build_local_defs(tree, module_name)

    dehydration_ctx = DehydrationContext(
        file_display=file_display,
        module_name=module_name,
        import_map=import_map,
        local_defs=local_defs
    )

    top_level_defs: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top_level_defs[node.name] = node

    return ModuleContext(
        file_path=file_path,
        tree=tree,
        top_level_defs=top_level_defs,
        dehydration_ctx=dehydration_ctx
    )


# =======================
# Dehydration / Output
# =======================

def tee_print(*args: object, **kwargs: Any) -> None:
    print(*args, **kwargs)
    if TRACE_FILE is not None:
        kw2 = dict(kwargs)
        kw2["file"] = TRACE_FILE
        print(*args, **kw2)


def dehydrate_node_wrapper(
    node: ast.AST,
    ctx: ModuleContext,
    target_name: str,
    pretty: bool
) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    dehydrator = Dehydrator()
    
    parent_name = None
    if isinstance(node, ast.FunctionDef) and "." in target_name:
        parent_name = target_name

    blocks = dehydrator.dehydrate_node(
        node, 
        ctx.dehydration_ctx, 
        parent_name=parent_name, 
        pretty=pretty
    )
    
    results = []
    for b in blocks:
        results.append((b.content, b.calls))
        
    return results


def print_final_report(grouped_results: Dict[str, List[str]], pretty: bool):
    """
    Prints the grouped results.
    """
    # Sort by file path for consistent output
    for file_display in sorted(grouped_results.keys()):
        blocks = grouped_results[file_display]
        tee_print(f"{{ FILE: {file_display}")
        for block_content in blocks:
            if pretty:
                for line in block_content.splitlines():
                    tee_print(f"  {line}")
            else:
                tee_print(f"  {block_content}")
        tee_print("}")


# =======================
# Target resolution & recursion helpers
# =======================

def resolve_target_to_file(target: str, roots: Dict[str, Root]) -> Optional[Path]:
    for root in roots.values():
        if root.kind != "package":
            continue

        if not target.startswith(root.prefix):
            continue

        rel = target[len(root.prefix):]
        parts = rel.split(".")

        for i in range(len(parts), 0, -1):
            mod_parts = parts[:i]
            base = root.path.joinpath(*mod_parts)

            candidate_py = base.with_suffix(".py")
            if candidate_py.is_file():
                if DEBUG: print(f"[DEBUG] Resolved {target} -> {candidate_py}")
                return candidate_py

            init_py = base / "__init__.py"
            if init_py.is_file():
                if DEBUG: print(f"[DEBUG] Resolved {target} -> {init_py} (__init__ fallback)")
                return init_py

    if DEBUG: print(f"[DEBUG] Could not resolve file for {target}")
    return None


def gather_call_targets(calls_structure: List[Tuple[str, Any]]) -> Set[str]:
    targets: Set[str] = set()

    def visit(items: List[Tuple[str, Any]]) -> None:
        for kind, payload in items:
            if kind == "call":
                name = str(payload)
                if name.startswith("<") and ">" in name:
                    type_part_end = name.find(">")
                    type_part = name[1:type_part_end]
                    rest = name[type_part_end + 2:]
                    canon = f"{type_part}.{rest}" if rest else type_part
                    targets.add(canon)
                else:
                    targets.add(name)
            elif kind in ("loop", "cond"):
                visit(payload)

    visit(calls_structure)
    return targets


def is_followable_target(target: str, roots: Dict[str, Root]) -> bool:
    for root in roots.values():
        if root.kind == "package" and target.startswith(root.prefix):
            return True
    return False


def process_one_target(
    target: str,
    repo_root: Path,
    roots: Dict[str, Root],
    pretty: bool,
    visited_aliases: Optional[Set[str]] = None  # <--- Added parameter
) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    
    # Initialize visited set for this resolution chain
    if visited_aliases is None:
        visited_aliases = set()

    # Cycle detection: If we've seen this target string in this chain, stop.
    if target in visited_aliases:
        if DEBUG:
            print(f"[DEBUG] Circular alias detected for {target}, breaking recursion.")
        return []
    
    visited_aliases.add(target)

    file_path = resolve_target_to_file(target, roots)
    if file_path is None:
        return []

    ctx = build_module_context(file_path, repo_root, roots)
    if ctx is None:
        return []

    # Use the moved helper
    node = find_ast_node(target, ctx.tree)
    
    if node is None:
        parts = target.split(".")
        name = parts[-1]
        parent = parts[-2] if len(parts) > 1 else None
        
        # Check Imports
        if name in ctx.dehydration_ctx.import_map:
            redirect_target = ctx.dehydration_ctx.import_map[name]
            # Pass visited_aliases to recursive call
            return process_one_target(redirect_target, repo_root, roots, pretty, visited_aliases)

        if parent and parent in ctx.dehydration_ctx.import_map:
            resolved_parent = ctx.dehydration_ctx.import_map[parent]
            redirect_target = f"{resolved_parent}.{name}"
            # Pass visited_aliases to recursive call
            return process_one_target(redirect_target, repo_root, roots, pretty, visited_aliases)
            
        # Check Inheritance
        if parent and parent in ctx.top_level_defs:
            parent_node = ctx.top_level_defs[parent]
            if isinstance(parent_node, ast.ClassDef):
                for base in parent_node.bases:
                    # Use the moved helper
                    raw_base = get_raw_call_name(base)
                    if not raw_base: continue
                    
                    if raw_base in ctx.dehydration_ctx.import_map:
                        qualified_base = ctx.dehydration_ctx.import_map[raw_base]
                    elif raw_base in ctx.dehydration_ctx.local_defs:
                         qualified_base = ctx.dehydration_ctx.local_defs[raw_base]
                    else:
                        qualified_base = raw_base

                    inherited_target = f"{qualified_base}.{name}"
                    # Pass visited_aliases to recursive call
                    result = process_one_target(inherited_target, repo_root, roots, pretty, visited_aliases)
                    if result:
                        return result
        return []

    return dehydrate_node_wrapper(node, ctx, target, pretty)

# =======================
# Main Logic
# =======================

def parse_cli_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("node_id", help="Pytest-style node id")
    parser.add_argument("-p", "--pretty", action="store_true", help="Pretty-print")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    return parser.parse_args(argv)


def resolve_entry_point(
    node_id: str, 
    roots: Dict[str, Root], 
    repo_root: Path
) -> Tuple[ModuleContext, ast.FunctionDef, str]:
    try:
        file_rel, cls_name, func_name = parse_node_id(node_id)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    file_path = resolve_node_file(file_rel, roots)
    
    if file_path is None:
        print(f"Test file not found: {file_rel}", file=sys.stderr)
        sys.exit(1)

    ctx = build_module_context(file_path, repo_root, roots)
    if ctx is None:
        sys.exit(1)

    # Use the moved helper
    func_node = get_test_function_node(ctx.tree, cls_name, func_name)
    
    if func_node is None:
        print(f"Test function not found: {func_name}", file=sys.stderr)
        sys.exit(1)

    function_name = build_function_qualified_name_from_module(
        ctx.dehydration_ctx.module_name, cls_name, func_name
    )
    
    return ctx, func_node, function_name


def run_recursion_loop(
    initial_targets: Set[str],
    ctx: ModuleContext,
    roots: Dict[str, Root],
    repo_root: Path,
    pretty: bool,
    debug: bool,
    grouped_results: Dict[str, List[str]]
) -> None:
    queue: List[str] = []
    visited: Set[str] = set()

    # Initialize Queue
    for t in sorted(initial_targets):
        if is_followable_target(t, roots):
            queue.append(t)
            continue
        if (ctx.dehydration_ctx.module_name is None 
            and "." not in t 
            and t in ctx.top_level_defs):
            queue.append(t)

    max_steps = 100
    steps = 0

    while queue and steps < max_steps:
        if debug:
            tee_print(f"\nQUEUE: {queue}")

        next_target = queue.pop(0)
        if next_target in visited:
            continue
        visited.add(next_target)

        if debug:
            tee_print(f"RECURSING: {next_target}")

        is_local_same_module = (
            ctx.dehydration_ctx.module_name is None
            and "." not in next_target
            and next_target in ctx.top_level_defs
        )

        results = []
        current_file_display = ""
        
        if is_local_same_module:
            node = ctx.top_level_defs[next_target]
            results = dehydrate_node_wrapper(node, ctx, next_target, pretty)
            current_file_display = ctx.dehydration_ctx.file_display
        else:
            fpath = resolve_target_to_file(next_target, roots)
            if fpath:
                try: 
                    current_file_display = str(fpath.relative_to(repo_root))
                except ValueError: 
                    current_file_display = str(fpath)
            
            results = process_one_target(next_target, repo_root, roots, pretty)

        if not results:
            continue

        for block_content, next_calls in results:
            if current_file_display not in grouped_results:
                grouped_results[current_file_display] = []
            
            grouped_results[current_file_display].append(block_content)

            new_targets = gather_call_targets(next_calls)
            for t in sorted(new_targets):
                if is_followable_target(t, roots):
                    if t not in visited and t not in queue:
                        queue.append(t)
                    continue
                if (ctx.dehydration_ctx.module_name is None and "." not in t and t in ctx.top_level_defs):
                    if t not in visited and t not in queue:
                        queue.append(t)

        steps += 1


def main(argv=None) -> None:
    args = parse_cli_args(argv)
    
    global DEBUG
    DEBUG = args.debug

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    roots = build_roots(repo_root)

    trace_dir = script_path.parent / "trace_history"
    trace_dir.mkdir(exist_ok=True)

    raw_name = args.node_id.replace(os.sep, "_").replace("::", "__")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_name)
    trace_path = trace_dir / f"{safe_name}.txt"

    global TRACE_FILE
    TRACE_FILE = trace_path.open("w", encoding="utf-8")

    grouped_results: Dict[str, List[str]] = {}

    try:
        ctx, func_node, function_name = resolve_entry_point(args.node_id, roots, repo_root)

        results = dehydrate_node_wrapper(func_node, ctx, function_name, args.pretty)

        if not results:
            tee_print(f"No logical calls found in: {function_name}")
            return

        for block_content, calls_structure in results:
            file_display = ctx.dehydration_ctx.file_display
            if file_display not in grouped_results:
                grouped_results[file_display] = []
            grouped_results[file_display].append(block_content)

            all_targets = gather_call_targets(calls_structure)
            
            run_recursion_loop(
                initial_targets=all_targets,
                ctx=ctx,
                roots=roots,
                repo_root=repo_root,
                pretty=args.pretty,
                debug=args.debug,
                grouped_results=grouped_results
            )
        
        print_final_report(grouped_results, args.pretty)

    finally:
        if TRACE_FILE is not None:
            TRACE_FILE.close()
            TRACE_FILE = None

if __name__ == "__main__":
    main()